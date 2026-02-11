import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

try:
    from .attention import CBAM1d, ChannelAttention1d
    from .istft_head import iSTFTHead
except ImportError:
    from attention import CBAM1d, ChannelAttention1d
    from istft_head import iSTFTHead

# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

# -----------------------------
# Utility factories
# -----------------------------

def wn_conv1d(in_ch, out_ch, k=7, s=1, p=None, d=1, groups=1, bias=True, norm='gn'):
    if p is None:
        p = ((k - 1) // 2) * d
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, groups=groups, bias=bias)
    if norm is None or norm == 'none':
        return conv
    return weight_norm(conv)

def wn_deconv1d(in_ch, out_ch, k=4, s=2, p=1, bias=True, norm='gn'):
    deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
    if norm is None or norm == 'none':
        return deconv
    return weight_norm(deconv)

def norm_1d(ch, kind='gn', groups=8):
    if kind is None or kind == 'none':
        return nn.Identity()
    if kind == 'bn':
        return nn.BatchNorm1d(ch)
    if kind == 'gn':
        g = min(groups, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, ch)
    if kind == 'ln':
        # channel-last normalization is more common for LN, but we keep simple LN over channels
        return nn.GroupNorm(1, ch)
    return nn.Identity()

def act_fn(name='lrelu', channels=None):
    if name == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'silu':
        return nn.SiLU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    if name == 'snake':
        if channels is None:
            raise ValueError("Channels must be specified for Snake activation")
        return Snake1d(channels)
    return nn.LeakyReLU(0.2, inplace=True)

# -----------------------------
# Blocks
# -----------------------------

class AttentionGate1d(nn.Module):
    def __init__(self, ch_g, ch_x, ch_int=None):
        super().__init__()
        if ch_int is None:
            ch_int = max(1, min(ch_g, ch_x) // 2)
        self.W_g = nn.Conv1d(ch_g, ch_int, 1, bias=True)
        self.W_x = nn.Conv1d(ch_x, ch_int, 1, bias=True)
        self.psi  = nn.Conv1d(ch_int, 1, 1, bias=True)
        self.act  = nn.ReLU(inplace=True)
        self.sig  = nn.Sigmoid()

    def forward(self, x, g):
        a = self.W_g(g) + self.W_x(x)
        a = self.sig(self.psi(self.act(a)))
        return x * a

class DepthwiseSeparableConv1d(nn.Module):
    # Optional: efficient conv block (depthwise + pointwise), both with weight norm
    def __init__(self, ch_in, ch_out, k=7, s=1, d=1, norm='gn', act='lrelu'):
        super().__init__()
        p = ((k - 1) // 2) * d
        self.dw = wn_conv1d(ch_in, ch_in, k=k, s=s, p=p, d=d, groups=ch_in, bias=True, norm=norm)
        self.pw = wn_conv1d(ch_in, ch_out, k=1, s=1, p=0, d=1, groups=1, bias=True, norm=norm)
        self.n1 = norm_1d(ch_in, norm)
        self.n2 = norm_1d(ch_out, norm)
        self.a1 = act_fn(act, ch_in)
        self.a2 = act_fn(act, ch_out)

    def forward(self, x):
        x = self.a1(self.n1(self.dw(x)))
        x = self.a2(self.n2(self.pw(x)))
        return x

class DoubleConv1d(nn.Module):
    # Standard 2x conv block with weight norm
    def __init__(self, ch_in, ch_out, k=7, d1=1, d2=1, norm='gn', act='lrelu', separable=False, residual=False):
        super().__init__()
        if separable:
            self.c1 = DepthwiseSeparableConv1d(ch_in, ch_out, k=k, s=1, d=d1, norm=norm, act=act)
            self.c2 = DepthwiseSeparableConv1d(ch_out, ch_out, k=k, s=1, d=d2, norm=norm, act=act)
        else:
            p1 = ((k - 1) // 2) * d1
            p2 = ((k - 1) // 2) * d2
            self.c1 = wn_conv1d(ch_in, ch_out, k=k, s=1, p=p1, d=d1, norm=norm)
            self.c2 = wn_conv1d(ch_out, ch_out, k=k, s=1, p=p2, d=d2, norm=norm)
            self.n1 = norm_1d(ch_out, norm)
            self.n2 = norm_1d(ch_out, norm)
            self.a1 = act_fn(act, ch_out)
            self.a2 = act_fn(act, ch_out)
        self.separable = separable
        self.residual = residual
        if not separable:
            self.block = None

    def forward(self, x):
        if self.separable:
            x1 = self.c1(x)
            x2 = self.c2(x1)
            if self.residual:
                min_len = min(x1.size(-1), x2.size(-1))
                return x1[..., :min_len] + x2[..., :min_len]
            return x2
        
        x1 = self.a1(self.n1(self.c1(x)))
        x2 = self.a2(self.n2(self.c2(x1)))
        if self.residual:
            min_len = min(x1.size(-1), x2.size(-1))
            return x1[..., :min_len] + x2[..., :min_len]
        return x2

class DownBlock1d(nn.Module):
    # Returns (x_down, skip)
    def __init__(self, ch_in, ch_out, k=7, norm='gn', act='lrelu', separable=False, channel_attn="se", stride=2, residual=False):
        super().__init__()
        self.pre = DoubleConv1d(ch_in, ch_out, k=k, d1=1, d2=1, norm=norm, act=act, separable=separable, residual=residual)
        # stride downsample with weight-normed conv
        # consistent with k=4, s=2, p=1: k = 2*s, p = s // 2
        self.down = wn_conv1d(ch_out, ch_out, k=2*stride, s=stride, p=stride//2, d=1, norm=norm)
        self.channel_attn = channel_attn

        if self.channel_attn is None:
            self.ca_block = nn.Identity()
        if self.channel_attn == "se":
            self.ca_block = ChannelAttention1d(ch_out, 16)
        elif self.channel_attn == "cbam":
            self.ca_block = CBAM1d(ch_out,16,7)

    def forward(self, x):
        skip = self.pre(x)

        skip = self.ca_block(skip)

        x = self.down(skip)
        return x, skip




class UpBlock1d(nn.Module):
    # Takes x (low-res) and skip, upsamples then fuses
    def __init__(self, ch_in, ch_skip, ch_out, k=7, norm='gn', act='lrelu', separable=False, use_deconv=True, channel_attn="se", use_attn_gate = False, stride=2, use_skip=True):
        super().__init__()
        self.use_deconv = use_deconv
        self.channel_attn = channel_attn
        self.use_attn_gate = use_attn_gate
        self.use_skip = use_skip

        if use_deconv:
            # consistent with k=4, s=2, p=1: k = 2*s, p = s // 2
            self.up = wn_deconv1d(ch_in, ch_out, k=2*stride, s=stride, p=stride//2, norm=norm)
        else:
            self.up = nn.Upsample(scale_factor=stride, mode='linear', align_corners=False)
            self.proj = wn_conv1d(ch_in, ch_out, k=1, s=1, p=0, d=1, norm=norm)

        # attention gate is owned by this block
        if use_attn_gate and use_skip:
            self.attn_gate = AttentionGate1d(ch_g=ch_out, ch_x=ch_skip)
        else:
            self.attn_gate = nn.Identity()
        self.fuse = wn_conv1d(ch_out, ch_out, k=3, p=1, norm=norm)
        self.afuse = act_fn(act, ch_out)


        self.glu_proj = wn_conv1d(
            ch_out,
            2 * ch_out,
            k=1, s=1, p=0,
            norm=None
        )
        # GLU initialization
        nn.init.constant_(self.glu_proj.bias[ch_out:], -2.0)

        if self.channel_attn is None:
            self.ca_block = nn.Identity()
        if self.channel_attn == "se":
            self.ca_block = ChannelAttention1d(ch_out, 16)
        elif self.channel_attn == "cbam":
            self.ca_block = CBAM1d(ch_out,16,7)


    def forward(self, x, skip=None):
        if self.use_deconv:
            x = self.up(x)
        else:
            x = self.proj(self.up(x))
        
        x = self.ca_block(x)

        if self.use_skip and skip is not None:
            # center-crop or pad if needed for odd lengths
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :skip.size(-1)]

            if self.use_attn_gate:
                skip = self.attn_gate(skip, x)

            x = x + skip

        # GLU gate (Demucs-style)
        a, b = self.glu_proj(x).chunk(2, dim=1)
        x = a * torch.sigmoid(b)

        res = self.fuse(x)
        res = self.afuse(res)
        x = x + res
        return x

class Bottleneck1d(nn.Module):
    def __init__(self, ch, k=7, dilations=(1, 2, 4, 8), norm='gn', act='lrelu', separable=False, residual=False):
        super().__init__()
        blocks = []
        for d in dilations:
            blocks.append(DoubleConv1d(ch, ch, k=k, d1=d, d2=1, norm=norm, act=act, separable=separable, residual=residual))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class LSTMBottleneck1d(nn.Module):
    def __init__(self, ch, layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(ch, ch, num_layers=layers, bidirectional=bidirectional, batch_first=True)
        self.proj = nn.Linear(2 * ch if bidirectional else ch, ch)

    def forward(self, x):
        # x: [B, C, T] -> LSTM (batch_first=True) expects [B, T, C]
        B, C, T = x.shape
        x = x.transpose(1, 2) # [B, T, C]
        x, _ = self.lstm(x)
        x = self.proj(x)
        x = x.transpose(1, 2) # [B, C, T]
        return x





# -----------------------------
# U-Net Refiner
# -----------------------------

class UNet1DRefiner(nn.Module):
    """
    General 1D U-Net for feature refinement.
    Expects input shape [B, in_ch, T].
    Outputs refined features of shape [B, out_ch, T].
    Auto-pads to a multiple of 2**depth during forward and unpads at the end.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        base_ch=64,
        depth=4,
        ch_mults=None,         # e.g., [1, 2, 4, 8]; if None, will be [1, 2, 4, ..., 2**(depth-1)]
        k=7,
        norm='gn',
        act='lrelu',
        separable=False,
        residual=False,
        use_deconv=True,
        bottleneck_dilations=(1, 2, 4, 8),
        learnable_alpha=True,
        alpha_init=0.1,
        decoder_k=None,
        stride=2,
        # Bottleneck settings
        use_lstm_bottleneck=False,
        lstm_layers=2,
        lstm_bidirectional=False,
        # iSTFT output head parameters
        use_istft_head=False,
        istft_n_fft=2048,
        istft_hop_length=512,
        istft_win_length=None,
        phase_eps=1e-8,
        skip_layer_indexes=None,
        skip_residual_scales=None,
    ):
        super().__init__()
        self.stride = stride
        if ch_mults is None:
            ch_mults = [2**i for i in range(depth)]
        assert len(ch_mults) == depth

        # Encoder
        self.enc = nn.ModuleList()
        self.decoder_k = k

        if decoder_k is not None:
            if decoder_k > 1:
                print(f"Using decoder k: {decoder_k}")
                self.decoder_k = decoder_k

        prev_ch = in_ch
        skips_ch = []

        ch_attns = [None] * len(ch_mults)
       # ch_attns = ["cbam"] * len(ch_mults)
       # ch_attns[0] = "cbam"
        #ch_attns[(len(ch_attns) // 2) - 1] = "cbam"
        #ch_attns[-1] = "cbam"

        for idx, m in enumerate(ch_mults):
            ch_out = base_ch * m
            self.enc.append(DownBlock1d(prev_ch, ch_out, k=k, norm=norm, act=act, separable=separable, channel_attn=ch_attns[idx], stride=stride, residual=residual))
            skips_ch.append(ch_out)
            prev_ch = ch_out

        # Bottleneck
        if use_lstm_bottleneck:
            self.bot = LSTMBottleneck1d(prev_ch, layers=lstm_layers, bidirectional=lstm_bidirectional)
        else:
            self.bot = Bottleneck1d(prev_ch, k=k, dilations=bottleneck_dilations, norm=norm, act=act, separable=separable, residual=residual)

        # Decoder (reverse channel schedule)
        self.dec = nn.ModuleList()
        rev_mults = list(reversed(ch_mults))
        rev_skips = list(reversed(skips_ch))
        ch_in = prev_ch

        if skip_layer_indexes is None:
            self.skip_layer_indexes = list(range(depth))
        else:
            # Handle negative indexing by mapping to positive indexes
            self.skip_layer_indexes = []
            for idx in skip_layer_indexes:
                if idx < 0:
                    idx = depth + idx
                self.skip_layer_indexes.append(idx)

        # Skip residual scales
        if skip_residual_scales is None:
            self.skip_scales = {idx: 1.0 for idx in self.skip_layer_indexes}
        else:
            if len(skip_residual_scales) != len(self.skip_layer_indexes):
                raise ValueError(f"Length of skip_residual_scales ({len(skip_residual_scales)}) must match length of skip_layer_indexes ({len(self.skip_layer_indexes)})")
            self.skip_scales = {idx: scale for idx, scale in zip(self.skip_layer_indexes, skip_residual_scales)}

        for idx, m in enumerate(rev_mults):
            # idx in decoder goes 0..depth-1
            # corresponding encoder idx is (depth - 1 - idx)
            enc_idx = depth - 1 - idx
            use_this_skip = enc_idx in self.skip_layer_indexes
            
            ch_skip = rev_skips[idx]
            ch_out = base_ch * m
            self.dec.append(UpBlock1d(
                ch_in, ch_skip, ch_out, 
                k=self.decoder_k, norm=norm, act=act, separable=separable, 
                use_deconv=use_deconv, channel_attn=ch_attns[idx], stride=stride,
                use_skip=use_this_skip, use_attn_gate=True,
            ))
            ch_in = ch_out

        # Head - either waveform or iSTFT
        self.use_istft_head = use_istft_head
        if use_istft_head:
            self.head = iSTFTHead(
                in_channels=ch_in,
                n_fft=istft_n_fft,
                hop_length=istft_hop_length,
                win_length=istft_win_length,
                phase_eps=phase_eps,
                use_weight_norm=(norm != 'none' and norm is not None),
            )
        else:
            self.head = wn_conv1d(ch_in, out_ch, k=1, s=1, p=0, d=1, norm=norm)
        
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))
        
        # Lightweight init
        self.apply(self._init_weights)

        self.noise_alpha = 0.003

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pad_to_factor(self, x, factor):
        T = x.size(-1)
        pad = (factor - (T % factor)) % factor
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x, T

    def forward(self, x):
        # x: [B, in_ch, T]
        factor = self.stride ** len(self.enc)
        x_padded, orig_T = self._pad_to_factor(x, factor)

        # Encoder path
        skips = []
        x_current = x_padded



        for down in self.enc:
            x_current, skip = down(x_current)
            skips.append(skip)

        # Bottleneck
        x_current = self.bot(x_current)

        # Decoder path
        for idx, up in enumerate(self.dec):
            skip = skips.pop()
            enc_idx = len(self.enc) - 1 - idx
            if enc_idx in self.skip_layer_indexes:
                skip = skip * self.skip_scales.get(enc_idx, 1.0)
                x_current = up(x_current, skip)
            else:
                x_current = up(x_current, None)

        # Output head
        if self.use_istft_head:
            # iSTFT synthesis: pass features and target length to get waveform
            refined = self.head(x_current, target_length=orig_T)
            refined = torch.clamp(refined, -1, 1)
        else:
            # Original waveform output path
            out = self.head(x_current)
            out = torch.clamp(out, -1, 1)
            
            # Ensure output length matches input length after padding adjustments
            if out.size(-1) != x_padded.size(-1):
                # This should ideally not happen with proper padding, but as a safeguard:
                if out.size(-1) > x_padded.size(-1):
                    out = out[..., :x_padded.size(-1)]
                else:
                    # This case indicates a potential issue in the architecture design
                    # or padding logic, but we pad with zeros for robustness.
                    pad_len = x_padded.size(-1) - out.size(-1)
                    out = F.pad(out, (0, pad_len))
            
            refined = out
            
            # Unpad to original length if necessary
            if refined.size(-1) != orig_T:
                refined = refined[..., :orig_T]

        return refined

    def remove_weight_norm_(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                try:
                    remove_weight_norm(m)
                except Exception:
                    pass
        return self
