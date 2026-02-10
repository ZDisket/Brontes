import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm
import torchaudio
from einops import rearrange

try:
    from .utils import get_padding
except ImportError:
    from utils import get_padding

LRELU_SLOPE = 0.1


class SEBlock1D(nn.Module):
    """
    Lightweight Squeeze-Excite attention.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBlock2D(nn.Module):
    """
    Squeeze-Excite attention for 2D convolutions.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, use_se_blocks=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_se_blocks = use_se_blocks
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        
        # Create conv layers
        conv_layers = [
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ]
        self.convs = nn.ModuleList(conv_layers)
        
        # Create single SE block if enabled (at the end)
        if self.use_se_blocks:
            self.se_block = SEBlock2D(1024)  # After the last conv layer
        else:
            self.se_block = None
        
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        # Apply convolutions
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        
        # Apply SE block at the end if enabled
        if self.use_se_blocks and self.se_block is not None:
            x = self.se_block(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_se_blocks=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(5, use_se_blocks=use_se_blocks),
            DiscriminatorP(7, use_se_blocks=False),
            DiscriminatorP(11, use_se_blocks=use_se_blocks),
            DiscriminatorP(17, use_se_blocks=False),
            DiscriminatorP(23, use_se_blocks=use_se_blocks),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, use_se_blocks=False):
        super(DiscriminatorS, self).__init__()
        self.use_se_blocks = use_se_blocks
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        
        # Create conv layers
        conv_layers = [
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ]
        self.convs = nn.ModuleList(conv_layers)
        
        # Create single SE block if enabled (at the end)
        if self.use_se_blocks:
            self.se_block = SEBlock1D(1024)   # After the last conv layer
        else:
            self.se_block = None
        
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        
        # Apply convolutions
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        
        # Apply SE block at the end if enabled
        if self.use_se_blocks and self.se_block is not None:
            x = self.se_block(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_se_blocks=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=False, use_se_blocks=False),
            DiscriminatorS(use_se_blocks=use_se_blocks),
            DiscriminatorS(use_se_blocks=False),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    spectral = kwargs.pop("spectral", False)

    conv = nn.Conv2d(*args, **kwargs)

    # Apply spectral norm OR weight norm, not both
    if spectral:
        conv = spectral_norm(conv)
    else:
        conv = weight_norm(conv)

    if not act:
        return conv

    return nn.Sequential(conv, nn.LeakyReLU(0.1))



DEFAULT_BANDS = [
    (0.0, 0.1),
    (0.1, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
]

class MultiBandSpecDiscriminator(nn.Module):
    def __init__(
        self,
        window_length,
        hop_factor=0.25,
        sample_rate=44100,
        bands=None,
        audio_channels=1,
        log_eps=1e-5,
        phase_gate_bias=-5.0,
        phase_gate_scale=2.0,
        log_mag_only=False,
    ):
        """
        Multi-band spectrogram discriminator using log-magnitude and masked phase.

        Uses log-magnitude + sin/cos phase representation instead of raw real/imag.
        Phase is weighted by a sigmoid gate based on magnitude, so low-energy
        frequency bins don't contribute noisy phase gradients.

        Parameters
        ----------
        window_length : int
            STFT window length (n_fft).
        hop_factor : float
            Hop factor as a fraction of window_length (e.g. 0.25 -> hop = window_length * 0.25).
        sample_rate : int
            Audio sample rate in Hz. Stored for reference.
        bands : list of (float, float), optional
            Frequency band boundaries as fractions of [0, 1] over the frequency axis.
            Each tuple is (start_frac, end_frac), non-overlapping and in ascending order.
            Defaults to DEFAULT_BANDS.
        audio_channels : int
            Number of audio channels in the input waveform (C in (B, C, T)).
        log_eps : float
            Epsilon added before log to avoid log(0).
        phase_gate_bias : float
            Bias for the sigmoid phase gate (controls threshold for phase masking).
        phase_gate_scale : float
            Scale for the sigmoid phase gate (controls sharpness).
        log_mag_only : bool
            If True, use only log-magnitude (1 channel per audio channel).
            If False, use log-magnitude + masked sin/cos phase (3 channels per audio channel).
        """
        super().__init__()

        if bands is None:
            bands = DEFAULT_BANDS

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.hop_length = int(window_length * hop_factor)
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.log_eps = log_eps
        self.phase_gate_bias = phase_gate_bias
        self.phase_gate_scale = phase_gate_scale
        self.log_mag_only = log_mag_only

        # Complex spectrogram (STFT) transform.
        # Input:  (B, C, T)
        # Output: complex tensor (B, C, F, T_frames)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=window_length,
            win_length=window_length,
            hop_length=self.hop_length,
            power=None,          # complex output
            center=True,
            pad_mode="reflect",
        )

        # One-sided STFT: n_freq = n_fft // 2 + 1
        n_freq = window_length // 2 + 1

        # Convert fractional bands [0,1] into integer bin ranges [start_bin, end_bin)
        band_bins = []
        for start_frac, end_frac in bands:
            start_bin = int(start_frac * n_freq)
            end_bin = int(end_frac * n_freq)
            # Ensure at least 1 bin per band
            if end_bin <= start_bin:
                end_bin = start_bin + 1
            # Clip to valid range just in case
            start_bin = max(0, min(start_bin, n_freq - 1))
            end_bin = max(1, min(end_bin, n_freq))
            band_bins.append((start_bin, end_bin))
        self.bands = band_bins  # list of (start_bin, end_bin)

        ch = 32
        # 1 channel per audio channel if log_mag_only, else 3 (log_mag, sin_phase, cos_phase)
        in_channels = audio_channels if log_mag_only else 3 * audio_channels

        def make_conv_stack():
            return nn.ModuleList(
                [
                    # Input channels = 2 * C (real/imag across all audio channels).
                    WNConv2d(in_channels, ch, (3, 9), (1, 1), padding=(1, 4), spectral=False),
                    WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
                ]
            )

        # One conv stack per frequency band
        self.band_convs = nn.ModuleList([make_conv_stack() for _ in range(len(self.bands))])

        # Final conv after concatenating all band outputs along frequency axis
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        """
        Compute STFT with log-magnitude and optionally masked sin/cos phase, split into frequency bands.

        Parameters
        ----------
        x : torch.Tensor
            Waveform of shape (B, C, T), values in [-1, 1].

        Returns
        -------
        x_bands : list of torch.Tensor
            Each element has shape (B, N*C, T_frames, F_band),
            where N=1 if log_mag_only else N=3 (log_mag, masked_sin_phase, masked_cos_phase per audio channel).
        """
        if x.dim() != 3:
            raise ValueError("Expected input x with shape (B, C, T)")

        B, C, T = x.shape
        if C != self.audio_channels:
            raise ValueError(
                "Input has {} channels but MRD was initialized with audio_channels={}".format(
                    C, self.audio_channels
                )
            )

        # STFT: (B, C, F, T_frames), complex dtype
        spec_complex = self.spectrogram_transform(x)

        # Compute magnitude
        mag = torch.abs(spec_complex)  # (B, C, F, T_frames)

        # Log-magnitude with log1p for numerical stability
        log_mag = torch.log1p(mag)  # (B, C, F, T_frames)

        if self.log_mag_only:
            # Use only log-magnitude: (B, C, F, T) -> (B, C, T, F)
            spec = rearrange(log_mag, "b c f t -> b c t f")
        else:
            # Compute phase and apply magnitude-based gating
            phase = torch.angle(spec_complex)  # (B, C, F, T_frames)

            # Compute sin/cos of phase
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)

            # Magnitude-based gate: suppress phase where magnitude is low
            # sigmoid((log_mag + bias) * scale) -> ~0 for low mag, ~1 for high mag
            log_mag_for_gate = torch.log(mag + self.log_eps)
            phase_gate = torch.sigmoid((log_mag_for_gate + self.phase_gate_bias) * self.phase_gate_scale)
            sin_phase = sin_phase * phase_gate
            cos_phase = cos_phase * phase_gate

            # Stack channels: (B, C, F, T, 3) -> (B, 3*C, T, F)
            spec = torch.stack([log_mag, sin_phase, cos_phase], dim=-1)  # (B, C, F, T, 3)
            spec = rearrange(spec, "b c f t ch -> b (c ch) t f")

        # Split along the frequency dimension into bands
        x_bands = []
        for start_bin, end_bin in self.bands:
            band = spec[..., start_bin:end_bin]  # (B, N*C, T_frames, F_band)
            x_bands.append(band)

        return x_bands

    def forward(self, x):
        """
        Forward pass of the multi-band discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Waveform with shape (B, C, T).

        Returns
        -------
        fmap : list of torch.Tensor
            Feature maps from all intermediate layers and the final output,
            typically used for GAN feature matching losses.
        """
        x_bands = self.spectrogram(x)

        fmap = []
        band_outputs = []

        # Process each frequency band with its own conv stack
        for band, conv_stack in zip(x_bands, self.band_convs):
            # band: (B, 2*C, T_frames, F_band)
            for layer in conv_stack:
                band = layer(band)
                fmap.append(band)
            band_outputs.append(band)

        # Concatenate final features from all bands along frequency axis
        # band_outputs[i]: (B, ch, T_frames, F_band_i)
        # x_cat: (B, ch, T_frames, sum(F_band_i))
        x_cat = torch.cat(band_outputs, dim=-1)

        # Final conv
        out = self.conv_post(x_cat)
        fmap.append(out)

        return fmap



class Discriminator(nn.Module):
    """
    Unified Discriminator that combines MultiPeriodDiscriminator, 
    MultiScaleDiscriminator, and MultiBandSpecDiscriminator.
    
    All three discriminators are run in a single forward pass and their 
    outputs (scores and feature maps) are combined into flat lists.
    """
    
    def __init__(
        self,
        use_se_blocks=False,
        # MultiBandSpecDiscriminator params
        mbsd_window_lengths=[2048, 1024, 512],
        mbsd_hop_factor=0.25,
        sample_rate=44100,
        bands=None,
        audio_channels=1,
        # Enable/disable individual discriminators
        enable_mpd=True,
        enable_msd=True,
        enable_mbsd=True,
        # Instance noise for stabilizing training
        instance_noise_std=0.0,
    ):
        """
        Initialize the unified discriminator.
        
        Parameters
        ----------
        use_se_blocks : bool
            Whether to use Squeeze-Excite blocks in MPD and MSD.
        mbsd_window_lengths : list of int
            Window lengths for each MultiBandSpecDiscriminator instance.
        mbsd_hop_factor : float
            Hop factor for MultiBandSpecDiscriminator.
        sample_rate : int
            Audio sample rate in Hz.
        bands : list of (float, float), optional
            Frequency band boundaries for MultiBandSpecDiscriminator.
        audio_channels : int
            Number of audio channels.
        enable_mpd : bool
            Whether to enable Multi-Period Discriminator.
        enable_msd : bool
            Whether to enable Multi-Scale Discriminator.
        enable_mbsd : bool
            Whether to enable Multi-Band Spectral Discriminators.
        instance_noise_std : float
            Standard deviation of Gaussian noise to add to discriminator inputs
            during training. Helps prevent discriminator overpowering. Set to 0
            to disable. Typical values: 0.01 to 0.1.
        """
        super().__init__()
        
        # Store enable flags
        self.enable_mpd = enable_mpd
        self.enable_msd = enable_msd
        self.enable_mbsd = enable_mbsd
        
        # Multi-Period Discriminator
        self.mpd = MultiPeriodDiscriminator(use_se_blocks=use_se_blocks) if enable_mpd else None
        
        # Multi-Scale Discriminator
        self.msd = MultiScaleDiscriminator(use_se_blocks=use_se_blocks) if enable_msd else None
        
        # Multi-Resolution/Multi-Band Spectral Discriminators (one per window length)
        if enable_mbsd:
            self.mbsds = nn.ModuleList([
                MultiBandSpecDiscriminator(
                    window_length=wl,
                    hop_factor=mbsd_hop_factor,
                    sample_rate=sample_rate,
                    bands=bands,
                    audio_channels=audio_channels,
                )
                for wl in mbsd_window_lengths
            ])
        else:
            self.mbsds = None
        
        # Instance noise standard deviation
        self.instance_noise_std = instance_noise_std
    
    def forward(self, y, y_hat):
        """
        Run all discriminators on real (y) and generated (y_hat) audio.
        
        Parameters
        ----------
        y : torch.Tensor
            Real audio of shape (B, C, T).
        y_hat : torch.Tensor
            Generated audio of shape (B, C, T).
        
        Returns
        -------
        y_d_rs : list of torch.Tensor
            Discriminator outputs for real audio from all discriminators.
        y_d_gs : list of torch.Tensor
            Discriminator outputs for generated audio from all discriminators.
        fmap_rs : list of list of torch.Tensor
            Feature maps for real audio from all discriminators.
        fmap_gs : list of list of torch.Tensor
            Feature maps for generated audio from all discriminators.
        """
        # Apply instance noise during training to stabilize G/D balance
        if self.training and self.instance_noise_std > 0:
            noise_real = torch.randn_like(y) * self.instance_noise_std
            noise_fake = torch.randn_like(y_hat) * self.instance_noise_std
            y = y + noise_real
            y_hat = y_hat + noise_fake
        
        # Collect all outputs
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        # Multi-Period Discriminator
        if self.mpd is not None:
            mpd_y_d_rs, mpd_y_d_gs, mpd_fmap_rs, mpd_fmap_gs = self.mpd(y, y_hat)
            y_d_rs.extend(mpd_y_d_rs)
            y_d_gs.extend(mpd_y_d_gs)
            fmap_rs.extend(mpd_fmap_rs)
            fmap_gs.extend(mpd_fmap_gs)
        
        # Multi-Scale Discriminator
        if self.msd is not None:
            msd_y_d_rs, msd_y_d_gs, msd_fmap_rs, msd_fmap_gs = self.msd(y, y_hat)
            y_d_rs.extend(msd_y_d_rs)
            y_d_gs.extend(msd_y_d_gs)
            fmap_rs.extend(msd_fmap_rs)
            fmap_gs.extend(msd_fmap_gs)
        
        # Multi-Band Spectral Discriminators
        if self.mbsds is not None:
            for mbsd in self.mbsds:
                # MBSD returns only feature maps (no separate real/fake outputs per sub-disc)
                # We process each separately and combine
                fmap_r = mbsd(y)
                fmap_g = mbsd(y_hat)
                
                # For MBSD, we treat the final feature map as the discriminator output
                # (it contains the prediction), and all maps including it as feature maps
                y_d_rs.append(fmap_r[-1].flatten(1, -1))  # Final output flattened
                y_d_gs.append(fmap_g[-1].flatten(1, -1))
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



def auto_slice_2nd_dim(tens1, tens2):
    if tens1.size(2) > tens2.size(2):
        tens1 = tens1[:,:,:tens2.size(2)]
    if tens2.size(2) > tens1.size(2):
        tens2 = tens2[:,:,:tens1.size(2)]

    return tens1, tens2
    

def feature_loss(fmap_r, fmap_g):
    loss = 0
    count = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl, gl = auto_slice_2nd_dim(rl, gl)
            loss += torch.mean(torch.abs(rl - gl))
            count += 1

    return loss / count


class DiscriminatorLoss(nn.Module):
    """Unified discriminator/generator loss supporting hinge and LSGAN.
    
    Args:
        loss_type: 'hinge' or 'lsgan'
    """
    def __init__(self, loss_type='hinge'):
        super().__init__()
        assert loss_type in ('hinge', 'lsgan'), f"loss_type must be 'hinge' or 'lsgan', got {loss_type}"
        self.loss_type = loss_type
    
    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        """Compute discriminator loss for real and generated outputs."""
        loss = 0
        r_losses = []
        g_losses = []
        
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            if self.loss_type == 'hinge':
                # Hinge loss: max(0, 1 - real) + max(0, 1 + fake)
                r_loss = torch.mean(F.relu(1.0 - dr))
                g_loss = torch.mean(F.relu(1.0 + dg))
            else:  # lsgan
                # LSGAN: (real - 1)^2 + fake^2
                r_loss = torch.mean((1 - dr) ** 2)
                g_loss = torch.mean(dg ** 2)
            
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())
        
        return loss, r_losses, g_losses
    
    def generator_loss(self, disc_outputs):
        """Compute generator loss for discriminator outputs."""
        loss = 0
        gen_losses = []
        
        for dg in disc_outputs:
            if self.loss_type == 'hinge':
                # Hinge loss for generator: -fake (maximize fake score)
                l = -torch.mean(dg)
            else:  # lsgan
                # LSGAN: (fake - 1)^2 (want fake to look real)
                l = torch.mean((1 - dg) ** 2)
            
            gen_losses.append(l)
            loss += l
        
        return loss, gen_losses


# Legacy function wrappers for backward compatibility (uses LSGAN by default)
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Legacy LSGAN discriminator loss function."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Legacy LSGAN generator loss function."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
