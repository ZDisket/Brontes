import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pesto import load_model as load_pesto_model


def cosine_feat_loss(f_act, r_act, w):
    # f_act, r_act: [B,T,C] or [B,C,T]
    # w: [B,T] weights (can include V/UV mask)

    # Ensure float32 (helps under mixed precision)
    f_act = f_act.float()
    r_act = r_act.float()
    w = w.float()

    # If activations are [B,T,C], make them [B,C,T]
    if f_act.dim() == 3 and f_act.size(1) == w.size(1):
        f_act = f_act.transpose(1, 2)
    if r_act.dim() == 3 and r_act.size(1) == w.size(1):
        r_act = r_act.transpose(1, 2)

    # Now features are [B,C,T]. Align time dim to w.size(1)
    T = w.size(1)
    if f_act.size(2) != T:
        f_act = F.interpolate(f_act, size=T, mode='nearest')
    if r_act.size(2) != T:
        r_act = F.interpolate(r_act, size=T, mode='nearest')

    # Channel-wise normalization, cosine distance per time step
    f_norm = F.normalize(f_act, dim=1)
    r_norm = F.normalize(r_act, dim=1)
    cos_dist = 1.0 - (f_norm * r_norm).sum(dim=1)  # [B,T]

    # Weight and reduce
    return (w * cos_dist).sum() / (w.sum() + 1e-8)


def charbonnier_loss(prediction, target, eps=1e-6):
    """Compute the Charbonnier loss (L1 + epsilon^2)"""
    diff = prediction - target
    loss = torch.mean(torch.sqrt(diff * diff + eps))
    return loss


_STFT_WINDOW_CACHE = {}


def _get_hann(win_length, device, dtype):
    key = (win_length, str(device), str(dtype))
    w = _STFT_WINDOW_CACHE.get(key)
    if w is None:
        w = torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
        _STFT_WINDOW_CACHE[key] = w
    return w


def _stft_mag(x, n_fft, hop_length, win_length):
    # x: [B,T], real
    window = _get_hann(win_length, x.device, x.dtype)
    X = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True
    )
    return torch.abs(X)  # [B,F,T]


def _safe_eps_for_dtype(dtype):
    # Use a floor that stays non-zero in halves, but small in fp32
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-4
    return 1e-7


class MultiScaleMelLoss(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        win_lengths=(32, 64, 128, 256, 512, 1024, 2048),
        n_mels=(5, 10, 20, 40, 80, 160, 320),
        hop_divisor=4,                        # hop = win_length // hop_divisor
        f_min=0.0,
        f_max=None,                           # default to sr/2 if None
        power=1.0,                            # stft power for mel (common: 2.0)
        log_mel=True,                         # log mel is standard for TTS/music
        log_eps=1e-5,
        loss_mode="l1",                       # "l1", "l1+l2", or "charbonnier"
        l2_weight=1.0,                        # only used if loss_mode == "l1+l2"
        charbonnier_eps=1e-6,                 # only used if loss_mode == "charbonnier"
        scale_weights=None,                   # list/tuple same length as win_lengths; if None → equal weights
        mel_scale="htk",                      # matches many TTS vocoder setups
        norm=None,                            # None or "slaney" per torchaudio docs
        clamp_mel_min=None,                   # optionally clamp mel floor after log (e.g., -8.0)
        in_device="cuda",
    ):
        super().__init__()
        assert len(win_lengths) == len(n_mels), "win_lengths and n_mels must match in length"
        self.sample_rate = sample_rate
        self.win_lengths = list(win_lengths)
        self.n_mels = list(n_mels)
        self.hops = [max(1, int(w // hop_divisor)) for w in self.win_lengths]
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.log_mel = log_mel
        self.log_eps = log_eps
        self.loss_mode = loss_mode
        self.l2_weight = l2_weight
        self.charbonnier_eps = charbonnier_eps
        self.mel_scale = mel_scale
        self.norm = norm
        self.clamp_mel_min = clamp_mel_min
        self.device = in_device

        if scale_weights is None:
            self.scale_weights = [1.0] * len(self.win_lengths)
        else:
            assert len(scale_weights) == len(self.win_lengths), "scale_weights length mismatch"
            self.scale_weights = list(scale_weights)

        # Build mel transforms (ModuleList so they move with .to(device))
        mels = []
        for wl, hop, nm in zip(self.win_lengths, self.hops, self.n_mels):
            mels.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=wl,
                    win_length=wl,
                    hop_length=hop,
                    n_mels=nm,
                    f_min=self.f_min,
                    f_max=(self.sample_rate / 2.0 if self.f_max is None else self.f_max),
                    window_fn=torch.hann_window,
                    power=self.power,
                    mel_scale=self.mel_scale,
                    norm=self.norm,
                    center=False
                )
            )
        self.mel_bank = nn.ModuleList(mels)

        self.l1 = nn.SmoothL1Loss(beta=0.1, reduction="mean")
        self.l2 = nn.MSELoss(reduction="mean")

    def _to_mono(self, x):
        # x: [B, T] or [B, C, T]
        if x.dim() == 3:
            return x.mean(dim=1)  # mix to mono for loss
        return x

    def _mel(self, transform, audio):
        # audio: [B, T], float
        mel = transform(audio)  # [B, n_mels, frames]
        if self.log_mel:
            mel = torch.log(mel + self.log_eps)
            if self.clamp_mel_min is not None:
                mel = torch.clamp(mel, min=self.clamp_mel_min)
        return mel

    def forward(self, gen_audio, ref_audio):
        """
        gen_audio, ref_audio: [B, T] or [B, C, T], float in [-1, 1] typically
        Returns: total_loss, details (dict)
        """
        gen = self._to_mono(gen_audio).to(self.device).float()
        ref = self._to_mono(ref_audio).to(self.device).float()

        device = self.device
        dtype = gen.dtype
        for t in self.mel_bank:
            t.to(device=device, dtype=torch.float32)
            # defensive: if a window was already created on another device, move it
            if hasattr(t.spectrogram, "window") and t.spectrogram.window is not None:
                t.spectrogram.window = t.spectrogram.window.to(device=device, dtype=dtype)

        # --- NEW: hard align waveform lengths ---
        audio_T = min(gen.size(-1), ref.size(-1))
        gen = gen[..., :audio_T]
        ref = ref[..., :audio_T]

        gen = torch.clamp(gen, min=-1.0, max=1.0)
        ref = torch.clamp(ref, min=-1.0, max=1.0)

        total = 0.0
        details = {}
        per_scale = []

        with torch.cuda.amp.autocast(enabled=False): # calculate losses in FP32
            for idx, (t, w) in enumerate(zip(self.mel_bank, self.scale_weights)):
                # explicit cast gen and ref audios to float32
                Mg = self._mel(t, gen.float())
                Mr = self._mel(t, ref.float())

                # --- NEW: make time frames match for this scale ---
                mel_T = min(Mg.size(-1), Mr.size(-1))
                Mg = Mg[..., :mel_T]
                Mr = Mr[..., :mel_T]

                if self.loss_mode == "l1":
                    L = self.l1(Mg, Mr)
                elif self.loss_mode == "l1+l2":
                    L = self.l1(Mg, Mr) + self.l2_weight * self.l2(Mg, Mr)
                elif self.loss_mode == "charbonnier":
                    diff = Mg - Mr
                    L = torch.mean(torch.sqrt(diff * diff + self.charbonnier_eps))
                else:
                    raise ValueError("loss_mode must be 'l1', 'l1+l2', or 'charbonnier'")

                L = torch.nan_to_num(L, nan=0.0, posinf=1e3, neginf=-1e3)
                L = torch.clamp(L, max=50.0)

                scaled = w * L
                total = total + scaled
                per_scale.append(L.detach())
                details[f"mel_scale_{idx}_w{int(t.win_length)}_m{t.n_mels}"] = float(L.detach().cpu())

        details["total"] = float(total.detach().cpu())
        details["per_scale_raw_mean"] = float(torch.stack(per_scale).mean().cpu())
        return total, details


class PitchLoss(nn.Module):
    def __init__(self, pesto_model, sample_rate,
                 tau=0.7, wmin=0.15, conf_clip_min=0.05, conf_clip_max=0.95,
                 vuv_thresh=0.5, eps=1e-5,
                 use_activation_loss=True, act_weight=0.1, normalize_acts=True,
                 use_charbonnier=False, charbonnier_eps=1e-6):
        super().__init__()
        self.pesto = pesto_model.eval()  # frozen feature extractor
        for p in self.pesto.parameters():
            p.requires_grad = False
        self.sr = sample_rate
        self.tau = tau
        self.wmin = wmin
        self.conf_clip_min = conf_clip_min
        self.conf_clip_max = conf_clip_max
        self.vuv_thresh = vuv_thresh
        self.eps = eps
        self.use_activation_loss = use_activation_loss
        self.act_weight = act_weight
        self.normalize_acts = normalize_acts
        self.use_charbonnier = use_charbonnier
        self.charbonnier_eps = charbonnier_eps
        self.device = next(pesto_model.parameters()).device

    def forward(self, fake_audio, real_audio):
        # fake_audio, real_audio: [B,T] or [T]; will be converted to mono if needed
        fake_audio = self._ensure_bt(fake_audio)
        real_audio = self._ensure_bt(real_audio)

        fake_audio = fake_audio.to(self.device, non_blocking=True)
        real_audio = real_audio.to(self.device, non_blocking=True)

        # real: no grad
        with torch.no_grad():
            r_pred, r_conf, _, r_act = self.pesto(real_audio, self.sr)  # r_pred: [B,Tr]
            f_pred, f_conf, _, f_act = self.pesto(fake_audio, self.sr)  # f_pred: [B,Tf]

        # fake: grad

        # ensure batch dims
        f_pred, f_conf, f_act = self._ensure_batch_dims(f_pred, f_conf, f_act)
        r_pred, r_conf, r_act = self._ensure_batch_dims(r_pred, r_conf, r_act)

        # time-align sequences by center-cropping to common length
        f_pred, r_pred = self._align_time(f_pred, r_pred)
        f_conf, r_conf = self._align_time(f_conf, r_conf)
        f_act, r_act = self._align_time_feats(f_act, r_act)

        # masks/weights
        vmask = (r_conf.clamp(0, 1) > self.vuv_thresh).float()  # [B,T]
        conf_w = r_conf.clamp(self.conf_clip_min, self.conf_clip_max).pow(self.tau)
        w = torch.maximum(conf_w, torch.tensor(self.wmin, device=conf_w.device)) * vmask

        # log-F0 loss (MSE or Charbonnier) (confidence-weighted, voiced-only)
        f_log = torch.log(f_pred + self.eps)
        r_log = torch.log(r_pred + self.eps)
        diff = f_log - r_log
        
        if self.use_charbonnier:
            # Charbonnier loss
            f0_loss_per_frame = torch.sqrt(diff * diff + self.charbonnier_eps)
        else:
            # MSE loss
            f0_loss_per_frame = diff.pow(2)
            
        f0_loss = (w * f0_loss_per_frame).sum() / (w.sum() + 1e-8)

        # activation L1 (optional)
        act_l1 = torch.tensor(0.0, device=f0_loss.device)
        if self.use_activation_loss:
            act_l1 = cosine_feat_loss(f_act, r_act, w)

        total = f0_loss + self.act_weight * act_l1
        return {"total": total, "f0_loss": f0_loss, "act_l1": act_l1}

    def _ensure_bt(self, x):
        # x: [T], [1,T], or [B,T] or stereo [C,T]; make [B,T] mono
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            # assume [B,C,T] -> mono
            return x.mean(dim=1)
        raise ValueError("audio must be [T], [B,T], or [B,C,T]")

    def _ensure_batch_dims(self, pred, conf, acts):
        # pred: [T] or [B,T]; conf same; acts: [T,C] or [B,T,C]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if conf.dim() == 1:
            conf = conf.unsqueeze(0)
        if acts.dim() == 2:
            acts = acts.unsqueeze(0)
        return pred, conf, acts

    def _align_time(self, a, b):
        # crop center to min length along time (dim=1)
        T = min(a.size(1), b.size(1))
        a = self._center_crop_time(a, T)
        b = self._center_crop_time(b, T)
        return a, b

    def _align_time_feats(self, a, b):
        # a,b: [B,T,C]
        T = min(a.size(1), b.size(1))
        a = self._center_crop_time(a, T)
        b = self._center_crop_time(b, T)
        return a, b

    def _center_crop_time(self, x, T):
        # x: [B,T,...]
        if x.size(1) == T:
            return x
        start = (x.size(1) - T) // 2
        end = start + T
        return x[:, start:end, ...]

    def _norm_feats(self, x, eps=1e-6):
        # x: [B,T,C] -> channel-wise standardization over time
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + eps
        return (x - mean) / std


def stft_loss(y_true, y_pred, n_fft, hop_length, win_length, device, use_charbonnier=False, charbonnier_eps=1e-6):
    # Force loss path to fp32 to avoid half underflow/inf logs
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)

    y_true_f = y_true.to(device=device, dtype=torch.float32)
    y_pred_f = y_pred.to(device=device, dtype=torch.float32)

    mag_true = _stft_mag(y_true_f, n_fft, hop_length, win_length)
    mag_pred = _stft_mag(y_pred_f, n_fft, hop_length, win_length)

    eps = _safe_eps_for_dtype(mag_true.dtype)

    # Spectral convergence (per-example)
    diff = mag_true - mag_pred
    num = torch.sqrt(torch.clamp((diff ** 2).sum(dim=(1, 2)), min=eps))
    denom = torch.sqrt(torch.clamp((mag_true ** 2).sum(dim=(1, 2)), min=eps))

    # Optional: zero-out SC where denom is effectively silent to avoid inf/huge grads
    silent = denom <= eps
    sc = num / denom
    if silent.any():
        sc = torch.where(silent, sc.new_zeros(()).expand_as(sc), sc)
    spectral_conv = sc.mean()

    # Log-mag loss (L1 or Charbonnier)
    log_true = torch.log1p(torch.clamp(mag_true, min=eps))
    log_pred = torch.log1p(torch.clamp(mag_pred, min=eps))
    
    if use_charbonnier:
        mag_loss = charbonnier_loss(log_true, log_pred, charbonnier_eps)
    else:
        mag_loss = F.l1_loss(log_true, log_pred)

    return spectral_conv + mag_loss


def mr_stft_loss(y_true, y_pred, h, device, cap=100.0):
    if y_true.dim() == 3 and y_true.size(1) > 1:
        y_true = y_true.mean(dim=1)
    if y_pred.dim() == 3 and y_pred.size(1) > 1:
        y_pred = y_pred.mean(dim=1)

    # Get MR-STFT parameters from config with defaults
    n_ffts = h.get('mr_stft_n_ffts', [2048, 1024, 512, 256, 128])
    hop_sizes = h.get('mr_stft_hop_sizes', [512, 256, 128, 64, 32])
    win_sizes = h.get('mr_stft_win_sizes', [2048, 1024, 512, 256, 128])
    
    assert len(n_ffts) == len(hop_sizes) == len(win_sizes), \
        "Multi-res lists must be the same length."

    # Get Charbonnier option from config
    use_charbonnier = h.get('mr_stft_use_charbonnier', False)
    charbonnier_eps = h.get('mr_stft_charbonnier_eps', 1e-6)

    total = 0.0
    with torch.cuda.amp.autocast(enabled=False):
        for n_fft, hop, win in zip(n_ffts, hop_sizes, win_sizes):
            total = total + stft_loss(y_true, y_pred, n_fft, hop, win, device, use_charbonnier, charbonnier_eps)


        total /= len(n_ffts)

    # prevent training from eating shit early on, as the raw loss value can be like 5-10k+ in the first few thousand steps.
    total = torch.clamp(total, max=cap)
    return total


class AudioLoss(nn.Module):
    def __init__(self, h, device, pesto_model=None):
        super().__init__()
        self.h = h
        self.device = device
        
        # Initialize pitch loss if enabled
        self.pitch_loss_fn = None
        if h.get('use_pitch_loss', False):
            if pesto_model is None:
                pesto_model = load_pesto_model(
                    h.get("pitch_loss_model", "mir-1k_g7"),
                    step_size=h.get("pitch_loss_step_size", 20.0)
                ).to(device)

            self.pitch_loss_fn = PitchLoss(
                pesto_model,
                h.get('sampling_rate', 16000),
                use_activation_loss=h.get('pitch_loss_use_activation_loss', False),
                act_weight=h.get('pitch_loss_act_weight', 0.1),
                normalize_acts=False,
                eps=1e-4 if h.get('fp16_run', False) else 1e-5,
                use_charbonnier=h.get('pitch_loss_use_charbonnier', False),
                charbonnier_eps=h.get('pitch_loss_charbonnier_eps', 1e-6),
                tau=h.get('pitch_loss_tau', 0.7),
                wmin=h.get('pitch_loss_wmin', 0.15),
                conf_clip_min=h.get('pitch_loss_conf_clip_min', 0.05),
                conf_clip_max=h.get('pitch_loss_conf_clip_max', 0.95),
                vuv_thresh=h.get('pitch_loss_vuv_thresh', 0.5),
            )

        # Initialize multi-scale mel loss
        self.use_multi_scale_mel_loss = h.get('use_multi_scale_mel_loss', False)
        self.multi_scale_mel_loss_fn = None
        if self.use_multi_scale_mel_loss:
            self.multi_scale_mel_loss_fn = MultiScaleMelLoss(
                sample_rate=h.get('sampling_rate', 16000),
                win_lengths=h.get('multi_scale_mel_win_lengths', [32, 128, 512, 1024, 2048]),
                n_mels=h.get('multi_scale_mel_n_mels', [5, 20, 80, 160, 320]),
                hop_divisor=h.get('multi_scale_mel_hop_divisor', 4),
                loss_mode=h.get('multi_scale_mel_loss_mode', 'l1'),
                scale_weights=h.get('multi_scale_mel_scale_weights', None),
                in_device=device,
                log_eps=h.get('multi_scale_mel_log_eps', 1e-5),
                l2_weight=h.get('multi_scale_mel_l2_weight', 1.0),
                charbonnier_eps=h.get('multi_scale_mel_charbonnier_eps', 1e-6),
                f_min=h.get('multi_scale_mel_f_min', 0.0),
                f_max=h.get('multi_scale_mel_f_max', None),
                power=h.get('multi_scale_mel_power', 1.0),
                log_mel=h.get('multi_scale_mel_log_mel', True),
                mel_scale=h.get('multi_scale_mel_scale', 'htk'),
                norm=h.get('multi_scale_mel_norm', None),
                clamp_mel_min=h.get('multi_scale_mel_clamp_min', None),
            )

        # Store loss weights
        self.mel_loss_weight = h.get('mel_loss_weight', 10.0)
        self.pitch_loss_weight = h.get('pitch_loss_weight', 1.0)
        self.mr_stft_loss_weight = h.get('mr_stft_loss_weight', 0.0)
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=h.get('sampling_rate', 16000),
            n_fft=h.get('n_fft', 1024),
            win_length=h.get('win_size', 1024),
            hop_length=h.get('hop_size', 256),
            n_mels=h.get('num_mels', 80),
            f_min=h.get('fmin', 0.0),
            f_max=h.get('fmax_for_loss', None),
            window_fn=torch.hann_window,
            power=2.0,
            center=False
        ).to(device)
        
    def mel_spectrogram(self, y):
        """
        Generate mel spectrogram from audio waveform using torchaudio
        
        Args:
            y: audio waveform [B, T]
            
        Returns:
            mel: mel spectrogram [B, num_mels, T_mel]
        """
        if y.dim() == 3:
            y = y.squeeze(1)
            
        # Ensure correct device and dtype
        y = y.to(self.device, dtype=torch.float32)
        
        # Generate mel spectrogram
        melspec = self.mel_transform(y)
        
        # Apply logarithm
        log_melspec = torch.log(torch.clamp(melspec, min=1e-5))
        
        return log_melspec
        
    def forward(self, y, y_g_hat):
        """
        Compute all audio losses
        
        Args:
            y: ground truth audio [B, 1, T] or [B, T]
            y_g_hat: generated audio [B, 1, T] or [B, T]
            
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary with individual losses
        """
        # Ensure correct dimensions
        if y.dim() == 3:
            y = y.squeeze(1)
        if y_g_hat.dim() == 3:
            y_g_hat = y_g_hat.squeeze(1)
            
        # Move to device
        y = y.to(self.device)
        y_g_hat = y_g_hat.to(self.device)
            
        # Initialize loss dictionary
        loss_dict = {}
        
        # Generate mel spectrograms
        y_mel = self.mel_spectrogram(y)
        y_g_hat_mel = self.mel_spectrogram(y_g_hat)
        
        # Multi-Scale Mel-Spectrogram Loss (fallback to L1 if not enabled)
        if self.use_multi_scale_mel_loss and self.multi_scale_mel_loss_fn is not None:
            loss_mel, mel_loss_details = self.multi_scale_mel_loss_fn(y_g_hat, y)
            loss_mel = loss_mel * self.mel_loss_weight
            loss_dict.update(mel_loss_details)
        else:
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * self.mel_loss_weight
            
        loss_dict['mel_loss'] = loss_mel.item()

        # Initialize MR-STFT loss
        loss_mr_stft = torch.tensor(0.0, device=self.device)

        # Compute MR-STFT loss if enabled
        if self.h.get('use_mr_stft_loss', False):
            loss_mr_stft = mr_stft_loss(y, y_g_hat, self.h, self.device, cap=self.h.get('mr_stft_cap', 100.0)) * self.mr_stft_loss_weight
            
        loss_dict['mr_stft_loss'] = loss_mr_stft.item()

        # Initialize pitch loss
        loss_pitch = torch.tensor(0.0, device=self.device)

        # Compute pitch loss if enabled
        if self.pitch_loss_fn is not None:
            with torch.cuda.amp.autocast(enabled=False):
                # PESTO uses CQT which doesn't like Bfloat16
                pitch_losses = self.pitch_loss_fn(y_g_hat.float(), y.float())

            loss_pitch = pitch_losses["total"] * self.pitch_loss_weight
            loss_dict['pitch_loss'] = loss_pitch.item()
            loss_dict['pitch_f0_loss'] = pitch_losses["f0_loss"].item()
            if self.h.get('pitch_loss_use_activation_loss', False):
                loss_dict['pitch_act_l1'] = pitch_losses["act_l1"].item()

        # Calculate total loss
        total_loss = loss_mel + loss_mr_stft + loss_pitch
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict