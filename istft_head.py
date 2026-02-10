"""
iSTFT Output Head for Wave U-Net

Provides numerically stable phase reconstruction via safe atan2 and
iSTFT synthesis for waveform generation from predicted spectral components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


@torch.jit.script
def safe_atan2(sin_phase: torch.Tensor, cos_phase: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable atan2 that avoids NaN gradients when sin=cos=0.
    
    Standard atan2 has undefined gradients at the origin (0, 0).
    This version normalizes inputs to the unit circle before computing
    the angle, with epsilon clamping to prevent division by zero.
    
    Args:
        sin_phase: Sine of phase angle, any shape
        cos_phase: Cosine of phase angle, same shape as sin_phase
        eps: Small epsilon for numerical stability
        
    Returns:
        Phase angle in radians, same shape as inputs
    """
    norm = torch.sqrt(sin_phase**2 + cos_phase**2).clamp(min=eps)
    sin_normed = sin_phase / norm
    cos_normed = cos_phase / norm
    return torch.atan2(sin_normed, cos_normed)


class iSTFTHead(nn.Module):
    """
    Converts U-Net feature maps to waveform via learned STFT prediction + iSTFT.
    
    Takes a feature map [B, C, T'] from the U-Net decoder, projects it to
    [log_magnitude, sin_phase, cos_phase] predictions, then synthesizes
    waveform via torch.istft.
    
    Args:
        in_channels: Number of input channels from U-Net decoder
        n_fft: FFT size for iSTFT synthesis
        hop_length: Hop length for iSTFT
        win_length: Window length (defaults to n_fft if None)
        phase_eps: Epsilon for safe atan2
        use_weight_norm: Whether to apply weight normalization to projection
    """
    
    def __init__(
        self,
        in_channels: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = None,
        phase_eps: float = 1e-8,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.phase_eps = phase_eps
        self.n_freqs = n_fft // 2 + 1
        
        # Projection to STFT components: log_mag + sin_phase + cos_phase
        proj = nn.Conv1d(in_channels, 3 * self.n_freqs, kernel_size=1)
        if use_weight_norm:
            proj = weight_norm(proj)
        self.proj = proj
        
        # Hann window for iSTFT (registered as buffer so it moves with model)
        self.register_buffer('window', torch.hann_window(self.win_length))
        
        # Initialize projection to small values for stable start
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights to small values for stable start
        # Zero init can cause issues when combined with extreme activations
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        if self.proj.bias is not None:
            # Split bias: first n_freqs is log_mag, rest is sin/cos phase
            with torch.no_grad():
                # log(0.1) ≈ -2.3, start with small magnitudes
                self.proj.bias[:self.n_freqs].fill_(-2.3)
                # sin/cos initialized to produce ~zero phase
                self.proj.bias[self.n_freqs:].fill_(0.0)
    
    def forward(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Convert U-Net features to waveform via iSTFT.
        
        Args:
            features: [B, C, T'] feature map from U-Net decoder
            target_length: Desired output waveform length in samples
            
        Returns:
            waveform: [B, 1, target_length] synthesized audio
        """
        B = features.shape[0]
        
        # Project to STFT components
        pred = self.proj(features)  # [B, 3*F, T']
        
        # Calculate expected frame count for target length
        # Formula: n_frames = (length + hop_length) // hop_length for center=True
        n_frames = (target_length + self.hop_length) // self.hop_length + 1
        
        # Interpolate to match expected frame count
        if pred.shape[-1] != n_frames:
            pred = F.interpolate(pred, size=n_frames, mode='linear', align_corners=False)
        
        # Split into components: [B, F, T_frames] each
        log_mag = pred[:, :self.n_freqs, :]
        sin_phase = pred[:, self.n_freqs:2*self.n_freqs, :]
        cos_phase = pred[:, 2*self.n_freqs:, :]
        
        # Clamp log-magnitude to prevent exp() overflow/underflow
        # log(1e-5) ≈ -11.5, log(1e3) ≈ 6.9
        log_mag = log_mag.clamp(min=-11.5, max=7.0)
        
        # Convert log-magnitude to linear magnitude with safe clamping
        magnitude = torch.exp(log_mag).clamp(min=1e-8, max=1e4)
        
        # Recover phase angle with safe atan2
        phase = safe_atan2(sin_phase, cos_phase, self.phase_eps)
        
        # Convert to complex STFT representation
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        stft_complex = torch.complex(real, imag)
        
        # iSTFT reconstruction
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=target_length,
        )
        
        return waveform.unsqueeze(1)  # [B, 1, L]
    
    def get_spectral_components(self, features: torch.Tensor, target_length: int):
        """
        Get the predicted spectral components without synthesizing waveform.
        Useful for spectral-domain losses.
        
        Args:
            features: [B, C, T'] feature map from U-Net decoder
            target_length: Reference length for frame count calculation
            
        Returns:
            log_mag: [B, F, T_frames] log-magnitude
            sin_phase: [B, F, T_frames] sin of phase
            cos_phase: [B, F, T_frames] cos of phase
        """
        pred = self.proj(features)
        
        n_frames = (target_length + self.hop_length) // self.hop_length + 1
        if pred.shape[-1] != n_frames:
            pred = F.interpolate(pred, size=n_frames, mode='linear', align_corners=False)
        
        log_mag = pred[:, :self.n_freqs, :]
        sin_phase = pred[:, self.n_freqs:2*self.n_freqs, :]
        cos_phase = pred[:, 2*self.n_freqs:, :]
        
        return log_mag, sin_phase, cos_phase
