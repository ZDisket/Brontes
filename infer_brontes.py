#!/usr/bin/env python3
"""
Minimal inference example for Brontes.

This script demonstrates how to:
1. Load a trained Brontes model checkpoint
2. Load an audio file
3. Process it through the model
4. Save the output

Usage:
    python infer_brontes.py --config configs/config_brontes_48khz_demucs.yaml \
                               --checkpoint checkpoints/brontes_best \
                               --input audio.wav \
                               --output output.wav
"""

import argparse
import yaml
import torch
import torchaudio
import math
import torch.nn.functional as F

try:
    from .brontes import Brontes
except ImportError:
    from brontes import Brontes


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config, device):
    """Create Brontes model from configuration."""
    model_config = config['model']
    unet_config = model_config.get('unet_config', {})
    
    model = Brontes(
        unet_config=unet_config
    ).to(device)
    
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint.
    
    Args:
        model: The Brontes model
        checkpoint_path: Path to the checkpoint file
        device: Torch device
    
    Returns:
        The model with loaded weights
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both direct state dict and wrapped state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully!")
    
    return model


def load_audio(audio_path, target_sr, mono=True, normalize=True):
    """Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True
        normalize: Normalize to [-1, 1] if True
    
    Returns:
        audio: Tensor of shape (1, 1, samples) for mono
        original_sr: Original sample rate
    """
    print(f"Loading audio: {audio_path}")
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Convert to mono if needed
    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    # Normalize
    if normalize:
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
    
    # Add batch dimension: (channels, samples) -> (1, channels, samples)
    audio = audio.unsqueeze(0)
    
    print(f"Audio shape: {audio.shape}, duration: {audio.shape[-1] / target_sr:.2f}s")
    
    return audio, sr


def process_audio(model, audio, device=None):
    """Process audio through the model.
    
    Args:
        model: The Brontes model
        audio: Input audio tensor of shape (1, 1, samples)
        device: Torch device
    
    Returns:
        Processed audio tensor of shape (1, 1, samples)
    """
    if device is None:
        device = next(model.parameters()).device
    
    audio = audio.to(device)
    
    model.eval()
    with torch.no_grad():
        # Brontes returns: (processed, latent, original_components, processed_components)
        output, _, _, _ = model(audio)
    
    return output


def save_audio(audio, path, sample_rate):
    """Save audio tensor to file.
    
    Args:
        audio: Tensor of shape (1, 1, samples) or (1, samples)
        path: Output file path
        sample_rate: Sample rate
    """
    # Remove batch dimension if present
    if audio.dim() == 3:
        audio = audio.squeeze(0)  # (1, samples)
    
    # Ensure audio is on CPU
    audio = audio.cpu()
    
    # Normalize before saving to prevent clipping
    max_val = audio.abs().max()
    if max_val > 1.0:
        audio = audio / max_val
    
    torchaudio.save(path, audio, sample_rate)
    print(f"Saved output to: {path}")


class BrontesInference:
    """Inference class for loading and using exported TorchScript Brontes models.
    
    This class provides a simple interface for:
    - Loading CPU or GPU TorchScript models
    - Processing audio with automatic resampling if input sample rate differs
    - Loading/saving audio files
    
    Example:
        >>> inference = BrontesInference('brontes_cuda.pt', device='cuda')
        >>> output = inference.process_file('input.wav', 'output.wav')
        
        # Or with explicit sample rate:
        >>> audio, sr = torchaudio.load('input.wav')
        >>> output = inference.process(audio, input_sr=sr)
    """
    
    def __init__(self, model_path: str, device: str = None, sample_rate: int = 44100):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to the TorchScript model file (.pt)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            sample_rate: Model's expected sample rate (audio will be resampled to this)
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.sample_rate = sample_rate
        
        # Cache for resamplers (to avoid recreating them)
        self._resamplers = {}
        
        # Load model
        print(f"Loading TorchScript model from: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"Model loaded on: {self.device}")
        print(f"Model sample rate: {self.sample_rate} Hz")
    
    def _get_resampler(self, from_sr: int, to_sr: int) -> torchaudio.transforms.Resample:
        """Get or create a cached resampler."""
        key = (from_sr, to_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(from_sr, to_sr).to(self.device)
        return self._resamplers[key]
    
    def process(self, audio: torch.Tensor, input_sr: int = None, 
                normalize: bool = True) -> torch.Tensor:
        """Process audio tensor through the model.
        
        Args:
            audio: Input tensor of shape (batch, channels, samples), (channels, samples), or (samples,)
            input_sr: Sample rate of the input audio. If different from model's sample rate,
                      the audio will be resampled before processing and output resampled back.
                      If None, assumes audio is already at model's sample rate.
            normalize: Whether to normalize input to [-1, 1]
            
        Returns:
            Processed audio tensor at the input sample rate (if input_sr was provided),
            or at the model's sample rate otherwise.
        """
        # Ensure correct shape: (batch, channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        
        # Store original length for output resampling
        original_length = audio.shape[-1]
        output_sr = input_sr if input_sr is not None else self.sample_rate
        
        # Normalize if requested
        if normalize:
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val
        
        # Resample to model's sample rate if needed
        if input_sr is not None and input_sr != self.sample_rate:
            resampler = self._get_resampler(input_sr, self.sample_rate)
            audio = resampler(audio)
        
        # Process
        with torch.no_grad():
            output = self.model(audio)
        
        return output
    

    
    def load_audio(self, path: str, resample: bool = True, 
                   normalize: bool = True) -> tuple[torch.Tensor, int]:
        """Load audio file.
        
        Args:
            path: Path to audio file
            resample: If True, resample to model's sample rate. If False, keep original.
            normalize: Whether to normalize to [-1, 1]
            
        Returns:
            Tuple of (audio tensor of shape (1, 1, samples), sample_rate)
        """
        audio, sr = torchaudio.load(path)
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if requested
        if resample and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            sr = self.sample_rate
        
        # Normalize
        if normalize:
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val
        
        return audio.unsqueeze(0), sr
    
    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: int = None):
        """Save audio tensor to file.
        
        Args:
            audio: Audio tensor (any shape, will be properly formatted)
            path: Output file path
            sample_rate: Sample rate for output file (default: model's sample rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        if audio.dim() == 3:
            audio = audio.squeeze(0)
        
        audio = audio.cpu()
        max_val = audio.abs().max()
        if max_val > 1.0:
            audio = audio / max_val
        
        torchaudio.save(path, audio, sample_rate)
    
    def process_file(self, input_path: str, output_path: str = None,
                     preserve_sample_rate: bool = True) -> torch.Tensor:
        """Load, process, and optionally save an audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save output (optional)
            preserve_sample_rate: If True, output will be at the same sample rate as input.
                                  If False, output will be at model's sample rate.
            
        Returns:
            Processed audio tensor
        """
        # Load without resampling to get original sample rate
        audio_orig, input_sr = torchaudio.load(input_path)
        
        # Convert to mono
        if audio_orig.shape[0] > 1:
            audio_orig = audio_orig.mean(dim=0, keepdim=True)
        
        # Add batch dimension
        audio = audio_orig.unsqueeze(0)
        
        # Process with automatic resampling
        if preserve_sample_rate:
            output = self.process(audio, input_sr=input_sr)
            output_sr = input_sr
        else:
            # Resample input to model's sample rate first
            if input_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(input_sr, self.sample_rate)
                audio = resampler(audio)
            output = self.process(audio, input_sr=None)
            output_sr = self.sample_rate
        
        if output_path:
            self.save_audio(output, output_path, output_sr)
            print(f"Saved to: {output_path}")
        
        return output


def main():
    parser = argparse.ArgumentParser(description='Brontes Inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output audio file')

    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    sample_rate = config['dataset']['sample_rate']
    
    # Create and load model
    model = create_model(config, device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Load audio
    audio, _ = load_audio(args.input, sample_rate, mono=True, normalize=True)
    audio = audio.to(device)
    
    # Process audio
    print("Processing audio...")
    output = process_audio(model, audio, device=device)
    
    # Save output
    save_audio(output, args.output, sample_rate)
    print("Done!")


if __name__ == '__main__':
    main()
