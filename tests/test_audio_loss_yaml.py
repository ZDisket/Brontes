#!/usr/bin/env python3
"""
Test script for AudioLoss with YAML configuration.
This script validates that AudioLoss works correctly with actual YAML config files.
"""

import torch
import yaml
import numpy as np
import unittest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from audio_loss import AudioLoss


def create_test_signal(length=16000, sample_rate=16000):
    """Create a simple test signal (sum of sinusoids) for testing."""
    t = np.linspace(0, length/sample_rate, length, endpoint=False)
    
    # Create a signal with multiple frequency components
    signal = (np.sin(2 * np.pi * 440 * t) +      # A4 note
              0.5 * np.sin(2 * np.pi * 880 * t) +  # A5 note (octave)
              0.3 * np.sin(2 * np.pi * 220 * t))   # A3 note (lower octave)
    
    # Normalize to [-1, 1]
    signal = signal / np.max(np.abs(signal))
    
    return signal.astype(np.float32)


class TestAudioLossWithYAMLConfig(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 16000
        self.length = 16000
        
        # Create test signals
        self.gt_signal = create_test_signal(self.length, self.sample_rate)
        self.recon_signal = self.gt_signal + 0.01 * np.random.randn(self.length)  # Add small noise
        
        # Convert to torch tensors
        self.gt_tensor = torch.from_numpy(self.gt_signal).float().unsqueeze(0).to(self.device)  # [1, T]
        self.recon_tensor = torch.from_numpy(self.recon_signal).float().unsqueeze(0).to(self.device)  # [1, T]

    def test_with_yaml_config(self):
        """Test AudioLoss with a real YAML configuration."""
        print("Testing AudioLoss with YAML configuration...")
        
        # Load a sample configuration from the project
        # Since we don't have a real YAML file in this test context, 
        # we'll simulate what a loaded YAML config would look like
        yaml_config = {
            # Basic audio parameters
            'sampling_rate': 16000,
            'n_fft': 1024,
            'win_size': 1024,
            'hop_size': 256,
            'num_mels': 80,
            'fmin': 0.0,
            'fmax_for_loss': 8000.0,
            
            # Loss weights
            'mel_loss_weight': 10.0,
            'mr_stft_loss_weight': 0.0,
            'pitch_loss_weight': 1.0,
            
            # Multi-scale mel loss parameters
            'use_multi_scale_mel_loss': False,
            'multi_scale_mel_win_lengths': [32, 128, 512, 1024, 2048],
            'multi_scale_mel_n_mels': [5, 20, 80, 160, 320],
            'multi_scale_mel_hop_divisor': 4,
            'multi_scale_mel_loss_mode': 'l1',
            'multi_scale_mel_log_eps': 1e-5,
            'multi_scale_mel_l2_weight': 1.0,
            'multi_scale_mel_charbonnier_eps': 1e-6,
            'multi_scale_mel_f_min': 0.0,
            'multi_scale_mel_f_max': None,
            'multi_scale_mel_power': 1.0,
            'multi_scale_mel_log_mel': True,
            'multi_scale_mel_scale': 'htk',
            'multi_scale_mel_norm': None,
            'multi_scale_mel_clamp_min': None,
            
            # MR-STFT loss parameters
            'use_mr_stft_loss': False,
            'mr_stft_n_ffts': [2048, 1024, 512, 256, 128],
            'mr_stft_hop_sizes': [512, 256, 128, 64, 32],
            'mr_stft_win_sizes': [2048, 1024, 512, 256, 128],
            'mr_stft_use_charbonnier': False,
            'mr_stft_charbonnier_eps': 1e-6,
            
            # Pitch loss parameters
            'use_pitch_loss': False,
            'pitch_loss_use_activation_loss': False,
            'pitch_loss_act_weight': 0.1,
            'pitch_loss_use_charbonnier': False,
            'pitch_loss_charbonnier_eps': 1e-6,
            'pitch_loss_tau': 0.7,
            'pitch_loss_wmin': 0.15,
            'pitch_loss_conf_clip_min': 0.05,
            'pitch_loss_conf_clip_max': 0.95,
            'pitch_loss_vuv_thresh': 0.5,
            
            # Training parameters
            'fp16_run': False
        }
        
        # Initialize AudioLoss with YAML configuration
        audio_loss = AudioLoss(yaml_config, self.device)
        
        # Compute loss
        total_loss, loss_dict = audio_loss(self.gt_tensor, self.recon_tensor)
        
        # Check return types
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        
        # Check that total_loss is a scalar
        self.assertEqual(total_loss.dim(), 0)
        
        # Check that loss_dict contains expected keys
        self.assertIn('mel_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        
        # Check that losses are non-negative
        self.assertGreaterEqual(loss_dict['mel_loss'], 0.0)
        self.assertGreaterEqual(loss_dict['total_loss'], 0.0)
        
        print(f"YAML config test passed. Total loss: {loss_dict['total_loss']:.4f}")

    def test_with_enabled_features(self):
        """Test AudioLoss with multiple features enabled."""
        print("Testing AudioLoss with multiple features enabled...")
        
        # Configuration with multiple features enabled
        yaml_config = {
            # Basic audio parameters
            'sampling_rate': 16000,
            'n_fft': 1024,
            'win_size': 1024,
            'hop_size': 256,
            'num_mels': 80,
            'fmin': 0.0,
            'fmax_for_loss': 8000.0,
            
            # Loss weights
            'mel_loss_weight': 10.0,
            'mr_stft_loss_weight': 1.0,
            'pitch_loss_weight': 1.0,
            
            # Multi-scale mel loss parameters
            'use_multi_scale_mel_loss': True,
            'multi_scale_mel_win_lengths': [32, 128, 512],
            'multi_scale_mel_n_mels': [5, 20, 80],
            'multi_scale_mel_hop_divisor': 4,
            'multi_scale_mel_loss_mode': 'l1',
            
            # MR-STFT loss parameters
            'use_mr_stft_loss': True,
            'mr_stft_n_ffts': [512, 256, 128],
            'mr_stft_hop_sizes': [128, 64, 32],
            'mr_stft_win_sizes': [512, 256, 128],
            
            # Training parameters
            'fp16_run': False
        }
        
        # Initialize AudioLoss with configuration
        audio_loss = AudioLoss(yaml_config, self.device)
        
        # Compute loss
        total_loss, loss_dict = audio_loss(self.gt_tensor, self.recon_tensor)
        
        # Check return types
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        
        # Check that total_loss is a scalar
        self.assertEqual(total_loss.dim(), 0)
        
        # Check that loss_dict contains expected keys
        self.assertIn('mel_loss', loss_dict)
        self.assertIn('mr_stft_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        
        # Check for multi-scale mel loss details
        self.assertTrue(any('mel_scale_' in key for key in loss_dict.keys()))
        
        print(f"Multi-feature test passed. Total loss: {loss_dict['total_loss']:.4f}")


if __name__ == '__main__':
    print("Running AudioLoss YAML configuration tests...")
    unittest.main(verbosity=2)
