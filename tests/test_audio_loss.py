#!/usr/bin/env python3
"""
Test script for the AudioLoss class.
This script validates the functionality of the AudioLoss by:
1. Creating test audio signals
2. Testing basic loss computation
3. Testing with different loss configurations from YAML
"""

import torch
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


class TestAudioLoss(unittest.TestCase):
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
        
        # Basic configuration (similar to YAML config)
        self.basic_config = {
            'sampling_rate': self.sample_rate,
            'n_fft': 1024,
            'win_size': 1024,
            'hop_size': 256,
            'num_mels': 80,
            'fmin': 0.0,
            'fmax_for_loss': 8000.0,
            'mel_loss_weight': 10.0,
            'use_multi_scale_mel_loss': False,
            'use_mr_stft_loss': False,
            'use_pitch_loss': False,
            'fp16_run': False
        }

    def test_basic_audio_loss(self):
        """Test basic AudioLoss functionality."""
        print("Testing basic AudioLoss functionality...")
        
        # Initialize AudioLoss with basic configuration
        audio_loss = AudioLoss(self.basic_config, self.device)
        
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
        
        print(f"Basic loss test passed. Total loss: {loss_dict['total_loss']:.4f}")

    def test_multi_scale_mel_loss(self):
        """Test AudioLoss with multi-scale mel loss enabled."""
        print("Testing multi-scale mel loss...")
        
        # Configuration with multi-scale mel loss enabled
        config = self.basic_config.copy()
        config.update({
            'use_multi_scale_mel_loss': True,
            'multi_scale_mel_win_lengths': [32, 128, 512],
            'multi_scale_mel_n_mels': [5, 20, 80],
            'multi_scale_mel_hop_divisor': 4,
            'multi_scale_mel_loss_mode': 'l1'
        })
        
        # Initialize AudioLoss with multi-scale mel configuration
        audio_loss = AudioLoss(config, self.device)
        
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
        
        # Check for multi-scale mel loss details
        self.assertTrue(any('mel_scale_' in key for key in loss_dict.keys()))
        
        print(f"Multi-scale mel loss test passed. Total loss: {loss_dict['total_loss']:.4f}")

    def test_mr_stft_loss(self):
        """Test AudioLoss with MR-STFT loss enabled."""
        print("Testing MR-STFT loss...")
        
        # Configuration with MR-STFT loss enabled
        config = self.basic_config.copy()
        config.update({
            'use_mr_stft_loss': True,
            'mr_stft_n_ffts': [512, 256, 128],
            'mr_stft_hop_sizes': [128, 64, 32],
            'mr_stft_win_sizes': [512, 256, 128],
            'mr_stft_loss_weight': 1.0
        })
        
        # Initialize AudioLoss with MR-STFT configuration
        audio_loss = AudioLoss(config, self.device)
        
        # Compute loss
        total_loss, loss_dict = audio_loss(self.gt_tensor, self.recon_tensor)
        
        # Check return types
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        
        # Check that total_loss is a scalar
        self.assertEqual(total_loss.dim(), 0)
        
        # Check that loss_dict contains expected keys
        self.assertIn('mr_stft_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        
        print(f"MR-STFT loss test passed. Total loss: {loss_dict['total_loss']:.4f}")

    def test_identical_signals(self):
        """Test that loss is near zero for identical signals."""
        print("Testing identical signals...")
        
        # Use the same signal for both ground truth and reconstruction
        identical_tensor = self.gt_tensor.clone()
        
        # Initialize AudioLoss with basic configuration
        audio_loss = AudioLoss(self.basic_config, self.device)
        
        # Compute loss
        total_loss, loss_dict = audio_loss(self.gt_tensor, identical_tensor)
        
        # Loss should be very small for identical signals
        self.assertLess(loss_dict['mel_loss'], 0.1)
        self.assertLess(loss_dict['total_loss'], 0.1)
        
        print(f"Identical signals test passed. Loss: {loss_dict['total_loss']:.6f}")

    def test_batch_processing(self):
        """Test AudioLoss with batched inputs."""
        print("Testing batch processing...")
        
        # Create batched inputs [Batch, Time]
        batch_size = 4
        gt_batch = self.gt_tensor.repeat(batch_size, 1)
        recon_batch = self.recon_tensor.repeat(batch_size, 1)
        
        # Initialize AudioLoss with basic configuration
        audio_loss = AudioLoss(self.basic_config, self.device)
        
        # Compute loss
        total_loss, loss_dict = audio_loss(gt_batch, recon_batch)
        
        # Check return types
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        
        # Check that total_loss is a scalar
        self.assertEqual(total_loss.dim(), 0)
        
        # Check that loss_dict contains expected keys
        self.assertIn('mel_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        
        print(f"Batch processing test passed. Total loss: {loss_dict['total_loss']:.4f}")

    def test_config_defaults(self):
        """Test that AudioLoss works with minimal configuration."""
        print("Testing configuration defaults...")
        
        # Minimal configuration
        minimal_config = {
            'sampling_rate': self.sample_rate
        }
        
        # Initialize AudioLoss with minimal configuration
        audio_loss = AudioLoss(minimal_config, self.device)
        
        # Compute loss
        total_loss, loss_dict = audio_loss(self.gt_tensor, self.recon_tensor)
        
        # Check return types
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(loss_dict, dict)
        
        # Check that total_loss is a scalar
        self.assertEqual(total_loss.dim(), 0)
        
        print(f"Config defaults test passed. Total loss: {loss_dict['total_loss']:.4f}")


if __name__ == '__main__':
    print("Running AudioLoss tests...")
    unittest.main(verbosity=2)
