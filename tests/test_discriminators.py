#!/usr/bin/env python3
"""
Test script for the discriminators.
This script validates the functionality of the discriminators by:
1. Testing imports
2. Testing forward passes for MPD and MSD
3. Testing loss functions
4. Testing with and without SE blocks
"""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiBandSpecDiscriminator,
    discriminator_loss,
    generator_loss,
    feature_loss,
    SEBlock1D,
    SEBlock2D
)


def create_test_audio(batch_size=2, length=16384):
    """Create a simple test audio signal for testing."""
    # Create random audio data (normalized to [-1, 1])
    audio = torch.randn(batch_size, 1, length)
    audio = audio / torch.max(torch.abs(audio))
    return audio


def test_imports():
    """Test that all discriminator components can be imported."""
    print("Testing imports...")
    
    # Check that all classes can be imported
    assert MultiPeriodDiscriminator is not None
    assert MultiScaleDiscriminator is not None
    assert discriminator_loss is not None
    assert generator_loss is not None
    assert feature_loss is not None
    assert SEBlock1D is not None
    assert SEBlock2D is not None
    
    print("All imports successful!")


def test_se_blocks():
    """Test Squeeze-and-Exitation blocks."""
    print("Testing SE blocks...")
    
    # Test SEBlock1D
    se1d = SEBlock1D(64)
    x1d = torch.randn(2, 64, 100)
    y1d = se1d(x1d)
    assert y1d.shape == x1d.shape, f"SEBlock1D output shape mismatch: {y1d.shape} vs {x1d.shape}"
    
    # Test SEBlock2D
    se2d = SEBlock2D(64)
    x2d = torch.randn(2, 64, 32, 32)
    y2d = se2d(x2d)
    assert y2d.shape == x2d.shape, f"SEBlock2D output shape mismatch: {y2d.shape} vs {x2d.shape}"
    
    print("SE blocks test passed!")


def test_discriminator_p():
    """Test individual period discriminator."""
    print("Testing individual period discriminator...")
    
    from discriminators import DiscriminatorP
    
    # Test without SE blocks
    disc_p = DiscriminatorP(5, use_se_blocks=False)
    x = create_test_audio(2, 16384)
    y, fmap = disc_p(x)
    
    # Check output shapes
    assert y.dim() == 2, f"Expected 2D output, got {y.dim()}D"
    assert len(fmap) == 6, f"Expected 6 feature maps, got {len(fmap)}"  # 5 conv layers + 1 conv_post
    
    # Test with SE blocks
    disc_p_se = DiscriminatorP(5, use_se_blocks=True)
    y_se, fmap_se = disc_p_se(x)
    
    # Check output shapes
    assert y_se.dim() == 2, f"Expected 2D output, got {y_se.dim()}D"
    assert len(fmap_se) == 6, f"Expected 6 feature maps, got {len(fmap_se)}"
    
    print("Individual period discriminator test passed!")


def test_mpd():
    """Test Multi-Period Discriminator."""
    print("Testing Multi-Period Discriminator...")
    
    # Test without SE blocks
    mpd = MultiPeriodDiscriminator(use_se_blocks=False)
    x = create_test_audio(2, 16384)
    y = create_test_audio(2, 16384)  # Generated audio
    
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(x, y)
    
    # Check output shapes
    assert len(y_d_rs) == 5, f"Expected 5 real outputs, got {len(y_d_rs)}"  # 5 periods
    assert len(y_d_gs) == 5, f"Expected 5 generated outputs, got {len(y_d_gs)}"
    assert len(fmap_rs) == 5, f"Expected 5 real feature maps, got {len(fmap_rs)}"
    assert len(fmap_gs) == 5, f"Expected 5 generated feature maps, got {len(fmap_gs)}"
    
    # Check that each output has the right shape
    for i in range(5):
        assert y_d_rs[i].dim() == 2, f"Real output {i} should be 2D"
        assert y_d_gs[i].dim() == 2, f"Generated output {i} should be 2D"
        assert len(fmap_rs[i]) == 6, f"Real feature map {i} should have 6 layers"
        assert len(fmap_gs[i]) == 6, f"Generated feature map {i} should have 6 layers"
    
    # Test with SE blocks
    mpd_se = MultiPeriodDiscriminator(use_se_blocks=True)
    y_d_rs_se, y_d_gs_se, fmap_rs_se, fmap_gs_se = mpd_se(x, y)
    
    # Check output shapes
    assert len(y_d_rs_se) == 5, f"Expected 5 real outputs, got {len(y_d_rs_se)}"
    assert len(y_d_gs_se) == 5, f"Expected 5 generated outputs, got {len(y_d_gs_se)}"
    
    print("Multi-Period Discriminator test passed!")


def test_msd():
    """Test Multi-Scale Discriminator."""
    print("Testing Multi-Scale Discriminator...")
    
    # Test without SE blocks
    msd = MultiScaleDiscriminator(use_se_blocks=False)
    x = create_test_audio(2, 16384)
    y = create_test_audio(2, 16384)  # Generated audio
    
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = msd(x, y)
    
    # Check output shapes
    assert len(y_d_rs) == 3, f"Expected 3 real outputs, got {len(y_d_rs)}"  # 3 scales
    assert len(y_d_gs) == 3, f"Expected 3 generated outputs, got {len(y_d_gs)}"
    assert len(fmap_rs) == 3, f"Expected 3 real feature maps, got {len(fmap_rs)}"
    assert len(fmap_gs) == 3, f"Expected 3 generated feature maps, got {len(fmap_gs)}"
    
    # Check that each output has the right shape
    for i in range(3):
        assert y_d_rs[i].dim() == 2, f"Real output {i} should be 2D"
        assert y_d_gs[i].dim() == 2, f"Generated output {i} should be 2D"
        assert len(fmap_rs[i]) == 8, f"Real feature map {i} should have 8 layers"  # 7 conv layers + 1 conv_post
        assert len(fmap_gs[i]) == 8, f"Generated feature map {i} should have 8 layers"
    
    # Test with SE blocks
    msd_se = MultiScaleDiscriminator(use_se_blocks=True)
    y_d_rs_se, y_d_gs_se, fmap_rs_se, fmap_gs_se = msd_se(x, y)
    
    # Check output shapes
    assert len(y_d_rs_se) == 3, f"Expected 3 real outputs, got {len(y_d_rs_se)}"
    assert len(y_d_gs_se) == 3, f"Expected 3 generated outputs, got {len(y_d_gs_se)}"
    
    print("Multi-Scale Discriminator test passed!")


def test_mbsd():
    """Test Multi-Band Spectral Discriminator forward pass."""
    print("Testing Multi-Band Spectral Discriminator...")
    
    mbsd = MultiBandSpecDiscriminator(window_length=512, hop_factor=0.25, audio_channels=1)
    x = create_test_audio(2, 4096)
    
    fmap = mbsd(x)
    
    # Expect a non-empty list of feature maps ending with a 2D conv output
    assert isinstance(fmap, list), "MBSD should return a list of feature maps"
    assert len(fmap) > 0, "MBSD should return at least one feature map"
    assert fmap[-1].dim() == 4, f"Expected 4D final feature map, got {fmap[-1].dim()}D"
    
    print("Multi-Band Spectral Discriminator test passed!")


def test_loss_functions():
    """Test discriminator loss functions."""
    print("Testing loss functions...")
    
    # Create dummy discriminator outputs
    batch_size = 2
    
    # Real outputs (should be close to 1)
    disc_real_outputs = [torch.ones(batch_size, 10) * 0.9]  # Close to 1
    
    # Generated outputs (should be close to 0)
    disc_generated_outputs = [torch.ones(batch_size, 10) * 0.1]  # Close to 0
    
    # Test discriminator loss
    disc_loss, r_losses, g_losses = discriminator_loss(disc_real_outputs, disc_generated_outputs)
    
    # Check that loss is a scalar tensor
    assert isinstance(disc_loss, torch.Tensor), "Discriminator loss should be a tensor"
    assert disc_loss.dim() == 0, f"Discriminator loss should be scalar, got {disc_loss.dim()}D"
    assert len(r_losses) == 1, f"Expected 1 real loss, got {len(r_losses)}"
    assert len(g_losses) == 1, f"Expected 1 generated loss, got {len(g_losses)}"
    
    # Test generator loss (should try to make generated outputs close to 1)
    gen_loss, gen_losses = generator_loss(disc_generated_outputs)
    
    # Check that loss is a scalar tensor
    assert isinstance(gen_loss, torch.Tensor), "Generator loss should be a tensor"
    assert gen_loss.dim() == 0, f"Generator loss should be scalar, got {gen_loss.dim()}D"
    assert len(gen_losses) == 1, f"Expected 1 generator loss, got {len(gen_losses)}"
    
    # Test feature loss with dummy feature maps (3D tensors to match discriminator outputs)
    fmap_r = [torch.randn(2, 64, 100, 1), torch.randn(2, 128, 50, 1)]
    fmap_g = [torch.randn(2, 64, 100, 1), torch.randn(2, 128, 50, 1)]
    
    feat_loss = feature_loss(fmap_r, fmap_g)
    
    # Check that loss is a scalar tensor
    assert isinstance(feat_loss, torch.Tensor), "Feature loss should be a tensor"
    assert feat_loss.dim() == 0, f"Feature loss should be scalar, got {feat_loss.dim()}D"
    
    print("Loss functions test passed!")


def test_full_forward_pass():
    """Test full forward pass with both discriminators."""
    print("Testing full forward pass...")
    
    # Create discriminators
    mpd = MultiPeriodDiscriminator(use_se_blocks=False)
    msd = MultiScaleDiscriminator(use_se_blocks=False)
    
    # Create test audio
    x = create_test_audio(2, 16384)  # Real audio
    y = create_test_audio(2, 16384)  # Generated audio
    
    # Test MPD forward pass
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(x, y)
    
    # Test MSD forward pass
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(x, y)
    
    # Verify that all outputs are properly formed
    assert len(y_df_hat_r) == 5, "MPD should return 5 real outputs"
    assert len(y_df_hat_g) == 5, "MPD should return 5 generated outputs"
    assert len(fmap_f_r) == 5, "MPD should return 5 real feature maps"
    assert len(fmap_f_g) == 5, "MPD should return 5 generated feature maps"
    
    assert len(y_ds_hat_r) == 3, "MSD should return 3 real outputs"
    assert len(y_ds_hat_g) == 3, "MSD should return 3 generated outputs"
    assert len(fmap_s_r) == 3, "MSD should return 3 real feature maps"
    assert len(fmap_s_g) == 3, "MSD should return 3 generated feature maps"
    
    print("Full forward pass test passed!")


def test_with_se_blocks():
    """Test discriminators with SE blocks enabled."""
    print("Testing discriminators with SE blocks...")
    
    # Create discriminators with SE blocks
    mpd_se = MultiPeriodDiscriminator(use_se_blocks=True)
    msd_se = MultiScaleDiscriminator(use_se_blocks=True)
    
    # Create test audio
    x = create_test_audio(2, 16384)  # Real audio
    y = create_test_audio(2, 16384)  # Generated audio
    
    # Test MPD forward pass with SE blocks
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd_se(x, y)
    
    # Test MSD forward pass with SE blocks
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd_se(x, y)
    
    # Verify that all outputs are properly formed
    assert len(y_df_hat_r) == 5, "MPD with SE should return 5 real outputs"
    assert len(y_df_hat_g) == 5, "MPD with SE should return 5 generated outputs"
    
    assert len(y_ds_hat_r) == 3, "MSD with SE should return 3 real outputs"
    assert len(y_ds_hat_g) == 3, "MSD with SE should return 3 generated outputs"
    
    print("SE blocks test passed!")


def run_all_tests():
    """Run all discriminator tests."""
    print("Running discriminator tests...")
    
    try:
        test_imports()
        test_se_blocks()
        test_discriminator_p()
        test_mpd()
        test_msd()
        test_mbsd()
        test_loss_functions()
        test_full_forward_pass()
        test_with_se_blocks()
        
        print("\nAll discriminator tests passed successfully!")
        return True
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run all tests
    run_all_tests()
