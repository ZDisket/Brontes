import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unet1d import UNet1DRefiner

def test_skip_scales():
    # Model parameters
    in_ch = 1
    out_ch = 1
    base_ch = 16
    depth = 2 # Small depth for quick testing
    
    # Input tensor
    x = torch.randn(1, in_ch, 256)
    
    print("Testing default scales (1.0)...")
    model_default = UNet1DRefiner(in_ch, out_ch, base_ch=base_ch, depth=depth)
    out_default = model_default(x)
    print(f"Output shape: {out_default.shape}")

    print("\nTesting selective skip with negative indexing and scales...")
    # skip_layer_indexes=[-1] means the last encoder layer's skip connection
    # depth=2, so -1 corresponds to enc_idx=1
    model_scaled = UNet1DRefiner(
        in_ch, out_ch, base_ch=base_ch, depth=depth,
        skip_layer_indexes=[-1, -2],
        skip_residual_scales=[0.1, 0.5]
    )
    out_scaled = model_scaled(x)
    print(f"Output shape: {out_scaled.shape}")
    
    print("\nTesting zero scale (should be different from default)...")
    model_zero = UNet1DRefiner(
        in_ch, out_ch, base_ch=base_ch, depth=depth,
        skip_layer_indexes=[-1],
        skip_residual_scales=[0.0]
    )
    out_zero = model_zero(x)
    
    model_no_skip = UNet1DRefiner(
        in_ch, out_ch, base_ch=base_ch, depth=depth,
        skip_layer_indexes=[] # No skips at all
    )
    out_no_skip = model_no_skip(x)
    
    # If scale is 0.0 for all layers (here only -1), it should be similar to no skip for that layer
    # Note: UpBlock1d handles None skip by skipping x = x + skip
    # Let's check if it runs.
    
    print("\nAll runs completed successfully!")

if __name__ == "__main__":
    test_skip_scales()
