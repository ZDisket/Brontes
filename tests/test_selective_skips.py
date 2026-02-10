import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unet1d import UNet1DRefiner

def test_selective_skips():
    # Model parameters
    in_ch = 1
    out_ch = 1
    base_ch = 32
    depth = 4
    T = 1024
    
    x = torch.randn(1, in_ch, T)
    
    print("--- Testing default (all skips) ---")
    model_default = UNet1DRefiner(in_ch, out_ch, base_ch=base_ch, depth=depth)
    out_default = model_default(x)
    print(f"Output shape: {out_default.shape}")
    assert out_default.shape == x.shape
    assert model_default.skip_layer_indexes == [0, 1, 2, 3]

    print("\n--- Testing selective skips (positive indexes: [0, 2]) ---")
    model_selective = UNet1DRefiner(in_ch, out_ch, base_ch=base_ch, depth=depth, skip_layer_indexes=[0, 2])
    out_selective = model_selective(x)
    print(f"Output shape: {out_selective.shape}")
    assert out_selective.shape == x.shape
    assert model_selective.skip_layer_indexes == [0, 2]

    print("\n--- Testing negative indexing ([-1, -3]) ---")
    # depth=4, so -1 -> 3, -3 -> 1. Combined: [1, 3]
    model_negative = UNet1DRefiner(in_ch, out_ch, base_ch=base_ch, depth=depth, skip_layer_indexes=[-1, -3])
    out_negative = model_negative(x)
    print(f"Output shape: {out_negative.shape}")
    assert out_negative.shape == x.shape
    assert model_negative.skip_layer_indexes == [3, 1]

    print("\n--- Testing no skips ([]) ---")
    model_no_skips = UNet1DRefiner(in_ch, out_ch, base_ch=base_ch, depth=depth, skip_layer_indexes=[])
    out_no_skips = model_no_skips(x)
    print(f"Output shape: {out_no_skips.shape}")
    assert out_no_skips.shape == x.shape
    assert model_no_skips.skip_layer_indexes == []

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_selective_skips()
