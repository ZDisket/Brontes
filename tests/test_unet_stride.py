import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unet1d import UNet1DRefiner

def test_stride(stride, T=48000):
    print(f"Testing stride={stride}, T={T}")
    model = UNet1DRefiner(
        in_ch=1,
        out_ch=1,
        base_ch=16,
        depth=3,
        stride=stride
    )
    
    x = torch.randn(1, 1, T)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} != {y.shape}"
    print("Test passed!\n")

def test_lstm_bottleneck(bidirectional=False):
    print(f"Testing LSTM bottleneck (bidirectional={bidirectional})")
    model = UNet1DRefiner(
        in_ch=1,
        out_ch=1,
        base_ch=16,
        depth=3,
        use_lstm_bottleneck=True,
        lstm_bidirectional=bidirectional
    )
    x = torch.randn(1, 1, 48000)
    with torch.no_grad():
        y = model(x)
    assert x.shape == y.shape, f"Shape mismatch with LSTM: {x.shape} != {y.shape}"
    print("LSTM Test passed!\n")

if __name__ == "__main__":
    test_stride(stride=2)
    test_stride(stride=4)
    test_stride(stride=2, T=32000)
    test_stride(stride=3, T=24000)
    test_lstm_bottleneck(bidirectional=False)
    test_lstm_bottleneck(bidirectional=True)
