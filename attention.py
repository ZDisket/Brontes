import torch
import torch.nn as nn


class ChannelAttention1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * att


class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.cat([mean_map, max_map], dim=1)
        att = torch.sigmoid(self.conv(m))
        return x * att


class CBAM1d(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention1d(channels, reduction=reduction)
        self.sa = SpatialAttention1d(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
