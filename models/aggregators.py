"""
This module contains custom aggregation layers for neural networks.
"""
import torch
import torch.nn.functional as F
from torch import nn

class GeM(nn.Module):
    """
    Generalized Mean (GeM) Pooling layer.
    """
    def __init__(self, p=3, eps=1e-6, input_size=512, output_size=512):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        """
        Forward pass for the GeM layer.
        """
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = self.pool(x)
        x = x.pow(1. / self.p)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x

class Average(nn.Module):
    """
    Average Pooling layer.
    """
    def __init__(self, input_size=512, output_size=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        """
        Forward pass for the Average layer.
        """
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x

class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=512,
                 in_h=14,
                 in_w=14,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x
    

