"""
Multi-task U-Net: shared encoder with three output heads.

Heads:
  1. Displacement fields (2, 64, 64) — U-Net decoder with skip connections
  2. Strain energy (7,) — FC layers from bottleneck
  3. Reaction forces (28,) — FC layers from bottleneck

~8.1M parameters.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskUNet(nn.Module):
    """
    Shared encoder U-Net with three output heads.

    Input:  (B, 1, 64, 64)
    Outputs:
        disp:  (B, 2, 64, 64)  — displacement fields
        se:    (B, 7)           — strain energy
        rf:    (B, 28)          — reaction forces
    """

    def __init__(self, in_channels: int = 1, features: list = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)

        # Shared encoder
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.encoder_blocks.append(ConvBlock(prev_ch, f))
            prev_ch = f

        bottleneck_ch = features[-1] * 2  # 512
        self.bottleneck = ConvBlock(features[-1], bottleneck_ch)

        # Head 1: Displacement decoder (U-Net style with skip connections)
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_ch = bottleneck_ch
        for f in reversed_features:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, 2, stride=2))
            self.decoder_blocks.append(ConvBlock(2 * f, f))
            prev_ch = f
        self.disp_head = nn.Conv2d(features[0], 2, 1)

        # Head 2: Strain energy (bottleneck -> global avg pool -> FC)
        self.se_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        # Head 3: Reaction forces (bottleneck -> global avg pool -> FC)
        self.rf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28),
        )

    def forward(self, x):
        # Shared encoder
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)  # (B, 512, 4, 4)

        # Scalar heads from bottleneck
        se = self.se_head(x)    # (B, 7)
        rf = self.rf_head(x)    # (B, 28)

        # Displacement decoder
        for upconv, dec, skip in zip(
            self.upconvs, self.decoder_blocks, reversed(skips)
        ):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        disp = self.disp_head(x)  # (B, 2, 64, 64)

        return disp, se, rf
