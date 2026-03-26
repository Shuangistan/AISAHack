"""
Smaller 4-level U-Net for 64×64 input images.

Adapted from prototype_unet/model.py to conform to the MechMNISTModel interface.
~8.1M parameters.
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from config import Config
from models.base import MechMNISTModel


@dataclass
class UNetSmallConfig(Config):
    model_name: str = "unet_small"

    # Architecture
    in_channels: int = 1
    features: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # Dataset — small U-Net expects 64×64 inputs
    img_size: int = 64
    batch_size: int = 64

    # Optimizer / scheduler tuned for the smaller model
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    scheduler: str = "plateau"


class _ConvBlock(nn.Module):
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


class MultiTaskUNet(MechMNISTModel):
    """
    Shared-encoder U-Net with three output heads.

    Input:  (B, 1, 64, 64)
    Outputs (dict):
        disp:  (B, 2, 64, 64)
        psi:   (B, 7)
        force: (B, 28)
    """

    def __init__(self, in_channels: int = 1, features: List[int] = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.pool = nn.MaxPool2d(2, 2)

        # Shared encoder
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for f in features:
            self.encoder_blocks.append(_ConvBlock(prev_ch, f))
            prev_ch = f

        bottleneck_ch = features[-1] * 2
        self.bottleneck = _ConvBlock(features[-1], bottleneck_ch)

        # Displacement decoder (U-Net with skip connections)
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, 2, stride=2))
            self.decoder_blocks.append(_ConvBlock(2 * f, f))
            prev_ch = f
        self.disp_head = nn.Conv2d(features[0], 2, 1)

        # Strain energy head
        self.psi_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        # Reaction force head
        self.force_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28),
        )

    @classmethod
    def from_config(cls, config) -> "MultiTaskUNet":
        return cls(
            in_channels=config.in_channels,
            features=list(config.features),
        )

    def forward(self, x: torch.Tensor) -> dict:
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        psi = self.psi_head(x)
        force = self.force_head(x)

        for upconv, dec, skip in zip(
            self.upconvs, self.decoder_blocks, reversed(skips)
        ):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        disp = self.disp_head(x)

        return {"psi": psi, "force": force, "disp": disp}
