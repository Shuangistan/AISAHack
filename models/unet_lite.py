"""
Multi-task U-Net (64x64 architecture).
Adapted for the new registry-based project structure.
"""

from dataclasses import dataclass, field
import torch
import torch.nn as nn

from config import Config
from models.base import MechMNISTModel

# ── 1. Configuration ────────────────────────────────────────────────────────
@dataclass
class UNetLiteConfig(Config):
    model_name: str = "unet_lite"
    img_size: int = 64  # Overrides the default 256 to feed the model 64x64 images!
    in_channels: int = 1
    features: list = field(default_factory=lambda: [32, 64, 128, 256])


# ── 2. Building Blocks ──────────────────────────────────────────────────────
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


# ── 3. Main Model ───────────────────────────────────────────────────────────
class UNetLite(MechMNISTModel): # <-- Inherits from the new Base Class!
    """
    Shared encoder U-Net with three output heads (64x64).
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

        # Head 1: Displacement decoder
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        prev_ch = bottleneck_ch
        for f in reversed_features:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, 2, stride=2))
            self.decoder_blocks.append(ConvBlock(2 * f, f))
            prev_ch = f
        self.disp_head = nn.Conv2d(features[0], 2, 1)

        # Head 2: Strain energy 
        self.se_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        # Head 3: Reaction forces 
        self.rf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28),
        )

    # NEW: Factory Method required by base.py
    @classmethod
    def from_config(cls, config) -> "UNetLite":
        return cls(
            in_channels=config.in_channels,
            features=config.features
        )

    def forward(self, x):
        # Shared encoder
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Scalar heads 
        se = self.se_head(x)   
        rf = self.rf_head(x)   

        # Displacement decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        disp = self.disp_head(x)

        # NEW: Must return the strict Dictionary contract!
        return {"psi": se, "force": rf, "disp": disp}