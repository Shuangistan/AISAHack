"""
U-Net with auxiliary scalar heads for multi-output FE surrogate modeling.

Architecture
------------
Encoder:  5 levels of double-conv + max-pool  (1 → 32 → 64 → 128 → 256 → 512)
Bottleneck: 512-channel feature map
Scalar head: GAP → MLP → 35 outputs (7 strain energy + 28 reaction force)
Decoder:  5 levels of up-conv + skip-concat + double-conv → 2-channel displacement
"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.base import MechMNISTModel


# ═══════════════════════════════════════════════════════════════════════════
# Model-specific configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UNetConfig(Config):
    """Configuration for UNetMultiRegression. Extends shared Config."""
    model_name: str = "unet"
    in_channels: int = 1                 # binary image input
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    use_batchnorm: bool = True
    dropout: float = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two sequential 3×3 conv → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i, (ic, oc) in enumerate([(in_ch, out_ch), (out_ch, out_ch)]):
            layers.append(nn.Conv2d(ic, oc, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0 and i == 1:
                layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv followed by 2×2 max-pool for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, use_bn, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)      # skip connection source
        downsampled = self.pool(features)
        return features, downsampled


class DecoderBlock(nn.Module):
    """Upsample → concatenate skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, use_bn, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════
# Scalar regression head
# ═══════════════════════════════════════════════════════════════════════════

class ScalarHead(nn.Module):
    """
    Global average pooling → MLP → scalar outputs.

    Produces two groups:
      - strain_energy  (n_psi values)
      - reaction_force (n_force values)
    """

    def __init__(self, in_ch: int, n_psi: int = 7, n_force: int = 28, hidden: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.head_psi = nn.Linear(hidden, n_psi)
        self.head_force = nn.Linear(hidden, n_force)

    def forward(self, x: torch.Tensor):
        x = self.gap(x).flatten(1)       # (B, C)
        h = self.shared(x)               # (B, hidden)
        psi = self.head_psi(h)            # (B, n_psi)
        force = self.head_force(h)        # (B, n_force)
        return psi, force


# ═══════════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════════

class UNetMultiRegression(MechMNISTModel):
    """
    U-Net with auxiliary scalar heads for Mechanical MNIST CH.

    Inputs:
        x: (B, 1, H, W) binary Cahn-Hilliard image

    Outputs (dict):
        psi:   (B, 7)           strain energy at each displacement level
        force: (B, 28)          reaction force at 4 boundaries × 7 levels
        disp:  (B, 2, H, W)    full-field displacement (u_x, u_y)
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: list = None,
        n_psi: int = 7,
        n_force: int = 28,
        disp_channels: int = 2,
        use_bn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 512]

        ec = encoder_channels  # shorthand

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, ec[0], use_bn, dropout)
        self.enc2 = EncoderBlock(ec[0], ec[1], use_bn, dropout)
        self.enc3 = EncoderBlock(ec[1], ec[2], use_bn, dropout)
        self.enc4 = EncoderBlock(ec[2], ec[3], use_bn, dropout)

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = DoubleConv(ec[3], ec[4], use_bn, dropout)

        # ── Scalar head (branches from bottleneck) ───────────────────────
        self.scalar_head = ScalarHead(ec[4], n_psi, n_force, hidden=256)

        # ── Decoder ──────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(ec[4], ec[3], ec[3], use_bn, dropout)
        self.dec3 = DecoderBlock(ec[3], ec[2], ec[2], use_bn, dropout)
        self.dec2 = DecoderBlock(ec[2], ec[1], ec[1], use_bn, dropout)
        self.dec1 = DecoderBlock(ec[1], ec[0], ec[0], use_bn, dropout)

        # ── Output projection ────────────────────────────────────────────
        self.out_conv = nn.Conv2d(ec[0], disp_channels, kernel_size=1)

    @classmethod
    def from_config(cls, config) -> "UNetMultiRegression":
        return cls(
            in_channels=config.in_channels,
            encoder_channels=config.encoder_channels,
            n_psi=config.n_psi,
            n_force=config.n_force,
            disp_channels=config.disp_channels,
            use_bn=config.use_batchnorm,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Encoder path (save skip connections)
        s1, x = self.enc1(x)     # s1: (B, 32, H, W)
        s2, x = self.enc2(x)     # s2: (B, 64, H/2, W/2)
        s3, x = self.enc3(x)     # s3: (B, 128, H/4, W/4)
        s4, x = self.enc4(x)     # s4: (B, 256, H/8, W/8)

        # Bottleneck
        x = self.bottleneck(x)   # (B, 512, H/16, W/16)

        # Scalar predictions (from bottleneck features)
        psi, force = self.scalar_head(x)

        # Decoder path (with skip connections)
        x = self.dec4(x, s4)     # (B, 256, H/8, W/8)
        x = self.dec3(x, s3)     # (B, 128, H/4, W/4)
        x = self.dec2(x, s2)     # (B, 64, H/2, W/2)
        x = self.dec1(x, s1)     # (B, 32, H, W)

        # Displacement field
        disp = self.out_conv(x)  # (B, 2, H, W)

        return {"psi": psi, "force": force, "disp": disp}


# ═══════════════════════════════════════════════════════════════════════════
# Multi-task loss with learned uncertainty weights
# ═══════════════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task learning.

    Learns log(σ²) for each task to automatically balance losses.
    Reference: Kendall, Gal & Cipolla, "Multi-Task Learning Using
    Uncertainty to Weigh Losses" (CVPR 2018).

    L = Σ_i [ (1 / 2σ_i²) · L_i + log(σ_i) ]
    """

    def __init__(self, n_tasks: int = 3, use_learned: bool = True):
        super().__init__()
        self.use_learned = use_learned
        if use_learned:
            # Initialize log(σ²) to 0 → σ² = 1 → equal weighting initially
            self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self,
        loss_psi: torch.Tensor,
        loss_force: torch.Tensor,
        loss_disp: torch.Tensor,
        fixed_weights: tuple = (1.0, 1.0, 0.1),
    ) -> tuple:
        losses = torch.stack([loss_psi, loss_force, loss_disp])

        if self.use_learned:
            # precision = 1 / σ²  = exp(-log_var)
            log_vars_clamped = self.log_vars.clamp(-6, 6)
            precisions = torch.exp(-log_vars_clamped)
            weighted = precisions * losses + log_vars_clamped
            total = weighted.sum()
            weights_display = precisions.detach()
        else:
            w = torch.tensor(fixed_weights, device=losses.device)
            total = (w * losses).sum()
            weights_display = w

        return total, weights_display


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: model summary
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = UNetMultiRegression()
    x = torch.randn(2, 1, 256, 256)
    out = model(x)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Input shape:      {x.shape}")
    print(f"Strain energy:    {out['psi'].shape}")      # (2, 7)
    print(f"Reaction force:   {out['force'].shape}")     # (2, 28)
    print(f"Displacement:     {out['disp'].shape}")      # (2, 2, 256, 256)

    # Test loss
    criterion = MultiTaskLoss(n_tasks=3, use_learned=True)
    l1 = F.mse_loss(out["psi"], torch.randn_like(out["psi"]))
    l2 = F.mse_loss(out["force"], torch.randn_like(out["force"]))
    l3 = F.mse_loss(out["disp"], torch.randn_like(out["disp"]))
    total, weights = criterion(l1, l2, l3)
    print(f"\nTotal loss: {total.item():.4f}")
    print(f"Task weights: {weights.cpu().numpy()}")
