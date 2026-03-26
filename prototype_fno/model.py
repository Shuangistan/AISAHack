"""
Multi-task Fourier Neural Operator: spectral convolutions with three output heads.

Heads:
  1. Displacement fields (2, 64, 64) — FNO decoder path
  2. Strain energy (7,) — FC layers from latent
  3. Reaction forces (28,) — FC layers from latent

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution via FFT: learns weights in Fourier space."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1  # number of Fourier modes to keep (height)
        self.modes2 = modes2  # number of Fourier modes to keep (width)

        scale = 1 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        # (B, in_ch, H, W) x (in_ch, out_ch, H, W) -> (B, out_ch, H, W)
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        B = x.shape[0]
        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_ch, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock(nn.Module):
    """One FNO layer: spectral conv + pointwise conv + residual + activation."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.bn = nn.BatchNorm2d(width)

    def forward(self, x):
        return F.gelu(self.bn(self.spectral(x) + self.pointwise(x)))


class MultiTaskFNO(nn.Module):
    """
    FNO with three output heads for Mechanical MNIST CH.

    Input:  (B, 1, 64, 64)
    Outputs:
        disp:  (B, 2, 64, 64)  — displacement fields
        se:    (B, 7)           — strain energy
        rf:    (B, 28)          — reaction forces
    """

    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        self.modes = modes
        self.width = width

        # Lift: 1 channel -> width channels
        self.lift = nn.Conv2d(1, width, 1)

        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(width, modes, modes) for _ in range(n_layers)
        ])

        # Head 1: Displacement (project back to spatial output)
        self.disp_head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 2, 1),
        )

        # Head 2: Strain energy (global avg pool -> FC)
        self.se_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        # Head 3: Reaction forces (global avg pool -> FC)
        self.rf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28),
        )

    def forward(self, x):
        # Lift to higher dimensional space
        x = self.lift(x)  # (B, width, 64, 64)

        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)  # (B, width, 64, 64)

        # Scalar heads from latent features
        se = self.se_head(x)    # (B, 7)
        rf = self.rf_head(x)    # (B, 28)

        # Displacement head
        disp = self.disp_head(x)  # (B, 2, 64, 64)

        return disp, se, rf
