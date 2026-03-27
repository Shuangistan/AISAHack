"""
Multi-task Fourier Neural Operator for 64×64 input images.

Adapted from prototype_fno/model.py to conform to the MechMNISTModel interface.

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.base import MechMNISTModel


@dataclass
class FNOConfig(Config):
    model_name: str = "fno"

    # Architecture
    modes: int = 16       # Fourier modes to keep per spatial axis
    width: int = 64       # channel dimension throughout FNO layers
    n_layers: int = 4     # number of FNO blocks

    # Dataset
    img_size: int = 64
    batch_size: int = 64

    # Training
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    scheduler: str = "plateau"
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 3
    patience: int = 15

    # Multi-task loss weights
    use_learned_loss_weights: bool = False
    lambda_psi: float = 1.0
    lambda_force: float = 1.0
    lambda_disp: float = 1.0


class _SpectralConv2d(nn.Module):
    """Spectral convolution via FFT: learns weights in Fourier space."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            B, self.out_ch, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class _FNOBlock(nn.Module):
    """One FNO layer: spectral conv + pointwise conv + residual + activation."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = _SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.bn = nn.BatchNorm2d(width)

    def forward(self, x):
        return F.gelu(self.bn(self.spectral(x) + self.pointwise(x)))


class MultiTaskFNO(MechMNISTModel):
    """
    FNO with three output heads for Mechanical MNIST CH.

    Input:  (B, 1, 64, 64)
    Outputs (dict):
        disp:  (B, 2, 64, 64)
        psi:   (B, 7)
        force: (B, 28)
    """

    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(1, width, 1)

        self.fno_layers = nn.ModuleList([
            _FNOBlock(width, modes, modes) for _ in range(n_layers)
        ])

        self.disp_head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 2, 1),
        )

        self.se_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        self.rf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28),
        )

    @classmethod
    def from_config(cls, config) -> "MultiTaskFNO":
        return cls(
            modes=config.modes,
            width=config.width,
            n_layers=config.n_layers,
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.lift(x)

        for layer in self.fno_layers:
            x = layer(x)

        psi = self.se_head(x)
        force = self.rf_head(x)
        disp = self.disp_head(x)

        return {"psi": psi, "force": force, "disp": disp}
