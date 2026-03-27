"""
Multi-task Swin Transformer for 64×64 input images.

Architecture:
  - Patch embedding (patch_size=2): 64×64 → 32×32 tokens, dim=64
  - 3 Swin stages with alternating W-MSA / SW-MSA blocks:
      Stage 1: 32×32 tokens, dim= 64, heads=2, depth=2  → PatchMerge
      Stage 2: 16×16 tokens, dim=128, heads=4, depth=2  → PatchMerge
      Stage 3:  8×8  tokens, dim=256, heads=8, depth=2
  - CNN decoder with U-Net skip connections → 64×64 displacement field
  - Global-average-pool scalar heads for ψ (7) and reaction force (28)

Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer
           using Shifted Windows" (ICCV 2021)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.base import MechMNISTModel


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SwinConfig(Config):
    model_name: str = "swin"

    # Architecture
    patch_size: int = 2
    embed_dim: int = 64
    depths: List[int] = field(default_factory=lambda: [2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [2, 4, 8])
    window_size: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # Dataset
    img_size: int = 64
    batch_size: int = 32

    # Training
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    patience: int = 15

    # Multi-task loss weights
    use_learned_loss_weights: bool = False
    lambda_psi: float = 1.0
    lambda_force: float = 1.0
    lambda_disp: float = 1.0


# ── Swin building blocks ───────────────────────────────────────────────────────

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition (B, H, W, C) into non-overlapping windows.

    Returns
    -------
    (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition: (num_windows*B, ws, ws, C) → (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size ** 2))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """
    Window multi-head self-attention with learnable relative position bias.

    Handles both W-MSA (mask=None) and SW-MSA (mask provided by SwinBlock).
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.window_size = window_size

        # Relative position bias table: (2W-1)² × num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute flat relative-position index for each token pair
        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        )  # (2, W, W)
        coords_flat = coords.flatten(1)  # (2, W²)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, W², W²)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))  # (W², W²)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : (num_windows * B, N, C)   N = window_size²
        mask : (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B_, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    """
    One Swin Transformer block.

    Even-indexed blocks use W-MSA (shift_size=0).
    Odd-indexed blocks use SW-MSA (shift_size=window_size//2) with cyclic shift.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int, shift_size: int,
                 mlp_ratio: float, drop: float, attn_drop: float):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def _attn_mask(self, H: int, W: int, device) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x : (B, H*W, C)"""
        B, _, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_win = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        x_win = self.attn(x_win, mask=self._attn_mask(H, W, x.device))
        x = window_reverse(x_win.view(-1, self.window_size, self.window_size, C),
                           self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C) + shortcut
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Embed image patches via stride-p convolution: (B,1,H,W) → (B, H/p*W/p, dim)."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


class PatchMerging(nn.Module):
    """2× spatial downsampling + 2× channel expansion."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = self.reduction(self.norm(x.view(B, -1, 4 * C)))
        return x, H // 2, W // 2


class SwinStage(nn.Module):
    """Sequence of Swin blocks + optional PatchMerging downsampler."""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float, drop: float, attn_drop: float, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            x = blk(x, H, W)
        skip = x  # tokens before downsampling — used as decoder skip connection
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, skip


# ── Decoder helpers ────────────────────────────────────────────────────────────

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


# ── Top-level model ────────────────────────────────────────────────────────────

class MultiTaskSwin(MechMNISTModel):
    """
    Swin Transformer encoder + CNN decoder for multi-task FE surrogate.

    Input:  (B, 1, 64, 64)
    Outputs (dict):
        psi:   (B, 7)       strain energy at 7 displacement levels
        force: (B, 28)      reaction forces at 4 boundaries × 7 levels
        disp:  (B, 2, 64, 64)  displacement field (u_x, u_y)

    Encoder (patch_size=2, embed_dim=64, 3 stages):
        patch embed  → (B, 32×32, 64)
        stage 1      → skip at 32×32, dim=64   then merge → 16×16, dim=128
        stage 2      → skip at 16×16, dim=128  then merge →  8×8,  dim=256
        stage 3      → bottleneck at 8×8, dim=256

    Decoder:
        8×8  →(up)→ 16×16 cat skip2 → 16×16
        16×16→(up)→ 32×32 cat skip1 → 32×32
        32×32→(up)→ 64×64            → disp head
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 2,
        embed_dim: int = 64,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2]
        if num_heads is None:
            num_heads = [2, 4, 8]

        self.window_size = window_size
        num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(num_stages)]  # [64, 128, 256]

        # Encoder
        self.patch_embed = PatchEmbed(1, embed_dim, patch_size)
        self.stages = nn.ModuleList([
            SwinStage(
                dim=dims[i], depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                drop=dropout, attn_drop=attn_dropout,
                downsample=(i < num_stages - 1),
            )
            for i in range(num_stages)
        ])

        bottleneck_dim = dims[-1]  # 256

        # Scalar heads
        self.psi_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )
        self.force_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 28),
        )

        # CNN decoder with skip connections from encoder stages
        self.up1 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)   # 8→16, 256→128
        self.dec1 = _ConvBlock(dims[1] * 2, dims[1])                    # cat skip from stage 2

        self.up2 = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)   # 16→32, 128→64
        self.dec2 = _ConvBlock(dims[0] * 2, dims[0])                    # cat skip from stage 1

        self.up3 = nn.ConvTranspose2d(dims[0], 32, 2, stride=2)        # 32→64
        self.dec3 = _ConvBlock(32, 32)

        self.disp_head = nn.Conv2d(32, 2, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @classmethod
    def from_config(cls, config) -> "MultiTaskSwin":
        return cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depths=list(config.depths),
            num_heads=list(config.num_heads),
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
        )

    @staticmethod
    def _to_spatial(tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W, C) → (B, C, H, W)"""
        B, _, C = tokens.shape
        return tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> dict:
        # ── Encoder ──────────────────────────────────────────────────────────
        x, H, W = self.patch_embed(x)   # (B, 32*32, 64)

        skip_tokens, skip_HWs = [], []
        for stage in self.stages:
            x, H, W, skip = stage(x, H, W)
            skip_tokens.append(skip)
            skip_HWs.append((H, W))     # spatial size AFTER downsampling / final stage

        # Spatial sizes BEFORE each downsampling (i.e. where the skip was captured):
        #   stage 0 skip: (32, 32), stage 1 skip: (16, 16), stage 2 skip: (8, 8)
        # Recover the pre-downsampling sizes:
        #   skip[0] was emitted before the first PatchMerging, so its H×W is
        #   skip_HWs[0] * 2 for stages that downsample, except the last stage.
        # Simpler: the skip H×W = number of tokens = len(skip_tokens[i]) / B
        B = x.shape[0]
        skip_sizes = []
        for s in skip_tokens:
            n = s.shape[1]
            hw = int(n ** 0.5)
            skip_sizes.append((hw, hw))

        # Bottleneck: tokens at 8×8, dim=256
        bottleneck = self._to_spatial(x, H, W)   # (B, 256, 8, 8)

        # ── Scalar heads ─────────────────────────────────────────────────────
        psi = self.psi_head(bottleneck)
        force = self.force_head(bottleneck)

        # ── Decoder ──────────────────────────────────────────────────────────
        sh1, sw1 = skip_sizes[0]   # 32×32
        sh2, sw2 = skip_sizes[1]   # 16×16

        skip2 = self._to_spatial(skip_tokens[1], sh2, sw2)  # (B, 128, 16, 16)
        skip1 = self._to_spatial(skip_tokens[0], sh1, sw1)  # (B,  64, 32, 32)

        d = self.dec1(torch.cat([self.up1(bottleneck), skip2], dim=1))  # (B, 128, 16, 16)
        d = self.dec2(torch.cat([self.up2(d), skip1], dim=1))           # (B,  64, 32, 32)
        d = self.dec3(self.up3(d))                                       # (B,  32, 64, 64)

        disp = self.disp_head(d)                                         # (B,   2, 64, 64)

        return {"psi": psi, "force": force, "disp": disp}
