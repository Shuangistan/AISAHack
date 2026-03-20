"""
Configuration for Mechanical MNIST Cahn-Hilliard U-Net training.
Adjust these parameters based on your GPU memory and dataset location.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────
    data_root: str = "./data"
    img_size: int = 256                  # resize 400×400 → 256×256
    train_split: float = 0.85
    val_split: float = 0.10              # remaining 0.05 = test
    num_workers: int = 4
    pin_memory: bool = True

    # ── Displacement levels in the FE simulation ─────────────────────────
    # d = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]  →  7 levels
    n_disp_levels: int = 7
    n_boundaries: int = 4                # 4 edges (equibiaxial)

    # ── Output dimensions ────────────────────────────────────────────────
    n_psi: int = 7                       # strain energy at each disp level
    n_force: int = 28                    # 4 boundaries × 7 levels
    n_scalar_total: int = 35             # n_psi + n_force
    disp_channels: int = 2              # (u_x, u_y)

    # ── Model ────────────────────────────────────────────────────────────
    in_channels: int = 1                 # binary image
    encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    use_batchnorm: bool = True
    dropout: float = 0.1

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 8                  # safe for RTX 3090/4090 at 256×256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"            # "cosine" | "step" | "plateau"
    warmup_epochs: int = 5

    # Multi-task loss: learnable uncertainty weights (Kendall et al.)
    use_learned_loss_weights: bool = True
    # Fallback fixed weights if learned weights disabled
    lambda_psi: float = 1.0
    lambda_force: float = 1.0
    lambda_disp: float = 0.1            # lower because field loss is much larger

    # ── Mixed precision ──────────────────────────────────────────────────
    use_amp: bool = True                 # automatic mixed precision

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10                 # save checkpoint every N epochs
    patience: int = 15                   # early stopping patience

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir: str = "./logs"
    print_every: int = 50                # print metrics every N batches


# Convenience instance
cfg = Config()
