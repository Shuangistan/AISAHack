"""
Configuration for Mechanical MNIST Cahn-Hilliard training.
Adjust these parameters based on your GPU memory and dataset location.
"""
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Model selection ──────────────────────────────────────────────────
    model_name: str = "unet"             # key into models.MODEL_REGISTRY

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

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 8                  # safe for RTX 3090/4090 at 256×256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"             # "adamw" | "adam"
    scheduler: str = "cosine"            # "cosine" | "plateau"
    warmup_epochs: int = 5               # only used when scheduler="cosine"
    lr_reduce_factor: float = 0.5        # only used when scheduler="plateau"
    lr_reduce_patience: int = 3          # only used when scheduler="plateau"

    # Multi-task loss: learnable uncertainty weights (Kendall et al.)
    use_learned_loss_weights: bool = False
    lambda_psi: float = 1.0
    lambda_force: float = 1.0
    lambda_disp: float = 1.0

    # ── Mixed precision ──────────────────────────────────────────────────
    use_amp: bool = False                 # automatic mixed precision

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 10                 # save checkpoint every N epochs
    patience: int = 15                   # early stopping patience

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir: str = "./logs"
    print_every: int = 1000              # print metrics every N batches

    def to_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))


# Convenience instance
cfg = Config()
