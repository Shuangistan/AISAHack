# Mechanical MNIST Cahn-Hilliard — Multi-Model FE Surrogate

A PyTorch surrogate modelling framework for predicting finite element (FE) simulation
results from binary Cahn-Hilliard microstructure images. Supports multiple model
architectures through a shared training pipeline, dataset, and experiment tracking system.

**Dataset:** [Mechanical MNIST Cahn-Hilliard](https://hdl.handle.net/2144/43971) —
~104,813 samples, CC BY-SA 4.0

## Prediction targets

| Target | Shape | Description |
|---|---|---|
| Strain energy (ΔΨ) | `(7,)` | Change in strain energy at 7 displacement levels |
| Reaction forces | `(28,)` | Total reaction force at 4 boundaries × 7 levels |
| Displacement field | `(2, H, W)` | Full-field (u_x, u_y) at final displacement d = 0.5 |

## Models

| Key | Architecture | Input | Parameters | Optimizer | Scheduler |
|---|---|---|---|---|---|
| `unet` | 5-level U-Net, 5-level encoder/decoder + scalar head | 256×256 | ~8M | AdamW | Cosine + warmup |
| `unet_small` | 4-level U-Net, shared encoder + 3 heads | 64×64 | ~2M | Adam | ReduceLROnPlateau |
| `fno` | Fourier Neural Operator, 4 FNO blocks + 3 heads | 64×64 | ~2M | Adam | ReduceLROnPlateau |

Each model has its own configuration class that pins all training hyperparameters
independently of the shared defaults in `config.py`.

## Repository structure

```
├── config.py               # Shared base Config dataclass
├── dataset.py              # Dataset, normalization, DataLoader factory
├── setup_data.py           # Dataset download, extraction, and consolidation
├── train.py                # Training loop, Trainer class, CLI entry point
├── visualize.py            # Inference, comparison plots, loss curves
│
├── models/
│   ├── base.py             # MechMNISTModel abstract base class
│   ├── __init__.py         # MODEL_REGISTRY, CONFIG_REGISTRY, get_model()
│   ├── unet.py             # UNetConfig + UNetMultiRegression
│   ├── unet_small.py       # UNetSmallConfig + MultiTaskUNet
│   └── fno.py              # FNOConfig + MultiTaskFNO
│
├── experiments/
│   ├── runs/               # Auto-generated per-run directories
│   │   └── <model>_<timestamp>/
│   │       ├── config.json         # Full config snapshot
│   │       ├── norm_stats.npz      # Normalization statistics
│   │       ├── best_model.pt       # Best checkpoint (lowest val loss)
│   │       ├── checkpoint_epoch*.pt
│   │       └── training_history.json
│   └── reference_models/   # Pre-trained reference checkpoints, one per model
│       ├── unet/
│       ├── unet_small/
│       └── fno/
│
├── data/                   # Downloaded dataset (created by setup_data.py)
```

## Setup

### Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm requests
```

### Download and prepare the dataset

```bash
# Full dataset (~3.5 GB) — all three cases, all outputs
python setup_data.py

# Scalars only — no displacement fields (~380 MB, much faster)
python setup_data.py --skip-disp

# Subset — Cases 1 and 2 only
python setup_data.py --cases 1 2

# Preview files without downloading
python setup_data.py --dry-run
```

After setup, the data directory will contain:

```
data/
  images/                      # 400×400 binary microstructure images (.txt)
  images_64x64/                # Pre-downsampled 64×64 case summary files
  summary_images_64x64.npy     # Consolidated image array  (N, 64, 64)
  summary_images_400x400.npy   # Consolidated image array  (N, 400, 400)
  summary_psi.npy              # Strain energy             (N, 7)
  summary_rxnforce.npy         # Reaction forces           (N, 28)
  summary_disp_x.npy           # x-displacement field      (N, 64, 64)
  summary_disp_y.npy           # y-displacement field      (N, 64, 64)
  norm_stats.npz               # Dataset-wide normalization statistics
```

The dataset loader uses memory-mapped `.npy` files for efficient loading. Image
resolution is selected automatically based on `img_size` in the model config.

## Training

### Basic usage

Select a model with `--model`. All hyperparameters default to the values defined
in that model's config class — no manual flag changes needed when switching models.

```bash
# Train the full U-Net (256×256, AdamW, cosine schedule)
python train.py --model unet --data_root ./data

# Train the small U-Net (64×64, Adam, plateau schedule)
python train.py --model unet_small --data_root ./data

# Train the FNO (64×64, Adam, plateau schedule)
python train.py --model fno --data_root ./data
```

### Common overrides

```bash
# Override specific hyperparameters
python train.py --model unet --epochs 50 --batch_size 16 --lr 5e-4

# Quick debug run on a small subset
python train.py --model fno --data_root ./data --max_samples 500 --epochs 3

# Disable mixed precision (e.g. on CPU)
python train.py --model unet --no_amp
```

### Resuming a run

```bash
python train.py --run_dir experiments/runs/unet_20260327_120000 --resume
```

### Evaluating a checkpoint

```bash
# Evaluate the best checkpoint from a run
python train.py --run_dir experiments/runs/fno_20260327_130000 --evaluate

# Evaluate a specific checkpoint
python train.py --run_dir experiments/runs/fno_20260327_130000 \
                --evaluate --checkpoint checkpoint_epoch050.pt
```

## Experiment runs

Each training run creates a timestamped directory under `experiments/runs/`:

```
experiments/runs/unet_20260327_120000/
```

This directory stores everything needed to reproduce or continue the run:
- `config.json` — full config snapshot, including all model-specific fields
- `norm_stats.npz` — normalization statistics computed from the training split
- `best_model.pt` — checkpoint with the lowest validation loss
- `training_history.json` — per-epoch train/val losses for all targets

## Visualization

```bash
# Plot predictions vs ground truth for 5 random test samples
python visualize.py --run_dir experiments/runs/unet_20260327_120000 \
                    --data_root ./data --n_samples 5

# Also plot training loss curves
python visualize.py --run_dir experiments/runs/unet_20260327_120000 \
                    --data_root ./data --history auto

# Save figures to a custom directory
python visualize.py --run_dir experiments/runs/fno_20260327_130000 \
                    --data_root ./data --out_dir ./figures/fno
```

Each sample produces a 3×3 panel figure:
- **Row 1:** Input microstructure | Strain energy curve (pred vs GT) | Reaction forces (pred vs GT)
- **Row 2:** Predicted u_x | Predicted u_y | Predicted |u|
- **Row 3:** Ground truth u_x | Ground truth u_y | Displacement error |Δu|

## Adding a new model

1. **Create `models/my_model.py`** — define a config subclass and a model class:

```python
from dataclasses import dataclass
from config import Config
from models.base import MechMNISTModel

@dataclass
class MyModelConfig(Config):
    model_name: str = "my_model"
    # pin all relevant hyperparameters here
    img_size: int = 64
    batch_size: int = 32
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    # model-specific fields
    hidden_dim: int = 128

class MyModel(MechMNISTModel):
    @classmethod
    def from_config(cls, config) -> "MyModel":
        return cls(hidden_dim=config.hidden_dim)

    def forward(self, x) -> dict:
        ...
        return {"psi": psi, "force": force, "disp": disp}
```

2. **Register in `models/__init__.py`:**

```python
from models.my_model import MyModelConfig, MyModel

MODEL_REGISTRY["my_model"] = MyModel
CONFIG_REGISTRY["my_model"] = MyModelConfig
```

3. **Train:**

```bash
python train.py --model my_model --data_root ./data
```

## Loss function

All models use a three-task MSE loss over strain energy, reaction forces, and
displacement. Two weighting modes are available, configured per model:

**Fixed weights** (default for all models):

```
L = λ_ψ · MSE(ψ) + λ_F · MSE(F) + λ_u · MSE(u)
```

**Learned weights** — Kendall et al. homoscedastic uncertainty weighting
(available for `unet`, enable via `use_learned_loss_weights=True` in `UNetConfig`):

```
L = (1/2σ_ψ²)·MSE(ψ) + (1/2σ_F²)·MSE(F) + (1/2σ_u²)·MSE(u) + log(σ_ψ σ_F σ_u)
```

The log-variance terms σ² are learnable parameters that automatically balance the
three tasks during training.

## Results

| Model | ψ R² | F R² | u MSE |
|---|---|---|---|
| `unet` | — | — | — |
| `unet_small` | — | — | — |
| `fno` | — | — | — |

*To be filled after benchmark runs.*
