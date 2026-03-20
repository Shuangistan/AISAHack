# Mechanical MNIST Cahn-Hilliard — U-Net Multi-Regression

A PyTorch implementation of a U-Net with auxiliary scalar heads for predicting
FE simulation results from Cahn-Hilliard binary microstructure images.

## Outputs predicted

| Target | Shape | Description |
|--------|-------|-------------|
| Strain energy (ΔΨ) | 7 | Change in strain energy at each displacement level |
| Reaction forces | 28 | Total reaction force at 4 boundaries × 7 displacement levels |
| Full-field displacement | 2 × 256 × 256 | (u_x, u_y) at final displacement d = 0.5 |

## Architecture

```
400×400 binary image
        │
   ┌────▼────┐
   │  Resize  │  → 256×256
   │  to 256  │
   └────┬────┘
        │
   Encoder (5 levels: 32 → 64 → 128 → 256 → 512)
        │
   ┌────▼────┐
   │Bottleneck│──────────────────────┐
   │  512 ch  │                      │
   └────┬────┘                  ┌────▼─────┐
        │                       │ GAP + MLP │
   Decoder (skip connections)   │  → 35     │
        │                       │ scalars   │
   ┌────▼────┐                  └───────────┘
   │  Conv 1×1│
   │  → 2 ch  │
   └────┬────┘
        │
   Full-field displacement (u_x, u_y)
```

## Quick start

```bash
# 1. Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn tqdm requests

# 2. Download & prepare dataset (~30 min on a fast connection)
python setup_data.py

# 3. Train
python train.py --data_root ./data --epochs 100 --batch_size 8
```


## Dataset setup options

```bash
python setup_data.py                          # full dataset (~3.5 GB)
python setup_data.py --skip-disp              # scalars only (~380 MB, much faster)
python setup_data.py --cases 1                # only Case 1 (37,523 samples)
python setup_data.py --cases 1 2              # Cases 1 and 2
python setup_data.py --dry-run                # preview files without downloading
python setup_data.py --keep-zips              # keep zip archives after extraction
```

After setup, the data directory will contain:

```
data/
  images/              # .txt files with 400×400 binary bitmaps
  summary_psi.txt      # (N, 7) strain energy at each displacement level
  summary_rxnforce.txt # (N, 28) reaction forces (4 boundaries × 7 levels)
  summary_disp_x.txt   # (N, M) x-displacement at d=0.5
  summary_disp_y.txt   # (N, M) y-displacement at d=0.5
```

Source: [OpenBU (hdl:2144/43971)](https://hdl.handle.net/2144/43971) — CC BY-SA 4.0

## Training

```bash
# Full training
python train.py --data_root ./data --epochs 100 --batch_size 8

# Quick test run (debug mode)
python train.py --data_root ./data --max_samples 500 --epochs 5

# Evaluate a checkpoint
python train.py --data_root ./data --evaluate --checkpoint best_model.pt

# Visualize predictions
python visualize.py --checkpoint checkpoints/best_model.pt --data_root ./data
```

## Loss Function

Multi-task weighted loss with learnable log-variance weights (Kendall et al.):

    L = (1/2σ₁²)·L_psi + (1/2σ₂²)·L_force + (1/2σ₃²)·L_disp + log(σ₁σ₂σ₃)

This automatically balances the three loss terms during training.
