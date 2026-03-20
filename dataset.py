"""
Dataset and data-loading utilities for Mechanical MNIST Cahn-Hilliard.

Supports two data layouts:

  Layout A — "summary files" (recommended for getting started):
    data_root/
      images/               # Image0001.txt, Image0002.txt, ... (400×400)
      summary_psi.txt       # (N, 7) strain energy at each disp level
      summary_rxnforce.txt  # (N, 28) reaction forces (4 boundaries × 7 levels)
      summary_disp_x.txt    # (N, M) x-displacements at final step  (or .npy)
      summary_disp_y.txt    # (N, M) y-displacements at final step  (or .npy)

  Layout B — "per-sample files":
    data_root/
      images/Image0001.txt
      psi/Image0001_psi.txt
      rxnforce/Image0001_rxnforce.txt
      displacement/Image0001_disp.txt

The dataset class auto-detects which layout is present.
"""

import os
import glob
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Normalization statistics (computed once, reused during training)
# ═══════════════════════════════════════════════════════════════════════════

class NormStats:
    """Stores per-target mean/std for standardization."""

    def __init__(self):
        self.psi_mean = None
        self.psi_std = None
        self.force_mean = None
        self.force_std = None
        self.disp_mean = None
        self.disp_std = None

    def compute(self, psi: np.ndarray, force: np.ndarray, disp: np.ndarray):
        self.psi_mean = psi.mean(axis=0)
        self.psi_std = psi.std(axis=0) + 1e-8
        self.force_mean = force.mean(axis=0)
        self.force_std = force.std(axis=0) + 1e-8
        self.disp_mean = disp.mean()
        self.disp_std = disp.std() + 1e-8
        return self

    def save(self, path: str):
        np.savez(
            path,
            psi_mean=self.psi_mean, psi_std=self.psi_std,
            force_mean=self.force_mean, force_std=self.force_std,
            disp_mean=self.disp_mean, disp_std=self.disp_std,
        )

    def load(self, path: str):
        d = np.load(path)
        self.psi_mean = d["psi_mean"]
        self.psi_std = d["psi_std"]
        self.force_mean = d["force_mean"]
        self.force_std = d["force_std"]
        self.disp_mean = d["disp_mean"]
        self.disp_std = d["disp_std"]
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class MechMNISTCahnHilliard(Dataset):
    """
    PyTorch Dataset for Mechanical MNIST Cahn-Hilliard.

    Parameters
    ----------
    data_root : str
        Path to dataset root directory.
    img_size : int
        Target image size (resized from 400×400).
    disp_size : int
        Target displacement field grid size (default: same as img_size).
    norm_stats : NormStats, optional
        Pre-computed normalization statistics. If None, targets are returned raw.
    indices : list[int], optional
        Subset of sample indices to use (for train/val/test splits).
    max_samples : int, optional
        Cap the number of samples (useful for debugging).
    """

    def __init__(
        self,
        data_root: str,
        img_size: int = 256,
        disp_size: int = 256,
        norm_stats: Optional[NormStats] = None,
        indices: Optional[list] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.disp_size = disp_size
        self.norm_stats = norm_stats

        # ── Discover image files ─────────────────────────────────────────
        img_dir = self.data_root / "images"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                "Expected structure: data_root/images/Image0001.txt ..."
            )

        self.image_files = sorted(glob.glob(str(img_dir / "*.txt")))
        if not self.image_files:
            # Try .npy format
            self.image_files = sorted(glob.glob(str(img_dir / "*.npy")))

        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {img_dir}")

        # ── Load summary targets (Layout A) or set up per-file (Layout B)─
        self._load_targets()

        # ── Apply index subsetting ───────────────────────────────────────
        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices]
            self.psi_data = self.psi_data[indices]
            self.force_data = self.force_data[indices]
            if self.disp_data is not None:
                self.disp_data = self.disp_data[indices]

        if max_samples is not None:
            n = min(max_samples, len(self.image_files))
            self.image_files = self.image_files[:n]
            self.psi_data = self.psi_data[:n]
            self.force_data = self.force_data[:n]
            if self.disp_data is not None:
                self.disp_data = self.disp_data[:n]

        self.n_samples = len(self.image_files)
        print(f"[Dataset] Loaded {self.n_samples} samples from {data_root}")

    def _load_targets(self):
        """Load target data from summary files or per-sample files."""
        root = self.data_root

        # ── Try summary file layout first ────────────────────────────────
        psi_path = self._find_file("summary_psi")
        force_path = self._find_file("summary_rxnforce")

        if psi_path and force_path:
            print("[Dataset] Using summary file layout")
            self.psi_data = self._load_array(psi_path)       # (N, 7)
            self.force_data = self._load_array(force_path)    # (N, 28)
        else:
            # Fall back to per-sample file layout
            print("[Dataset] Using per-sample file layout")
            self.psi_data, self.force_data = self._load_per_sample_scalars()

        # ── Displacement field (may be large — loaded lazily if needed) ──
        disp_x_path = self._find_file("summary_disp_x")
        disp_y_path = self._find_file("summary_disp_y")

        if disp_x_path and disp_y_path:
            dx = self._load_array(disp_x_path)
            dy = self._load_array(disp_y_path)
            # Stack into (N, 2, sqrt(M), sqrt(M)) if flat, or keep as-is
            self.disp_data = self._reshape_disp(dx, dy)
            self.disp_layout = "summary"
        else:
            # Check for pre-gridded .npy displacement
            disp_npy = root / "displacement_fields.npy"
            if disp_npy.exists():
                self.disp_data = np.load(str(disp_npy))  # (N, 2, H, W)
                self.disp_layout = "summary"
            else:
                self.disp_data = None
                self.disp_layout = "per_file"
                print("[Dataset] Displacement will be loaded per-sample "
                      "(slower — consider pre-processing into .npy)")

    def _find_file(self, prefix: str) -> Optional[str]:
        """Look for a summary file with various extensions."""
        root = self.data_root
        for ext in [".txt", ".npy", ".csv", ".gz"]:
            p = root / f"{prefix}{ext}"
            if p.exists():
                return str(p)
        return None

    def _load_array(self, path: str) -> np.ndarray:
        """Load a numeric array from .txt, .npy, or .csv."""
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".gz"):
            return np.loadtxt(path)
        else:
            return np.loadtxt(path)

    def _reshape_disp(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Reshape flat displacement vectors into 2D grids."""
        n = dx.shape[0]
        m = dx.shape[1]
        side = int(np.sqrt(m))
        if side * side == m:
            dx = dx.reshape(n, side, side)
            dy = dy.reshape(n, side, side)
        else:
            # Non-square: keep flat and interpolate in __getitem__
            pass
        return np.stack([dx, dy], axis=1).astype(np.float32)  # (N, 2, H, W)

    def _load_per_sample_scalars(self):
        """Load psi and force from individual files."""
        psi_list, force_list = [], []
        for img_path in self.image_files:
            name = Path(img_path).stem  # e.g., "Image0001"
            psi_path = self.data_root / "psi" / f"{name}_psi.txt"
            force_path = self.data_root / "rxnforce" / f"{name}_rxnforce.txt"
            psi_list.append(np.loadtxt(str(psi_path)))
            force_list.append(np.loadtxt(str(force_path)))
        return np.array(psi_list, dtype=np.float32), np.array(force_list, dtype=np.float32)

    def _load_single_disp(self, idx: int) -> np.ndarray:
        """Load displacement for a single sample from per-file layout."""
        name = Path(self.image_files[idx]).stem
        disp_path = self.data_root / "displacement" / f"{name}_disp.txt"
        if disp_path.exists():
            data = np.loadtxt(str(disp_path))
            # Expect columns: [node_x, node_y, u_x, u_y] or just [u_x, u_y]
            if data.ndim == 2 and data.shape[1] >= 4:
                ux, uy = data[:, 2], data[:, 3]
            elif data.ndim == 2 and data.shape[1] == 2:
                ux, uy = data[:, 0], data[:, 1]
            else:
                ux = uy = data.flatten()[:data.shape[0] // 2], data.flatten()[data.shape[0] // 2:]
            side = int(np.sqrt(len(ux)))
            ux = ux.reshape(side, side)
            uy = uy.reshape(side, side)
            return np.stack([ux, uy], axis=0).astype(np.float32)  # (2, H, W)
        else:
            # Return zeros as fallback (will be masked in loss)
            return np.zeros((2, self.disp_size, self.disp_size), dtype=np.float32)

    def _load_image(self, path: str) -> np.ndarray:
        """Load a single image (400×400 binary bitmap) from .txt or .npy."""
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        else:
            return np.loadtxt(path).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        # ── Image ────────────────────────────────────────────────────────
        img = self._load_image(self.image_files[idx])
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # (1, 400, 400)

        # Resize to target size
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 1, 400, 400)
        img_tensor = F.interpolate(
            img_tensor, size=(self.img_size, self.img_size), mode="nearest"
        ).squeeze(0)  # (1, H, W) — nearest-neighbor preserves binary values

        # ── Scalar targets ───────────────────────────────────────────────
        psi = self.psi_data[idx].astype(np.float32)
        force = self.force_data[idx].astype(np.float32)

        # ── Displacement field ───────────────────────────────────────────
        if self.disp_data is not None:
            disp = self.disp_data[idx]  # (2, H_orig, W_orig)
            if disp.ndim == 3:
                disp_tensor = torch.from_numpy(disp).unsqueeze(0)
                disp_tensor = F.interpolate(
                    disp_tensor, size=(self.disp_size, self.disp_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            else:
                disp_tensor = torch.from_numpy(disp)
        else:
            disp = self._load_single_disp(idx)
            disp_tensor = torch.from_numpy(disp).unsqueeze(0)
            disp_tensor = F.interpolate(
                disp_tensor, size=(self.disp_size, self.disp_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        # ── Normalize targets ────────────────────────────────────────────
        if self.norm_stats is not None:
            ns = self.norm_stats
            psi = (psi - ns.psi_mean) / ns.psi_std
            force = (force - ns.force_mean) / ns.force_std
            disp_tensor = (disp_tensor - ns.disp_mean) / ns.disp_std

        psi_tensor = torch.as_tensor(psi, dtype=torch.float32)
        force_tensor = torch.as_tensor(force, dtype=torch.float32)
        disp_tensor = disp_tensor.float()  # ensure float32

        return {
            "image": img_tensor,        # (1, 256, 256)
            "psi": psi_tensor,           # (7,)
            "force": force_tensor,       # (28,)
            "disp": disp_tensor,         # (2, 256, 256)
        }


# ═══════════════════════════════════════════════════════════════════════════
# Data pipeline: splits + dataloaders
# ═══════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    data_root: str,
    img_size: int = 256,
    batch_size: int = 8,
    train_split: float = 0.85,
    val_split: float = 0.10,
    num_workers: int = 4,
    max_samples: int = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, NormStats]:
    """
    Create train/val/test dataloaders with normalization.

    Returns
    -------
    train_loader, val_loader, test_loader, norm_stats
    """
    # Load full dataset (without normalization) to compute stats
    full_ds = MechMNISTCahnHilliard(
        data_root, img_size=img_size, max_samples=max_samples
    )

    n = len(full_ds)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    # Deterministic split
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=gen
    )
    train_idx = list(train_idx)
    val_idx = list(val_idx)
    test_idx = list(test_idx)

    print(f"[Split] Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Compute normalization stats on training set only
    norm = NormStats().compute(
        full_ds.psi_data[train_idx],
        full_ds.force_data[train_idx],
        full_ds.disp_data[train_idx] if full_ds.disp_data is not None
        else np.zeros(1),
    )

    # Create split datasets with normalization
    train_ds = MechMNISTCahnHilliard(
        data_root, img_size, norm_stats=norm, indices=train_idx, max_samples=max_samples
    )
    val_ds = MechMNISTCahnHilliard(
        data_root, img_size, norm_stats=norm, indices=val_idx
    )
    test_ds = MechMNISTCahnHilliard(
        data_root, img_size, norm_stats=norm, indices=test_idx
    )

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader, norm
