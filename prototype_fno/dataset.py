"""
PyTorch Dataset and DataLoader for preprocessed Mechanical MNIST CH data.

Multi-task: displacement fields, strain energy, and reaction forces.
Expects .npy files produced by preprocess.py in the data directory.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class MechMNISTDataset(Dataset):
    """
    Multi-task dataset for Mechanical MNIST Cahn-Hilliard.

    Returns:
        inp:    (1, 64, 64)  — downsampled binary microstructure
        disp:   (2, 64, 64)  — displacement fields (x, y)
        se:     (7,)         — strain energy at 7 displacement levels
        rf:     (28,)        — reaction forces (4 boundaries x 7 levels)
    """

    def __init__(self, data_dir: str, normalize_output: bool = True):
        data_dir = Path(data_dir)
        self.inputs = np.load(data_dir / "inputs_64x64.npy", mmap_mode="r")
        self.disp_x = np.load(data_dir / "disp_x_64x64.npy", mmap_mode="r")
        self.disp_y = np.load(data_dir / "disp_y_64x64.npy", mmap_mode="r")
        self.strain_energy = np.load(data_dir / "strain_energy.npy", mmap_mode="r")
        self.rxn_force = np.load(data_dir / "rxn_force.npy", mmap_mode="r")

        self.normalize_output = normalize_output
        if normalize_output:
            stats = np.load(data_dir / "norm_stats.npy", allow_pickle=True).item()
            self.dx_mean = stats["disp_x_mean"]
            self.dx_std = stats["disp_x_std"]
            self.dy_mean = stats["disp_y_mean"]
            self.dy_std = stats["disp_y_std"]
            self.se_mean = np.array(stats["se_mean"], dtype=np.float32)
            self.se_std = np.array(stats["se_std"], dtype=np.float32)
            self.rf_mean = np.array(stats["rf_mean"], dtype=np.float32)
            self.rf_std = np.array(stats["rf_std"], dtype=np.float32)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        inp = torch.from_numpy(self.inputs[idx].copy()).unsqueeze(0)  # (1,64,64)

        dx = self.disp_x[idx].copy()
        dy = self.disp_y[idx].copy()
        se = self.strain_energy[idx].copy()
        rf = self.rxn_force[idx].copy()

        if self.normalize_output:
            dx = (dx - self.dx_mean) / self.dx_std
            dy = (dy - self.dy_mean) / self.dy_std
            se = (se - self.se_mean) / (self.se_std + 1e-10)
            rf = (rf - self.rf_mean) / (self.rf_std + 1e-10)

        disp = torch.from_numpy(np.stack([dx, dy], axis=0))  # (2,64,64)
        se = torch.from_numpy(se)    # (7,)
        rf = torch.from_numpy(rf)    # (28,)

        return inp, disp, se, rf


def get_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """Create train/val/test DataLoaders with a reproducible random split."""
    dataset = MechMNISTDataset(data_dir)

    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
