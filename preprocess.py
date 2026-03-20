"""
Preprocessing script for Mechanical MNIST Cahn-Hilliard dataset.

Downloads data from OpenBU and converts into the summary file layout expected
by the dataset class. Run this once before training.

Usage:
    python preprocess.py --raw_dir ./raw_download --out_dir ./data

The raw dataset from OpenBU contains:
    - Individual .txt files per image (400×400 binary bitmaps)
    - Individual .txt files per simulation result
    
This script consolidates them into efficient summary files:
    data/
      images/             (symlinked or copied .txt files)
      summary_psi.txt     (N × 7)
      summary_rxnforce.txt (N × 28)
      displacement_fields.npy  (N × 2 × grid × grid)  [optional, large]
"""

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def natural_sort_key(s: str):
    """Sort strings with embedded numbers naturally (Image1 < Image10)."""
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", str(s))
    ]


def consolidate_images(raw_dir: str, out_dir: str) -> list:
    """
    Find all image .txt files and organize them in the output directory.
    Returns sorted list of image basenames.
    """
    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    # Search common locations
    patterns = [
        os.path.join(raw_dir, "**", "Image*.txt"),
        os.path.join(raw_dir, "**", "image*.txt"),
        os.path.join(raw_dir, "images", "*.txt"),
        os.path.join(raw_dir, "CH_images", "*.txt"),
    ]

    all_files = set()
    for pattern in patterns:
        all_files.update(glob.glob(pattern, recursive=True))

    # Filter to only image files (400×400 bitmaps)
    image_files = sorted(all_files, key=natural_sort_key)
    print(f"Found {len(image_files)} image files")

    basenames = []
    for src in tqdm(image_files, desc="Organizing images"):
        name = os.path.basename(src)
        dst = os.path.join(img_out, name)
        if not os.path.exists(dst):
            # Copy (or symlink for speed)
            try:
                os.symlink(os.path.abspath(src), dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
        basenames.append(os.path.splitext(name)[0])

    return basenames


def consolidate_scalar_results(
    raw_dir: str, out_dir: str, basenames: list, result_type: str
) -> np.ndarray:
    """
    Consolidate per-sample scalar results into a single summary file.

    Parameters
    ----------
    result_type : str
        "psi" for strain energy, "rxnforce" for reaction forces.
    """
    # Check if a summary file already exists
    for pattern in [
        f"summary_{result_type}*.txt",
        f"*{result_type}*summary*.txt",
        f"FEA_{result_type}_results*.txt",
    ]:
        existing = glob.glob(os.path.join(raw_dir, "**", pattern), recursive=True)
        if existing:
            print(f"Found existing summary: {existing[0]}")
            data = np.loadtxt(existing[0])
            out_path = os.path.join(out_dir, f"summary_{result_type}.txt")
            np.savetxt(out_path, data, fmt="%.8e")
            print(f"  → Saved ({data.shape}) to {out_path}")
            return data

    # Otherwise, consolidate from individual files
    print(f"Consolidating {result_type} from individual files...")

    # Try common directory names
    search_dirs = [
        os.path.join(raw_dir, result_type),
        os.path.join(raw_dir, f"FEA_{result_type}_results"),
        os.path.join(raw_dir, f"{result_type}_results"),
    ]

    result_dir = None
    for d in search_dirs:
        if os.path.isdir(d):
            result_dir = d
            break

    if result_dir is None:
        print(f"  WARNING: Could not find {result_type} directory. "
              f"Searched: {search_dirs}")
        return None

    data_rows = []
    missing = 0
    for name in tqdm(basenames, desc=f"Loading {result_type}"):
        # Try common file naming patterns
        candidates = [
            os.path.join(result_dir, f"{name}_{result_type}.txt"),
            os.path.join(result_dir, f"{name}.txt"),
            os.path.join(result_dir, f"{name}_{result_type}_results.txt"),
        ]
        found = False
        for path in candidates:
            if os.path.exists(path):
                row = np.loadtxt(path).flatten()
                data_rows.append(row)
                found = True
                break
        if not found:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing}/{len(basenames)} samples missing {result_type} data")

    if data_rows:
        data = np.array(data_rows, dtype=np.float64)
        out_path = os.path.join(out_dir, f"summary_{result_type}.txt")
        np.savetxt(out_path, data, fmt="%.8e")
        print(f"  → Saved {data.shape} to {out_path}")
        return data
    return None


def consolidate_displacement(
    raw_dir: str, out_dir: str, basenames: list, grid_size: int = 64
) -> np.ndarray:
    """
    Consolidate displacement fields and regrid onto a uniform grid.

    The raw displacement data is at mesh nodes (unstructured). We interpolate
    onto a regular grid for CNN training.

    Parameters
    ----------
    grid_size : int
        Output grid resolution (default 64 to keep file size manageable;
        the dataset class will interpolate up to img_size during training).
    """
    # Check for pre-existing consolidated file
    for pattern in ["displacement_fields.npy", "summary_disp*.npy"]:
        existing = glob.glob(os.path.join(raw_dir, "**", pattern), recursive=True)
        if existing:
            print(f"Found existing displacement file: {existing[0]}")
            data = np.load(existing[0])
            out_path = os.path.join(out_dir, "displacement_fields.npy")
            if existing[0] != out_path:
                np.save(out_path, data)
            return data

    # Check for summary text files (x and y separately)
    dx_path = None
    dy_path = None
    for pattern in ["summary_disp_x*.txt", "*disp*x*.txt", "FEA_disp*x*.txt"]:
        found = glob.glob(os.path.join(raw_dir, "**", pattern), recursive=True)
        if found:
            dx_path = found[0]
    for pattern in ["summary_disp_y*.txt", "*disp*y*.txt", "FEA_disp*y*.txt"]:
        found = glob.glob(os.path.join(raw_dir, "**", pattern), recursive=True)
        if found:
            dy_path = found[0]

    if dx_path and dy_path:
        print(f"Loading displacement x from {dx_path}")
        print(f"Loading displacement y from {dy_path}")
        dx = np.loadtxt(dx_path, dtype=np.float32)
        dy = np.loadtxt(dy_path, dtype=np.float32)
        n = dx.shape[0]
        m = dx.shape[1]
        side = int(np.sqrt(m))
        if side * side == m:
            dx = dx.reshape(n, side, side)
            dy = dy.reshape(n, side, side)
        data = np.stack([dx, dy], axis=1)  # (N, 2, side, side)
        out_path = os.path.join(out_dir, "displacement_fields.npy")
        np.save(out_path, data)
        print(f"  → Saved {data.shape} to {out_path}")

        # Also save as separate summary files for the dataset loader
        np.savetxt(os.path.join(out_dir, "summary_disp_x.txt"),
                   dx.reshape(n, -1), fmt="%.6e")
        np.savetxt(os.path.join(out_dir, "summary_disp_y.txt"),
                   dy.reshape(n, -1), fmt="%.6e")
        return data

    # Per-file loading (slow but handles unstructured meshes)
    print(f"Loading displacement from individual files (grid_size={grid_size})...")

    disp_dir = None
    for d in ["displacement", "FEA_displacement_results", "disp_results"]:
        p = os.path.join(raw_dir, d)
        if os.path.isdir(p):
            disp_dir = p
            break

    if disp_dir is None:
        print("  WARNING: No displacement directory found. Skipping.")
        return None

    from scipy.interpolate import griddata

    xi = np.linspace(0, 1, grid_size)
    yi = np.linspace(0, 1, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)

    fields = []
    missing = 0

    for name in tqdm(basenames, desc="Interpolating displacements"):
        candidates = [
            os.path.join(disp_dir, f"{name}_disp.txt"),
            os.path.join(disp_dir, f"{name}.txt"),
        ]
        found = False
        for path in candidates:
            if os.path.exists(path):
                data = np.loadtxt(path)
                # Assume columns: x_coord, y_coord, u_x, u_y
                if data.shape[1] >= 4:
                    coords = data[:, :2]
                    ux = griddata(coords, data[:, 2], (grid_x, grid_y), method="linear", fill_value=0)
                    uy = griddata(coords, data[:, 3], (grid_x, grid_y), method="linear", fill_value=0)
                elif data.shape[1] == 2:
                    # Already on a grid
                    side = int(np.sqrt(data.shape[0]))
                    ux = data[:, 0].reshape(side, side)
                    uy = data[:, 1].reshape(side, side)
                else:
                    ux = uy = np.zeros((grid_size, grid_size))
                fields.append(np.stack([ux, uy], axis=0))
                found = True
                break
        if not found:
            fields.append(np.zeros((2, grid_size, grid_size)))
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing}/{len(basenames)} displacement files missing")

    data = np.array(fields, dtype=np.float32)
    out_path = os.path.join(out_dir, "displacement_fields.npy")
    np.save(out_path, data)
    print(f"  → Saved {data.shape} to {out_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Preprocess Mechanical MNIST CH data")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing raw downloaded files")
    parser.add_argument("--out_dir", type=str, default="./data",
                        help="Output directory for processed data")
    parser.add_argument("--disp_grid", type=int, default=64,
                        help="Grid size for displacement field interpolation")
    parser.add_argument("--skip_disp", action="store_true",
                        help="Skip displacement field processing (fastest)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Mechanical MNIST Cahn-Hilliard — Data Preprocessing")
    print("=" * 60)

    # Step 1: Images
    print("\n[1/3] Organizing images...")
    basenames = consolidate_images(args.raw_dir, args.out_dir)

    if not basenames:
        print("ERROR: No images found. Check --raw_dir path.")
        return

    # Step 2: Scalar results
    print("\n[2/3] Consolidating scalar results...")
    psi = consolidate_scalar_results(args.raw_dir, args.out_dir, basenames, "psi")
    force = consolidate_scalar_results(args.raw_dir, args.out_dir, basenames, "rxnforce")

    # Step 3: Displacement fields
    if not args.skip_disp:
        print("\n[3/3] Processing displacement fields...")
        disp = consolidate_displacement(
            args.raw_dir, args.out_dir, basenames, grid_size=args.disp_grid
        )
    else:
        print("\n[3/3] Skipping displacement fields (--skip_disp)")

    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"Output directory: {args.out_dir}")
    print(f"  images/          : {len(basenames)} files")
    if psi is not None:
        print(f"  summary_psi.txt  : {psi.shape}")
    if force is not None:
        print(f"  summary_rxnforce.txt : {force.shape}")
    if not args.skip_disp:
        disp_path = os.path.join(args.out_dir, "displacement_fields.npy")
        if os.path.exists(disp_path):
            d = np.load(disp_path)
            print(f"  displacement_fields.npy : {d.shape}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
