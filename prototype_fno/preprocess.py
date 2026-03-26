"""
Preprocess Mechanical MNIST Cahn-Hilliard dataset.

Converts raw text files (400x400 input patterns + displacement fields)
into compact .npy arrays for fast training.

Usage:
    python preprocess.py --data_dir ../data --output_dir ./processed
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def downsample_400_to_64(img_400: np.ndarray) -> np.ndarray:
    """Downsample 400x400 binary image to 64x64 via center-crop + block average."""
    # Crop from 400 to 384 (center crop, 384 = 64 * 6)
    offset = (400 - 384) // 2  # = 8
    cropped = img_400[offset:offset + 384, offset:offset + 384]
    # Block average: reshape to (64, 6, 64, 6), mean over block axes
    return cropped.reshape(64, 6, 64, 6).mean(axis=(1, 3)).astype(np.float32)


def load_case_inputs(case_dir: Path, image_numbers: list) -> np.ndarray:
    """Load input images for one case, sorted by image number."""
    arrays = []
    for img_num in tqdm(image_numbers, desc=f"Loading {case_dir.name}"):
        filepath = case_dir / f"Image{img_num}.txt"
        img = np.loadtxt(filepath, dtype=np.float32)
        arrays.append(downsample_400_to_64(img))
    return np.stack(arrays)


def load_case_displacements(data_dir: Path, case: str, direction: str) -> np.ndarray:
    """
    Load and concatenate all 20 displacement files for one case/direction.

    Files 1-10 are in {case}_disp_0.5_{dir}_1/,
    files 11-20 are in {case}_disp_0.5_{dir}_2/.
    Each row = 4096 floats = one 64x64 flattened displacement field.
    """
    all_rows = []
    for file_num in tqdm(range(1, 21), desc=f"{case} disp_{direction}"):
        subdir = f"{case}_disp_0.5_{direction}_1" if file_num <= 10 else f"{case}_disp_0.5_{direction}_2"
        filename = f"{case}_disp_0.5_{direction}_{file_num}.txt"
        filepath = data_dir / subdir / filename
        data = np.loadtxt(filepath, dtype=np.float32)
        all_rows.append(data)

    combined = np.vstack(all_rows)
    return combined.reshape(-1, 64, 64)


def main():
    parser = argparse.ArgumentParser(description="Preprocess Mechanical MNIST CH data")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="./processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Case1: images 1..37523, Case2: images 37524..75203, Case3: images 75204..104813
    cases = [
        ("Case1", list(range(1, 37524))),
        ("Case2", list(range(37524, 75204))),
        ("Case3", list(range(75204, 104814))),
    ]

    # Load inputs
    print("=== Loading input patterns ===")
    input_parts = []
    for case_name, nums in cases:
        part = load_case_inputs(data_dir / f"{case_name}_input_patterns", nums)
        input_parts.append(part)
    all_inputs = np.concatenate(input_parts, axis=0)

    # Load displacements
    print("\n=== Loading displacement fields ===")
    disp_x_parts, disp_y_parts = [], []
    for case_name, _ in cases:
        disp_x_parts.append(load_case_displacements(data_dir, case_name, "x"))
        disp_y_parts.append(load_case_displacements(data_dir, case_name, "y"))
    all_disp_x = np.concatenate(disp_x_parts, axis=0)
    all_disp_y = np.concatenate(disp_y_parts, axis=0)

    # Load strain energy (7 values per sample: d=0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5)
    print("\n=== Loading strain energy ===")
    se_parts = []
    for case_name, _ in cases:
        filepath = data_dir / "Case123_strain_energy" / "Case123_strain_energy" / f"{case_name}_strain_energy.txt"
        se_parts.append(np.loadtxt(filepath, dtype=np.float32))
        print(f"  {case_name}: {se_parts[-1].shape}")
    all_strain_energy = np.concatenate(se_parts, axis=0)

    # Load reaction forces (28 values per sample: 4 boundaries x 7 displacement levels)
    print("\n=== Loading reaction forces ===")
    rf_parts = []
    for case_name, _ in cases:
        filepath = data_dir / "Case123_rxn_force" / "Case123_rxn_force" / f"{case_name}_rxn_force.txt"
        rf_parts.append(np.loadtxt(filepath, dtype=np.float32))
        print(f"  {case_name}: {rf_parts[-1].shape}")
    all_rxn_force = np.concatenate(rf_parts, axis=0)

    # Verify alignment
    n = all_inputs.shape[0]
    assert n == all_disp_x.shape[0] == all_disp_y.shape[0] == \
           all_strain_energy.shape[0] == all_rxn_force.shape[0], (
        f"Mismatch: inputs={n}, disp_x={all_disp_x.shape[0]}, "
        f"disp_y={all_disp_y.shape[0]}, strain_energy={all_strain_energy.shape[0]}, "
        f"rxn_force={all_rxn_force.shape[0]}"
    )

    print(f"\nTotal samples: {n}")
    print(f"  Inputs:         {all_inputs.shape}")
    print(f"  Disp X:         {all_disp_x.shape}")
    print(f"  Disp Y:         {all_disp_y.shape}")
    print(f"  Strain energy:  {all_strain_energy.shape}")
    print(f"  Rxn force:      {all_rxn_force.shape}")

    # Compute and save normalization statistics
    stats = {
        "disp_x_mean": float(all_disp_x.mean()), "disp_x_std": float(all_disp_x.std()),
        "disp_y_mean": float(all_disp_y.mean()), "disp_y_std": float(all_disp_y.std()),
        "se_mean": all_strain_energy.mean(axis=0).tolist(),
        "se_std": all_strain_energy.std(axis=0).tolist(),
        "rf_mean": all_rxn_force.mean(axis=0).tolist(),
        "rf_std": all_rxn_force.std(axis=0).tolist(),
    }

    print(f"\nNorm stats: disp_x mean={stats['disp_x_mean']:.6f} std={stats['disp_x_std']:.6f}")
    print(f"            disp_y mean={stats['disp_y_mean']:.6f} std={stats['disp_y_std']:.6f}")
    print(f"            strain_energy shape=({len(stats['se_mean'])},)")
    print(f"            rxn_force shape=({len(stats['rf_mean'])},)")

    # Save
    np.save(output_dir / "inputs_64x64.npy", all_inputs)
    np.save(output_dir / "disp_x_64x64.npy", all_disp_x)
    np.save(output_dir / "disp_y_64x64.npy", all_disp_y)
    np.save(output_dir / "strain_energy.npy", all_strain_energy)
    np.save(output_dir / "rxn_force.npy", all_rxn_force)
    np.save(output_dir / "norm_stats.npy", stats, allow_pickle=True)

    print(f"\nPreprocessing complete. Files saved to {output_dir}")


if __name__ == "__main__":
    main()
