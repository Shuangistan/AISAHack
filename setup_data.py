#!/usr/bin/env python3
"""
setup_data.py — One-command dataset setup for Mechanical MNIST Cahn-Hilliard.

Run this after cloning the repository:

    python setup_data.py

Options:
    python setup_data.py                          # full dataset (~3.5 GB download)
    python setup_data.py --skip-disp              # skip displacement (~380 MB)
    python setup_data.py --cases 1                # only Case 1
    python setup_data.py --cases 1 2              # Cases 1 and 2
    python setup_data.py --out-dir ./data         # custom output directory
    python setup_data.py --dry-run                # list files without downloading
    python setup_data.py --keep-zips              # keep zip archives after extraction

Requires: Python >= 3.8, requests, numpy
    pip install requests numpy
"""

import argparse
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required.")
    print("Install it with:  pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset file manifest
# ═══════════════════════════════════════════════════════════════════════════

HANDLE = "2144/43971"
SEARCH_API = "https://open.bu.edu/server/api/core/bitstreams/search/byItemHandle"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": f"https://open.bu.edu/bitstream/handle/{HANDLE}/",
}

# (filename, size_bytes, category, case_number)
# case_number 0 = shared across all cases
FILE_MANIFEST = [
    # ── Case 1 (37,523 samples) ──────────────────────────────────────
    ("Case1_input_patterns.zip",           135_004_160, "images",    1),
    ("Case1_disp_0.5_x_1.zip",            302_624_358, "disp",      1),
    ("Case1_disp_0.5_x_2.zip",            302_435_738, "disp",      1),
    ("Case1_disp_0.5_y_1.zip",            290_193_408, "disp",      1),
    ("Case1_disp_0.5_y_2.zip",            290_130_534, "disp",      1),
    # ── Case 2 (37,680 samples) ──────────────────────────────────────
    ("Case2_input_patterns.zip",           138_763_264, "images",    2),
    ("Case2_disp_0.5_x_1.zip",            306_840_986, "disp",      2),
    ("Case2_disp_0.5_x_2.zip",            306_704_589, "disp",      2),
    ("Case2_disp_0.5_y_1.zip",            289_313_587, "disp",      2),
    ("Case2_disp_0.5_y_2.zip",            289_041_818, "disp",      2),
    # ── Case 3 (29,610 samples) ──────────────────────────────────────
    ("Case3_input_patterns.zip",            96_902_758, "images",    3),
    ("Case3_disp_0.5_x_1.zip",            242_064_179, "disp",      3),
    ("Case3_disp_0.5_x_2.zip",            242_483_507, "disp",      3),
    ("Case3_disp_0.5_y_1.zip",            226_294_989, "disp",      3),
    ("Case3_disp_0.5_y_2.zip",            226_410_342, "disp",      3),
    # ── Shared across all cases ──────────────────────────────────────
    ("Case123_input_patterns_64_x_64.zip",  21_217_075, "images_64", 0),
    ("Case123_rxn_force.zip",                7_401_062, "rxnforce",  0),
    ("Case123_strain_energy.zip",            2_328_166, "psi",       0),
    ("Case123_simulation_information.zip",     811_479, "meta",      0),
    ("description.pdf",                      3_041_689, "meta",      0),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def fmt_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def validate_file(filepath: str) -> bool:
    """Check that a downloaded file is a real zip/pdf, not an HTML error page."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) < 1000:
        return False
    with open(filepath, "rb") as f:
        magic = f.read(8)
    if filepath.endswith(".zip"):
        return magic[:2] == b"PK"
    if filepath.endswith(".pdf"):
        return magic[:5] == b"%PDF-"
    return not magic.lstrip().startswith(b"<")


def resolve_download_url(filename: str) -> str:
    """
    Use the DSpace 7 search API to resolve a filename to a direct download URL.

    Endpoint: /server/api/core/bitstreams/search/byItemHandle
    Returns the '_links.content.href' from the bitstream metadata.
    """
    resp = requests.get(
        SEARCH_API,
        headers=HEADERS,
        params={"handle": HANDLE, "filename": filename},
        timeout=30,
    )
    resp.raise_for_status()
    metadata = resp.json()
    return metadata["_links"]["content"]["href"]


def download_file(filename: str, dest_dir: str, max_retries: int = 3) -> bool:
    """
    Resolve and download a single file from OpenBU.
    Skips files that already exist and pass validation.
    """
    dest = os.path.join(dest_dir, filename)

    # Already downloaded and valid
    if os.path.exists(dest) and validate_file(dest):
        print(f"  ✓ {filename} (already exists)")
        return True

    # Remove invalid previous download
    if os.path.exists(dest):
        os.remove(dest)

    for attempt in range(1, max_retries + 1):
        try:
            # Step 1: resolve filename → download URL via search API
            download_url = resolve_download_url(filename)

            # Step 2: stream download
            with requests.get(download_url, headers=HEADERS, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0
                bar_w = 35

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1_048_576):  # 1 MB
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            frac = downloaded / total
                            filled = int(bar_w * frac)
                            bar = "█" * filled + "░" * (bar_w - filled)
                            print(
                                f"\r  {filename:<45} |{bar}| "
                                f"{frac*100:5.1f}% {fmt_size(downloaded)}/{fmt_size(total)}",
                                end="", flush=True,
                            )

            print()

            # Validate
            if validate_file(dest):
                return True

            print(f"  ⚠ {filename}: downloaded but file is invalid (removing)")
            os.remove(dest)

        except Exception as e:
            print(f"\n  ⚠ Attempt {attempt}/{max_retries} failed: {e}")
            if os.path.exists(dest):
                os.remove(dest)

        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"    Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ✗ FAILED: {filename}")
    return False


def extract_zip(zip_path: str, dest: str) -> int:
    """Extract a zip, flattening all files into dest/. Returns file count."""
    os.makedirs(dest, exist_ok=True)
    count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if not basename or basename.startswith(".") or "__MACOSX" in member:
                    continue
                target = os.path.join(dest, basename)
                if not os.path.exists(target):
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                count += 1
    except zipfile.BadZipFile:
        print(f"  ⚠ Corrupt zip: {zip_path}")
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline stages
# ═══════════════════════════════════════════════════════════════════════════

def build_manifest(cases: List[int], skip_disp: bool) -> List[Tuple[str, int, str]]:
    """Filter the file manifest based on user options."""
    selected = []
    for filename, size, category, case_num in FILE_MANIFEST:
        if case_num != 0 and case_num not in cases:
            continue
        if skip_disp and category == "disp":
            continue
        selected.append((filename, size, category))
    return selected


def download_all(manifest, dl_dir: str) -> List[str]:
    total_size = sum(s for _, s, _ in manifest)
    print(f"\n[1/4] Downloading {len(manifest)} files ({fmt_size(total_size)})")
    print(f"      → {dl_dir}/\n")

    os.makedirs(dl_dir, exist_ok=True)
    failed = []
    for i, (filename, size, category) in enumerate(manifest, 1):
        print(f"  [{i}/{len(manifest)}]", end="")
        if not download_file(filename, dl_dir):
            failed.append(filename)
        time.sleep(1.0)  # rate-limit courtesy
    return failed


def extract_all(dl_dir: str, data_dir: str, manifest, skip_disp: bool) -> dict:
    print(f"\n[2/4] Extracting archives → {data_dir}/\n")
    stats = {}

    for filename, _, category in manifest:
        src = os.path.join(dl_dir, filename)
        if not os.path.exists(src):
            continue

        # Non-zip files just get copied
        if not filename.endswith(".zip"):
            dst = os.path.join(data_dir, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            print(f"  {filename:<45} → data/")
            continue

        # Validate before extracting
        if not validate_file(src):
            size = os.path.getsize(src)
            print(f"  ⚠ {filename:<45} SKIPPED (invalid, {fmt_size(size)})")
            continue

        # Route to target directory
        if category == "images":
            dest = os.path.join(data_dir, "images")
        elif category == "images_64":
            dest = os.path.join(data_dir, "images_64x64")
        elif category == "psi":
            dest = os.path.join(data_dir, "psi")
        elif category == "rxnforce":
            dest = os.path.join(data_dir, "rxnforce")
        elif category == "disp":
            if "_x_" in filename:
                dest = os.path.join(data_dir, "disp_x")
            elif "_y_" in filename:
                dest = os.path.join(data_dir, "disp_y")
            else:
                dest = os.path.join(data_dir, "displacement")
        else:
            dest = data_dir

        n = extract_zip(src, dest)
        rel = os.path.relpath(dest, data_dir)
        stats[filename] = n
        print(f"  {filename:<45} → {rel}/ ({n:,} files)")

    return stats


def _consolidate_scalar(root: Path, subfolder: str, out_name: str):
    """
    Consolidate scalar target files into a single summary file.

    Handles two layouts:
      A) Per-case summaries: 3 files (one per case), each with many rows.
         e.g., Case1_strain_energy.txt is (37523, 7)
         → vertically stack them.
      B) Per-sample files: one file per simulation, each a single row.
         → load each, stack as rows.
    """
    import numpy as np

    folder = root / subfolder
    if not folder.exists():
        return

    files = sorted(folder.glob("*.txt"))
    if not files:
        return

    print(f"  Reading {len(files):,} {subfolder} files...")

    # Peek at the first file to decide layout
    first = np.loadtxt(str(files[0]))

    if first.ndim == 2 and first.shape[0] > 1:
        # Layout A: each file is already a multi-row summary → vstack
        print(f"    Detected per-case summary format (first file: {first.shape})")
        chunks = [np.loadtxt(str(f)) for f in files]
        # Verify all have the same number of columns
        ncols = chunks[0].shape[1] if chunks[0].ndim == 2 else chunks[0].shape[0]
        for i, c in enumerate(chunks):
            if c.ndim == 1:
                chunks[i] = c.reshape(1, -1)
        arr = np.vstack(chunks)
    else:
        # Layout B: each file is one sample → stack as rows
        print(f"    Detected per-sample format (first file: {first.shape})")
        rows = [np.loadtxt(str(f)).flatten() for f in files]
        arr = np.array(rows, dtype=np.float64)

    out_path = root / out_name
    np.savetxt(str(out_path), arr, fmt="%.8e")
    np.save(str(out_path.with_suffix(".npy")), arr.astype(np.float32))
    print(f"  → {out_name} + .npy: {arr.shape}")


def consolidate(data_dir: str, skip_disp: bool):
    print(f"\n[3/4] Building consolidated summary files\n")

    try:
        import numpy as np
    except ImportError:
        print("  ⚠ NumPy not installed — skipping consolidation.")
        print("    pip install numpy && python setup_data.py --skip-extract")
        return

    root = Path(data_dir)

    # ── Strain energy ────────────────────────────────────────────────
    _consolidate_scalar(root, "psi", "summary_psi.txt")

    # ── Reaction force ───────────────────────────────────────────────
    _consolidate_scalar(root, "rxnforce", "summary_rxnforce.txt")

    # ── Displacement fields ──────────────────────────────────────────
    if skip_disp:
        print("  Skipping displacement consolidation (--skip-disp)")
        return

    disp_x_dir = root / "disp_x"
    disp_y_dir = root / "disp_y"
    if disp_x_dir.exists() and disp_y_dir.exists():
        x_files = sorted(disp_x_dir.glob("*.txt"))
        y_files = sorted(disp_y_dir.glob("*.txt"))
        n = min(len(x_files), len(y_files))
        if n == 0:
            return

        import numpy as np

        # Peek at first file: if it has multiple rows, these are per-case
        # summaries (each file = thousands of samples). If it's 1D or a
        # single row, these are per-sample files.
        first = np.loadtxt(str(x_files[0]))
        is_summary = first.ndim == 2 and first.shape[0] > 1

        if is_summary:
            print(f"  Concatenating {n} displacement summary files "
                  f"(first: {first.shape})...")
            x_chunks = [np.loadtxt(str(f)) for f in x_files]
            y_chunks = [np.loadtxt(str(f)) for f in y_files]
            arr_x = np.vstack(x_chunks)
            arr_y = np.vstack(y_chunks)
            np.savetxt(str(root / "summary_disp_x.txt"), arr_x, fmt="%.6e")
            np.savetxt(str(root / "summary_disp_y.txt"), arr_y, fmt="%.6e")
            np.save(str(root / "summary_disp_x.npy"), arr_x.astype(np.float32))
            np.save(str(root / "summary_disp_y.npy"), arr_y.astype(np.float32))
            side = int(math.sqrt(arr_x.shape[1]))
            print(f"  → summary_disp_x.txt + .npy: {arr_x.shape}")
            print(f"  → summary_disp_y.txt + .npy: {arr_y.shape} ({side}×{side} grid)")
        else:
            print(f"  Reading {n:,} per-sample displacement pairs (chunked)...")

            out_x = root / "summary_disp_x.txt"
            out_y = root / "summary_disp_y.txt"
            chunk_size = 5000
            all_x, all_y = [], []

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                cx = [np.loadtxt(str(x_files[i])).flatten() for i in range(start, end)]
                cy = [np.loadtxt(str(y_files[i])).flatten() for i in range(start, end)]
                mode = "ab" if start > 0 else "wb"
                with open(str(out_x), mode) as f:
                    np.savetxt(f, np.array(cx), fmt="%.6e")
                with open(str(out_y), mode) as f:
                    np.savetxt(f, np.array(cy), fmt="%.6e")
                all_x.extend(cx)
                all_y.extend(cy)
                print(f"    {end:,}/{n:,}")

            arr_x = np.array(all_x, dtype=np.float32)
            arr_y = np.array(all_y, dtype=np.float32)
            np.save(str(root / "summary_disp_x.npy"), arr_x)
            np.save(str(root / "summary_disp_y.npy"), arr_y)
            side = int(math.sqrt(arr_x.shape[1]))
            print(f"  → summary_disp_x.txt + .npy, summary_disp_y.txt + .npy ({side}×{side} per sample)")


def verify(data_dir: str, skip_disp: bool):
    print(f"\n[4/4] Verification\n")
    root = Path(data_dir)
    rows = []

    img_dir = root / "images"
    n = len(list(img_dir.glob("*.txt"))) if img_dir.exists() else 0
    rows.append(("✓" if n > 0 else "✗", "Input images (400×400)", f"{n:,} files"))

    img64_dir = root / "images_64x64"
    n64 = len(list(img64_dir.glob("*.txt"))) if img64_dir.exists() else 0
    if n64 > 0:
        rows.append(("✓", "Downsampled (64×64)", f"{n64:,} files"))

    for label, fname in [("Strain energy", "summary_psi.txt"),
                         ("Reaction force", "summary_rxnforce.txt")]:
        p = root / fname
        if p.exists():
            try:
                import numpy as np
                arr = np.loadtxt(str(p), max_rows=2)
                ncols = arr.shape[-1] if arr.ndim > 1 else arr.shape[0]
                with open(str(p)) as f:
                    nrows = sum(1 for _ in f)
                rows.append(("✓", label, f"({nrows:,}, {ncols})"))
            except Exception:
                rows.append(("✓", label, "file exists"))
        else:
            rows.append(("✗", label, "not found"))

    if not skip_disp:
        dx = root / "summary_disp_x.txt"
        dy = root / "summary_disp_y.txt"
        if dx.exists() and dy.exists():
            rows.append(("✓", "Displacement fields", "ready"))
        else:
            n_dx = len(list((root / "disp_x").glob("*.txt"))) if (root / "disp_x").exists() else 0
            status = "⚠" if n_dx > 0 else "✗"
            detail = f"{n_dx:,} files (raw)" if n_dx > 0 else "not found"
            rows.append((status, "Displacement fields", detail))
    else:
        rows.append(("–", "Displacement fields", "skipped"))

    print(f"  ╔{'═'*56}╗")
    print(f"  ║  {'Dataset summary':<53}║")
    print(f"  ╠{'═'*56}╣")
    for status, label, detail in rows:
        print(f"  ║  {status} {label:<28} {detail:<23}║")
    print(f"  ╠{'═'*56}╣")
    total_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"  ║  {'Disk usage:':<31} {fmt_size(total_bytes):<23}║")
    print(f"  ╚{'═'*56}╝")

    print(f"""
  Ready! Next steps:

    # Quick test (small subset):
    python train.py --data_root {data_dir} --max_samples 500 --epochs 5

    # Full training:
    python train.py --data_root {data_dir} --epochs 100 --batch_size 8
""")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Download and set up the Mechanical MNIST Cahn-Hilliard dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_data.py                          # full dataset
  python setup_data.py --skip-disp              # without displacement (~380 MB)
  python setup_data.py --cases 1 --skip-disp    # Case 1 scalars only (~130 MB)
  python setup_data.py --dry-run                # preview download list
        """,
    )
    p.add_argument("--out-dir", default="./data",
                   help="Output directory (default: ./data)")
    p.add_argument("--download-dir", default=None,
                   help="Zip download cache (default: OUT_DIR/downloads)")
    p.add_argument("--cases", nargs="+", type=int, default=[1, 2, 3],
                   choices=[1, 2, 3], help="Which cases to download (default: all)")
    p.add_argument("--skip-disp", action="store_true",
                   help="Skip displacement fields (saves ~3 GB)")
    p.add_argument("--skip-extract", action="store_true",
                   help="Download only, skip extraction and consolidation")
    p.add_argument("--skip-consolidate", action="store_true",
                   help="Download and extract, skip summary file creation")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be downloaded, then exit")
    p.add_argument("--keep-zips", action="store_true",
                   help="Keep zip archives after extraction")
    args = p.parse_args()

    data_dir = args.out_dir
    dl_dir = args.download_dir or os.path.join(data_dir, "downloads")

    manifest = build_manifest(args.cases, args.skip_disp)
    total_size = sum(s for _, s, _ in manifest)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Mechanical MNIST Cahn-Hilliard — Dataset Setup           ║")
    print("║   Source: https://hdl.handle.net/2144/43971                ║")
    print("║   License: CC BY-SA 4.0                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    print(f"  Cases:             {args.cases}")
    print(f"  Skip displacement: {args.skip_disp}")
    print(f"  Files:             {len(manifest)}")
    print(f"  Total download:    ~{fmt_size(total_size)}")
    print(f"  Output:            {data_dir}/")

    if args.dry_run:
        print(f"\n  {'File':<45} {'Size':>10}  Category")
        print(f"  {'─'*45} {'─'*10}  {'─'*10}")
        for filename, size, category in manifest:
            print(f"  {filename:<45} {fmt_size(size):>10}  {category}")
        print(f"\n  {'TOTAL':<45} {fmt_size(total_size):>10}")
        print("\n  [Dry run — nothing downloaded]")
        return

    # ── Step 1: Download ─────────────────────────────────────────────
    failed = download_all(manifest, dl_dir)
    if failed:
        print(f"\n  ⚠ {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"    • {f}")
        print(f"\n  Re-run this script to retry (existing files are skipped).\n")

    if args.skip_extract:
        print(f"\n  Done (--skip-extract). Zips are in: {dl_dir}")
        return

    # ── Step 2: Extract ──────────────────────────────────────────────
    extract_all(dl_dir, data_dir, manifest, args.skip_disp)

    # ── Step 3: Consolidate ──────────────────────────────────────────
    if not args.skip_consolidate:
        consolidate(data_dir, args.skip_disp)

    # ── Clean up zips ────────────────────────────────────────────────
    if not args.keep_zips:
        zips = list(Path(dl_dir).glob("*.zip"))
        if zips:
            freed = sum(z.stat().st_size for z in zips)
            for z in zips:
                z.unlink()
            try:
                os.rmdir(dl_dir)
            except OSError:
                pass
            print(f"\n  Cleaned up zip files (freed {fmt_size(freed)})")

    # ── Step 4: Verify ───────────────────────────────────────────────
    verify(data_dir, args.skip_disp)


if __name__ == "__main__":
    main()
