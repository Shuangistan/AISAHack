"""
preprocess_image.py
====================
Converts images into binary patterns ready for:
  - The Mechanical MNIST FEA pipeline (400x400 .txt, values 0/255)
  - ML model inference (256x256 float, values 0/1)

TWO MODES (auto-detected from image content):

  RGB mode:
    - Circle detection via HoughCircles
    - Dark circles → black (0) = fibres (E=10)
    - Light background → white (255) = matrix (E=1)

  Binary mode:
    - Image used as-is (no HoughCircles, no processing)
    - Just resize to 400x400 if needed
    - Works for personal B&W images AND dataset images (0/1 or 0/255)

Usage:
    python3 preprocess_image.py <input_image> [output_name] [options]

Examples:
    python3 preprocess_image.py photo.jpg carbon_fibre          # RGB → HoughCircles
    python3 preprocess_image.py pattern.png my_pattern           # Binary → use as-is
    python3 preprocess_image.py photo.jpg carbon_fibre --param2 8 --min-dist 20

RGB-only options:
    --param2 INT         Accumulator threshold (default: 12, lower=more circles)
    --min-dist INT       Min distance between circle centers in px (default: 25)
    --min-radius INT     Min circle radius in px (default: 8)
    --max-radius INT     Max circle radius in px (default: 14)

General options:
    --size INT           Output size in pixels (default: 400)
    --no-preview         Skip saving preview PNG files
    --output-dir PATH    Directory for output files (default: current directory)

Requirements:
    pip install numpy opencv-python
"""

import argparse
import os
import cv2
import numpy as np


def is_binary_image(img_gray):
    """Return True if image contains only 2 distinct values (binary)."""
    unique_vals = np.unique(img_gray)
    return len(unique_vals) <= 2


def preprocess_image(
    input_path,
    output_name="my_pattern",
    param2=12,
    min_dist=25,
    min_radius=8,
    max_radius=14,
    size=400,
    save_preview=True,
    output_dir=".",
):
    # --- Load image ---
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")
    print(f"Loaded: {input_path}  ({img.shape[1]}x{img.shape[0]} px)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Auto-detect mode ---
    if is_binary_image(gray):
        mode = "binary"
    else:
        mode = "rgb"
    print(f"Mode: {mode}")

    if mode == "binary":
        # ── Binary mode: use as-is, just resize ──
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
        # Hard binarize (handles 0/1 and 0/255)
        if resized.max() <= 1:
            binary = (resized * 255).astype(np.uint8)
        else:
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        black_pct = 100 * np.sum(binary == 0) / binary.size
        print(f"Black (fibre E=10): {black_pct:.1f}%  |  White (matrix E=1): {100-black_pct:.1f}%")

    else:
        # ── RGB mode: HoughCircles ──
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (5, 5), 1)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1,
            minDist=min_dist, param1=50, param2=param2,
            minRadius=min_radius, maxRadius=max_radius,
        )

        binary = np.ones((size, size), dtype=np.uint8) * 255  # white background
        n_circles = 0
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            # Remove circles touching the border (breaks Gmsh)
            circles = [c for c in circles if
                       c[0] - c[2] > 1 and c[0] + c[2] < size - 1 and
                       c[1] - c[2] > 1 and c[1] + c[2] < size - 1]
            n_circles = len(circles)
            for (x, y, r) in circles:
                cv2.circle(binary, (x, y), r, 0, -1)  # black circles = fibres
            print(f"Detected {n_circles} circles (border circles removed)")
        else:
            print("WARNING: no circles detected. Try lowering --param2.")

        black_pct = 100 * np.sum(binary == 0) / binary.size
        print(f"Black (fibre E=10): {black_pct:.1f}%  |  White (matrix E=1): {100-black_pct:.1f}%")
        if black_pct < 5:
            print("WARNING: very few fibres detected. Try lowering --param2.")
        if black_pct > 80:
            print("WARNING: too many fibres detected. Try raising --param2.")

    os.makedirs(output_dir, exist_ok=True)

    # --- Save 400x400 .txt for FEA (values 0/255) ---
    fea_dir = os.path.join(output_dir, "input_patterns")
    os.makedirs(fea_dir, exist_ok=True)
    txt_path = os.path.join(fea_dir, f"{output_name}.txt")
    np.savetxt(txt_path, binary, fmt="%d")
    print(f"FEA input ({size}x{size}, values 0/255) saved: {txt_path}")

    # --- Save 256x256 float for ML model (values 0/1) ---
    resized_256 = cv2.resize(binary, (256, 256), interpolation=cv2.INTER_AREA)
    binary_256  = (resized_256 >= 128).astype(np.float32)  # 0=fibre, 1=matrix
    ml_path = os.path.join(output_dir, f"{output_name}_256x256.npy")
    np.save(ml_path, binary_256)
    print(f"ML input (256x256, values 0/1) saved: {ml_path}")

    # --- Save previews ---
    if save_preview:
        preview_400 = os.path.join(output_dir, f"{output_name}_preview_{size}.png")
        preview_256 = os.path.join(output_dir, f"{output_name}_preview_256.png")
        cv2.imwrite(preview_400, binary)
        cv2.imwrite(preview_256, (binary_256 * 255).astype(np.uint8))
        print(f"Previews saved: {preview_400}  |  {preview_256}")

    print(f"\nNext steps:")
    print(f"  python3 NumpyImageToGmsh.py   (reads input_patterns/)")
    print(f"  python3 Equibiaxial_Hyperelastic.py mesh_files/{output_name}.xdmf")

    return binary, binary_256


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image for Mechanical MNIST FEA + ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  help="Path to input image")
    parser.add_argument("name",   nargs="?", default="my_pattern",
                        help="Output base name (default: my_pattern)")
    parser.add_argument("--param2",     type=int,   default=12,
                        help="RGB mode: HoughCircles accumulator threshold (default: 12)")
    parser.add_argument("--min-dist",   type=int,   default=25,
                        help="RGB mode: min distance between circle centers (default: 25)")
    parser.add_argument("--min-radius", type=int,   default=8,
                        help="RGB mode: min circle radius in px (default: 8)")
    parser.add_argument("--max-radius", type=int,   default=14,
                        help="RGB mode: max circle radius in px (default: 14)")
    parser.add_argument("--size",       type=int,   default=400,
                        help="Output size in pixels (default: 400)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip preview PNG files")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory (default: current)")

    args = parser.parse_args()

    preprocess_image(
        input_path=args.input,
        output_name=args.name,
        param2=args.param2,
        min_dist=args.min_dist,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        size=args.size,
        save_preview=not args.no_preview,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
