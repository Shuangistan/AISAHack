"""
Predict displacement fields, strain energy, and reaction forces from an image.

Usage:
    python predict.py --image ../new.jpeg
    python predict.py --image ../data/Case1_input_patterns/Image100.txt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from model import MultiTaskFNO


def load_image(image_path: str) -> np.ndarray:
    """Load image as 64x64 float32 array. Supports .txt, .jpeg, .png."""
    path = Path(image_path)

    if path.suffix == ".txt":
        img = np.loadtxt(path, dtype=np.float32)
    else:
        pil_img = Image.open(path).convert("L")
        img = np.array(pil_img, dtype=np.float32) / 255.0
        img = (img > 0.5).astype(np.float32)

    h, w = img.shape
    if h == 64 and w == 64:
        return img
    crop_h = (h // 64) * 64
    crop_w = (w // 64) * 64
    offset_h = (h - crop_h) // 2
    offset_w = (w - crop_w) // 2
    cropped = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
    block_h = crop_h // 64
    block_w = crop_w // 64
    return cropped.reshape(64, block_h, 64, block_w).mean(axis=(1, 3)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Predict all outputs from image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./runs")
    parser.add_argument("--stats_dir", type=str, default="./processed")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model with saved hparams
    ckpt = torch.load(
        Path(args.model_dir) / "best_model.pt",
        map_location=device, weights_only=False,
    )
    hparams = ckpt.get("hparams", {"modes": 16, "width": 64, "n_layers": 4})
    model = MultiTaskFNO(**hparams).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stats = np.load(Path(args.stats_dir) / "norm_stats.npy", allow_pickle=True).item()

    img_64 = load_image(args.image)
    print(f"Input: {args.image} -> {img_64.shape}")

    inp = torch.from_numpy(img_64).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        disp_pred, se_pred, rf_pred = model(inp)

    # De-normalize
    disp_x = (disp_pred[0, 0].cpu().numpy() * stats["disp_x_std"] + stats["disp_x_mean"])
    disp_y = (disp_pred[0, 1].cpu().numpy() * stats["disp_y_std"] + stats["disp_y_mean"])

    se_mean = np.array(stats["se_mean"])
    se_std = np.array(stats["se_std"]) + 1e-10
    se = se_pred[0].cpu().numpy() * se_std + se_mean

    rf_mean = np.array(stats["rf_mean"])
    rf_std = np.array(stats["rf_std"]) + 1e-10
    rf = rf_pred[0].cpu().numpy() * rf_std + rf_mean

    # Print results
    print(f"\nDisplacement: disp_x [{disp_x.min():.4f}, {disp_x.max():.4f}]  "
          f"disp_y [{disp_y.min():.4f}, {disp_y.max():.4f}]")

    disp_levels = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("\nStrain Energy:")
    for d, v in zip(disp_levels, se):
        print(f"  d={d:.3f}: {v:.6f}")

    print("\nReaction Forces (4 boundaries x 7 levels):")
    rf_reshaped = rf.reshape(7, 4)
    print(f"  {'d':>6s}  {'F_left':>10s}  {'F_right':>10s}  {'F_bottom':>10s}  {'F_top':>10s}")
    for i, d in enumerate(disp_levels):
        print(f"  {d:6.3f}  {rf_reshaped[i,0]:10.4f}  {rf_reshaped[i,1]:10.4f}  "
              f"{rf_reshaped[i,2]:10.4f}  {rf_reshaped[i,3]:10.4f}")

    # Save
    stem = Path(args.image).stem
    np.savetxt(output_dir / f"{stem}_disp_x.txt", disp_x, fmt="%.6e")
    np.savetxt(output_dir / f"{stem}_disp_y.txt", disp_y, fmt="%.6e")
    np.savetxt(output_dir / f"{stem}_strain_energy.txt", se, fmt="%.6e")
    np.savetxt(output_dir / f"{stem}_rxn_force.txt", rf, fmt="%.6e")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_64, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input Pattern")

    im1 = axes[1].imshow(disp_x, cmap="RdBu_r")
    axes[1].set_title("Predicted disp_x")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(disp_y, cmap="RdBu_r")
    axes[2].set_title("Predicted disp_y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add strain energy text
    se_text = "Strain Energy:\n" + "\n".join(
        f"d={d:.3f}: {v:.4f}" for d, v in zip(disp_levels, se)
    )
    fig.text(0.02, 0.02, se_text, fontsize=8, family="monospace",
             verticalalignment="bottom")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_dir / f"{stem}_prediction.png", dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
