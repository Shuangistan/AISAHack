"""
Generate a comprehensive FEA-style results figure from a new image.

Replicates the layout of user-api.jpeg:
  Top row:    Original image | Binary pattern | disp_x | disp_y | |u| magnitude
  Bottom row: Strain energy vs displacement | Reaction forces vs displacement

Usage:
    python visualize.py --image ../new.jpeg
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from model import MultiTaskUNet


def load_image_raw(image_path: str):
    """Load raw image for display and return both raw and 64x64 versions."""
    path = Path(image_path)

    if path.suffix == ".txt":
        img = np.loadtxt(path, dtype=np.float32)
        raw = img.copy()
    else:
        pil_img = Image.open(path).convert("L")
        raw = np.array(pil_img, dtype=np.float32) / 255.0
        img = (raw > 0.5).astype(np.float32)

    # Downsample to 64x64
    h, w = img.shape
    if h == 64 and w == 64:
        img_64 = img
    else:
        crop_h = (h // 64) * 64
        crop_w = (w // 64) * 64
        offset_h = (h - crop_h) // 2
        offset_w = (w - crop_w) // 2
        cropped = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
        block_h = crop_h // 64
        block_w = crop_w // 64
        img_64 = cropped.reshape(64, block_h, 64, block_w).mean(axis=(1, 3)).astype(np.float32)

    return raw, img_64


def main():
    parser = argparse.ArgumentParser(description="Generate FEA results visualization")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./runs")
    parser.add_argument("--stats_dir", type=str, default="./processed")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MultiTaskUNet(in_channels=1).to(device)
    ckpt = torch.load(
        Path(args.model_dir) / "best_model.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stats = np.load(Path(args.stats_dir) / "norm_stats.npy", allow_pickle=True).item()

    # Load image
    raw_img, img_64 = load_image_raw(args.image)
    binary_64 = (img_64 > 0.5).astype(np.float32)

    # Predict
    inp = torch.from_numpy(img_64).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        disp_pred, se_pred, rf_pred = model(inp)

    # De-normalize
    disp_x = disp_pred[0, 0].cpu().numpy() * stats["disp_x_std"] + stats["disp_x_mean"]
    disp_y = disp_pred[0, 1].cpu().numpy() * stats["disp_y_std"] + stats["disp_y_mean"]
    disp_mag = np.sqrt(disp_x**2 + disp_y**2)

    se = se_pred[0].cpu().numpy() * (np.array(stats["se_std"]) + 1e-10) + np.array(stats["se_mean"])
    rf = rf_pred[0].cpu().numpy() * (np.array(stats["rf_std"]) + 1e-10) + np.array(stats["rf_mean"])

    disp_levels = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Reshape rf: (28,) -> (7 levels, 4 boundaries)
    # Order: F_left, F_right, F_bottom, F_top per level
    rf_reshaped = rf.reshape(7, 4)

    stem = Path(args.image).stem

    # --- Create figure ---
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"FEA Results \u2014 {stem}", fontsize=16, fontweight="bold")

    # Top row: 5 panels
    gs_top = fig.add_gridspec(1, 5, left=0.03, right=0.97, top=0.92, bottom=0.52,
                               wspace=0.35)

    # 1. Original image (RGB/grayscale)
    ax0 = fig.add_subplot(gs_top[0])
    ax0.imshow(raw_img, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Original image", fontsize=10)
    ax0.axis("off")

    # 2. Input pattern (binary 64x64)
    ax1 = fig.add_subplot(gs_top[1])
    ax1.imshow(binary_64, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Input pattern\n(binary)", fontsize=10)
    ax1.axis("off")

    # 3. Displacement u_x
    ax2 = fig.add_subplot(gs_top[2])
    im2 = ax2.imshow(disp_x, cmap="RdBu_r")
    ax2.set_title("Displacement $u_x$ (d=0.5)", fontsize=10)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 4. Displacement u_y
    ax3 = fig.add_subplot(gs_top[3])
    im3 = ax3.imshow(disp_y, cmap="RdBu_r")
    ax3.set_title("Displacement $u_y$ (d=0.5)", fontsize=10)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 5. Displacement magnitude
    ax4 = fig.add_subplot(gs_top[4])
    im4 = ax4.imshow(disp_mag, cmap="viridis")
    ax4.set_title("Displacement magnitude\n$|u|$ (d=0.5)", fontsize=10)
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Bottom row: 2 plots
    gs_bot = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.44, bottom=0.06,
                               wspace=0.3)

    # 6. Strain energy vs displacement
    ax5 = fig.add_subplot(gs_bot[0])
    ax5.plot(disp_levels, se, "o-", color="tab:blue", markersize=6, linewidth=1.5)
    for d, v in zip(disp_levels, se):
        ax5.annotate(f"{v:.3f}", (d, v), textcoords="offset points",
                     xytext=(5, 8), fontsize=8)
    ax5.set_xlabel("Applied displacement d", fontsize=10)
    ax5.set_ylabel("Strain energy ($\\Delta\\Psi$)", fontsize=10)
    ax5.set_title("Strain energy vs displacement", fontsize=11)
    ax5.set_xlim(-0.02, 0.52)
    ax5.grid(True, alpha=0.3)

    # 7. Reaction forces vs displacement
    ax6 = fig.add_subplot(gs_bot[1])
    labels = ["$F_{left,x}$", "$F_{right,x}$", "$F_{bottom,y}$", "$F_{top,y}$"]
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax6.plot(disp_levels, rf_reshaped[:, i], "o-", color=color, label=label,
                 markersize=5, linewidth=1.5)
    ax6.set_xlabel("Applied displacement d", fontsize=10)
    ax6.set_ylabel("Reaction force", fontsize=10)
    ax6.set_title("Reaction forces vs displacement", fontsize=11)
    ax6.legend(fontsize=9, loc="upper left")
    ax6.set_xlim(-0.02, 0.52)
    ax6.grid(True, alpha=0.3)

    out_path = output_dir / f"{stem}_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
