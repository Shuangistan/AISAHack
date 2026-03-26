"""
Inference and visualization for trained Mechanical MNIST CH models.

Usage:
    python visualize.py --checkpoint checkpoints/best_model.pt \
                        --data_root ./data --n_samples 5
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from models import get_model, default_config, load_config
from dataset import MechMNISTCahnHilliard, NormStats


def load_model(checkpoint_path: str, cfg, device: torch.device):
    """Load a trained model from checkpoint."""
    model = get_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def denormalize_psi(psi: np.ndarray, norm: NormStats) -> np.ndarray:
    return psi * norm.psi_std + norm.psi_mean


def denormalize_force(force: np.ndarray, norm: NormStats) -> np.ndarray:
    return force * norm.force_std + norm.force_mean


def denormalize_disp(disp: np.ndarray, norm: NormStats) -> np.ndarray:
    # disp: (2, H, W), disp_mean/std: (2,) — broadcast over spatial dims
    return disp * norm.disp_std[:, None, None] + norm.disp_mean[:, None, None]


@torch.no_grad()
def predict_single(model, image_tensor, device):
    """Run inference on a single image."""
    x = image_tensor.unsqueeze(0).to(device)
    out = model(x)
    return out["psi"].cpu().numpy()[0], out["force"].cpu().numpy()[0], out["disp"].cpu().numpy()[0]


def plot_sample_comparison(
    image: np.ndarray,
    psi_pred: np.ndarray, psi_gt: np.ndarray,
    force_pred: np.ndarray, force_gt: np.ndarray,
    disp_pred: np.ndarray, disp_gt: np.ndarray,
    sample_idx: int = 0,
    save_path: str = None,
):
    """
    Create a comprehensive comparison plot for a single sample.

    Layout:
      Row 1: Input image | Strain energy curve | Reaction force curve
      Row 2: Predicted u_x | Predicted u_y | Displacement magnitude
      Row 3: Ground truth u_x | Ground truth u_y | Error magnitude
    """
    disp_levels = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # ── Row 1: Input + scalar curves ─────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image.squeeze(), cmap="gray", interpolation="nearest")
    ax_img.set_title(f"Input image (sample {sample_idx})", fontsize=11)
    ax_img.axis("off")

    ax_psi = fig.add_subplot(gs[0, 1])
    ax_psi.plot(disp_levels, psi_gt, "ko-", label="FE (ground truth)", markersize=5)
    ax_psi.plot(disp_levels, psi_pred, "r^--", label="Predicted", markersize=5)
    ax_psi.set_xlabel("Applied displacement d")
    ax_psi.set_ylabel("ΔΨ (strain energy)")
    ax_psi.set_title("Strain energy vs. displacement")
    ax_psi.legend(fontsize=9)
    ax_psi.grid(True, alpha=0.3)

    ax_force = fig.add_subplot(gs[0, 2])
    # Plot force for each boundary (4 boundaries × 7 levels → reshape)
    force_gt_2d = force_gt.reshape(4, 7) if force_gt.shape[0] == 28 else force_gt.reshape(-1, 7)
    force_pred_2d = force_pred.reshape(4, 7) if force_pred.shape[0] == 28 else force_pred.reshape(-1, 7)
    boundary_names = ["Bottom", "Right", "Top", "Left"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i in range(min(4, force_gt_2d.shape[0])):
        ax_force.plot(disp_levels, force_gt_2d[i], "o-", color=colors[i],
                      label=f"{boundary_names[i]} (GT)", markersize=4, alpha=0.7)
        ax_force.plot(disp_levels, force_pred_2d[i], "^--", color=colors[i],
                      label=f"{boundary_names[i]} (pred)", markersize=4, alpha=0.7)
    ax_force.set_xlabel("Applied displacement d")
    ax_force.set_ylabel("Reaction force")
    ax_force.set_title("Reaction forces vs. displacement")
    ax_force.legend(fontsize=7, ncol=2)
    ax_force.grid(True, alpha=0.3)

    # ── Row 2: Predicted displacement fields ─────────────────────────────
    pred_ux, pred_uy = disp_pred[0], disp_pred[1]
    pred_mag = np.sqrt(pred_ux**2 + pred_uy**2)

    for i, (field, title) in enumerate([
        (pred_ux, "Predicted u_x"),
        (pred_uy, "Predicted u_y"),
        (pred_mag, "Predicted |u|"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(field, cmap="RdBu_r" if i < 2 else "viridis",
                        interpolation="bilinear")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Row 3: Ground truth + error ──────────────────────────────────────
    gt_ux, gt_uy = disp_gt[0], disp_gt[1]
    error_mag = np.sqrt((pred_ux - gt_ux)**2 + (pred_uy - gt_uy)**2)

    for i, (field, title, cmap) in enumerate([
        (gt_ux, "Ground truth u_x", "RdBu_r"),
        (gt_uy, "Ground truth u_y", "RdBu_r"),
        (error_mag, "Displacement error |Δu|", "hot"),
    ]):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(field, cmap=cmap, interpolation="bilinear")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_training_history(history_path: str, save_path: str = None):
    """Plot training and validation loss curves."""
    import json
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    epochs = range(1, len(history["train"]) + 1)

    for ax, key, title in zip(
        axes,
        ["total", "psi", "force", "disp"],
        ["Total loss", "Strain energy loss", "Reaction force loss", "Displacement loss"],
    ):
        train_vals = [h[key] for h in history["train"]]
        val_vals = [h[key] for h in history["val"]]
        ax.plot(epochs, train_vals, label="Train", alpha=0.8)
        ax.plot(epochs, val_vals, label="Val", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory (e.g. experiments/runs/unet_20260326_120000). "
                             "If provided, config.json, best_model.pt, and norm_stats.npz "
                             "are loaded from there automatically.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--norm_path", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--history", type=str, default=None,
                        help="Path to training_history.json for loss plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve paths — run_dir is the convenient single-argument form
    if args.run_dir:
        run_dir = args.run_dir
        config_path = os.path.join(run_dir, "config.json")
        cfg = load_config(config_path) if os.path.exists(config_path) else default_config()
        checkpoint_path = os.path.join(run_dir, args.checkpoint)
        norm_path = args.norm_path or os.path.join(run_dir, "norm_stats.npz")
        if args.history is None and os.path.exists(os.path.join(run_dir, "training_history.json")):
            args.history = os.path.join(run_dir, "training_history.json")
    else:
        cfg = default_config()
        checkpoint_path = args.checkpoint
        norm_path = args.norm_path or "checkpoints/norm_stats.npz"

    # Load model
    model = load_model(checkpoint_path, cfg, device)
    norm = NormStats().load(norm_path)

    # Load a few test samples
    ds = MechMNISTCahnHilliard(
        args.data_root, img_size=cfg.img_size, norm_stats=norm
    )

    # Pick random samples
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=min(args.n_samples, len(ds)), replace=False)

    for i, idx in enumerate(indices):
        sample = ds[idx]
        psi_pred, force_pred, disp_pred = predict_single(
            model, sample["image"], device
        )

        # Denormalize for plotting
        psi_pred_dn = denormalize_psi(psi_pred, norm)
        psi_gt_dn = denormalize_psi(sample["psi"].numpy(), norm)
        force_pred_dn = denormalize_force(force_pred, norm)
        force_gt_dn = denormalize_force(sample["force"].numpy(), norm)
        disp_pred_dn = denormalize_disp(disp_pred, norm)
        disp_gt_dn = denormalize_disp(sample["disp"].numpy(), norm)

        plot_sample_comparison(
            image=sample["image"].numpy(),
            psi_pred=psi_pred_dn, psi_gt=psi_gt_dn,
            force_pred=force_pred_dn, force_gt=force_gt_dn,
            disp_pred=disp_pred_dn, disp_gt=disp_gt_dn,
            sample_idx=int(idx),
            save_path=os.path.join(args.out_dir, f"prediction_sample_{idx:05d}.png"),
        )

    # Plot training history if available
    if args.history:
        plot_training_history(
            args.history,
            save_path=os.path.join(args.out_dir, "training_history.png"),
        )

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
