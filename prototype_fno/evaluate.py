"""
Evaluate trained multi-task FNO model: test metrics and visualizations.

Usage:
    python evaluate.py --data_dir ./processed --run_dir ./runs --output_dir ./eval_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import MechMNISTDataset, get_data_loaders
from model import MultiTaskFNO


def load_stats(data_dir):
    return np.load(Path(data_dir) / "norm_stats.npy", allow_pickle=True).item()


def denorm_disp(pred, stats):
    """De-normalize displacement predictions."""
    p = pred.clone()
    p[:, 0] = p[:, 0] * stats["disp_x_std"] + stats["disp_x_mean"]
    p[:, 1] = p[:, 1] * stats["disp_y_std"] + stats["disp_y_mean"]
    return p


def denorm_se(pred, stats):
    se_mean = torch.tensor(stats["se_mean"], device=pred.device)
    se_std = torch.tensor(stats["se_std"], device=pred.device) + 1e-10
    return pred * se_std + se_mean


def denorm_rf(pred, stats):
    rf_mean = torch.tensor(stats["rf_mean"], device=pred.device)
    rf_std = torch.tensor(stats["rf_std"], device=pred.device) + 1e-10
    return pred * rf_std + rf_mean


@torch.no_grad()
def compute_metrics(model, loader, device, stats):
    model.eval()
    metrics = {"disp_mse": 0, "disp_rel_l2": 0, "se_mse": 0, "rf_mse": 0}
    n = 0

    for inputs, disp_tgt, se_tgt, rf_tgt in loader:
        inputs = inputs.to(device)
        disp_tgt, se_tgt, rf_tgt = disp_tgt.to(device), se_tgt.to(device), rf_tgt.to(device)

        disp_pred, se_pred, rf_pred = model(inputs)

        # De-normalize all
        dp = denorm_disp(disp_pred, stats)
        dt = denorm_disp(disp_tgt, stats)
        sp = denorm_se(se_pred, stats)
        st = denorm_se(se_tgt, stats)
        rp = denorm_rf(rf_pred, stats)
        rt = denorm_rf(rf_tgt, stats)

        # Displacement metrics
        diff = dp - dt
        metrics["disp_mse"] += (diff ** 2).mean(dim=(1, 2, 3)).sum().item()
        l2_err = torch.sqrt((diff ** 2).sum(dim=(1, 2, 3)))
        l2_true = torch.sqrt((dt ** 2).sum(dim=(1, 2, 3)))
        metrics["disp_rel_l2"] += (l2_err / (l2_true + 1e-10)).sum().item()

        # Scalar metrics
        metrics["se_mse"] += ((sp - st) ** 2).mean(dim=1).sum().item()
        metrics["rf_mse"] += ((rp - rt) ** 2).mean(dim=1).sum().item()

        n += inputs.shape[0]

    return {k: v / n for k, v in metrics.items()}


def visualize_predictions(model, dataset, device, stats, output_dir,
                          n_samples=5, seed=42):
    model.eval()
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), n_samples, replace=False)

    for idx in indices:
        inp, disp_tgt, se_tgt, rf_tgt = dataset[idx]

        with torch.no_grad():
            disp_pred, se_pred, rf_pred = model(inp.unsqueeze(0).to(device))

        # De-normalize
        dp = denorm_disp(disp_pred, stats).cpu().squeeze(0).numpy()
        dt = denorm_disp(disp_tgt.unsqueeze(0), stats).cpu().squeeze(0).numpy()
        sp = denorm_se(se_pred, stats).cpu().squeeze(0).numpy()
        st = denorm_se(se_tgt.unsqueeze(0), stats).cpu().squeeze(0).numpy()
        inp_np = inp.squeeze(0).numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for row, (label, ch) in enumerate([("disp_x", 0), ("disp_y", 1)]):
            p, t = dp[ch], dt[ch]
            err = p - t
            vmin, vmax = min(p.min(), t.min()), max(p.max(), t.max())

            if row == 0:
                axes[row, 0].imshow(inp_np, cmap="gray", vmin=0, vmax=1)
                axes[row, 0].set_title("Input")
            else:
                axes[row, 0].axis("off")

            im1 = axes[row, 1].imshow(p, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f"Pred {label}")
            plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

            im2 = axes[row, 2].imshow(t, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f"True {label}")
            plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

            im3 = axes[row, 3].imshow(err, cmap="RdBu_r")
            axes[row, 3].set_title(f"Error {label}")
            plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        # Add strain energy comparison as text
        disp_levels = [0.0, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5]
        se_text = "Strain Energy (pred / true):\n"
        for i, d in enumerate(disp_levels):
            se_text += f"  d={d}: {sp[i]:.4f} / {st[i]:.4f}\n"
        fig.text(0.02, 0.02, se_text, fontsize=8, family="monospace",
                 verticalalignment="bottom")

        plt.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(output_dir / f"prediction_sample_{idx}.png", dpi=150)
        plt.close(fig)
        print(f"Saved visualization for sample {idx}")


def plot_training_history(run_dir: Path, output_dir: Path):
    with open(run_dir / "history.json") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, key, title in zip(
        axes[:3],
        ["disp", "se", "rf"],
        ["Displacement", "Strain Energy", "Reaction Force"],
    ):
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"], label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[3].plot(epochs, history["lr"])
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Learning Rate")
    axes[3].set_title("LR Schedule")
    axes[3].set_yscale("log")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task FNO model")
    parser.add_argument("--data_dir", type=str, default="./processed")
    parser.add_argument("--run_dir", type=str, default="./runs")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--n_vis", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.data_dir)

    # Load model with saved hparams
    ckpt = torch.load(
        Path(args.run_dir) / "best_model.pt",
        map_location=device, weights_only=False,
    )
    hparams = ckpt.get("hparams", {"modes": 16, "width": 64, "n_layers": 4})
    model = MultiTaskFNO(**hparams).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from epoch {ckpt['epoch']}")

    _, val_loader, test_loader = get_data_loaders(
        args.data_dir, batch_size=64, seed=42,
    )

    val_metrics = compute_metrics(model, val_loader, device, stats)
    test_metrics = compute_metrics(model, test_loader, device, stats)

    print(f"\nVal:  disp_MSE={val_metrics['disp_mse']:.6e}  disp_RelL2={val_metrics['disp_rel_l2']:.4f}  "
          f"se_MSE={val_metrics['se_mse']:.6e}  rf_MSE={val_metrics['rf_mse']:.6e}")
    print(f"Test: disp_MSE={test_metrics['disp_mse']:.6e}  disp_RelL2={test_metrics['disp_rel_l2']:.4f}  "
          f"se_MSE={test_metrics['se_mse']:.6e}  rf_MSE={test_metrics['rf_mse']:.6e}")

    all_metrics = {"val": val_metrics, "test": test_metrics, "epoch": ckpt["epoch"]}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    dataset = MechMNISTDataset(args.data_dir)
    visualize_predictions(model, dataset, device, stats, output_dir, n_samples=args.n_vis)
    plot_training_history(Path(args.run_dir), output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
