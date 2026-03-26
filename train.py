"""
Training script for Mechanical MNIST Cahn-Hilliard U-Net multi-regression.

Usage:
    python train.py --data_root ./data --epochs 100 --batch_size 8
    python train.py --data_root ./data --evaluate --checkpoint best_model.pt

Features:
    - Automatic mixed precision (AMP) for memory-efficient single-GPU training
    - Learned multi-task loss weighting (Kendall et al.)
    - Cosine annealing with linear warmup
    - Early stopping on validation loss
    - Per-target metrics (MSE, MAE, R²) logged each epoch
    - Checkpoint saving with best-model tracking
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from model import UNetMultiRegression, MultiTaskLoss, count_parameters
from dataset import create_dataloaders, NormStats


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute R² score (coefficient of determination)."""
    ss_res = ((target - pred) ** 2).sum().item()
    ss_tot = ((target - target.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-8)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute MSE, MAE, and R² for a batch of predictions."""
    with torch.no_grad():
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        r2 = compute_r2(pred, target)
    return {"mse": mse, "mae": mae, "r2": r2}


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device

        # ── Model ────────────────────────────────────────────────────────
        self.model = UNetMultiRegression(
            in_channels=cfg.in_channels,
            encoder_channels=cfg.encoder_channels,
            n_psi=cfg.n_psi,
            n_force=cfg.n_force,
            disp_channels=cfg.disp_channels,
            use_bn=cfg.use_batchnorm,
            dropout=cfg.dropout,
        ).to(device)

        print(f"[Model] Parameters: {count_parameters(self.model):,}")

        # ── Loss ─────────────────────────────────────────────────────────
        self.criterion = MultiTaskLoss(
            n_tasks=3, use_learned=cfg.use_learned_loss_weights
        ).to(device)

        # ── Optimizer (include loss params if learned) ───────────────────
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # ── Scheduler: linear warmup → cosine annealing ──────────────────
        warmup = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[cfg.warmup_epochs]
        )

        # ── AMP (only effective on CUDA) ─────────────────────────────────
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.use_amp = cfg.use_amp and device.type == "cuda"
        self.scaler = GradScaler(device=self.device_type, enabled=self.use_amp)

        # ── Tracking ─────────────────────────────────────────────────────
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train": [], "val": []}

        # ── Directories ──────────────────────────────────────────────────
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

    def train_epoch(self, loader) -> dict:
        self.model.train()
        epoch_losses = {"total": 0, "psi": 0, "force": 0, "disp": 0}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            img = batch["image"].to(self.device)
            psi_gt = batch["psi"].to(self.device)
            force_gt = batch["force"].to(self.device)
            disp_gt = batch["disp"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(self.device_type, enabled=self.use_amp):
                psi_pred, force_pred, disp_pred = self.model(img)

                loss_psi = F.mse_loss(psi_pred, psi_gt)
                loss_force = F.mse_loss(force_pred, force_gt)
                loss_disp = F.mse_loss(disp_pred, disp_gt)

                total_loss, weights = self.criterion(
                    loss_psi, loss_force, loss_disp,
                    fixed_weights=(self.cfg.lambda_psi, self.cfg.lambda_force, self.cfg.lambda_disp),
                )

            # Skip batch if forward pass produced NaN/inf (GradScaler only
            # catches inf in gradients, not NaN — so we guard here instead)
            if not torch.isfinite(total_loss):
                print(f"  [Warning] Non-finite loss at batch {batch_idx}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            scale_before = self.scaler.get_scale()
            self.scaler.scale(total_loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            all_params = list(self.model.parameters()) + list(self.criterion.parameters())
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # If scaler reduced its scale, gradients overflowed — step was skipped
            if self.scaler.get_scale() < scale_before:
                print(f"  [Warning] Gradient overflow at batch {batch_idx}, step skipped")
                continue

            epoch_losses["total"] += total_loss.item()
            epoch_losses["psi"] += loss_psi.item()
            epoch_losses["force"] += loss_force.item()
            epoch_losses["disp"] += loss_disp.item()
            n_batches += 1

            if batch_idx % self.cfg.print_every == 0 and batch_idx > 0:
                print(
                    f"  [Batch {batch_idx}/{len(loader)}] "
                    f"loss={total_loss.item():.4f} "
                    f"ψ={loss_psi.item():.4f} F={loss_force.item():.4f} "
                    f"u={loss_disp.item():.4f} "
                    f"w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"
                )

        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self, loader) -> dict:
        self.model.eval()
        epoch_losses = {"total": 0, "psi": 0, "force": 0, "disp": 0}
        metrics = {"psi": [], "force": [], "disp": []}
        n_batches = 0

        for batch in loader:
            img = batch["image"].to(self.device)
            psi_gt = batch["psi"].to(self.device)
            force_gt = batch["force"].to(self.device)
            disp_gt = batch["disp"].to(self.device)

            with autocast(self.device_type, enabled=self.use_amp):
                psi_pred, force_pred, disp_pred = self.model(img)

                loss_psi = F.mse_loss(psi_pred, psi_gt)
                loss_force = F.mse_loss(force_pred, force_gt)
                loss_disp = F.mse_loss(disp_pred, disp_gt)

                total_loss, _ = self.criterion(
                    loss_psi, loss_force, loss_disp,
                    fixed_weights=(self.cfg.lambda_psi, self.cfg.lambda_force, self.cfg.lambda_disp),
                )

            epoch_losses["total"] += total_loss.item()
            epoch_losses["psi"] += loss_psi.item()
            epoch_losses["force"] += loss_force.item()
            epoch_losses["disp"] += loss_disp.item()

            metrics["psi"].append(compute_metrics(psi_pred, psi_gt))
            metrics["force"].append(compute_metrics(force_pred, force_gt))
            metrics["disp"].append(compute_metrics(
                disp_pred.flatten(1), disp_gt.flatten(1)
            ))
            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        # Average per-target metrics
        avg_metrics = {}
        for target in ["psi", "force", "disp"]:
            avg_metrics[target] = {
                k: np.mean([m[k] for m in metrics[target]])
                for k in ["mse", "mae", "r2"]
            }

        return {**avg_losses, "metrics": avg_metrics}

    def fit(self, train_loader, val_loader, start_epoch: int = 1):
        print(f"\n{'='*70}")
        print(f"Training U-Net multi-regression ({self.cfg.epochs} epochs)")
        if start_epoch > 1:
            print(f"Resuming from epoch {start_epoch}")
        print(f"{'='*70}\n")

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            t0 = time.time()

            # ── Train ────────────────────────────────────────────────────
            train_loss = self.train_epoch(train_loader)

            # ── Validate ─────────────────────────────────────────────────
            val_result = self.validate(val_loader)
            val_loss = val_result["total"]

            # ── Scheduler step ───────────────────────────────────────────
            self.scheduler.step()

            # ── Logging ──────────────────────────────────────────────────
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            # Get learned weights if applicable
            weight_str = ""
            if self.cfg.use_learned_loss_weights:
                w = torch.exp(-self.criterion.log_vars).detach().cpu().numpy()
                weight_str = f" w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"

            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs} │ "
                f"Train {train_loss['total']:.4f} │ "
                f"Val {val_loss:.4f} │ "
                f"ψ_R²={val_result['metrics']['psi']['r2']:.4f} "
                f"F_R²={val_result['metrics']['force']['r2']:.4f} "
                f"u_MSE={val_result['metrics']['disp']['mse']:.4f} │ "
                f"lr={lr:.2e}{weight_str} │ "
                f"{elapsed:.1f}s"
            )

            self.history["train"].append(train_loss)
            self.history["val"].append({**val_result})

            # ── Checkpointing ────────────────────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch)
                print(f"  ★ New best model saved (val_loss={val_loss:.6f})")
            else:
                self.patience_counter += 1

            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch:03d}.pt", epoch)

            # ── Early stopping ───────────────────────────────────────────
            if self.patience_counter >= self.cfg.patience:
                print(f"\n[Early stopping] No improvement for {self.cfg.patience} epochs.")
                break

        # Save training history
        self._save_history()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")

    def _save_checkpoint(self, filename: str, epoch: int):
        path = os.path.join(self.cfg.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, path: str, resume: bool = False) -> int:
        """Load checkpoint. Returns the epoch to resume from (start_epoch)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "criterion_state_dict" in ckpt:
            self.criterion.load_state_dict(ckpt["criterion_state_dict"])
        saved_epoch = ckpt.get("epoch", 0)
        if resume:
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "best_val_loss" in ckpt:
                self.best_val_loss = ckpt["best_val_loss"]
            print(f"[Checkpoint] Resuming from {path} (epoch {saved_epoch})")
            return saved_epoch + 1
        print(f"[Checkpoint] Loaded from {path} (epoch {saved_epoch})")
        return 1

    def _save_history(self):
        path = os.path.join(self.cfg.log_dir, "training_history.json")

        def serialize(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=serialize)
        print(f"[History] Saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation on test set
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(trainer: Trainer, test_loader, norm_stats: NormStats):
    """Full evaluation on test set with denormalized metrics."""
    print(f"\n{'='*70}")
    print("Evaluating on test set")
    print(f"{'='*70}\n")

    result = trainer.validate(test_loader)

    print(f"{'Target':<20} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 55)
    for target in ["psi", "force", "disp"]:
        m = result["metrics"][target]
        print(f"{target:<20} {m['mse']:10.6f} {m['mae']:10.6f} {m['r2']:10.4f}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on Mechanical MNIST CH")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from --checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    # Override config with CLI args
    cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.no_amp:
        cfg.use_amp = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    print(f"\n[Data] Loading from {cfg.data_root}")
    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        data_root=cfg.data_root,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        num_workers=cfg.num_workers,
        max_samples=args.max_samples,
    )

    # Save normalization stats for inference
    norm_path = os.path.join(cfg.checkpoint_dir, "norm_stats.npz")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    norm_stats.save(norm_path)
    print(f"[Norm] Stats saved to {norm_path}")

    # Trainer
    trainer = Trainer(cfg, device)

    if args.evaluate:
        ckpt_path = os.path.join(cfg.checkpoint_dir, args.checkpoint)
        trainer.load_checkpoint(ckpt_path)
        evaluate(trainer, test_loader, norm_stats)
    else:
        start_epoch = 1
        if args.resume:
            ckpt_path = os.path.join(cfg.checkpoint_dir, args.checkpoint)
            start_epoch = trainer.load_checkpoint(ckpt_path, resume=True)
        trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
        # Final test evaluation
        trainer.load_checkpoint(os.path.join(cfg.checkpoint_dir, "best_model.pt"))
        evaluate(trainer, test_loader, norm_stats)


if __name__ == "__main__":
    main()
