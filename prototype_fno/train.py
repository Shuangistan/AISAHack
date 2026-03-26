"""
Train the multi-task FNO surrogate model.

Predicts displacement fields, strain energy, and reaction forces simultaneously.

Usage:
    python train.py --data_dir ./processed --output_dir ./runs --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_data_loaders
from model import MultiTaskFNO


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = {"total": 0, "disp": 0, "se": 0, "rf": 0}
    n = 0
    for inputs, disp_tgt, se_tgt, rf_tgt in loader:
        inputs = inputs.to(device)
        disp_tgt = disp_tgt.to(device)
        se_tgt = se_tgt.to(device)
        rf_tgt = rf_tgt.to(device)

        optimizer.zero_grad()
        disp_pred, se_pred, rf_pred = model(inputs)

        loss_disp = criterion(disp_pred, disp_tgt)
        loss_se = criterion(se_pred, se_tgt)
        loss_rf = criterion(rf_pred, rf_tgt)
        loss = loss_disp + loss_se + loss_rf

        loss.backward()
        optimizer.step()

        losses["total"] += loss.item()
        losses["disp"] += loss_disp.item()
        losses["se"] += loss_se.item()
        losses["rf"] += loss_rf.item()
        n += 1

    return {k: v / n for k, v in losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = {"total": 0, "disp": 0, "se": 0, "rf": 0}
    n = 0
    for inputs, disp_tgt, se_tgt, rf_tgt in loader:
        inputs = inputs.to(device)
        disp_tgt = disp_tgt.to(device)
        se_tgt = se_tgt.to(device)
        rf_tgt = rf_tgt.to(device)

        disp_pred, se_pred, rf_pred = model(inputs)

        loss_disp = criterion(disp_pred, disp_tgt)
        loss_se = criterion(se_pred, se_tgt)
        loss_rf = criterion(rf_pred, rf_tgt)

        losses["total"] += (loss_disp + loss_se + loss_rf).item()
        losses["disp"] += loss_disp.item()
        losses["se"] += loss_se.item()
        losses["rf"] += loss_rf.item()
        n += 1

    return {k: v / n for k, v in losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Train multi-task FNO surrogate")
    parser.add_argument("--data_dir", type=str, default="./processed")
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, _ = get_data_loaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = MultiTaskFNO(
        modes=args.modes, width=args.width, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {
        "train_total": [], "train_disp": [], "train_se": [], "train_rf": [],
        "val_total": [], "val_disp": [], "val_se": [], "val_rf": [],
        "lr": [], "epoch_time": [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        for k in ["total", "disp", "se", "rf"]:
            history[f"train_{k}"].append(train_losses[k])
            history[f"val_{k}"].append(val_losses[k])
        history["lr"].append(lr)
        history["epoch_time"].append(elapsed)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_losses['total']:.6f} "
              f"(disp={train_losses['disp']:.4f} se={train_losses['se']:.4f} rf={train_losses['rf']:.4f}) | "
              f"Val: {val_losses['total']:.6f} | LR: {lr:.2e} | {elapsed:.1f}s")

        scheduler.step(val_losses["total"])

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_losses,
                "train_loss": train_losses,
                "hparams": {"modes": args.modes, "width": args.width, "n_layers": args.n_layers},
            }, output_dir / "best_model.pt")
            print(f"  -> Saved best model (val_total={val_losses['total']:.6f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
