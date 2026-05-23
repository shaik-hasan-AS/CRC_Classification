"""
train.py
Main training script for MedLite-CRC.
Run: python train.py
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import wandb

from data.dataset import get_train_val_loaders, get_crossval_loader, CRC_CLASSES
from models.medlite_crc import build_model, count_parameters
from utils.metrics import (
    compute_metrics, print_classification_report,
    CheckpointManager, EarlyStopping, AverageMeter, load_checkpoint
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Optimizer & Scheduler ─────────────────────────────────────────────────────

def build_optimizer(model, cfg):
    tr = cfg["training"]
    if tr["optimizer"].lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
        )
    elif tr["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
        )
    raise ValueError(f"Unknown optimizer: {tr['optimizer']}")


def build_scheduler(optimizer, cfg):
    tr = cfg["training"]
    total_steps   = tr["epochs"]
    warmup_epochs = tr.get("warmup_epochs", 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_steps - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Train One Epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    use_amp   = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    log_freq  = cfg["wandb"].get("log_freq", 50)

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), imgs.size(0))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if step % log_freq == 0:
            print(f"  Step [{step:4d}/{len(loader)}] loss={loss_meter.avg:.4f}")
            if cfg["wandb"]["enabled"]:
                wandb.log({"train/step_loss": loss_meter.avg,
                           "epoch": epoch, "step": step})

    metrics = compute_metrics(all_preds, all_labels, CRC_CLASSES)
    metrics["loss"] = round(loss_meter.avg, 5)
    return metrics


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, split_name="val"):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), imgs.size(0))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels, CRC_CLASSES)
    metrics["loss"] = round(loss_meter.avg, 5)

    if split_name in ("cross_val", "unitopatho"):
        print(f"\n[{split_name.upper()}] Detailed Report:")
        print_classification_report(all_preds, all_labels, CRC_CLASSES)

    return metrics


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(cfg_path: str, resume: str = None):
    cfg    = load_config(cfg_path)
    seed   = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} | CUDA: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    # ── Wandb ────────────────────────────────────────────────────────────────
    if cfg["wandb"]["enabled"]:
        wandb.init(
            project = cfg["wandb"]["project"],
            entity  = cfg["wandb"]["entity"] or None,
            config  = cfg,
            name    = f"MedLite-CRC_run_{int(time.time())}",
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[DATA] Loading datasets...")
    train_loader, val_loader = get_train_val_loaders(cfg)
    cross_val_loader         = get_crossval_loader(cfg)

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[MODEL] Building MedLite-CRC...")
    model = build_model(cfg).to(device)
    stats = count_parameters(model)
    print(f"  Parameters: {stats['total_M']}M total | {stats['trainable_M']}M trainable")

    if cfg["wandb"]["enabled"]:
        wandb.config.update({"model_params_M": stats["total_M"]})
        wandb.watch(model, log="gradients", log_freq=200)

    # ── Optimizer / Scheduler / Loss ─────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=cfg["training"].get("mixed_precision", True)
                           and device.type == "cuda")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    if resume:
        start_epoch, best_val_acc = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1

    # ── Checkpoint / Early Stop ───────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(
        cfg["outputs"]["checkpoint_dir"],
        top_k=cfg["outputs"].get("save_top_k", 3)
    )
    early_stop = EarlyStopping(
        patience  = cfg["training"].get("early_stopping_patience", 20),
        min_delta = 1e-4,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting training for {cfg['training']['epochs']} epochs...\n")
    epochs = cfg["training"]["epochs"]

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"{'='*60}")
        print(f"Epoch [{epoch+1}/{epochs}]  lr={lr:.6f}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
        )

        # Val (in-distribution)
        val_metrics = evaluate(model, val_loader, criterion, device, cfg, "val")

        scheduler.step()

        val_acc = val_metrics["accuracy"]
        epoch_time = time.time() - t0

        print(f"\n  [TRAIN] loss={train_metrics['loss']:.4f}  "
              f"acc={train_metrics['accuracy']:.4f}  "
              f"F1={train_metrics['macro_f1']:.4f}")
        print(f"  [VAL]   loss={val_metrics['loss']:.4f}  "
              f"acc={val_acc:.4f}  "
              f"F1={val_metrics['macro_f1']:.4f}  "
              f"time={epoch_time:.1f}s")

        # Cross-dataset eval every 10 epochs
        if cross_val_loader and (epoch + 1) % 10 == 0:
            cv_metrics = evaluate(model, cross_val_loader, criterion, device, cfg, "cross_val")
            print(f"  [CROSS-VAL] acc={cv_metrics['accuracy']:.4f}  "
                  f"F1={cv_metrics['macro_f1']:.4f}")
            if cfg["wandb"]["enabled"]:
                wandb.log({f"cross_val/{k}": v for k, v in cv_metrics.items()},
                          step=epoch)

        # Wandb logging
        if cfg["wandb"]["enabled"]:
            wandb.log({
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v   for k, v in val_metrics.items()},
                "lr": lr, "epoch": epoch,
            }, step=epoch)

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_mgr.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc":         best_val_acc,
                "config":               cfg,
            }, val_acc, epoch)

        # Early stopping
        if early_stop.step(val_acc):
            print(f"\n[EARLY STOP] Triggered at epoch {epoch+1}")
            break

    print(f"\n[DONE] Best val accuracy: {best_val_acc:.4f}")
    print(f"[DONE] Best checkpoint: {ckpt_mgr.best_checkpoint()}")

    if cfg["wandb"]["enabled"]:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.finish()

    return ckpt_mgr.best_checkpoint()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedLite-CRC")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    best_ckpt = train(args.config, args.resume)
    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
