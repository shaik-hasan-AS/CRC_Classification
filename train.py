"""
train.py
Main training script for MedLite-CRC.
Run: python train.py
      python train.py --config configs/finetune_v2.yaml --finetune outputs/checkpoints/ckpt_epoch175_acc0.9984.pt
"""



import time
import random
import argparse


import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast

import wandb

from data.dataset import get_train_val_loaders, get_crossval_loader, CRC_CLASSES, HYBRID_CLASSES
from models.medlite_crc import build_model, count_parameters
from utils.metrics import (
    compute_metrics, print_classification_report,
    CheckpointManager, EarlyStopping, AverageMeter, load_checkpoint
)
from utils.losses import compute_class_weights


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


def build_criterion(cfg, train_dataset=None):
    """Build loss function based on config: 'ce' or 'focal'."""
    tr = cfg["training"]
    loss_type = tr.get("loss", "ce").lower()
    label_smoothing = tr.get("label_smoothing", 0.1)

    # Compute class weights if requested
    class_weights = None
    if tr.get("use_class_weights", False) and train_dataset is not None:
        class_weights = compute_class_weights(
            train_dataset,
            num_classes=cfg["data"]["num_classes"],
            method=tr.get("class_weight_method", "inverse_freq")
        )
        print(f"  [LOSS] Class weights: {class_weights.numpy().round(3)}")

    print(f"  [LOSS] CrossEntropyLoss(smoothing={label_smoothing})")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )
    return criterion


# ── Mixup Augmentation ────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """
    Mixup: linearly interpolate between random pairs of training examples.
    Improves generalisation by creating virtual training samples.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 so the dominant sample stays dominant
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """Compute mixup loss as weighted combination."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ── CutMix Augmentation ───────────────────────────────────────────────────────

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lam to exactly match the pixel ratio that was swapped
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[index], lam


# ── Train One Epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch, teacher=None):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    grad_clip   = cfg["training"].get("grad_clip", 1.0)
    use_amp     = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    log_freq    = cfg["wandb"].get("log_freq", 50)
    mixup_alpha = cfg["training"].get("mixup_alpha", 0.0)
    cutmix_alpha = cfg["training"].get("cutmix_alpha", 0.0)
    use_mixup   = mixup_alpha > 0
    use_cutmix  = cutmix_alpha > 0



    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Optionally apply Mixup or Cutmix
        if use_cutmix:
            mixed_imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels, cutmix_alpha)
        elif use_mixup:
            mixed_imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, mixup_alpha)
        else:
            mixed_imgs, targets_a, targets_b, lam = imgs, labels, labels, 1.0

        with autocast(enabled=use_amp):
            logits = model(mixed_imgs)
            if use_cutmix or use_mixup:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, labels)
                
            # Add Knowledge Distillation Loss if Teacher is present
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(mixed_imgs)
                
                kd_T = cfg["training"].get("kd_temperature", 3.0)
                kd_alpha = cfg["training"].get("kd_alpha", 0.5)
                
                soft_targets = F.softmax(teacher_logits / kd_T, dim=1)
                log_probs = F.log_softmax(logits / kd_T, dim=1)
                
                kd_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (kd_T ** 2)
                loss = (1. - kd_alpha) * loss + kd_alpha * kd_loss
                


        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # For accuracy tracking, use original labels (not mixed)
        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), imgs.size(0))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if step % log_freq == 0:
            print(f"  Step [{step:4d}/{len(loader)}] loss={loss_meter.avg:.4f}")
            if cfg["wandb"]["enabled"]:
                wandb.log({"train/step_loss": loss_meter.avg,
                           "epoch": epoch, "step": step})

    target_classes = HYBRID_CLASSES if cfg["data"].get("is_hybrid", False) else CRC_CLASSES
    metrics = compute_metrics(all_preds, all_labels, target_classes)
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

    target_classes = HYBRID_CLASSES if cfg["data"].get("is_hybrid", False) else CRC_CLASSES
    metrics = compute_metrics(all_preds, all_labels, target_classes)
    metrics["loss"] = round(loss_meter.avg, 5)

    if split_name == "cross_val":
        print(f"\n[{split_name.upper()}] Detailed Report:")
        print_classification_report(all_preds, all_labels, target_classes)

    return metrics


# ── Main Training Loop ────────────────────────────────────────────────────────

def train(cfg_path: str, resume: str = None, finetune: str = None):
    cfg    = load_config(cfg_path)
    seed   = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} | CUDA: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    # ── Wandb ────────────────────────────────────────────────────────────────
    if cfg["wandb"]["enabled"]:
        run_tag = "finetune" if finetune else "train"
        wandb.init(
            project = cfg["wandb"]["project"],
            entity  = cfg["wandb"]["entity"] or None,
            config  = cfg,
            name    = f"MedLite-CRC_{run_tag}_{int(time.time())}",
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

    # ── Teacher Model (Knowledge Distillation) ───────────────────────────────
    teacher_model = None
    if cfg["training"].get("use_kd", False):
        print("\n[KD] Loading Teacher Model (MobileNetV2)...")
        teacher_model = models.mobilenet_v2(weights=None)
        # Modify classifier to output 9 classes
        teacher_model.classifier[1] = nn.Linear(teacher_model.last_channel, cfg["data"]["num_classes"])
        teacher_ckpt = cfg["training"].get("teacher_checkpoint")
        if teacher_ckpt:
            load_checkpoint(teacher_ckpt, teacher_model)
            print(f"  [KD] Teacher weights loaded from {teacher_ckpt}")
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    # ── Fine-tune: load ONLY model weights (fresh optimizer/scheduler) ────────
    if finetune:
        print(f"\n[FINETUNE] Loading model weights from: {finetune}")
        load_checkpoint(finetune, model)  # model only, no optimizer/scheduler
        print(f"[FINETUNE] Fresh optimizer + scheduler will be created at lr={cfg['training']['lr']}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    # Get train dataset for class weight computation
    train_dataset = train_loader.dataset
    criterion = build_criterion(cfg, train_dataset).to(device)
    if hasattr(criterion, 'alpha') and criterion.alpha is not None:
        criterion.alpha = criterion.alpha.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=cfg["training"].get("mixed_precision", True)
                           and device.type == "cuda")

    # ── Resume (restores optimizer/scheduler state, unlike finetune) ─────────
    start_epoch = 0
    best_val_acc = 0.0
    if resume and not finetune:
        start_epoch, best_val_acc = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1

    # ── Checkpoint / Early Stop ───────────────────────────────────────────────
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    if finetune:
        ckpt_dir = ckpt_dir.rstrip("/") + "_v2"  # separate dir for fine-tune runs
    ckpt_mgr = CheckpointManager(
        ckpt_dir,
        top_k=cfg["outputs"].get("save_top_k", 3)
    )
    early_stop = EarlyStopping(
        patience  = cfg["training"].get("early_stopping_patience", 20),
        min_delta = 1e-4,
    )

    # ── Cross-val eval frequency ──────────────────────────────────────────────
    crossval_freq = cfg["training"].get("crossval_eval_freq", 10)

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting training for {cfg['training']['epochs']} epochs...")
    if cfg["training"].get("cutmix_alpha", 0) > 0:
        print(f"[TRAIN] CutMix enabled (alpha={cfg['training']['cutmix_alpha']})")
    elif cfg["training"].get("mixup_alpha", 0) > 0:
        print(f"[TRAIN] Mixup enabled (alpha={cfg['training']['mixup_alpha']})")
    print()
    epochs = cfg["training"]["epochs"]

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        print(f"{'='*60}")
        print(f"Epoch [{epoch+1}/{epochs}]  lr={lr:.6f}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch, teacher=teacher_model
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

        # Cross-dataset eval at configured frequency
        if cross_val_loader and (epoch + 1) % crossval_freq == 0:
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
                        help="Path to checkpoint to resume from (restores optimizer/scheduler)")
    parser.add_argument("--finetune", default=None,
                        help="Path to checkpoint for fine-tuning (model weights only, fresh optimizer)")
    args = parser.parse_args()

    best_ckpt = train(args.config, args.resume, args.finetune)
    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
