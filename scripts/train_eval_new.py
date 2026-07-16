import os
import time
import json
import argparse
import yaml
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

from models.medlite_crc import build_model, count_parameters
from data.transforms import get_train_transforms, get_val_transforms
from utils.metrics import (
    compute_metrics, print_classification_report,
    CheckpointManager, AverageMeter, load_checkpoint
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    warmup_epochs = tr.get("warmup_epochs", 1)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_steps - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_dataset_filtered(root_dir, transform=None):
    dataset = ImageFolder(root_dir, transform=transform)
    # Check if this is EBHI-SEG (contains 'image' and 'label' structure)
    # We can detect this by checking if any path contains 'label'
    has_labels = any("label" in Path(p).parts for p, _ in dataset.samples)
    if has_labels:
        filtered_samples = [
            (p, t) for p, t in dataset.samples
            if "image" in Path(p).parts and "label" not in Path(p).parts
        ]
        dataset.samples = filtered_samples
        dataset.imgs = filtered_samples
        print(f"  [FILTER] Filtered dataset to {len(dataset)} images (removed segmentation masks).")
    return dataset


def get_dataloaders(cfg):
    bs = cfg["training"]["batch_size"]
    val_bs = cfg["training"]["val_batch_size"]
    nw = cfg["data"]["num_workers"]
    pin = cfg["data"]["pin_memory"]
    seed = cfg.get("project", {}).get("seed", 42)

    if "train_dir" in cfg["data"] and "test_dir" in cfg["data"]:
        train_dir = cfg["data"]["train_dir"]
        test_dir = cfg["data"]["test_dir"]
        
        train_dataset = load_dataset_filtered(train_dir, transform=get_train_transforms(cfg))
        val_dataset = load_dataset_filtered(test_dir, transform=get_val_transforms(cfg))
        
        train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=val_bs, shuffle=False,
            num_workers=nw, pin_memory=pin
        )
        print(f"[DATA] Train samples: {len(train_dataset):,} | Test samples: {len(val_dataset):,}")
        return train_loader, val_loader
    
    dataset_dir = cfg["data"]["dataset_dir"]
    full_train_dataset = load_dataset_filtered(dataset_dir, transform=get_train_transforms(cfg))
    full_val_dataset = load_dataset_filtered(dataset_dir, transform=get_val_transforms(cfg))
    
    total = len(full_train_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(total), [train_size, val_size], generator=generator)
    
    train_subset = Subset(full_train_dataset, train_idx.indices)
    val_subset = Subset(full_val_dataset, val_idx.indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=val_bs, shuffle=False,
        num_workers=nw, pin_memory=pin
    )
    print(f"[DATA] Total samples: {total:,} | Split -> Train: {len(train_subset):,} | Val: {len(val_subset):,}")
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), imgs.size(0))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    metrics = compute_metrics(all_preds, all_labels, cfg["data"]["classes"])
    metrics["loss"] = round(loss_meter.avg, 5)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
            
        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), imgs.size(0))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    metrics = compute_metrics(all_preds, all_labels, cfg["data"]["classes"])
    metrics["loss"] = round(loss_meter.avg, 5)
    return metrics, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, class_names, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [CM SAVED] Confusion matrix -> {save_path}")


def main(config_path, scratch=False):
    cfg = load_config(config_path)
    set_seed(cfg.get("project", {}).get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "SCRATCH" if scratch else "PRETRAINED"
    print(f"\n[START] Mode: {mode_str} | Dataset: {cfg['project']['name']} | Device: {device}")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(cfg)
    
    # 2. Build Model
    if scratch:
        print(f"[MODEL] Building model with {cfg['data']['num_classes']} classes from scratch...")
        model = build_model(cfg).to(device)
    else:
        print("[MODEL] Initializing model with 9 classes to load SOTA checkpoint...")
        cfg_9 = copy.deepcopy(cfg)
        cfg_9["data"]["num_classes"] = 9
        model = build_model(cfg_9).to(device)
        
        pretrained_ckpt = "outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"
        if os.path.exists(pretrained_ckpt):
            load_checkpoint(pretrained_ckpt, model)
        else:
            print(f"[WARN] Pretrained SOTA checkpoint not found at {pretrained_ckpt}! Falling back to scratch...")
            
        print(f"[MODEL] Swapping classifier head to output {cfg['data']['num_classes']} classes...")
        C_channels = cfg["model"].get("base_channels", 32)
        model.classifier[4] = nn.Linear(C_channels * 8, cfg["data"]["num_classes"]).to(device)
    
    stats = count_parameters(model)
    print(f"  Parameters: {stats['total_M']}M total | {stats['trainable_M']}M trainable")
    
    # 3. Optimizer, Scheduler, Criterion
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg["training"].get("mixed_precision", True) and device.type == "cuda")
    
    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        targets = []
        if isinstance(train_loader.dataset, Subset):
            ds = train_loader.dataset.dataset
            for idx in train_loader.dataset.indices:
                targets.append(ds.samples[idx][1])
        else:
            for _, label in train_loader.dataset.samples:
                targets.append(label)
        counts = np.bincount(targets, minlength=cfg["data"]["num_classes"])
        class_weights = 1.0 / (counts + 1e-6)
        class_weights = class_weights / np.sum(class_weights) * cfg["data"]["num_classes"]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"  [LOSS] Class weights: {class_weights.cpu().numpy().round(3)}")
        
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["training"].get("label_smoothing", 0.1)
    )
    
    # 4. Checkpoint Manager
    suffix = "_scratch" if scratch else ""
    checkpoint_dir = f"{cfg['outputs']['checkpoint_dir']}{suffix}"
    ckpt_mgr = CheckpointManager(checkpoint_dir, top_k=1)
    
    # 5. Training Loop
    best_val_acc = 0.0
    epochs = cfg["training"]["epochs"]
    print(f"\n[TRAIN] Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        val_metrics, preds, labels = evaluate(model, val_loader, criterion, device, cfg)
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        val_acc = val_metrics["accuracy"]
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc:.4f} F1: {val_metrics['macro_f1']:.4f} | "
              f"Time: {epoch_time:.1f}s")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_mgr.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "config": cfg,
            }, val_acc, epoch)
            
    # 6. Final Evaluation
    best_ckpt = ckpt_mgr.best_checkpoint()
    print(f"\n[EVAL] Loading best checkpoint for final evaluation: {best_ckpt}")
    load_checkpoint(best_ckpt, model)
    
    val_metrics, final_preds, final_labels = evaluate(model, val_loader, criterion, device, cfg)
    print(f"\n{'='*60}")
    print(f" Final Evaluation Report ({mode_str}): {cfg['project']['name']}")
    print(f"{'='*60}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {val_metrics['macro_f1']:.4f}")
    print()
    print_classification_report(final_preds, final_labels, cfg["data"]["classes"])
    
    # Save Confusion Matrix
    cm_filename = f"confusion_matrix{suffix}.png"
    cm_path = Path(cfg["outputs"]["gradcam_dir"]) / cm_filename
    plot_confusion_matrix(final_preds, final_labels, cfg["data"]["classes"], cm_path)
    
    # Save results JSON
    results = {
        "dataset": cfg["project"]["name"],
        "mode": mode_str,
        "num_classes": cfg["data"]["num_classes"],
        "final_accuracy": val_metrics["accuracy"],
        "final_macro_f1": val_metrics["macro_f1"],
        "class_metrics": val_metrics
    }
    
    log_dir = f"{cfg['outputs']['log_dir']}{suffix}"
    results_path = Path(log_dir) / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MedLite-CRC on a downstream dataset")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch (no pretrained weights)")
    args = args = parser.parse_args()
    main(args.config, args.scratch)
