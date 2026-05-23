"""
utils/metrics.py
Training utilities: metrics, checkpoint management, early stopping, AMP scaler.
"""

import os
import json
import heapq
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)


# ── Metric Computation ────────────────────────────────────────────────────────

def compute_metrics(all_preds, all_labels, class_names=None):
    """
    Compute accuracy, macro-F1, per-class F1 from prediction lists.
    Returns a flat dict suitable for wandb logging.
    """
    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    w_f1     = f1_score(labels, preds, average="weighted", zero_division=0)
    per_cls  = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {
        "accuracy":   round(float(acc), 5),
        "macro_f1":   round(float(macro_f1), 5),
        "weighted_f1": round(float(w_f1), 5),
    }

    if class_names is not None:
        for i, cls in enumerate(class_names):
            if i < len(per_cls):
                metrics[f"f1_{cls}"] = round(float(per_cls[i]), 5)

    return metrics


def print_classification_report(all_preds, all_labels, class_names):
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names, digits=4))


# ── Checkpoint Manager ────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Saves top-K checkpoints by validation accuracy.
    Deletes older checkpoints automatically.
    """

    def __init__(self, save_dir: str, top_k: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k    = top_k
        self._heap    = []   # min-heap: (val_acc, filepath)

    def save(self, state: dict, val_acc: float, epoch: int) -> str:
        fname = self.save_dir / f"ckpt_epoch{epoch:03d}_acc{val_acc:.4f}.pt"
        torch.save(state, fname)

        heapq.heappush(self._heap, (val_acc, str(fname)))

        # Remove lowest if we exceed top_k
        if len(self._heap) > self.top_k:
            _, worst_path = heapq.heappop(self._heap)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"  [CKPT] Removed old checkpoint: {Path(worst_path).name}")

        print(f"  [CKPT] Saved: {fname.name}")
        return str(fname)

    def best_checkpoint(self) -> Optional[str]:
        if not self._heap:
            return None
        return max(self._heap, key=lambda x: x[0])[1]


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """Load a checkpoint and restore model/optimizer/scheduler state."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch    = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_val_acc", 0.0)
    print(f"[CKPT] Loaded from {path} (epoch {epoch}, best_acc={best_acc:.4f})")
    return epoch, best_acc


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val accuracy doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def step(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  [ES] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_acc
            self.counter    = 0
        return self.stop


# ── AverageMeter ──────────────────────────────────────────────────────────────

class AverageMeter:
    """Running average of a scalar value (loss, accuracy, etc.)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count
