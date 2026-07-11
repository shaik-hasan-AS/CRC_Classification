"""
evaluate.py
Run full evaluation on CRC-VAL-HE-7K (cross-patient test set).
Generates confusion matrix, per-class F1, and efficiency report.

Run: python evaluate.py --checkpoint outputs/checkpoints/ckpt_best.pt
"""

import argparse
import time
import json
from pathlib import Path

import yaml
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix

from data.dataset import get_crossval_loader, get_nonorm_loader, CRC_CLASSES, HYBRID_CLASSES
from models.medlite_crc import build_model, count_parameters
from utils.metrics import compute_metrics, print_classification_report, load_checkpoint


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(model, loader, device, split_name, class_names, use_tta=False):
    model.eval()
    all_preds, all_labels = [], []
    latencies = []
    use_amp = device.type == "cuda"

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        t0 = time.perf_counter()
        with autocast(enabled=use_amp):
            if use_tta:
                probs = torch.softmax(model(imgs), dim=1)
                probs += torch.softmax(model(torch.rot90(imgs, 1, [2, 3]).contiguous()), dim=1)
                probs += torch.softmax(model(torch.rot90(imgs, 2, [2, 3]).contiguous()), dim=1)
                probs += torch.softmax(model(torch.rot90(imgs, 3, [2, 3]).contiguous()), dim=1)
                probs /= 4.0
                logits = probs  # argmax will work identically on probabilities
            else:
                logits = model(imgs)
        latencies.append((time.perf_counter() - t0) * 1000 / imgs.size(0))  # ms/img
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_preds, all_labels, class_names)
    metrics["avg_latency_ms"] = round(float(np.mean(latencies)), 3)

    print(f"\n{'='*60}")
    print(f" Evaluation: {split_name}")
    print(f"{'='*60}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1   : {metrics['macro_f1']:.4f}")
    print(f"  Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"  Latency    : {metrics['avg_latency_ms']:.2f} ms/image")
    print()
    print_classification_report(all_preds, all_labels, class_names)

    return metrics, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, class_names, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
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
    print(f"  [SAVED] Confusion matrix → {save_path}")


def efficiency_report(model, device):
    """Report parameters, model size, and FLOPs."""
    stats = count_parameters(model)

    # Model size on disk
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / 1e6
        os.unlink(f.name)

    print(f"\n{'='*60}")
    print(" Efficiency Report")
    print(f"{'='*60}")
    print(f"  Total parameters : {stats['total_M']}M")
    print(f"  Trainable params : {stats['trainable_M']}M")
    print(f"  Model size (disk): {size_mb:.2f} MB")

    try:
        from thop import profile
        dummy = torch.randn(1, 3, 224, 224).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        print(f"  GFLOPs           : {macs / 1e9:.4f}")
    except ImportError:
        print("  GFLOPs           : install thop → pip install thop")

    return {"params_M": stats["total_M"], "size_mb": round(size_mb, 2)}


def main(args):
    cfg    = load_config(args.config)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Build model
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    # Efficiency report
    eff_stats = efficiency_report(model, device)

    results = {"checkpoint": args.checkpoint, "efficiency": eff_stats, "splits": {}}
    all_metrics = results["splits"]
    target_classes = HYBRID_CLASSES if cfg["data"].get("is_hybrid", False) else CRC_CLASSES

    # 1. CRC-VAL-HE-7K
    cross_val_loader = get_crossval_loader(cfg)
    if cross_val_loader:
        metrics, preds, labels = run_eval(
            model, cross_val_loader, device, "CRC-VAL-HE-7K (cross-patient)", target_classes, use_tta=args.tta
        )
        all_metrics["crc_val_7k"] = metrics
        plot_confusion_matrix(
            preds, labels, target_classes,
            save_path=Path(cfg["outputs"].get("gradcam_dir", "outputs/eval")) / "cm_crc_val_7k.png"
        )

    # 2. NCT-CRC-HE-100K-NONORM
    nonorm_loader = get_nonorm_loader(cfg)
    if nonorm_loader:
        metrics, preds, labels = run_eval(
            model, nonorm_loader, device, "NCT-CRC-HE-100K-NONORM (cross-stain)", target_classes, use_tta=args.tta
        )
        all_metrics["nonorm"] = metrics
        plot_confusion_matrix(
            preds, labels, target_classes,
            save_path=Path(cfg["outputs"].get("gradcam_dir", "outputs/eval")) / "cm_nonorm.png"
        )


    # Save results JSON
    out_path = Path(cfg["outputs"]["log_dir"]) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Evaluation results → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedLite-CRC")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--cpu",        action="store_true",
                        help="Force CPU evaluation for latency benchmarking")
    parser.add_argument("--tta",        action="store_true",
                        help="Use Test-Time Augmentation (4 rotations) to improve accuracy")
    args = parser.parse_args()
    main(args)
