"""
scripts/calibration_analysis.py

Confidence Calibration & Expected Calibration Error (ECE) Analysis for MedLite-CRC.

This script:
  1. Loads a model checkpoint.
  2. Runs inference on CRC-VAL-HE-7K to extract logits, probabilities, and labels.
  3. Computes ECE (Expected Calibration Error) with 15 bins.
  4. Optimizes a single scalar Temperature (T) using Negative Log Likelihood (NLL)
     on the logits to calibrate the probabilities.
  5. Recalculates ECE after temperature scaling.
  6. Plots a comparative reliability diagram (Before vs After calibration).

SOTA checkpoint (MobileNetV2 KD student, 95.97% OOD):
    python scripts/calibration_analysis.py \
        --checkpoint outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt \
        --config configs/kd_mobilenet_teacher.yaml \
        --output_dir assets/
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import build_model
from data.dataset import get_crossval_loader, get_train_val_loaders


# ── Temperature Scaling Class ─────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Optimizes a single temperature parameter to calibrate the model confidence.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        """
        Learn temperature on validation logits to minimize Negative Log Likelihood.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        # Run optimization
        def eval_val():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_val)
        print(f"  ✓ Optimized temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


# ── Expected Calibration Error (ECE) ───────────────────────────────────────────

def compute_ece(probs, labels, n_bins=15):
    """
    Computes ECE using uniform binning of confidences.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    bin_accuracies = []
    bin_confidences = []
    bin_weights = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
            bin_weights.append(prop_in_bin.item())
            
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_weights.append(0.0)
            
    return ece.item(), bin_accuracies, bin_confidences, bin_weights


# ── Reliability Plotter ───────────────────────────────────────────────────────

def plot_reliability_diagram(accuracies_before, confidences_before, ece_before,
                             accuracies_after, confidences_after, ece_after,
                             temp, save_path, n_bins=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#16213e')
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    
    # ── Left Plot: Before Calibration
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    # Ideal calibration line
    ax.plot([0, 1], [0, 1], '--', color='#888888', label='Perfect Calibration')
    # Actual calibration bars
    ax.bar(bin_centers, accuracies_before, width=1.0/n_bins, edgecolor='#1a1a2e',
           color='#ff4b5c', alpha=0.8, label=f'Model (ECE: {ece_before*100:.2f}%)')
    # Gap highlight
    ax.bar(bin_centers, bin_centers - accuracies_before, bottom=accuracies_before,
           width=1.0/n_bins, color='#ff4b5c', alpha=0.3, edgecolor='#ff4b5c',
           hatch='//', label='Calibration Gap')
    
    ax.set_title("Uncalibrated MedLite-CRC", color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel("Confidence", color='#aaaaaa')
    ax.set_ylabel("Accuracy", color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    ax.spines['bottom'].set_edgecolor('#444444')
    ax.spines['left'].set_edgecolor('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', facecolor='#2a2a4a', edgecolor='#555', labelcolor='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Right Plot: After Calibration
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    ax.plot([0, 1], [0, 1], '--', color='#888888', label='Perfect Calibration')
    ax.bar(bin_centers, accuracies_after, width=1.0/n_bins, edgecolor='#1a1a2e',
           color='#07d35f', alpha=0.8, label=f'Calibrated (ECE: {ece_after*100:.2f}%)')
    ax.bar(bin_centers, bin_centers - accuracies_after, bottom=accuracies_after,
           width=1.0/n_bins, color='#07d35f', alpha=0.3, edgecolor='#07d35f',
           hatch='//', label='Calibration Gap')
    
    ax.set_title(f"Calibrated via Temp Scaling (T={temp:.3f})", color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel("Confidence", color='#aaaaaa')
    ax.set_ylabel("Accuracy", color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    ax.spines['bottom'].set_edgecolor('#444444')
    ax.spines['left'].set_edgecolor('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', facecolor='#2a2a4a', edgecolor='#555', labelcolor='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.suptitle("Reliability Diagrams for Colorectal Cancer Classification\n"
                 "(Evaluated on cross-patient CRC-VAL-HE-7K)", color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved Diagram → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")

    # Load Config
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["num_workers"] = 0

    # ── Load Model ────────────────────────────────────────────────────────────
    print(f"\n[MODEL] Building and loading checkpoint: {args.checkpoint}")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ── Load loaders ──────────────────────────────────────────────────────────
    # We use NCT-100K validation split to find/optimize the Temperature (T)
    # then evaluate ECE improvements out-of-distribution on CRC-VAL-HE-7K.
    print("\n[DATA] Loading NCT-100K val loader (for calibration optimization)...")
    _, val_loader = get_train_val_loaders(cfg)
    
    print("[DATA] Loading CRC-VAL-HE-7K test loader (for cross-patient validation)...")
    test_loader = get_crossval_loader(cfg)

    # ── Collect Logits (Validation set) ───────────────────────────────────────
    print("\n[INFERENCE] Extracting calibration set logits (NCT-100K Val)...")
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(labels)
    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    # ── Collect Logits (Cross-Val Test set) ───────────────────────────────────
    print("[INFERENCE] Extracting test set logits (CRC-VAL-HE-7K)...")
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            test_logits_list.append(logits.cpu())
            test_labels_list.append(labels)
    test_logits = torch.cat(test_logits_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    # ── Optimize Temperature ──────────────────────────────────────────────────
    print("\n[CALIBRATION] Optimizing Temperature parameter...")
    scaler = TemperatureScaler()
    temp_scalar = scaler.calibrate(val_logits, val_labels)

    # ── Compute ECE Before & After Calibration on Test Set ────────────────────
    probs_before = F.softmax(test_logits, dim=1)
    ece_before, accs_before, confs_before, _ = compute_ece(probs_before, test_labels)

    calibrated_test_logits = scaler(test_logits)
    probs_after = F.softmax(calibrated_test_logits, dim=1)
    ece_after, accs_after, confs_after, _ = compute_ece(probs_after, test_labels)

    print(f"\n[RESULTS] Evaluation on OOD CRC-VAL-HE-7K:")
    print(f"  - ECE Before: {ece_before * 100:.4f}%")
    print(f"  - ECE After : {ece_after * 100:.4f}%")
    print(f"  - Absolute calibration error reduction: {(ece_before - ece_after) * 100:.4f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n[PLOT] Plotting reliability diagrams...")
    os.makedirs(args.output_dir, exist_ok=True)
    plot_reliability_diagram(
        accs_before, confs_before, ece_before,
        accs_after, confs_after, ece_after,
        temp_scalar,
        save_path=os.path.join(args.output_dir, 'calibration_diagram.png')
    )
    print("✓ All done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibration & ECE Analysis')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--config', default='configs/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--output_dir', default='assets/',
                        help='Directory to save reliability diagram')
    args = parser.parse_args()
    main(args)
