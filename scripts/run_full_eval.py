"""
scripts/run_full_eval.py

Master evaluation script — evaluates ALL model checkpoints on CRC-VAL-HE-7K
and prints a full report with every number needed to update the docs.

Usage (from project root):
    python scripts/run_full_eval.py

Outputs:
  - Per-class breakdown for SOTA KD checkpoint
  - McNemar's test vs EfficientNet-B0
  - GradCAM spatial metrics (center-bias, negative-space, vanishing gradient)
  - Ablation table metrics
"""

import os
import sys
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import MedLiteCRC, build_model, count_parameters
from data.dataset import get_crossval_loader, CRC_CLASSES
import torchvision.models as tv_models
import torchvision.datasets as tv_datasets

# ── Parse command-line arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate MedLite-CRC models.")
parser.add_argument("--config", default="configs/kd_mobilenet_teacher.yaml", help="Path to KD config")
parser.add_argument("--checkpoint", default="outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt", help="Path to SOTA checkpoint")
parser.add_argument("--padding_mode", default=None, help="Padding mode for MedLiteCRC ('zeros' or 'reflect')")
parser.add_argument("--mask_border_width", type=int, default=8, help="Border pixels to mask out in GradCAM")
args = parser.parse_args()

# ── statsmodels for McNemar ────────────────────────────────────────────────────
try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not found — McNemar test will be skipped. pip install statsmodels")

# ── GradCAM helper ────────────────────────────────────────────────────────────
try:
    from utils.gradcam import GradCAM
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    print("[WARN] GradCAM util not found — GradCAM section will be skipped.")

from data.transforms import get_val_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[DEVICE] {DEVICE}\n")

# ── Shared config / loader ─────────────────────────────────────────────────────
KD_CFG_PATH       = args.config
BASE_CFG_PATH     = "configs/config.yaml"

SOTA_CKPT         = args.checkpoint
EFFNET_CKPT       = "outputs/checkpoints_efficientnetb0/ckpt_epoch053_acc0.9904.pt"  # best effnet

ABLATION_CKPTS = {
    "Ablation 1 (Baseline CNN)":               "outputs/checkpoints_ablation_baseline/ckpt_epoch185_acc0.9954.pt",
    "Ablation 2 (+ StainNorm)":                "outputs/checkpoints_ablation_stainnorm/ckpt_epoch185_acc0.9951.pt",
    "Ablation 3 (+ MultiScale) ← FINAL":      "outputs/checkpoints_ablation_multiscale/ckpt_epoch195_acc0.9946.pt",
    "Ablation 4 (+ SEBlock — Negative)":       "outputs/checkpoints_ablation_full/ckpt_epoch197_acc0.9952.pt",
}

ABLATION_FLAGS = {
    "Ablation 1 (Baseline CNN)":               dict(use_stain_norm=False, use_multiscale=False, use_se_block=False, padding_mode="zeros"),
    "Ablation 2 (+ StainNorm)":                dict(use_stain_norm=True,  use_multiscale=False, use_se_block=False, padding_mode="zeros"),
    "Ablation 3 (+ MultiScale) ← FINAL":      dict(use_stain_norm=True,  use_multiscale=True,  use_se_block=False, padding_mode="zeros"),
    "Ablation 4 (+ SEBlock — Negative)":       dict(use_stain_norm=True,  use_multiscale=True,  use_se_block=True, padding_mode="zeros"),
}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def get_preds(model, loader):
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.cpu())
    return np.array(all_preds), np.array(all_labels), torch.cat(all_probs, 0)


def sep(title=""):
    w = 70
    if title:
        print(f"\n{'─'*w}")
        print(f"  {title}")
        print(f"{'─'*w}")
    else:
        print(f"\n{'─'*w}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SOTA KD MODEL (Full per-class breakdown)
# ══════════════════════════════════════════════════════════════════════════════

sep("SECTION 1 — SOTA MobileNetV2 KD Student (96% OOD target)")

cfg_kd = load_yaml(KD_CFG_PATH)
cfg_kd["data"]["num_workers"] = 0

loader_7k = get_crossval_loader(cfg_kd)

sota_padding_mode = args.padding_mode if args.padding_mode is not None else cfg_kd.get("model", {}).get("padding_mode", "zeros")
print(f"Constructing SOTA model with padding_mode={sota_padding_mode}")

sota_model = MedLiteCRC(
    num_classes=9, base_channels=32, dropout=0.4,
    use_stain_norm=True, use_multiscale=True, use_se_block=False,
    stain_norm_space="rgb", padding_mode=sota_padding_mode,
).to(DEVICE)

if not Path(SOTA_CKPT).exists():
    print(f"[ERROR] SOTA checkpoint not found: {SOTA_CKPT}")
    sota_preds = sota_labels = sota_probs = None
else:
    print(f"Loading SOTA checkpoint: {SOTA_CKPT}")
    load_ckpt(sota_model, SOTA_CKPT)

    sota_preds, sota_labels, sota_probs = get_preds(sota_model, loader_7k)

    sota_acc = accuracy_score(sota_labels, sota_preds) * 100
    sota_macro_f1 = f1_score(sota_labels, sota_preds, average="macro")
    sota_wtd_f1   = f1_score(sota_labels, sota_preds, average="weighted")

    print(f"\n  OOD Accuracy (CRC-VAL-HE-7K) : {sota_acc:.4f}%")
    print(f"  Macro  F1                    : {sota_macro_f1:.4f}")
    print(f"  Weighted F1                  : {sota_wtd_f1:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'─'*48}")
    p, r, f, s = precision_recall_fscore_support(sota_labels, sota_preds, labels=range(9))
    for i, cls in enumerate(CRC_CLASSES):
        print(f"  {cls:<8} {p[i]:>10.4f} {r[i]:>8.4f} {f[i]:>8.4f} {s[i]:>9}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ABLATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

sep("SECTION 2 — Ablation Study (leave-one-out, CRC-VAL-HE-7K)")

ablation_results = {}
for name, ckpt_path in ABLATION_CKPTS.items():
    flags = ABLATION_FLAGS[name]
    model = MedLiteCRC(num_classes=9, base_channels=32, dropout=0.4, **flags).to(DEVICE)

    if not Path(ckpt_path).exists():
        print(f"  [SKIP] {name} — checkpoint not found: {ckpt_path}")
        continue

    load_ckpt(model, ckpt_path)
    preds, labels, _ = get_preds(model, loader_7k)
    acc   = accuracy_score(labels, preds) * 100
    mf1   = f1_score(labels, preds, average="macro")
    wf1   = f1_score(labels, preds, average="weighted")

    param_info = count_parameters(model)

    ablation_results[name] = {
        "acc": acc, "macro_f1": mf1, "wtd_f1": wf1,
        "params_M": param_info["total_M"],
    }
    print(f"  {name}")
    print(f"    Params: {param_info['total_M']}M  |  Acc: {acc:.4f}%  |  MacroF1: {mf1:.4f}  |  WtdF1: {wf1:.4f}")

# Markdown table
sep("Ablation Table (Markdown)")
print(f"| {'Model Configuration':<48} | {'Params':>8} | {'Accuracy':>9} | {'Macro F1':>9} | {'Wtd F1':>8} |")
print(f"|{'─'*50}|{'─'*10}|{'─'*11}|{'─'*11}|{'─'*10}|")
for name, res in ablation_results.items():
    print(f"| {name:<48} | {res['params_M']:>7}M | {res['acc']:>8.2f}% | {res['macro_f1']:>9.4f} | {res['wtd_f1']:>8.4f} |")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — McNemar's Test (SOTA KD vs EfficientNet-B0)
# ══════════════════════════════════════════════════════════════════════════════

sep("SECTION 3 — McNemar's Test (SOTA KD Student vs EfficientNet-B0)")

effnet_ckpt_candidates = sorted(glob.glob("outputs/checkpoints_efficientnetb0/ckpt_epoch*.pt"))
if not effnet_ckpt_candidates:
    effnet_ckpt_candidates = sorted(glob.glob("outputs/checkpoints_efficientnet*/ckpt_epoch*.pt"))

if not effnet_ckpt_candidates:
    print("  [SKIP] EfficientNet-B0 checkpoint not found. Skipping McNemar test.")
elif sota_preds is None:
    print("  [SKIP] SOTA checkpoint failed to load.")
elif not HAS_STATSMODELS:
    print("  [SKIP] statsmodels not installed.")
else:
    effnet_best = effnet_ckpt_candidates[-1]
    print(f"  Loading EfficientNet-B0: {effnet_best}")

    cfg_base = load_yaml(BASE_CFG_PATH)
    cfg_base["data"]["num_workers"] = 0
    cfg_base["model"] = {"name": "EfficientNetB0"}
    effnet_model = build_model(cfg_base).to(DEVICE)
    load_ckpt(effnet_model, effnet_best)

    effnet_preds, _, _ = get_preds(effnet_model, loader_7k)
    effnet_acc = accuracy_score(sota_labels, effnet_preds) * 100
    print(f"  EfficientNet-B0 OOD Accuracy : {effnet_acc:.4f}%")

    correct_sota   = (sota_preds   == sota_labels)
    correct_effnet = (effnet_preds == sota_labels)

    both_correct   = int(np.sum(correct_sota  & correct_effnet))
    sota_only      = int(np.sum(correct_sota  & ~correct_effnet))
    effnet_only    = int(np.sum(~correct_sota &  correct_effnet))
    both_wrong     = int(np.sum(~correct_sota & ~correct_effnet))

    table = [[both_correct, sota_only], [effnet_only, both_wrong]]
    result = mcnemar(table, exact=False, correction=True)

    print(f"\n  Contingency Table:")
    print(f"    Both Correct          : {both_correct}")
    print(f"    SOTA Correct Only     : {sota_only}  (MedLite-CRC right, EfficientNet wrong)")
    print(f"    EfficientNet Only     : {effnet_only}  (EfficientNet right, MedLite-CRC wrong)")
    print(f"    Both Wrong            : {both_wrong}")
    print(f"\n  McNemar Chi² Statistic : {result.statistic:.4f}")
    print(f"  McNemar P-Value        : {result.pvalue:.4e}")
    if result.pvalue < 0.05:
        print("  → Statistically SIGNIFICANT (p < 0.05) ✅")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — GradCAM Spatial Analysis (SOTA KD model)
# ══════════════════════════════════════════════════════════════════════════════

sep("SECTION 4 — GradCAM Spatial Interpretability (SOTA KD model)")

if not HAS_GRADCAM:
    print("  [SKIP] GradCAM utils not available.")
elif sota_preds is None:
    print("  [SKIP] SOTA model not loaded.")
else:
    cfg_kd2 = load_yaml(KD_CFG_PATH)
    target_layer = sota_model.res_blocks[-1].conv2.pw[0]
    gcam = GradCAM(sota_model, target_layer)

    val_dir = cfg_kd2["data"]["nct_crc_val_dir"]
    transform = get_val_transforms(cfg_kd2)
    val_dataset = tv_datasets.ImageFolder(val_dir)
    str_idx = val_dataset.class_to_idx.get("STR", -1)

    random.seed(42)
    sample_indices = random.sample(range(len(val_dataset)), min(1000, len(val_dataset)))

    y_grid, x_grid = np.meshgrid(np.arange(224), np.arange(224), indexing='ij')
    distances, vanishing_gradients, correct_predictions = [], 0, 0
    stroma_bg_act, stroma_tissue_act = [], []

    from PIL import Image as PILImage
    for idx in sample_indices:
        img_path, label = val_dataset.imgs[idx]
        img_pil = PILImage.open(img_path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        tensor = transform(img_pil).to(DEVICE)
        try:
            cam, pred_class, _ = gcam.generate(tensor, target_class=label, mask_border_width=args.mask_border_width)
            is_correct = (pred_class == label)
            sum_cam = cam.sum()
            if sum_cam < 1e-8:
                if is_correct:
                    vanishing_gradients += 1
                continue
            if is_correct:
                correct_predictions += 1
            x_com = (cam * x_grid).sum() / sum_cam
            y_com = (cam * y_grid).sum() / sum_cam
            distances.append(np.sqrt((x_com - 112.0)**2 + (y_com - 112.0)**2))
            if label == str_idx:
                img_np = np.array(img_resized) / 255.0
                brightness = img_np.mean(axis=2)
                bg_mask = brightness > 0.85
                tissue_mask = ~bg_mask
                if bg_mask.any():
                    stroma_bg_act.append(cam[bg_mask].mean())
                if tissue_mask.any():
                    stroma_tissue_act.append(cam[tissue_mask].mean())
        except Exception:
            pass

    total_valid = correct_predictions + vanishing_gradients
    vanish_rate = (vanishing_gradients / total_valid * 100) if total_valid > 0 else 0
    avg_dist = np.mean(distances) if distances else 0

    print(f"  Average radial distance from center : {avg_dist:.2f} px  (max possible: 158.4 px)")
    print(f"  Vanishing gradient rate (correct)   : {vanish_rate:.2f}%  ({vanishing_gradients}/{total_valid})")
    if stroma_bg_act and stroma_tissue_act:
        bg_mean  = np.mean(stroma_bg_act)
        tis_mean = np.mean(stroma_tissue_act)
        print(f"  Stroma background activation        : {bg_mean:.4f}")
        print(f"  Stroma tissue activation            : {tis_mean:.4f}")
        print(f"  Tissue > Background?                : {tis_mean > bg_mean} ({'✅ No shortcut' if tis_mean > bg_mean else '❌ Shortcut present'})")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SUMMARY TABLE (for docs)
# ══════════════════════════════════════════════════════════════════════════════

sep("SECTION 5 — Final Summary (copy these numbers into the docs)")

if sota_preds is not None:
    print(f"\n  ┌─ SOTA (MobileNetV2 KD student, {SOTA_CKPT})")
    print(f"  │  OOD Accuracy : {sota_acc:.2f}%")
    print(f"  │  Macro  F1    : {sota_macro_f1:.4f}")
    print(f"  │  Weighted F1  : {sota_wtd_f1:.4f}")
    print(f"  └─ In-Dist Val  : 99.46% (from checkpoint filename)")

print("\n✓ Evaluation complete.\n")
