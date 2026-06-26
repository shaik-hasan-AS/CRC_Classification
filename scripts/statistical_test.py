"""
scripts/statistical_test.py

Runs McNemar's test to determine if the difference in predictions between
MedLite-CRC and a baseline (EfficientNetB0) is statistically significant.
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

# Try to import statsmodels, provide instructions if missing
try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    print("Please install statsmodels to run this script: pip install statsmodels")
    exit(1)

from data.dataset import get_crossval_loader, CRC_CLASSES
from models.medlite_crc import build_model
from utils.metrics import load_checkpoint

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    return np.array(all_preds), np.array(all_labels)

def main(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading CRC-VAL-HE-7K Cross-Patient Dataset...")
    loader = get_crossval_loader(cfg)
    if loader is None:
        print("Error: Could not load cross-validation dataset.")
        return

    # 2. Setup Models
    # Model A: MedLite-CRC
    print(f"Loading Model A: MedLite-CRC from {args.ckpt_a}")
    cfg_a = cfg.copy()
    cfg_a['model'] = {'name': 'MedLiteCRC', 'base_channels': 32, 'attention_reduction': 16, 'dropout': 0.4}
    model_a = build_model(cfg_a).to(device)
    load_checkpoint(args.ckpt_a, model_a)
    
    # Model B: Baseline (EfficientNetB0)
    print(f"Loading Model B: Baseline from {args.ckpt_b}")
    cfg_b = cfg.copy()
    cfg_b['model'] = {'name': 'EfficientNetB0'}
    model_b = build_model(cfg_b).to(device)
    load_checkpoint(args.ckpt_b, model_b)
    
    # 3. Get Predictions
    print("Generating predictions for Model A...")
    preds_a, labels = get_predictions(model_a, loader, device)
    
    print("Generating predictions for Model B...")
    preds_b, _ = get_predictions(model_b, loader, device)
    
    # 4. Calculate Accuracy
    acc_a = (preds_a == labels).mean() * 100
    acc_b = (preds_b == labels).mean() * 100
    print(f"\nModel A (MedLite-CRC) Accuracy: {acc_a:.2f}%")
    print(f"Model B (Baseline)    Accuracy: {acc_b:.2f}%")
    
    # 5. Prepare Contingency Table for McNemar's Test
    # b = Model A correct, Model B incorrect
    # c = Model A incorrect, Model B correct
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    
    both_correct = np.sum(correct_a & correct_b)
    a_corr_b_inc = np.sum(correct_a & ~correct_b)  # b
    a_inc_b_corr = np.sum(~correct_a & correct_b)  # c
    both_incorrect = np.sum(~correct_a & ~correct_b)
    
    table = [[both_correct, a_corr_b_inc],
             [a_inc_b_corr, both_incorrect]]
             
    print("\nContingency Table:")
    print(f"Both Correct: {both_correct} | Model A Correct / Model B Incorrect: {a_corr_b_inc}")
    print(f"Model A Incorrect / Model B Correct: {a_inc_b_corr} | Both Incorrect: {both_incorrect}")
    
    # 6. Run McNemar's Test
    # exact=False uses Chi-Square distribution (standard for large sample sizes > 25 discordant pairs)
    result = mcnemar(table, exact=False, correction=True)
    
    print("\n" + "="*50)
    print(" McNemar's Test Results (Statistical Significance)")
    print("="*50)
    print(f"Statistic (Chi-Squared): {result.statistic:.4f}")
    print(f"P-Value                : {result.pvalue:.4e}")
    
    if result.pvalue < 0.05:
        print("\nConclusion: The difference between the models is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("\nConclusion: The difference between the models is NOT statistically significant (p >= 0.05).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt_a", required=True, help="Checkpoint for MedLite-CRC")
    parser.add_argument("--ckpt_b", required=True, help="Checkpoint for Baseline Model")
    args = parser.parse_args()
    main(args)
