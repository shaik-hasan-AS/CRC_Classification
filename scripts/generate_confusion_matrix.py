"""
scripts/generate_confusion_matrix.py

Generates a publication-ready confusion matrix for MedLite-CRC 
evaluated on the cross-patient CRC-VAL-HE-7K dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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

def plot_publication_matrix(preds, labels, class_names, save_path):
    # Calculate normalized confusion matrix
    cm = confusion_matrix(labels, preds, normalize="true")
    
    # Set up the matplotlib figure (high DPI for publication)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Modern styling
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.5)
    
    # Custom color map
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Draw heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax, 
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    
    # Set labels and title with professional fonts
    ax.set_xlabel("Predicted Class", fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel("True Class", fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title("MedLite-CRC Cross-Patient Generalization (CRC-VAL-HE-7K)", 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n[SAVED] Publication-ready confusion matrix -> {save_path}")

def plot_per_class_metrics(preds, labels, class_names, save_path):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, labels=range(len(class_names)))
    
    # Plotting setup
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor('#16213e')
    ax.set_facecolor('#1a1a2e')
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Draw bars
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.9, edgecolor='none')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#07d35f', alpha=0.9, edgecolor='none')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#ff4b5c', alpha=0.9, edgecolor='none')
    
    ax.set_ylabel('Score', color='white', fontsize=12, fontweight='bold')
    ax.set_title('MedLite-CRC Per-Class Classification Metrics (CRC-VAL-HE-7K)', 
                 color='white', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, color='white', fontsize=10, rotation=45)
    ax.tick_params(colors='#aaaaaa')
    ax.set_ylim(0, 1.05)
    
    # Grid and spines styling
    ax.grid(True, linestyle='--', alpha=0.1, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='lower left', facecolor='#2a2a4a', edgecolor='#555', labelcolor='white')
    
    # Annotate low-performing bars to highlight where the model struggles (e.g. STR, MUS)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height < 0.90:  # Show exact score for hard classes
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color='#e0e0e0', fontsize=8)
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] Per-class metrics bar chart -> {save_path}")

def main(args):
    cfg = load_config(args.config)
    # Force CPU to avoid CUDA OOM conflicts with training
    device = torch.device("cpu")
    print(f"Using device: {device} (Forced CPU to avoid OOM)")
    
    print("Loading CRC-VAL-HE-7K Cross-Patient Dataset...")
    loader = get_crossval_loader(cfg)
    if loader is None:
        print("Error: Could not load dataset.")
        return

    print(f"Loading MedLite-CRC from {args.checkpoint}")
    # Explicitly set final architecture flags — use_se_block MUST be False (Ablation 3 = final model)
    cfg['model'] = {
        'name': 'MedLiteCRC',
        'base_channels': 32,
        'attention_reduction': 16,
        'dropout': 0.4,
        'use_stain_norm': True,
        'use_multiscale': True,
        'use_se_block': False,   # Final architecture — SEBlock permanently removed (see ablation §9.3)
    }
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    
    print("Running inference to generate predictions...")
    preds, labels = get_predictions(model, loader, device)
    
    acc = (preds == labels).mean() * 100
    print(f"Accuracy on 7K Dataset: {acc:.2f}%")
    
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path_cm = out_dir / "cm_publication_ready.png"
    save_path_bar = out_dir / "per_class_metrics_bar.png"
    
    print("Generating High-Resolution Plots...")
    plot_publication_matrix(preds, labels, CRC_CLASSES, save_path_cm)
    plot_per_class_metrics(preds, labels, CRC_CLASSES, save_path_bar)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint for MedLite-CRC")
    args = parser.parse_args()
    main(args)
