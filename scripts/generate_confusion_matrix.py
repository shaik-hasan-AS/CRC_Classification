"""
scripts/generate_confusion_matrix.py

Generates a publication-ready confusion matrix for MedLite-CRC 
evaluated on the cross-patient CRC-VAL-HE-7K dataset.
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def main(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading CRC-VAL-HE-7K Cross-Patient Dataset...")
    loader = get_crossval_loader(cfg)
    if loader is None:
        print("Error: Could not load dataset.")
        return

    print(f"Loading MedLite-CRC from {args.checkpoint}")
    cfg['model'] = {'name': 'MedLiteCRC', 'base_channels': 32, 'attention_reduction': 16, 'dropout': 0.4}
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    
    print("Running inference to generate predictions...")
    preds, labels = get_predictions(model, loader, device)
    
    acc = (preds == labels).mean() * 100
    print(f"Accuracy on 7K Dataset: {acc:.2f}%")
    
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "cm_publication_ready.png"
    
    print("Generating High-Resolution Plot...")
    plot_publication_matrix(preds, labels, CRC_CLASSES, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint for MedLite-CRC")
    args = parser.parse_args()
    main(args)
