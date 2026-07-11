"""
scripts/visualize_embeddings.py

t-SNE / UMAP visualization of MedLite-CRC learned feature embeddings.

Extracts 256-dim GAP features from:
  - NCT-CRC-HE-100K validation split (Heidelberg scanner, in-distribution)
  - CRC-VAL-HE-7K test set (Mannheim scanner, out-of-distribution)

Produces two plots:
  1. assets/tsne_class_separation.png  — points colored by tissue class (9 colors)
  2. assets/tsne_scanner_origin.png    — same points colored by scanner origin

If class clusters are tight AND scanner colors are MIXED within each cluster,
the model has learned scanner-invariant morphological features.

Usage:
    python scripts/visualize_embeddings.py \\
        --checkpoint outputs/checkpoints_ablation_multiscale/ckpt_epoch195_acc0.9946.pt \\
        --n_samples 800 \\
        --output_dir assets/
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import MedLiteCRC
from data.dataset import CRC_CLASSES


# ── Class colours (consistent with paper figures) ─────────────────────────────

CLASS_COLORS = {
    'ADI':  '#F4A460',   # sandy brown
    'BACK': '#708090',   # slate grey
    'DEB':  '#8B4513',   # saddle brown
    'LYM':  '#1E90FF',   # dodger blue
    'MUC':  '#9370DB',   # medium purple
    'MUS':  '#FF8C00',   # dark orange
    'NORM': '#32CD32',   # lime green
    'STR':  '#DC143C',   # crimson
    'TUM':  '#FF1493',   # deep pink
}

SCANNER_COLORS = {
    'Heidelberg (NCT-100K)': '#2196F3',   # blue
    'Mannheim (VAL-7K)':     '#F44336',   # red
}


# ── Lightweight ImageFolder-like dataset ───────────────────────────────────────

class FolderDataset(torch.utils.data.Dataset):
    """Read class subfolders from a root directory."""

    def __init__(self, root, class_list, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}

        for cls in class_list:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = [e for e in os.scandir(cls_dir) if not e.is_dir()]
            if max_per_class:
                random.shuffle(files)
                files = files[:max_per_class]
            for e in files:
                self.samples.append((e.path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'rb') as f:
            import io
            img = Image.open(io.BytesIO(f.read())).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Feature extractor hook ─────────────────────────────────────────────────────

def extract_features(model, loader, device):
    """Run forward pass and capture 256-d GAP features (before classifier)."""
    model.eval()
    feats, labels = [], []

    # Hook the GAP output
    gap_out = []
    def hook(module, inp, out):
        gap_out.append(out.detach().cpu())

    handle = model.pool2.register_forward_hook(hook)

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            _ = model(imgs)
            feats.append(gap_out[-1])
            labels.extend(lbls.numpy())
            gap_out.clear()

    handle.remove()
    return torch.cat(feats, 0).numpy(), np.array(labels)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_tsne(embedding, labels, color_map, title, legend_labels, save_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#16213e')

    colors_arr = np.array([color_map[legend_labels[l]] for l in labels])

    for label_name, color in color_map.items():
        mask = np.array([legend_labels[l] == label_name for l in labels])
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=14, alpha=0.75, linewidths=0,
                   label=label_name)

    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    legend = ax.legend(
        loc='upper right', fontsize=9, framealpha=0.2,
        facecolor='#2a2a4a', edgecolor='#555',
        labelcolor='white', markerscale=1.8,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[MODEL] Loading checkpoint: {args.checkpoint}")
    model = MedLiteCRC(
        num_classes=9, base_channels=32, dropout=0.4,
        use_stain_norm=True, use_multiscale=True, use_se_block=False,
        stain_norm_space='rgb',
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    print(f"  Loaded successfully.")

    # ── Transforms (same as val — no augmentation) ────────────────────────────
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.7406, 0.5331, 0.7059],
                    std =[0.1651, 0.2174, 0.1574]),
    ])

    # ── Load datasets ─────────────────────────────────────────────────────────
    n_per_cls = args.n_samples // 9

    print(f"\n[DATA] Loading NCT-100K val split ({n_per_cls}/class) ...")
    ds_nct = FolderDataset(
        args.nct_val_dir, CRC_CLASSES, transform, max_per_class=n_per_cls
    )
    print(f"  NCT-100K: {len(ds_nct)} samples")

    print(f"[DATA] Loading CRC-VAL-7K ({n_per_cls}/class) ...")
    ds_val = FolderDataset(
        args.val7k_dir, CRC_CLASSES, transform, max_per_class=n_per_cls
    )
    print(f"  CRC-VAL-7K: {len(ds_val)} samples")

    loader_nct = DataLoader(ds_nct, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n[FEAT] Extracting features from NCT-100K val ...")
    feats_nct, labels_nct = extract_features(model, loader_nct, device)
    print(f"  Shape: {feats_nct.shape}")

    print("[FEAT] Extracting features from CRC-VAL-7K ...")
    feats_val, labels_val = extract_features(model, loader_val, device)
    print(f"  Shape: {feats_val.shape}")

    # ── Combine ───────────────────────────────────────────────────────────────
    all_feats   = np.concatenate([feats_nct, feats_val], axis=0)
    all_labels  = np.concatenate([labels_nct, labels_val], axis=0)
    scanner_ids = np.array(
        [0] * len(feats_nct) + [1] * len(feats_val)
    )

    print(f"\n[t-SNE] Running on {len(all_feats)} samples (this may take ~2 min) ...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=42, n_jobs=-1)
    embedding = tsne.fit_transform(all_feats)
    print(f"  Done. Embedding shape: {embedding.shape}")

    # ── Plot 1: color by class ─────────────────────────────────────────────────
    print("\n[PLOT] Plot 1: class separation ...")
    idx_to_class = {i: c for i, c in enumerate(CRC_CLASSES)}
    plot_tsne(
        embedding, all_labels,
        color_map=CLASS_COLORS,
        title="MedLite-CRC Feature Embeddings — Colored by Tissue Class\n"
              "(NCT-100K Heidelberg + CRC-VAL-7K Mannheim combined)",
        legend_labels=idx_to_class,
        save_path=os.path.join(args.output_dir, 'tsne_class_separation.png'),
    )

    # ── Plot 2: color by scanner origin ───────────────────────────────────────
    print("[PLOT] Plot 2: scanner origin ...")
    scanner_legend = {0: 'Heidelberg (NCT-100K)', 1: 'Mannheim (VAL-7K)'}
    plot_tsne(
        embedding, scanner_ids,
        color_map=SCANNER_COLORS,
        title="MedLite-CRC Feature Embeddings — Colored by Scanner Origin\n"
              "(Mixed scanner colors within class clusters = scanner-invariant learning)",
        legend_labels=scanner_legend,
        save_path=os.path.join(args.output_dir, 'tsne_scanner_origin.png'),
    )

    print("\n✓ All done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t-SNE feature embedding visualization')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to MedLite-CRC checkpoint (.pt)')
    parser.add_argument('--nct_val_dir', default='data/NCT-CRC-HE-100K',
                        help='NCT-100K val directory (used as in-distribution source)')
    parser.add_argument('--val7k_dir', default='data/CRC-VAL-HE-7K',
                        help='CRC-VAL-HE-7K directory (OOD Mannheim scanner)')
    parser.add_argument('--n_samples', type=int, default=810,
                        help='Total samples (split equally across 9 classes per dataset)')
    parser.add_argument('--output_dir', default='assets/',
                        help='Directory to save plots')
    args = parser.parse_args()
    main(args)
