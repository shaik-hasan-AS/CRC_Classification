# MedLite-CRC
## Lightweight Stain-Robust CNN for Colon Histopathology Classification

---

### Project Structure
```
medlite_crc/
├── configs/
│   └── config.yaml          # All hyperparameters and paths
├── data/
│   ├── transforms.py        # Augmentation + stain augmentation
│   └── dataset.py           # DataLoader factories
├── models/
│   └── medlite_crc.py       # MedLite-CRC architecture
├── utils/
│   ├── metrics.py           # Accuracy, F1, checkpointing, early stopping
│   └── gradcam.py           # GradCAM visualisation
├── scripts/
│   └── download_data.py     # Auto-download NCT-CRC + CRC-VAL datasets
├── outputs/
│   ├── checkpoints/         # Saved model weights
│   ├── logs/                # Eval results JSON
│   └── gradcam/             # GradCAM output images
├── train.py                 # Main training script
├── evaluate.py              # Cross-dataset evaluation
└── requirements.txt
```

---

### Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (~8GB total)
python scripts/download_data.py

# 3. Set your wandb username in configs/config.yaml (wandb.entity)

# 4. Train
python train.py

# 5. Evaluate cross-dataset
python evaluate.py --checkpoint outputs/checkpoints/<best_ckpt>.pt

# 6. GradCAM on a single image
python utils/gradcam.py --checkpoint outputs/checkpoints/<best_ckpt>.pt --img path/to/image.tif
```

---

### Datasets
| Dataset | Size | Purpose | Link |
|---|---|---|---|
| NCT-CRC-HE-100K | 100K patches | Train / Val | https://zenodo.org/records/1214456 |
| CRC-VAL-HE-7K | 7K patches | Cross-patient test | Same Zenodo record |

---

### Architecture: MedLite-CRC
```
Input (224×224×3)
  → LearnableStainNorm
  → Stem (Conv + DW Conv)
  → MultiScaleBranch (3×3 + 5×5 + 7×7 parallel DW-Sep branches)
  → MaxPool
  → DWResBlock × 3 (128ch → 256ch → 256ch)
  → SE Channel Attention
  → GlobalAvgPool
  → Classifier (FC → Dropout → FC(9))
```
Target: <5M params | <1 GFLOPs | ≥98% on NCT-CRC | ≥93% on CRC-VAL
