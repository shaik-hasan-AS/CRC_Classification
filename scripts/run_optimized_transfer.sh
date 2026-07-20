#!/bin/bash
set -e
echo "Starting Optimized Transfer Learning on 3 Datasets"
PYTHON=.venv/bin/python

SOTA_CKPT="outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"

echo "1. EBHI-SEG Fine-tuning (40 epochs + Mixup 0.2)"
$PYTHON train.py --config configs/ebhi_finetune.yaml --finetune $SOTA_CKPT

echo "2. CRC-HGD-v1 Fine-tuning (40 epochs + Mixup 0.2)"
$PYTHON train.py --config configs/hgd_finetune.yaml --finetune $SOTA_CKPT

echo "3. Kather MSI/MSS Fine-tuning (15 epochs + Mixup 0.1)"
$PYTHON train.py --config configs/kather_finetune.yaml --finetune $SOTA_CKPT

echo "All optimized transfer learning tasks completed successfully!"
