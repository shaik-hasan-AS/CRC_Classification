#!/bin/bash
# Evaluate all trained baseline models on CRC-VAL-HE-7K
# Results will prove MedLite-CRC efficiency vs heavy baselines

EVAL_CONFIG="configs/crc7k_eval.yaml"
PYTHON=".venv/bin/python"

echo "========================================================"
echo " Baseline Evaluation on CRC-VAL-HE-7K (Cross-Patient)"
echo "========================================================"

echo ""
echo "[1/4] MobileNetV2..."
$PYTHON evaluate.py --config configs/baseline_mobilenetv2.yaml \
    --checkpoint outputs/checkpoints_mobilenetv2/ckpt_epoch057_acc0.9918.pt

echo ""
echo "[2/4] EfficientNet-B0..."
$PYTHON evaluate.py --config configs/baseline_efficientnetb0.yaml \
    --checkpoint outputs/checkpoints_efficientnetb0/ckpt_epoch053_acc0.9904.pt

echo ""
echo "[3/4] ShuffleNetV2..."
$PYTHON evaluate.py --config configs/baseline_shufflenetv2.yaml \
    --checkpoint outputs/checkpoints_shufflenetv2/ckpt_epoch080_acc0.9918.pt

echo ""
echo "[4/4] ResNet-50..."
$PYTHON evaluate.py --config configs/baseline_resnet50.yaml \
    --checkpoint outputs/checkpoints_resnet50/ckpt_epoch036_acc0.9853.pt

echo ""
echo "========================================================"
echo " All baseline evaluations complete!"
echo "========================================================"
