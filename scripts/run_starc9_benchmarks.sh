#!/bin/bash
# Sequential execution of all STARC-9 10% benchmarking runs.
# MedLite-CRC + 4 Baselines.

PYTHON=".venv/bin/python"

echo "========================================================"
echo " STARC-9 10% Benchmarking (MedLite-CRC vs Baselines)"
echo "========================================================"

echo ""
echo "[1/5] Training MedLite-CRC (V1)..."
$PYTHON train.py --config configs/starc9_train.yaml

echo ""
echo "[2/5] Training MobileNetV2..."
$PYTHON train.py --config configs/baseline_mobilenetv2_starc9.yaml

echo ""
echo "[3/5] Training EfficientNet-B0..."
$PYTHON train.py --config configs/baseline_efficientnetb0_starc9.yaml

echo ""
echo "[4/5] Training ShuffleNetV2..."
$PYTHON train.py --config configs/baseline_shufflenetv2_starc9.yaml

echo ""
echo "[5/5] Training ResNet-50..."
$PYTHON train.py --config configs/baseline_resnet50_starc9.yaml

echo ""
echo "========================================================"
echo " All STARC-9 benchmarking runs complete!"
echo "========================================================"
