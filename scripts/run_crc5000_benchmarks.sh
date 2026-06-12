#!/bin/bash
# Sequential execution of all CRC-5000 benchmarking runs.
# MedLite-CRC + 4 Baselines.

PYTHON=".venv/bin/python"

echo "========================================================"
echo " CRC-5000 Benchmarking (MedLite-CRC vs Baselines)"
echo "========================================================"

echo ""
echo "[1/5] Training MedLite-CRC (V1)..."
$PYTHON train.py --config configs/crc5000_train.yaml

echo ""
echo "[2/5] Training MobileNetV2..."
$PYTHON train.py --config configs/baseline_mobilenetv2_crc5000.yaml

echo ""
echo "[3/5] Training EfficientNet-B0..."
$PYTHON train.py --config configs/baseline_efficientnetb0_crc5000.yaml

echo ""
echo "[4/5] Training ShuffleNetV2..."
$PYTHON train.py --config configs/baseline_shufflenetv2_crc5000.yaml

echo ""
echo "[5/5] Training ResNet-50..."
$PYTHON train.py --config configs/baseline_resnet50_crc5000.yaml

echo ""
echo "========================================================"
echo " All CRC-5000 benchmarking runs complete!"
echo "========================================================"
