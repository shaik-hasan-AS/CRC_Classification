#!/bin/bash
# scripts/run_architectural_ablation.sh
# Automates the training of the 4 architectural ablation variants.
# These configs assume you have set up your main config.yaml to read the boolean flags.
# Since we modified the Python code, we will override the config using sed or python args.
# For simplicity, we create temporary config files here and run train.py.

set -e

# Create an ablation directory
mkdir -p configs/ablation
cp configs/config.yaml configs/ablation/base_config.yaml

echo "Setting up Ablation 1: Baseline CNN..."
sed -e 's/name: "MedLiteCRC"/name: "MedLiteCRC"\n  use_stain_norm: false\n  use_multiscale: false\n  use_se_block: false/g' \
    -e 's/checkpoint_dir: "outputs\/checkpoints"/checkpoint_dir: "outputs\/checkpoints_ablation_baseline"/g' \
    -e 's/log_dir: "outputs\/logs"/log_dir: "outputs\/logs_ablation_baseline"/g' \
    configs/ablation/base_config.yaml > configs/ablation/config_1_baseline.yaml

echo "Setting up Ablation 2: + LearnableStainNorm..."
sed -e 's/name: "MedLiteCRC"/name: "MedLiteCRC"\n  use_stain_norm: true\n  use_multiscale: false\n  use_se_block: false/g' \
    -e 's/checkpoint_dir: "outputs\/checkpoints"/checkpoint_dir: "outputs\/checkpoints_ablation_stainnorm"/g' \
    -e 's/log_dir: "outputs\/logs"/log_dir: "outputs\/logs_ablation_stainnorm"/g' \
    configs/ablation/base_config.yaml > configs/ablation/config_2_stainnorm.yaml

echo "Setting up Ablation 3: + MultiScaleBranch..."
sed -e 's/name: "MedLiteCRC"/name: "MedLiteCRC"\n  use_stain_norm: true\n  use_multiscale: true\n  use_se_block: false/g' \
    -e 's/checkpoint_dir: "outputs\/checkpoints"/checkpoint_dir: "outputs\/checkpoints_ablation_multiscale"/g' \
    -e 's/log_dir: "outputs\/logs"/log_dir: "outputs\/logs_ablation_multiscale"/g' \
    configs/ablation/base_config.yaml > configs/ablation/config_3_multiscale.yaml

echo "Setting up Ablation 4: + SEBlock (Negative Finding — not the final architecture)..."
sed -e 's/name: "MedLiteCRC"/name: "MedLiteCRC"\n  use_stain_norm: true\n  use_multiscale: true\n  use_se_block: true/g' \
    -e 's/checkpoint_dir: "outputs\/checkpoints"/checkpoint_dir: "outputs\/checkpoints_ablation_full"/g' \
    -e 's/log_dir: "outputs\/logs"/log_dir: "outputs\/logs_ablation_full"/g' \
    configs/ablation/base_config.yaml > configs/ablation/config_4_full.yaml


echo "Starting Ablation Training Pipeline..."

echo "---------------------------------------------------------"
echo "Running Ablation 1/4: Baseline CNN"
echo "---------------------------------------------------------"
python train.py --config configs/ablation/config_1_baseline.yaml

echo "---------------------------------------------------------"
echo "Running Ablation 2/4: + LearnableStainNorm"
echo "---------------------------------------------------------"
python train.py --config configs/ablation/config_2_stainnorm.yaml

echo "---------------------------------------------------------"
echo "Running Ablation 3/4: + MultiScaleBranch"
echo "---------------------------------------------------------"
python train.py --config configs/ablation/config_3_multiscale.yaml

echo "---------------------------------------------------------"
echo "Running Ablation 4/4: + SEBlock (Negative Finding)"
echo "---------------------------------------------------------"
python train.py --config configs/ablation/config_4_full.yaml

echo "Done!"
