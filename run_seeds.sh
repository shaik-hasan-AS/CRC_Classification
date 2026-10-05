#!/bin/bash
set -e

# Config 2 Seed 44 (RESUME)
echo "Resuming Config 2 (Stain Adaptation) - Seed 44"
sed -i 's/seed: 43/seed: 44/g' configs/ablation/config_2_stainnorm.yaml
LATEST_CKPT=$(ls -t outputs/checkpoints_ablation_stainnorm/ckpt_epoch*.pt | head -n 1)
.venv/bin/python train.py --config configs/ablation/config_2_stainnorm.yaml --resume "$LATEST_CKPT" >> outputs/logs/config2_seed44.log 2>&1
echo "Evaluating Config 2 - Seed 44 on CRC-7K"
CKPT2_44=$(ls -t outputs/checkpoints_ablation_stainnorm/ckpt_epoch*.pt | head -n 1)
.venv/bin/python evaluate.py --config configs/ablation/config_2_stainnorm.yaml --checkpoint "$CKPT2_44" > outputs/logs/eval_config2_seed44.log 2>&1
sed -i 's/seed: 44/seed: 42/g' configs/ablation/config_2_stainnorm.yaml

# Config 3 Seed 43
echo "Starting Config 3 (MultiScale) - Seed 43"
sed -i 's/seed: 42/seed: 43/g' configs/ablation/config_3_multiscale.yaml
.venv/bin/python train.py --config configs/ablation/config_3_multiscale.yaml > outputs/logs/config3_seed43.log 2>&1
echo "Evaluating Config 3 - Seed 43 on CRC-7K"
CKPT3_43=$(ls -t outputs/checkpoints_ablation_multiscale/ckpt_epoch*.pt | head -n 1)
.venv/bin/python evaluate.py --config configs/ablation/config_3_multiscale.yaml --checkpoint "$CKPT3_43" > outputs/logs/eval_config3_seed43.log 2>&1

# Config 3 Seed 44
echo "Starting Config 3 (MultiScale) - Seed 44"
sed -i 's/seed: 43/seed: 44/g' configs/ablation/config_3_multiscale.yaml
.venv/bin/python train.py --config configs/ablation/config_3_multiscale.yaml > outputs/logs/config3_seed44.log 2>&1
echo "Evaluating Config 3 - Seed 44 on CRC-7K"
CKPT3_44=$(ls -t outputs/checkpoints_ablation_multiscale/ckpt_epoch*.pt | head -n 1)
.venv/bin/python evaluate.py --config configs/ablation/config_3_multiscale.yaml --checkpoint "$CKPT3_44" > outputs/logs/eval_config3_seed44.log 2>&1
sed -i 's/seed: 44/seed: 42/g' configs/ablation/config_3_multiscale.yaml

echo "All runs completed!"
