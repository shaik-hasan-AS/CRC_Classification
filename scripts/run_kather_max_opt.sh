#!/bin/bash
set -e

# ANSI escape codes
BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
RESET="\033[0m"

echo -e "${BOLD}${BLUE}Starting Kather MSI/MSS Maximum Optimization (30 epochs + Unfreezing + Focal Loss)${RESET}"

SOTA_CKPT="outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"
if [ ! -f "$SOTA_CKPT" ]; then
    echo "ERROR: SOTA checkpoint not found at $SOTA_CKPT"
    exit 1
fi

.venv/bin/python train.py --config configs/kather_finetune.yaml --finetune "$SOTA_CKPT"

echo -e "${BOLD}${GREEN}Kather MSI/MSS optimization completed successfully!${RESET}"
