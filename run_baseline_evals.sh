#!/usr/bin/env bash
# run_baseline_evals.sh
# Evaluates all 4 baseline models on CRC-VAL-HE-7K and saves per-model JSON results.
set -e

VENV=".venv/bin/python"
LOG_DIR="outputs/logs/baselines"
mkdir -p "$LOG_DIR"

declare -A MODELS=(
  ["mobilenetv2"]="outputs/checkpoints_mobilenetv2/ckpt_epoch057_acc0.9918.pt configs/baseline_mobilenetv2.yaml"
  ["efficientnetb0"]="outputs/checkpoints_efficientnetb0/ckpt_epoch053_acc0.9904.pt configs/baseline_efficientnetb0.yaml"
  ["shufflenetv2"]="outputs/checkpoints_shufflenetv2/ckpt_epoch080_acc0.9918.pt configs/baseline_shufflenetv2.yaml"
  ["resnet50"]="outputs/checkpoints_resnet50/ckpt_epoch036_acc0.9853.pt configs/baseline_resnet50.yaml"
)

for MODEL in mobilenetv2 efficientnetb0 shufflenetv2 resnet50; do
  read -r CKPT CFG <<< "${MODELS[$MODEL]}"
  echo ""
  echo "============================================================"
  echo " Evaluating: $MODEL"
  echo " Checkpoint: $CKPT"
  echo " Config    : $CFG"
  echo "============================================================"

  # Temporarily patch outputs.log_dir to write per-model JSON
  $VENV evaluate.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    2>&1 | tee "$LOG_DIR/eval_${MODEL}.txt"

  # Copy the just-written eval_results.json to a named file
  cp outputs/logs/eval_results.json "$LOG_DIR/eval_results_${MODEL}.json"
  echo "[DONE] Results saved to $LOG_DIR/eval_results_${MODEL}.json"
done

echo ""
echo "============================================================"
echo " All baselines evaluated. Results in: $LOG_DIR/"
echo "============================================================"
