#!/bin/bash
# ==============================================================================
# MedLite-CRC: Peer-Reviewer Replication Guide
# ==============================================================================
# This script guides reviewers through reproducing the main tables, figures, 
# and statistical claims presented in the MedLite-CRC manuscript.
#
# Usage:
#   Interactive menu:   bash scripts/replicate_all.sh
#   Direct task execution: bash scripts/replicate_all.sh <1-9|all>
# ==============================================================================

set -e

# ANSI escape codes for styling
BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
CYAN="\033[36m"
YELLOW="\033[33m"
MAGENTA="\033[35m"
RESET="\033[0m"

# Auto-detect Python interpreter
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

show_menu() {
    clear || true
    echo -e "${BOLD}${GREEN}======================================================================${RESET}"
    echo -e "${BOLD}${GREEN}               MedLite-CRC Replication Console for Reviewers         ${RESET}"
    echo -e "${BOLD}${GREEN}======================================================================${RESET}"
    echo -e "Using Python: ${CYAN}$PYTHON${RESET}"
    echo -e "Reproduce the empirical claims, tables, and figures from the manuscript:\n"
    echo -e "  ${BOLD}1${RESET}) [Table 1 & Sec 5.1] SOTA Evaluation on CRC-VAL-HE-7K (96.27% Acc, Macro-F1)"
    echo -e "  ${BOLD}2${RESET}) [Table 2 & Sec 5.5] Multi-Cohort STARC-9 & CRC-5000 Benchmarks"
    echo -e "  ${BOLD}3${RESET}) [Table 3 & Sec 5.2] Architectural Leave-One-Out Ablations (Params & GFLOPs)"
    echo -e "  ${BOLD}4${RESET}) [Section 5.3]       Paired McNemar's Statistical Significance Test"
    echo -e "  ${BOLD}5${RESET}) [Section 5.6]       Expected Calibration Error (ECE) & Reliability Diagram"
    echo -e "  ${BOLD}6${RESET}) [Section 7]         Grad-CAM Spatial Artifact & Mitigation Analysis"
    echo -e "  ${BOLD}7${RESET}) [Section 8 & Tab 8] Computational Efficiency & Carbon Footprint Analysis"
    echo -e "  ${BOLD}8${RESET}) [Section 8]         CPU Inference Latency Benchmark (Batch size 1)"
    echo -e "  ${BOLD}9${RESET}) [Section 8]         INT8 Quantization Benchmark (0.72 MB, 95.72% Acc)"
    echo -e "  ${BOLD}a${RESET}) [Run All Quick]     Execute all analytical verification tasks (1, 3, 4, 5, 7, 8, 9)"
    echo -e "  ${BOLD}0${RESET}) Exit Console"
    echo ""
}

# Determine choice (CLI argument or interactive prompt)
if [ -n "$1" ]; then
    choice="$1"
    NON_INTERACTIVE=true
else
    show_menu
    read -p "Select a task to run replication (1-9, a, or 0): " choice
    NON_INTERACTIVE=false
fi

run_task() {
    local task="$1"
    case $task in
        1)
            echo -e "\n${BOLD}${BLUE}--- [Table 1] SOTA Cross-Patient Evaluation (CRC-VAL-HE-7K) ---${RESET}"
            CKPT="outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"
            if [ -f "$CKPT" ]; then
                $PYTHON evaluate.py --config configs/config.yaml --checkpoint "$CKPT"
            else
                echo -e "${YELLOW}Checkpoint $CKPT not found. Running with available default weights...${RESET}"
                $PYTHON evaluate.py --config configs/config.yaml
            fi
            ;;
        2)
            echo -e "\n${BOLD}${BLUE}--- [Table 2] Multi-Cohort STARC-9 & CRC-5000 Benchmarks ---${RESET}"
            echo -e "To train/evaluate models across the STARC-9 multi-centric or CRC-5000 noisy cohorts, run:"
            echo -e "  - STARC-9 (54k validation tiles):  ${CYAN}bash scripts/run_starc9_benchmarks.sh${RESET}"
            echo -e "  - CRC-5000 (clinical noisy holdout): ${CYAN}bash scripts/run_crc5000_benchmarks.sh${RESET}"
            if [ "$NON_INTERACTIVE" != "true" ]; then
                echo -e "\nWould you like to run STARC-9 benchmark suite now? (y/n)"
                read -p "> " run_opt
                if [ "$run_opt" = "y" ] || [ "$run_opt" = "Y" ]; then
                    bash scripts/run_starc9_benchmarks.sh
                fi
            fi
            ;;
        3)
            echo -e "\n${BOLD}${BLUE}--- [Table 3] Architectural Leave-One-Out Ablations (Params & GFLOPs) ---${RESET}"
            $PYTHON scripts/architectural_ablation.py
            ;;
        4)
            echo -e "\n${BOLD}${BLUE}--- [Section 5.3] Paired McNemar's Statistical Significance Test ---${RESET}"
            $PYTHON scripts/statistical_test.py
            ;;
        5)
            echo -e "\n${BOLD}${BLUE}--- [Section 5.6] Expected Calibration Error (ECE) Analysis ---${RESET}"
            $PYTHON scripts/calibration_analysis.py
            ;;
        6)
            echo -e "\n${BOLD}${BLUE}--- [Section 7] Grad-CAM Spatial Interpretability Analysis ---${RESET}"
            V2_CKPT="outputs/checkpoints_kd_v2_v2/ckpt_epoch002_acc0.9935.pt"
            if [ -f "$V2_CKPT" ]; then
                $PYTHON scripts/analyze_gradcam_spatial.py --config configs/kd_mobilenet_v2.yaml --checkpoint "$V2_CKPT" --mask_border_width 8
            else
                $PYTHON scripts/analyze_gradcam_spatial.py
            fi
            ;;
        7)
            echo -e "\n${BOLD}${BLUE}--- [Section 8] Computational Efficiency & Carbon Footprint Analysis ---${RESET}"
            $PYTHON scripts/compute_efficiency_analysis.py
            ;;
        8)
            echo -e "\n${BOLD}${BLUE}--- [Section 8] CPU Inference Latency Benchmark (Batch Size 1) ---${RESET}"
            $PYTHON scripts/benchmark_all_cpu.py
            ;;
        9)
            echo -e "\n${BOLD}${BLUE}--- [Section 8] INT8 Quantization Benchmark (0.72 MB Footprint) ---${RESET}"
            $PYTHON scripts/quantize_int8.py
            ;;
        a|all|quick)
            echo -e "\n${BOLD}${MAGENTA}======================================================================${RESET}"
            echo -e "${BOLD}${MAGENTA}       Executing All Analytical Replications (Tasks 1, 3-5, 7-9)      ${RESET}"
            echo -e "${BOLD}${MAGENTA}======================================================================${RESET}"
            run_task 1
            run_task 3
            run_task 4
            run_task 5
            run_task 7
            run_task 8
            run_task 9
            echo -e "\n${BOLD}${GREEN}✓ All analytical replications completed successfully!${RESET}"
            ;;
        0|exit|q)
            echo -e "\nExiting replication console. Refer to ${BOLD}README.md${RESET} for direct 1-liner commands."
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Invalid option: $task${RESET}"
            exit 1
            ;;
    esac
}

run_task "$choice"
