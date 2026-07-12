#!/bin/bash
# ==============================================================================
# MedLite-CRC: Peer-Reviewer Replication Guide
# ==============================================================================
# This script guides reviewers through reproducing the main tables, figures, 
# and statistical claims presented in the MedLite-CRC manuscript.

set -e

# ANSI escape codes for beautiful styling
BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
CYAN="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

clear
echo -e "${BOLD}${GREEN}======================================================================${RESET}"
echo -e "${BOLD}${GREEN}               MedLite-CRC Replication Console for Reviewers         ${RESET}"
echo -e "${BOLD}${GREEN}======================================================================${RESET}"
echo -e "This utility displays replication commands and runs key evaluations to"
echo -e "verify the empirical claims made in the manuscript."
echo ""
echo -e "Available Verification Tasks:"
echo -e "  ${BOLD}1${RESET}) [Table 1] OOD Cross-Patient Baseline & KD Student Evaluations (CRC-7K)"
echo -e "  ${BOLD}2${RESET}) [Table 2] Multi-Cohort STARC-9 & CRC-5000 Benchmarks"
echo -e "  ${BOLD}3${RESET}) [Table 3] Architectural Leave-One-Out Ablation Study"
echo -e "  ${BOLD}4${RESET}) [Section 5.3] Statistical Significance Test (McNemar's Chi-Square)"
echo -e "  ${BOLD}5${RESET}) [Section 5.6] Expected Calibration Error (ECE) Analysis"
echo -e "  ${BOLD}6${RESET}) [Section 7] Grad-CAM Spatial Boundary Artifact & Mitigation Analysis"
echo -e "  ${BOLD}7${RESET}) [Section 8] Computational Efficiency, FLOPs & Latency Benchmarks"
echo -e "  ${BOLD}8${RESET}) Exit Console"
echo ""
read -p "Select a task to view instructions and run replication (1-8): " choice

case $choice in
    1)
        echo -e "\n${BOLD}${BLUE}--- [Table 1] OOD Cross-Patient Baseline & KD Student Evaluations ---${RESET}"
        echo -e "To evaluate baselines and the SOTA distilled models on the CRC-VAL-HE-7K cohort, run:"
        echo -e "  ${CYAN}python scripts/run_full_eval.py --config configs/config.yaml --checkpoint <checkpoint_path>${RESET}"
        echo -e "Would you like to run a validation check on the MobileNetV2 KD Student model? (y/n)"
        read -p "> " run_opt
        if [ "$run_opt" = "y" ] || [ "$run_opt" = "Y" ]; then
            # Look for the KD student checkpoint
            CKPT="outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"
            if [ -f "$CKPT" ]; then
                python scripts/run_full_eval.py --config configs/kd_mobilenet_teacher.yaml --checkpoint "$CKPT"
            else
                echo -e "${YELLOW}Checkpoint $CKPT not found. Running evaluation with available mock/temporary weights...${RESET}"
                python scripts/run_full_eval.py --config configs/config.yaml --dummy
            fi
        fi
        ;;
    2)
        echo -e "\n${BOLD}${BLUE}--- [Table 2] Multi-Cohort STARC-9 & CRC-5000 Benchmarks ---${RESET}"
        echo -e "To run evaluations across the STARC-9 multi-centric or CRC-5000 noisy cohorts, use:"
        echo -e "  - STARC-9:   ${CYAN}bash scripts/run_starc9_benchmarks.sh${RESET}"
        echo -e "  - CRC-5000:  ${CYAN}bash scripts/run_crc5000_benchmarks.sh${RESET}"
        ;;
    3)
        echo -e "\n${BOLD}${BLUE}--- [Table 3] Architectural Leave-One-Out Ablation Study ---${RESET}"
        echo -e "To reproduce the architectural component ablation study, run:"
        echo -e "  ${CYAN}bash scripts/run_architectural_ablation.sh${RESET}"
        echo -e "This script evaluates: Baseline CNN, +StainNorm, +MultiScale (Final), and +SEBlock."
        ;;
    4)
        echo -e "\n${BOLD}${BLUE}--- [Section 5.3] Statistical Significance Test (McNemar's) ---${RESET}"
        echo -e "To replicate the paired McNemar's Chi-Square significance test comparing the SOTA KD student"
        echo -e "vs. the 8x larger EfficientNet-B0 baseline under normal and masked scenarios, run:"
        echo -e "  ${CYAN}python scripts/statistical_test.py${RESET}"
        echo -e "Would you like to execute this test now? (y/n)"
        read -p "> " run_opt
        if [ "$run_opt" = "y" ] || [ "$run_opt" = "Y" ]; then
            python scripts/statistical_test.py
        fi
        ;;
    5)
        echo -e "\n${BOLD}${BLUE}--- [Section 5.6] Expected Calibration Error (ECE) Analysis ---${RESET}"
        echo -e "To calculate calibration curves and reliability metrics (ECE / MCE), run:"
        echo -e "  ${CYAN}python scripts/calibration_analysis.py${RESET}"
        echo -e "Would you like to run the ECE calculator now? (y/n)"
        read -p "> " run_opt
        if [ "$run_opt" = "y" ] || [ "$run_opt" = "Y" ]; then
            python scripts/calibration_analysis.py
        fi
        ;;
    6)
        echo -e "\n${BOLD}${BLUE}--- [Section 7] Grad-CAM Spatial Artifact & Mitigation Analysis ---${RESET}"
        echo -e "To compute spatial center-bias, vanishing gradient rates, and stroma tissue vs. background"
        echo -e "activation values for V1 (zero-padded) and V2 (reflect-padded + masked) models, run:"
        echo -e "  ${CYAN}python scripts/analyze_gradcam_spatial.py --config configs/kd_mobilenet_v2.yaml --checkpoint outputs/checkpoints_kd_v2_v2/ckpt_epoch002_acc0.9935.pt --mask_border_width 8${RESET}"
        echo -e "Would you like to run the spatial interpretability analysis on the V2 mitigated model? (y/n)"
        read -p "> " run_opt
        if [ "$run_opt" = "y" ] || [ "$run_opt" = "Y" ]; then
            python scripts/analyze_gradcam_spatial.py --config configs/kd_mobilenet_v2.yaml --checkpoint outputs/checkpoints_kd_v2_v2/ckpt_epoch002_acc0.9935.pt --mask_border_width 8
        fi
        ;;
    7)
        echo -e "\n${BOLD}${BLUE}--- [Section 8] Computational Efficiency, FLOPs & Latency Benchmarks ---${RESET}"
        echo -e "To profile CPU/GPU inference latencies, peak memory usage, and parameter counts, run:"
        echo -e "  ${CYAN}python scripts/compute_efficiency_analysis.py${RESET}"
        echo -e "Would you like to run the profiling benchmark suite now? (y/n)"
        read -p "> " run_opt
        if [ "$run_opt" = "y" ] || [ "$run_opt" = "Y" ]; then
            python scripts/compute_efficiency_analysis.py
        fi
        ;;
    8|*)
        echo -e "\nExiting replication console. Refer to the documentation in ${BOLD}README.md${RESET} and ${BOLD}docs/supplementary_materials.md${RESET} for detailed findings."
        exit 0
        ;;
esac
