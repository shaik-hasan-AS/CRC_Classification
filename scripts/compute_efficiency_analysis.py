"""
scripts/compute_efficiency_analysis.py

Calculates and plots the computational efficiency and carbon footprint (CO2 emissions) 
for MedLite-CRC versus baseline models (ResNet-50, EfficientNet-B0, MobileNetV2, ShuffleNetV2).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Constants and assumptions
CARBON_INTENSITY_INDIA = 0.82  # kg CO2 per kWh (India national grid average)
EDGE_CPU_WATTS = 28.0          # TDP of Intel i5-1135G7 in Watts under full load
GPU_TRAINING_WATTS = 85.0      # Average RTX 4060 GPU power draw in Watts during training
RESNET_GPU_WATTS = 105.0       # Higher power draw for ResNet-50 due to higher GPU utilization
SYSTEM_OVERHEAD_WATTS = 50.0   # CPU, RAM, and motherboard idle/system overhead during training

# 2. Model Specific Data (from Table 5.1 and experimental logs)
models_data = {
    "MedLite-CRC (INT8)": {
        "params_m": 0.48,
        "latency_ms": 2.08,
        "train_hours": 2.0,  # 30 epochs @ ~4 min/epoch
        "system_train_watts": GPU_TRAINING_WATTS + SYSTEM_OVERHEAD_WATTS,
    },
    "MedLite-CRC (FP32)": {
        "params_m": 0.48,
        "latency_ms": 8.28,
        "train_hours": 2.0,
        "system_train_watts": GPU_TRAINING_WATTS + SYSTEM_OVERHEAD_WATTS,
    },
    "ShuffleNetV2": {
        "params_m": 1.26,
        "latency_ms": 5.13,
        "train_hours": 2.6,  # 30 epochs @ ~5.2 min/epoch
        "system_train_watts": GPU_TRAINING_WATTS + SYSTEM_OVERHEAD_WATTS,
    },
    "MobileNetV2": {
        "params_m": 2.24,
        "latency_ms": 7.48,
        "train_hours": 2.75, # 30 epochs @ ~5.5 min/epoch
        "system_train_watts": GPU_TRAINING_WATTS + SYSTEM_OVERHEAD_WATTS,
    },
    "EfficientNet-B0": {
        "params_m": 4.02,
        "latency_ms": 11.72,
        "train_hours": 3.5,  # 30 epochs @ ~7 min/epoch
        "system_train_watts": GPU_TRAINING_WATTS + SYSTEM_OVERHEAD_WATTS,
    },
    "ResNet-50": {
        "params_m": 23.53,
        "latency_ms": 19.06,
        "train_hours": 5.0,  # 30 epochs @ ~10 min/epoch
        "system_train_watts": RESNET_GPU_WATTS + SYSTEM_OVERHEAD_WATTS,
    }
}

def calculate_metrics():
    results = {}
    for name, data in models_data.items():
        # Training Calculations
        train_kwh = (data["system_train_watts"] * data["train_hours"]) / 1000.0
        train_co2_kg = train_kwh * CARBON_INTENSITY_INDIA
        train_co2_g = train_co2_kg * 1000.0
        
        # Inference Calculations (per image)
        inf_latency_s = data["latency_ms"] / 1000.0
        inf_energy_j = EDGE_CPU_WATTS * inf_latency_s
        inf_energy_kwh = (EDGE_CPU_WATTS * inf_latency_s) / 3600000.0
        
        # Inference for 100,000 images
        inf_100k_kwh = inf_energy_kwh * 100000.0
        inf_100k_co2_g = inf_100k_kwh * CARBON_INTENSITY_INDIA * 1000.0
        
        results[name] = {
            "params_m": data["params_m"],
            "latency_ms": data["latency_ms"],
            "train_hours": data["train_hours"],
            "train_kwh": train_kwh,
            "train_co2_g": train_co2_g,
            "inf_energy_j": inf_energy_j,
            "inf_100k_co2_g": inf_100k_co2_g
        }
    return results

def print_markdown_table(results):
    print("\n### Computational Efficiency & Carbon Footprint Analysis (India Grid, 0.82 kg CO2/kWh)")
    print("| Model | Params (M) | Latency (ms) | Train Time (hrs) | Training Energy (kWh) | Training CO2 (g) | Inf Energy (J/img) | Inf CO2 (g / 100k img) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, res in results.items():
        print(f"| {name} | {res['params_m']:.2f} | {res['latency_ms']:.2f} | {res['train_hours']:.2f} | {res['train_kwh']:.3f} | {res['train_co2_g']:.1f} | {res['inf_energy_j']:.4f} | {res['inf_100k_co2_g']:.3f} |")
    print()

def plot_efficiency(results, save_path):
    names = list(results.keys())
    train_co2 = [results[n]["train_co2_g"] for n in names]
    inf_co2 = [results[n]["inf_100k_co2_g"] for n in names]
    
    # Exclude redundant QAT / INT8 from training plot if they share weights
    # Let's keep them all for full comparison
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor('#16213e')
    
    # Custom color palette matching previous assets
    colors = ['#ff4b5c', '#ff6b6b', '#ffbe0b', '#07d35f', '#2196F3', '#9c27b0']
    # Reverse to make ours stand out or style specifically
    bar_colors = ['#07d35f', '#2196F3', '#aaaaaa', '#cccccc', '#ffbe0b', '#ff4b5c']
    
    # Left plot: Training Carbon Footprint
    ax1.set_facecolor('#1a1a2e')
    bars1 = ax1.barh(names, train_co2, color=bar_colors, alpha=0.9, edgecolor='none', height=0.6)
    ax1.set_xlabel('CO2 Emissions (grams)', color='white', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_title('Training Carbon Footprint\n(30 Epochs on RTX 4060)', color='white', fontsize=13, fontweight='bold', pad=15)
    ax1.tick_params(colors='#aaaaaa')
    ax1.set_yticklabels(names, color='white', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.1, color='white')
    
    # Annotate bars
    for bar in bars1:
        width = bar.get_width()
        ax1.annotate(f'{width:.0f}g',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', color='white', fontsize=9, fontweight='bold')
                    
    # Right plot: Inference Carbon Footprint
    ax2.set_facecolor('#1a1a2e')
    bars2 = ax2.barh(names, inf_co2, color=bar_colors, alpha=0.9, edgecolor='none', height=0.6)
    ax2.set_xlabel('CO2 Emissions (grams)', color='white', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('Inference Carbon Footprint\n(per 100,000 Images on Edge CPU)', color='white', fontsize=13, fontweight='bold', pad=15)
    ax2.tick_params(colors='#aaaaaa')
    ax2.set_yticklabels([]) # Hide labels as they are shared on the left
    ax2.grid(True, linestyle='--', alpha=0.1, color='white')
    
    # Annotate bars
    for bar in bars2:
        width = bar.get_width()
        ax2.annotate(f'{width:.2f}g',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  # 5 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', color='white', fontsize=9, fontweight='bold')
    
    # Styling spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.suptitle('MedLite-CRC Carbon Footprint & Environmental Impact Analysis', 
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] Carbon efficiency plot -> {save_path}")

def main():
    results = calculate_metrics()
    print_markdown_table(results)
    
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "carbon_efficiency_comparison.png"
    plot_efficiency(results, save_path)
    
    # Copy to assets folder as well
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy(save_path, assets_dir / "carbon_efficiency_comparison.png")
    print(f"[COPIED] -> assets/carbon_efficiency_comparison.png")

if __name__ == "__main__":
    main()
