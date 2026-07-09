"""
scripts/plot_pareto_efficiency.py

Generates a publication-ready Pareto efficiency scatter plot comparing MedLite-CRC
against standard baselines (ShuffleNetV2, MobileNetV2, EfficientNet-B0, ResNet-50).
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Data for the models (accuracy, params, latency, disk size)
    models = {
        "MedLite-CRC (Ours)": {
            "params": 0.49,
            "accuracy": 94.05,
            "latency": 7.93,
            "size": 2.0,
            "color": "#1f77b4",
            "marker": "*",
            "size_multiplier": 300,
        },
        "ShuffleNetV2": {
            "params": 1.26,
            "accuracy": 95.08,
            "latency": 5.13,
            "size": 5.23,
            "color": "#ff7f0e",
            "marker": "o",
            "size_multiplier": 150,
        },
        "MobileNetV2": {
            "params": 2.24,
            "accuracy": 94.82,
            "latency": 7.48,
            "size": 9.19,
            "color": "#2ca02c",
            "marker": "s",
            "size_multiplier": 150,
        },
        "EfficientNetB0": {
            "params": 4.02,
            "accuracy": 94.81,
            "latency": 11.72,
            "size": 16.38,
            "color": "#d62728",
            "marker": "^",
            "size_multiplier": 150,
        },
        "ResNet50": {
            "params": 23.53,
            "accuracy": 94.33,
            "latency": 19.06,
            "size": 94.43,
            "color": "#9467bd",
            "marker": "d",
            "size_multiplier": 150,
        }
    }

    # Set up matplotlib style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Enable grid lines with soft coloring
    ax.grid(True, linestyle="--", alpha=0.5, color="#dddddd")
    ax.set_axisbelow(True)

    # Plot each model
    for name, data in models.items():
        ax.scatter(
            data["params"],
            data["accuracy"],
            s=data["size_multiplier"],
            color=data["color"],
            marker=data["marker"],
            label=f"{name} ({data['size']} MB)",
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
            zorder=3
        )
        
        # Add labels near points
        xytext_offsets = {
            "MedLite-CRC (Ours)": (10, -5),
            "ShuffleNetV2": (10, 5),
            "MobileNetV2": (10, -12),
            "EfficientNetB0": (10, 5),
            "ResNet50": (-120, -5)
        }
        
        offset = xytext_offsets.get(name, (10, 0))
        
        label_text = f"{name}\n({data['params']}M params, {data['accuracy']:.2f}%)"
        if name == "MedLite-CRC (Ours)":
            label_text = r"$\mathbf{" + name.replace(" (Ours)", "") + "}$" + "\n(Ours: 0.49M, 94.05%)"
            
        ax.annotate(
            label_text,
            (data["params"], data["accuracy"]),
            textcoords="offset points",
            xytext=offset,
            ha="left" if offset[0] > 0 else "right",
            va="center",
            fontsize=9,
            fontweight="bold" if name == "MedLite-CRC (Ours)" else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none", zorder=2)
        )

    # Draw Pareto frontier line
    # The frontier is defined by MedLite-CRC (0.49M, 94.05%) and ShuffleNetV2 (1.26M, 95.08%)
    frontier_x = [models["MedLite-CRC (Ours)"]["params"], models["ShuffleNetV2"]["params"]]
    frontier_y = [models["MedLite-CRC (Ours)"]["accuracy"], models["ShuffleNetV2"]["accuracy"]]
    ax.plot(frontier_x, frontier_y, linestyle=":", color="#333333", linewidth=1.5, zorder=1, label="Pareto Frontier")

    # Set axes labels and title
    ax.set_xlabel("Model Parameters (Millions)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Cross-Patient Accuracy (%) on CRC-VAL-HE-7K", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Pareto Efficiency: Accuracy vs. Model Complexity\n(Trained strictly from scratch)", fontsize=13, fontweight="bold", pad=15)
    
    # Adjust axes limits to frame well
    ax.set_xlim(-1, 26)
    ax.set_ylim(93.5, 95.5)

    # Clean spine lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend positioning
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    # Ensure output dir exists
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "pareto_efficiency.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[SAVED] Pareto efficiency plot saved to: {out_path}")

if __name__ == "__main__":
    main()
