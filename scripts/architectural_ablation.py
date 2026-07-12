"""
scripts/architectural_ablation.py

Instantiates MedLite-CRC architectural variants to measure their parameters
and FLOPs. Generates a markdown table of the results.
"""

import torch
from models.medlite_crc import MedLiteCRC, count_parameters

def main():
    variants = [
        {
            "name": "Baseline CNN",
            "stain_norm": False,
            "multiscale": False,
            "se_block": False,
        },
        {
            "name": "+ LearnableStainNorm",
            "stain_norm": True,
            "multiscale": False,
            "se_block": False,
        },
        {
            "name": "+ MultiScaleBranch",
            "stain_norm": True,
            "multiscale": True,
            "se_block": False,
        },
        {
            "name": "Ablation 4: + SEBlock (Negative Finding)",
            "stain_norm": True,
            "multiscale": True,
            "se_block": True,
        }
    ]

    print(f"{'Variant Name':<35} | {'Params (M)':<10} | {'GFLOPs':<10}")
    print("-" * 62)

    try:
        from thop import profile
        has_thop = True
    except ImportError:
        print("Note: Install 'thop' (pip install thop) to calculate GFLOPs.")
        has_thop = False

    dummy_input = torch.randn(1, 3, 224, 224)

    for v in variants:
        model = MedLiteCRC(
            num_classes=9,
            base_channels=32,
            use_stain_norm=v["stain_norm"],
            use_multiscale=v["multiscale"],
            use_se_block=v["se_block"]
        )
        
        stats = count_parameters(model)
        params_M = stats['total_M']
        
        flops_str = "N/A"
        if has_thop:
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_str = f"{macs / 1e9:.4f}"
            
        print(f"{v['name']:<35} | {params_M:<10} | {flops_str:<10}")

if __name__ == "__main__":
    main()
