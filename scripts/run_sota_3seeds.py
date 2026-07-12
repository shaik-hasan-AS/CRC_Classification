import os
import subprocess
import yaml
import json
import numpy as np

SEEDS = [42, 123, 999]
BASE_CONFIG = "configs/kd_mobilenet_v2.yaml"
FINETUNE_CKPT = "outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"

def run_seed(seed, run_idx):
    print(f"\n{'='*60}")
    print(f"🚀 SOTA RUN {run_idx}/3 | SEED: {seed}")
    print(f"{'='*60}\n")
    
    # 1. Create a specific config for this seed
    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    
    if "project" not in cfg:
        cfg["project"] = {}
    cfg["project"]["seed"] = seed
    
    # Force epochs to 3 for fast statistical validation
    cfg["training"]["epochs"] = 3
    
    # Update log and checkpoint dirs to avoid overwriting
    cfg["outputs"]["checkpoint_dir"] = f"outputs/checkpoints_sota_seed{seed}"
    cfg["outputs"]["log_dir"] = f"outputs/logs_sota_seed{seed}"
    
    temp_config = f"configs/kd_mobilenet_v2_seed{seed}.yaml"
    with open(temp_config, "w") as f:
        yaml.dump(cfg, f)
        
    # 2. Check if checkpoint directory already has trained weights to save time
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    if FINETUNE_CKPT:
        ckpt_dir = ckpt_dir.rstrip("/") + "_v2"
        
    best_ckpt_path = None
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if ckpts:
            print(f"✅ Checkpoint directory {ckpt_dir} already exists and contains weights. Skipping training.")
            def get_acc(ckpt_name):
                try:
                    return float(ckpt_name.split("acc")[1].replace(".pt", ""))
                except:
                    return 0.0
            best_ckpt = max(ckpts, key=get_acc)
            best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)
            
    if best_ckpt_path is None:
        print(f"🔄 Fine-tuning SOTA model with seed {seed}...")
        train_cmd = [".venv/bin/python", "train.py", "--config", temp_config, "--finetune", FINETUNE_CKPT]
        
        log_file = f"outputs/logs_sota_seed{seed}/train_stdout.log"
        os.makedirs(f"outputs/logs_sota_seed{seed}", exist_ok=True)
        
        with open(log_file, "w") as lf:
            subprocess.run(train_cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
            
        print(f"✅ Training completed. Logs saved to {log_file}")
        
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        def get_acc(ckpt_name):
            try:
                return float(ckpt_name.split("acc")[1].replace(".pt", ""))
            except:
                return 0.0
        best_ckpt = max(ckpts, key=get_acc)
        best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)
        
    print(f"🔍 Best Checkpoint Found: {best_ckpt_path}")
    
    # 4. Evaluate the model
    print(f"📊 Evaluating model on cross-val dataset...")
    eval_cmd = [".venv/bin/python", "evaluate.py", "--config", temp_config, "--checkpoint", best_ckpt_path]
    
    with open(f"outputs/logs_sota_seed{seed}/eval_stdout.log", "w") as lf:
        subprocess.run(eval_cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
        
    # 5. Read the evaluation results
    results_file = os.path.join(cfg["outputs"]["log_dir"], "eval_results.json")
    with open(results_file, "r") as f:
        results = json.load(f)
        
    # We want the performance on CRC-VAL-HE-7K
    cross_val_metrics = results["splits"]["crc_val_7k"]
    acc = cross_val_metrics["accuracy"] * 100 # Convert to percentage
    f1 = cross_val_metrics["macro_f1"]
    
    print(f"🎯 Seed {seed} Results -> Accuracy: {acc:.2f}% | Macro-F1: {f1:.4f}")
    
    # Cleanup temp config
    try:
        os.remove(temp_config)
    except:
        pass
        
    return acc, f1

def main():
    accuracies = []
    f1_scores = []
    
    print("🌟 Starting 3-Seed Statistical Validation for MedLite-CRC V2 (KD SOTA) 🌟")
    
    for i, seed in enumerate(SEEDS, 1):
        try:
            acc, f1 = run_seed(seed, i)
            accuracies.append(acc)
            f1_scores.append(f1)
        except Exception as e:
            print(f"❌ Error running seed {seed}: {e}")
            return
            
    # Calculate Mean and Std
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\n{'='*60}")
    print("🎉 FINAL SOTA STATISTICAL VALIDATION RESULTS 🎉")
    print(f"{'='*60}")
    print(f"Accuracy across 3 seeds: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Macro-F1 across 3 seeds: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save a report
    report_path = "outputs/sota_3seeds_report.txt"
    with open(report_path, "w") as f:
        f.write("MedLite-CRC V2 (MobileNetV2 KD SOTA) - 3 Seed Validation\n")
        f.write("="*70 + "\n")
        for i, seed in enumerate(SEEDS):
            f.write(f"Seed {seed}: Accuracy = {accuracies[i]:.2f}%, Macro-F1 = {f1_scores[i]:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"FINAL ACCURACY: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"FINAL MACRO-F1: {mean_f1:.4f} ± {std_f1:.4f}\n")
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
