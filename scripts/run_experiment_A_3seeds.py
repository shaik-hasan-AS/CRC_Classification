import os
import subprocess
import yaml
import json
import numpy as np

SEEDS = [42, 123, 999]
BASE_CONFIG = "configs/config.yaml"

def run_seed(seed, run_idx):
    print(f"\n{'='*60}")
    print(f"🚀 RUN {run_idx}/3 | SEED: {seed}")
    print(f"{'='*60}\n")
    
    # 1. Create a specific config for this seed
    with open(BASE_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    
    if "project" not in cfg:
        cfg["project"] = {}
    cfg["project"]["seed"] = seed
    
    # Update log and checkpoint dirs to avoid overwriting
    cfg["outputs"]["checkpoint_dir"] = f"outputs/checkpoints_seed{seed}"
    cfg["outputs"]["log_dir"] = f"outputs/logs_seed{seed}"
    
    # Disable wandb to prevent spamming the dashboard (optional, but cleaner)
    if "wandb" in cfg:
        cfg["wandb"]["enabled"] = False
        
    temp_config = f"configs/config_seed{seed}.yaml"
    with open(temp_config, "w") as f:
        yaml.dump(cfg, f)
        
    # 2. Train the model
    print(f"🔄 Training model with seed {seed}...")
    train_cmd = [".venv/bin/python", "train.py", "--config", temp_config]
    
    # We use subprocess.run and pipe stdout to a file to keep the console clean
    log_file = f"outputs/logs_seed{seed}/train_stdout.log"
    os.makedirs(f"outputs/logs_seed{seed}", exist_ok=True)
    
    with open(log_file, "w") as lf:
        subprocess.run(train_cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
        
    print(f"✅ Training completed. Logs saved to {log_file}")
    
    # 3. Find the best checkpoint from training logs or directory
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    
    # Assuming standard naming convention like 'ckpt_epoch020_acc0.9900.pt'
    # Sort by accuracy to get the best one
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
    
    with open(f"outputs/logs_seed{seed}/eval_stdout.log", "w") as lf:
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
    return acc, f1

def main():
    accuracies = []
    f1_scores = []
    
    print("🌟 Starting 3-Seed Statistical Validation for Experiment A 🌟")
    print("Note: Training 3 models from scratch will take several hours depending on your GPU.")
    
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
    print("🎉 FINAL STATISTICAL VALIDATION RESULTS 🎉")
    print(f"{'='*60}")
    print(f"Accuracy across 3 seeds: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Macro-F1 across 3 seeds: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save a report
    report_path = "outputs/experiment_A_3seeds_report.txt"
    with open(report_path, "w") as f:
        f.write("Experiment A (NCT-CRC-HE-100K -> CRC-VAL-HE-7K) - 3 Seed Validation\n")
        f.write("="*70 + "\n")
        for i, seed in enumerate(SEEDS):
            f.write(f"Seed {seed}: Accuracy = {accuracies[i]:.2f}%, Macro-F1 = {f1_scores[i]:.4f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"FINAL ACCURACY: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"FINAL MACRO-F1: {mean_f1:.4f} ± {std_f1:.4f}\n")
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
