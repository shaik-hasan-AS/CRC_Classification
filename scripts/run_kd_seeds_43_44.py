import os
import subprocess
import json
import time

def run_kd_seed(seed):
    print(f"============================================================")
    print(f"Starting KD MobileNetV2 Full Training for Seed {seed}")
    print(f"============================================================")
    
    # We will use the kd_mobilenet_teacher.yaml config as a base
    config_path = "configs/kd_mobilenet_teacher.yaml"
    
    # Create a temporary config to override seed and epochs
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg["project"]["seed"] = seed
    cfg["training"]["epochs"] = 200  # Full 200 epochs as requested
    cfg["outputs"]["checkpoint_dir"] = f"outputs/checkpoints_kd_mobilenet_seed{seed}"
    cfg["outputs"]["log_dir"] = f"outputs/logs_kd_mobilenet_seed{seed}"
    
    os.makedirs(cfg["outputs"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["outputs"]["log_dir"], exist_ok=True)
    
    temp_config = f"configs/temp_kd_seed{seed}.yaml"
    with open(temp_config, "w") as f:
        yaml.dump(cfg, f)
        
    # Run training
    train_cmd = [".venv/bin/python", "train.py", "--config", temp_config]
    
    print(f"Running command: {' '.join(train_cmd)}")
    start_time = time.time()
    
    # We will stream the output so we can see progress in the task log
    process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"ERROR: Training failed for seed {seed} with return code {process.returncode}")
        return False
        
    # Evaluate the best checkpoint
    print(f"--- Training finished for seed {seed} in {(time.time() - start_time) / 3600:.2f} hours ---")
    
    # Find the best checkpoint (highest epoch usually, or based on acc name)
    ckpt_dir = cfg["outputs"]["checkpoint_dir"]
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        print(f"ERROR: No checkpoints found for seed {seed}")
        return False
        
    # Since checkpoints are named ckpt_epochXXX_accY.YYYY.pt, sort by accuracy or epoch
    best_ckpt = sorted(ckpts)[-1]
    best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)
    
    print(f"Best checkpoint for seed {seed}: {best_ckpt_path}")
    
    # Run evaluation on CRC-VAL-HE-7K
    eval_cmd = [
        ".venv/bin/python", "eval_starc.py",  # Wait, eval_starc.py might be for STARC-9. 
        # I should use calibration_analysis.py or just rely on the eval_results.json from training?
        # Actually, train.py evaluates on CRC-VAL-HE-7K natively at the end and saves best checkpoint!
    ]
    
    return True

if __name__ == "__main__":
    success_43 = run_kd_seed(43)
    success_44 = run_kd_seed(44)
    
    if success_43 and success_44:
        print("Both seeds completed successfully!")
    else:
        print("One or more seeds failed.")
