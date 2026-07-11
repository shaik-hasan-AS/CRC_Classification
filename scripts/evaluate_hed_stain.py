import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import build_model
from data.dataset import get_crossval_loader
from utils.metrics import compute_metrics

def main():
    config_path = "configs/hed_stain_exp.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build model
    model = build_model(cfg)
    
    # Load best checkpoint
    import glob
    ckpt_list = glob.glob("outputs/checkpoints_hed_stain/ckpt_epoch*.pt")
    if not ckpt_list:
        print("Error: No checkpoints found in outputs/checkpoints_hed_stain/")
        return
    ckpt_path = max(ckpt_list, key=os.path.getmtime)
    print(f"Loading weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    
    # Get test loader
    # Override num_workers to 0 to avoid multiprocessing overhead during inference
    cfg["data"]["num_workers"] = 0
    test_loader = get_crossval_loader(cfg)
    if test_loader is None:
        print("Error: Could not load crossval loader.")
        return
        
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    metrics = compute_metrics(all_preds, all_labels, cfg["data"]["classes"])
    print("\n" + "="*50)
    print("HED Stain Model Evaluation Results on CRC-VAL-HE-7K:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
