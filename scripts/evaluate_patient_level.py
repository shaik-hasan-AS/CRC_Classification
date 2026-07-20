import os
import re
import yaml
import torch
import numpy as np
from collections import defaultdict
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import sys

# Add the parent directory to Python path to import MedLiteCRC and transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.medlite_crc import build_model
from data.transforms import get_val_transforms
from utils.metrics import load_checkpoint

def main():
    print("Loading config...")
    with open("configs/kather_finetune.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Revert to standard CrossEntropy loss settings just in case
    cfg["training"]["loss"] = "ce"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building model...")
    model = build_model(cfg).to(device)
    
    ckpt_path = "outputs/checkpoints_kather_v2/ckpt_epoch013_acc0.8797.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return
    load_checkpoint(ckpt_path, model)
    model.eval()

    print("Loading test dataset...")
    test_dir = cfg["data"]["test_dir"]
    val_transforms = get_val_transforms(cfg)
    dataset = ImageFolder(test_dir, transform=val_transforms)
    
    # shuffle=False is critical to match DataLoader output to dataset.samples
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    all_probs = []
    all_labels = []

    print("Running patch-level inference (this may take a few minutes)...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
            if len(all_probs) % 10000 < 256:
                print(f"  Processed {len(all_probs)} / {len(dataset)} patches")

    print("Grouping predictions by patient ID...")
    patient_probs = defaultdict(list)
    patient_labels = {}
    
    missing_ids = 0
    
    for i, (path, label) in enumerate(dataset.samples):
        # Extract TCGA patient barcode (e.g., TCGA-AZ-4615)
        match = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', path)
        if match:
            patient_id = match.group(1)
            patient_probs[patient_id].append(all_probs[i])
            patient_labels[patient_id] = label
        else:
            missing_ids += 1

    if missing_ids > 0:
        print(f"Warning: Could not extract patient ID for {missing_ids} patches.")

    print(f"Aggregating predictions for {len(patient_probs)} unique patients...")
    
    patient_preds_list = []
    patient_trues_list = []
    
    for pid, probs_list in patient_probs.items():
        # Average probability across all patches for the patient
        avg_probs = np.mean(probs_list, axis=0)
        pred_class = np.argmax(avg_probs)
        
        patient_preds_list.append(pred_class)
        patient_trues_list.append(patient_labels[pid])
        
    # Patch-level metrics
    patch_preds = np.argmax(all_probs, axis=1)
    patch_acc = accuracy_score(all_labels, patch_preds)
    
    # Patient-level metrics
    patient_acc = accuracy_score(patient_trues_list, patient_preds_list)
    
    print("\n" + "="*50)
    print("      KATHER MSI/MSS TTA AGGREGATION RESULTS      ")
    print("="*50)
    print(f"Total Patches Evaluated : {len(dataset)}")
    print(f"Total Patients Found    : {len(patient_probs)}")
    print("-" * 50)
    print(f"Patch-Level Accuracy    : {patch_acc * 100:.2f}%")
    print(f"Patient-Level Accuracy  : {patient_acc * 100:.2f}%")
    print("="*50)
    
    print("\nPatient-Level Classification Report:")
    print(classification_report(patient_trues_list, patient_preds_list, target_names=dataset.classes, digits=4))

if __name__ == "__main__":
    main()
