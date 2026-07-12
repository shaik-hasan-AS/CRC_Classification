import os
import sys
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.datasets as datasets

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import build_model
from utils.gradcam import GradCAM, overlay_heatmap
from data.transforms import get_val_transforms

def main():
    # Load V2 config
    config_path = "configs/kd_mobilenet_v2.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build model
    model = build_model(cfg).to(device)
    
    # Load V2 fine-tuned checkpoint
    checkpoint_path = "outputs/checkpoints_kd_v2_v2/ckpt_epoch002_acc0.9935.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    
    # Hook target layer
    target_layer = model.res_blocks[-1].conv2.pw[0]
    gcam = GradCAM(model, target_layer)
    
    # Load dataset
    val_dir = cfg["data"]["nct_crc_val_dir"]
    transform = get_val_transforms(cfg)
    val_dataset = datasets.ImageFolder(val_dir)
    print(f"Loaded {len(val_dataset)} images from {val_dir}")
    
    # Classes to plot in the grid
    classes_to_plot = ["STR", "MUS", "TUM", "LYM"]
    
    fig, axes = plt.subplots(len(classes_to_plot), 2, figsize=(10, 5 * len(classes_to_plot)))
    plt.suptitle("MedLite-CRC (V2) Mitigated Grad-CAM Visualizations", fontsize=18, y=0.98)
    
    # Set seed for reproducible image selection
    random.seed(42)
    
    for row, cls_name in enumerate(classes_to_plot):
        cls_idx = val_dataset.class_to_idx.get(cls_name, -1)
        if cls_idx == -1:
            print(f"Class {cls_name} not found in dataset!")
            continue
            
        # Get all image indices for this class
        indices = [i for i, (_, label) in enumerate(val_dataset.imgs) if label == cls_idx]
        
        # Shuffle indices to find a good representative sample that is predicted correctly
        random.shuffle(indices)
        
        selected_idx = None
        for idx in indices:
            img_path, label = val_dataset.imgs[idx]
            img_pil = Image.open(img_path).convert("RGB")
            tensor = transform(img_pil).to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(tensor.unsqueeze(0))
                pred_class = logits.argmax(dim=1).item()
                
            if pred_class == label:
                selected_idx = idx
                break
                
        if selected_idx is None:
            # Fallback to first image if none are correctly predicted
            selected_idx = indices[0]
            
        img_path, label = val_dataset.imgs[selected_idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        
        tensor = transform(img_pil).to(device)
        
        # Generate CAM with 8px boundary mask
        cam, pred_class, probs = gcam.generate(tensor, target_class=label, mask_border_width=8)
        
        # Create overlay
        overlay = overlay_heatmap(img_pil, cam, alpha=0.45)
        
        # Plot Original
        axes[row, 0].imshow(img_resized)
        axes[row, 0].set_title(f"Original: {cls_name}", fontsize=14)
        axes[row, 0].axis("off")
        
        # Plot Overlay
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title("GradCAM Heatmap", fontsize=14)
        axes[row, 1].axis("off")
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save to both locations
    os.makedirs("outputs/gradcam", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    
    save_path_outputs = "outputs/gradcam/gradcam_results.png"
    save_path_assets = "assets/gradcam_results.png"
    
    plt.savefig(save_path_outputs, dpi=150, bbox_inches="tight")
    plt.savefig(save_path_assets, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully generated and saved Grad-CAM V2 figure:")
    print(f"  - {save_path_outputs}")
    print(f"  - {save_path_assets}")

if __name__ == "__main__":
    main()
