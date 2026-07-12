import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import torchvision.datasets as datasets

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import build_model
from utils.gradcam import GradCAM
from data.transforms import get_val_transforms

def main():
    # Load config
    config_path = "configs/kd_mobilenet_teacher.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build and load model
    model = build_model(cfg).to(device)
    checkpoint_path = "outputs/checkpoints_kd_mobilenet/ckpt_epoch058_acc0.9946.pt"
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    
    # Hook target layer for GradCAM
    target_layer = model.res_blocks[-1].conv2.pw[0]
    gcam = GradCAM(model, target_layer)
    
    # Load dataset
    val_dir = cfg["data"]["nct_crc_val_dir"]
    transform = get_val_transforms(cfg)
    val_dataset = datasets.ImageFolder(val_dir)
    print(f"Loaded {len(val_dataset)} images from {val_dir}")
    
    # Setup coordinates for Center of Mass
    y_grid, x_grid = np.meshgrid(np.arange(224), np.arange(224), indexing='ij')
    
    distances = []
    vanishing_gradients = 0
    correct_predictions = 0
    total_images = len(val_dataset)
    
    # For Negative Space Shortcut (Stroma/STR class is index 7)
    # Let's find index of 'STR'
    str_idx = val_dataset.class_to_idx.get("STR", -1)
    print(f"Stroma class index: {str_idx}")
    
    stroma_bg_activations = []
    stroma_tissue_activations = []
    
    import random
    # To run quickly, let's analyze 1000 random samples
    random.seed(42)
    sample_indices = random.sample(range(total_images), min(1000, total_images))
    print(f"Analyzing {len(sample_indices)} random validation samples...")
    
    for count, idx in enumerate(sample_indices):
        img_path, label = val_dataset.imgs[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_resized = img_pil.resize((224, 224))
        
        # Convert to tensor and predict
        tensor = transform(img_pil).to(device)
        
        try:
            cam, pred_class, probs = gcam.generate(tensor, target_class=label)
            is_correct = (pred_class == label)
            
            # 1. Vanishing Gradient
            sum_cam = cam.sum()
            if sum_cam < 1e-8:
                if is_correct:
                    vanishing_gradients += 1
                continue
                
            if is_correct:
                correct_predictions += 1
                
            # 2. Center of Mass
            x_com = (cam * x_grid).sum() / sum_cam
            y_com = (cam * y_grid).sum() / sum_cam
            dist = np.sqrt((x_com - 112.0)**2 + (y_com - 112.0)**2)
            distances.append(dist)
            
            # 3. Negative Space Shortcut (Specifically for Stroma class)
            if label == str_idx:
                img_np = np.array(img_resized) / 255.0
                # slide background is white/bright (average channel value > 0.85)
                brightness = img_np.mean(axis=2)
                bg_mask = brightness > 0.85
                tissue_mask = ~bg_mask
                
                if bg_mask.any():
                    stroma_bg_activations.append(cam[bg_mask].mean())
                if tissue_mask.any():
                    stroma_tissue_activations.append(cam[tissue_mask].mean())
                    
        except Exception as e:
            # Catch potential exceptions
            pass
            
        if (count + 1) % 200 == 0:
            print(f"Processed {count + 1}/1000 samples...")
            
    print("\n" + "="*50)
    print("Grad-CAM Spatial Interpretability Results (SOTA KD model):")
    print(f"Average radial distance from center: {np.mean(distances):.2f} pixels (max possible: 158.4)")
    print(f"Vanishing gradient rate (empty CAMs for correct predictions): {vanishing_gradients / (correct_predictions + vanishing_gradients) * 100:.2f}%")
    if stroma_bg_activations and stroma_tissue_activations:
        print(f"Stroma Background Mean Activation: {np.mean(stroma_bg_activations):.4f}")
        print(f"Stroma Tissue Mean Activation: {np.mean(stroma_tissue_activations):.4f}")
        bg_ratio = np.mean(stroma_bg_activations) > np.mean(stroma_tissue_activations)
        print(f"Is background activation higher than tissue? {bg_ratio}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
