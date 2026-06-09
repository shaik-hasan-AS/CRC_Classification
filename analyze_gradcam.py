import torch
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets
from pytorch_grad_cam import GradCAM
from models.medlite_crc import build_model
import yaml

def analyze_heatmaps():
    print("="*60)
    print("  CRITICAL GRAD-CAM SPATIAL ANALYSIS")
    print("="*60)
    
    with open("configs/finetune_kd.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    
    ckpt = torch.load("outputs/checkpoints/ckpt_epoch175_acc0.9984.pt", map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    target_layers = [model.res_blocks[2].conv2]
    cam = GradCAM(model=model, target_layers=target_layers)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["data"]["augmentation"]["normalize_mean"], 
                             std=cfg["data"]["augmentation"]["normalize_std"])
    ])

    val_dataset = datasets.ImageFolder(cfg["data"]["nct_crc_val_dir"], transform=transform)
    class_to_idx = val_dataset.class_to_idx

    # We will test a few samples from each class and critically evaluate where the network is looking
    classes_to_test = ["STR", "MUS", "TUM", "LYM"]
    
    for cls_name in classes_to_test:
        cls_idx = class_to_idx[cls_name]
        indices = [i for i, (_, label) in enumerate(val_dataset.samples) if label == cls_idx]
        
        print(f"\nEvaluating Class: {cls_name}")
        
        # Test 5 random images per class
        for i in range(5):
            idx = np.random.choice(indices)
            img_tensor, _ = val_dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Generate CAM (returns 2D array [0, 1] representing activation strength)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            
            # Un-normalize image to get brightness (to find background/white space)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            std = np.array(cfg["data"]["augmentation"]["normalize_std"])
            mean = np.array(cfg["data"]["augmentation"]["normalize_mean"])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            # Calculate brightness (grayscale)
            brightness = np.mean(img_np, axis=2)
            
            # METRIC 1: Center Bias
            # Is the network just lazily looking at the exact center of the image?
            h, w = grayscale_cam.shape
            y, x = np.ogrid[:h, :w]
            center_dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
            
            # Weighted average distance of activations from center
            # Max possible distance is ~158 pixels
            weighted_dist = np.average(center_dist, weights=grayscale_cam)
            
            # METRIC 2: Background Activation
            # In H&E stains, empty space (background/adipose) is bright white (brightness > 0.8)
            # Is the network cheating by looking at empty space?
            background_mask = brightness > 0.85
            background_activation = np.mean(grayscale_cam[background_mask]) if np.any(background_mask) else 0.0
            tissue_activation = np.mean(grayscale_cam[~background_mask]) if np.any(~background_mask) else 0.0
            
            # METRIC 3: Activation Dispersion
            # Is it a single blob (variance low) or looking at complex high-frequency textures (variance high)?
            activation_std = np.std(grayscale_cam)
            
            print(f"  Sample {i+1}:")
            print(f"    - Avg Distance from Center: {weighted_dist:.1f} px (Low = Center Bias)")
            print(f"    - Tissue Activation Avg: {tissue_activation:.3f} | Background Activation Avg: {background_activation:.3f}")
            if background_activation > tissue_activation:
                print(f"      [WARNING] Network is looking at EMPTY WHITE SPACE instead of tissue!")
            
            if weighted_dist < 30.0:
                print(f"      [WARNING] Severe Center Bias detected! The network is ignoring the edges.")

if __name__ == "__main__":
    analyze_heatmaps()
