import torch
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from models.medlite_crc import build_model
from data.dataset import EvaluationHybridDataset, HYBRID_CLASSES
import yaml

# Load config
with open("configs/hybrid_eval.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)

ckpt = torch.load("outputs/checkpoints_hybrid11/ckpt_epoch014_acc0.9976.pt", map_location=device, weights_only=False)
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

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(cfg["data"]["augmentation"]["normalize_mean"], cfg["data"]["augmentation"]["normalize_std"])],
    std=[1/s for s in cfg["data"]["augmentation"]["normalize_std"]]
)

val_dataset = EvaluationHybridDataset(cfg["data"]["nct_crc_val_dir"], transform=transform)

classes_to_test = ["LYM", "STR", "NORM", "TUM", "DEB", "MUS"]
num_samples = 20

print("======================================================")
print(" Grad-CAM Structural Alignment Analysis")
print(" (What % of 'Heat' is actually on tissue vs background?)")
print("======================================================")

for cls_name in classes_to_test:
    cls_idx = HYBRID_CLASSES.index(cls_name)
    indices = [i for i, (_, label) in enumerate(val_dataset.samples) if label == cls_idx]
    
    if not indices:
        continue
        
    random_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
    
    total_tissue_heat_ratio = 0.0
    
    for idx in random_indices:
        img_tensor, _ = val_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Original RGB (0 to 1)
        rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # Create a simple tissue mask (White background is usually > 0.85 in all channels)
        # 1.0 = Tissue, 0.0 = Background
        grayscale_img = np.mean(rgb_img, axis=-1)
        tissue_mask = (grayscale_img < 0.85).astype(np.float32) 
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Get the top 20% hottest pixels in the CAM
        threshold = np.percentile(grayscale_cam, 80)
        hot_mask = (grayscale_cam >= threshold).astype(np.float32)
        
        # Calculate how much of the hot_mask falls inside the tissue_mask
        hot_pixels_total = np.sum(hot_mask)
        hot_pixels_on_tissue = np.sum(hot_mask * tissue_mask)
        
        if hot_pixels_total > 0:
            ratio = hot_pixels_on_tissue / hot_pixels_total
            total_tissue_heat_ratio += ratio
            
    avg_ratio = (total_tissue_heat_ratio / len(random_indices)) * 100
    print(f"[{cls_name}] Heatmap strictly on Tissue: {avg_ratio:.1f}%")
