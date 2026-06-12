import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.medlite_crc import build_model
from data.dataset import EvaluationHybridDataset, HYBRID_CLASSES
import yaml

# Load config
with open("configs/hybrid_eval.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)

# Load best weights
ckpt = torch.load("outputs/checkpoints_hybrid11/ckpt_epoch014_acc0.9976.pt", map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# We hook onto the final residual block's 2nd convolution
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

# Use EvaluationHybridDataset to map 9 folders to 11 classes correctly
val_dataset = EvaluationHybridDataset(cfg["data"]["nct_crc_val_dir"], transform=transform)

# We want to plot the classes present in CRC-VAL-HE-7K
classes_to_plot = ["LYM", "STR", "NORM", "ADI", "BACK", "TUM"]
fig, axes = plt.subplots(len(classes_to_plot), 2, figsize=(8, 4 * len(classes_to_plot)))
plt.suptitle("Hybrid 11-Class Model Analysis: Grad-CAM on CRC-VAL-HE-7K", fontsize=16)

for row, cls_name in enumerate(classes_to_plot):
    cls_idx = HYBRID_CLASSES.index(cls_name)
    
    # Find all indices for this class
    indices = [i for i, (_, label) in enumerate(val_dataset.samples) if label == cls_idx]
    if not indices:
        print(f"Skipping {cls_name}, no samples found.")
        continue
        
    random_idx = random.choice(indices)
    
    img_tensor, label = val_dataset[random_idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Get original image for plotting
    rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Overlay
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    axes[row, 0].imshow(rgb_img)
    axes[row, 0].set_title(f"Original: {cls_name}")
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(cam_image)
    axes[row, 1].set_title(f"GradCAM Heatmap")
    axes[row, 1].axis('off')

plt.tight_layout()
os.makedirs("outputs/gradcam_hybrid11", exist_ok=True)
save_path = "outputs/gradcam_hybrid11/hybrid11_gradcam_analysis.png"
plt.savefig(save_path, dpi=150)
print(f"Saved {save_path}")
