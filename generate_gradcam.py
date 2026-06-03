import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision.datasets as datasets

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.medlite_crc import build_model
import yaml

# Load config
with open("configs/finetune_kd.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)

# Load best KD weights
ckpt = torch.load("outputs/checkpoints_kd_v2/ckpt_epoch000_acc0.9971.pt", map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# We hook onto the final residual block's 2nd convolution
# In our MedLiteCRC model: model.res_blocks[2].conv2
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

val_dataset = datasets.ImageFolder(cfg["data"]["nct_crc_val_dir"], transform=transform)

classes_to_plot = ["STR", "MUS", "TUM", "LYM"]
fig, axes = plt.subplots(len(classes_to_plot), 2, figsize=(8, 4 * len(classes_to_plot)))
plt.suptitle("MedLite-CRC (V1+KD) GradCAM Visualizations", fontsize=16)

class_to_idx = val_dataset.class_to_idx

for row, cls_name in enumerate(classes_to_plot):
    cls_idx = class_to_idx[cls_name]
    
    # Find all indices for this class
    indices = [i for i, (_, label) in enumerate(val_dataset.samples) if label == cls_idx]
    random_idx = random.choice(indices)
    
    img_tensor, label = val_dataset[random_idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Get original image for plotting
    rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    
    # Generate CAM
    # We pass None for targets so it uses the highest scoring category
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
os.makedirs("/home/hasan/.gemini/antigravity/brain/d5b61fe8-871c-4985-a5fd-4aa2ebe2db57/artifacts", exist_ok=True)
plt.savefig("/home/hasan/.gemini/antigravity/brain/d5b61fe8-871c-4985-a5fd-4aa2ebe2db57/artifacts/gradcam_results.png", dpi=150)
print("Saved gradcam_results.png")
