import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.medlite_crc import build_model
from data.dataset import CRC_CLASSES
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)

# Load V1 best weights
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

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(cfg["data"]["augmentation"]["normalize_mean"], cfg["data"]["augmentation"]["normalize_std"])],
    std=[1/s for s in cfg["data"]["augmentation"]["normalize_std"]]
)

val_dataset = datasets.ImageFolder(cfg["data"]["nct_crc_val_dir"], transform=transform)
loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print("[INFO] Finding failure cases...")
failure_cases = []

with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu()
        labels = labels.cpu()
        
        for i in range(len(labels)):
            if labels[i].item() != preds[i].item():
                failure_cases.append({
                    "img_tensor": imgs[i].cpu(),
                    "true_label": labels[i].item(),
                    "pred_label": preds[i].item()
                })
        
        # Stop early if we have enough failures to plot
        if len(failure_cases) > 50:
            break

print(f"[INFO] Found {len(failure_cases)} failures in the scanned batches.")

# Select a random subset to plot
random.shuffle(failure_cases)
plot_cases = failure_cases[:8]

fig, axes = plt.subplots(len(plot_cases), 2, figsize=(8, 4 * len(plot_cases)))
plt.suptitle("Grad-CAM on Failure Cases (CRC-VAL-HE-7K)", fontsize=16)

for row, case in enumerate(plot_cases):
    img_tensor = case["img_tensor"]
    true_cls = CRC_CLASSES[case["true_label"]]
    pred_cls = CRC_CLASSES[case["pred_label"]]
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Original image for plotting
    rgb_img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Overlay
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    axes[row, 0].imshow(rgb_img)
    axes[row, 0].set_title(f"True: {true_cls} | Pred: {pred_cls}")
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(cam_image)
    axes[row, 1].set_title(f"GradCAM Heatmap")
    axes[row, 1].axis('off')

plt.tight_layout()
os.makedirs("outputs/gradcam", exist_ok=True)
save_path = "outputs/gradcam/failure_cases_gradcam.png"
plt.savefig(save_path, dpi=150)
print(f"[SAVED] {save_path}")
