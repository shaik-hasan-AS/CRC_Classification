import os
import torch

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np


from data.dataset import CRC_CLASSES
from models.medlite_crc import build_model
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")
model = build_model(cfg).to(device)

# Load V1 best weights
ckpt = torch.load("outputs/checkpoints/ckpt_epoch175_acc0.9984.pt", map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# Val transforms (No augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg["data"]["augmentation"]["normalize_mean"], 
                         std=cfg["data"]["augmentation"]["normalize_std"])
])

val_dataset = datasets.ImageFolder(cfg["data"]["nct_crc_val_dir"], transform=transform)
loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

str_idx = CRC_CLASSES.index("STR")
mus_idx = CRC_CLASSES.index("MUS")

str_mus_confusions = [] # True STR, Pred MUS
mus_str_confusions = [] # True MUS, Pred STR

# To easily get original unnormalized images for visualization
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(cfg["data"]["augmentation"]["normalize_mean"], cfg["data"]["augmentation"]["normalize_std"])],
    std=[1/s for s in cfg["data"]["augmentation"]["normalize_std"]]
)

with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu()
        labels = labels.cpu()
        
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = preds[i].item()
            if true_label == str_idx and pred_label == mus_idx:
                str_mus_confusions.append(imgs[i].cpu())
            elif true_label == mus_idx and pred_label == str_idx:
                mus_str_confusions.append(imgs[i].cpu())

print(f"Total True STR -> Pred MUS: {len(str_mus_confusions)}")
print(f"Total True MUS -> Pred STR: {len(mus_str_confusions)}")

# Save a grid of confusions
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
plt.suptitle("False Positives: Top 2 rows = True STR/Pred MUS | Bottom 2 = True MUS/Pred STR", fontsize=16)

confused_imgs = str_mus_confusions[:8] + mus_str_confusions[:8]
for i, ax in enumerate(axes.flatten()):
    if i < len(confused_imgs):
        img = inv_normalize(confused_imgs[i])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        title = "True: STR | Pred: MUS" if i < 8 else "True: MUS | Pred: STR"
        ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/str_vs_mus_confusions.png", dpi=150)
print("Saved outputs/str_vs_mus_confusions.png")
