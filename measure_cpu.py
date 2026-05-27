import time
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.medlite_crc import build_model
import yaml

with open("configs/finetune_kd.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Force CPU
device = torch.device("cpu")
print("[CPU] Building model...")
model = build_model(cfg).to(device)

ckpt = torch.load("outputs/checkpoints_kd_v2/ckpt_epoch000_acc0.9971.pt", map_location=device, weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg["data"]["augmentation"]["normalize_mean"], 
                         std=cfg["data"]["augmentation"]["normalize_std"])
])

val_dataset = datasets.ImageFolder(cfg["data"]["nct_crc_val_dir"], transform=transform)
loader = DataLoader(val_dataset, batch_size=1, shuffle=True) # batch size 1 for true latency

print("[CPU] Warming up...")
with torch.no_grad():
    # Warmup
    for i, (img, _) in enumerate(loader):
        if i >= 10: break
        model(img.to(device))

print("[CPU] Measuring latency on 500 images...")
times = []
with torch.no_grad():
    for i, (img, _) in enumerate(loader):
        if i >= 500: break
        img = img.to(device)
        
        t0 = time.time()
        _ = model(img)
        t1 = time.time()
        
        times.append((t1 - t0) * 1000) # Convert to ms

avg_latency = sum(times) / len(times)
print(f"=====================================")
print(f" CPU Inference Latency: {avg_latency:.2f} ms/image")
print(f"=====================================")
