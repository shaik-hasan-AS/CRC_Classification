import os
import time
import yaml
import torch
import copy
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

from models.medlite_crc import build_model
from utils.metrics import compute_metrics

# Load config
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu") # Quantization is meant for CPU inference
print(f"Using device: {device}")

# Load the best 175 epoch model
model = build_model(cfg)
ckpt = torch.load("outputs/checkpoints/ckpt_epoch175_acc0.9984.pt", map_location="cpu", weights_only=False)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()

# Original size
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

orig_path = "outputs/medlite_fp32.pt"
torch.save(model.state_dict(), orig_path)
orig_size = get_size_mb(orig_path)
print(f"\n[INFO] Original FP32 Model Size: {orig_size:.2f} MB")

# Define calibration dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg["data"]["augmentation"]["normalize_mean"], 
                         std=cfg["data"]["augmentation"]["normalize_std"])
])
val_dataset = datasets.ImageFolder(cfg["data"]["nct_crc_val_dir"], transform=transform)
calib_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# ── FX Graph Mode Static Quantization ──
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

print("[INFO] Preparing FX Graph for Static Quantization...")
qconfig_mapping = get_default_qconfig_mapping("fbgemm")
example_inputs = (torch.randn(1, 3, 224, 224),)

try:
    model_prepared = prepare_fx(copy.deepcopy(model), qconfig_mapping, example_inputs)
    
    print("[INFO] Calibrating on validation data (this may take a minute)...")
    with torch.no_grad():
        for i, (imgs, _) in enumerate(calib_loader):
            model_prepared(imgs)
            if i >= 10: # Calibrate on ~320 images
                break
                
    print("[INFO] Converting to INT8...")
    model_int8 = convert_fx(model_prepared)
    
    quant_path = "outputs/medlite_int8.pt"
    # Save quantized model
    traced_int8 = torch.jit.trace(model_int8, example_inputs)
    torch.jit.save(traced_int8, quant_path)
    
    quant_size = get_size_mb(quant_path)
    print(f"[INFO] Quantized INT8 Model Size: {quant_size:.2f} MB")
    print(f"[INFO] Size Reduction: {orig_size/quant_size:.1f}x")
    
except Exception as e:
    print(f"\n[WARN] FX Quantization failed: {e}")
    print("[INFO] Falling back to Dynamic Quantization...")
    # Dynamic Quantization shrinks the weights on disk
    model_int8 = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )
    quant_path = "outputs/medlite_int8_dynamic.pt"
    torch.save(model_int8.state_dict(), quant_path)
    quant_size = get_size_mb(quant_path)
    print(f"[INFO] Dynamic INT8 Model Size: {quant_size:.2f} MB")
    print(f"[INFO] Size Reduction: {orig_size/quant_size:.1f}x")

# Benchmarking CPU Latency
print("\n[INFO] Benchmarking CPU Latency (Batch Size 1)...")
dummy_input = torch.randn(1, 3, 224, 224)

def benchmark(net, name):
    net.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            net(dummy_input)
            
    # Measure
    start = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            net(dummy_input)
    total_time = (time.time() - start) * 1000 # ms
    avg_latency = total_time / iters
    print(f"  {name} Latency: {avg_latency:.2f} ms")
    return avg_latency

fp32_latency = benchmark(model, "FP32 MedLite-CRC")
int8_latency = benchmark(model_int8, "INT8 MedLite-CRC")

if int8_latency < fp32_latency:
    print(f"\n[SUCCESS] Speedup: {fp32_latency/int8_latency:.2f}x faster!")
else:
    print(f"\n[NOTE] INT8 speedups depend heavily on the specific CPU architecture and PyTorch backend (e.g. FBGEMM).")
