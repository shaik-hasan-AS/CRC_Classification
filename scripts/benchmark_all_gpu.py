import time
import torch
import torchvision.models as models
from models.medlite_crc import build_model
import yaml

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

if not torch.cuda.is_available():
    print("CUDA not available")
    exit()

device = torch.cuda.current_device()

medlite = build_model(cfg).to(device)
resnet = models.resnet50(weights=None).to(device)
mobilenet = models.mobilenet_v2(weights=None).to(device)
shufflenet = models.shufflenet_v2_x1_0(weights=None).to(device)
efficientnet = models.efficientnet_b0(weights=None).to(device)

def measure(model, name, batch_size=64):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    # warmup
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(10):
            model(x)
    torch.cuda.synchronize()
            
    # measure
    times = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(50):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    avg_latency = sum(times) / len(times) / batch_size
    print(f"{name:15s}: {avg_latency:.3f} ms/img")

print("GPU Latency (Batch Size 64)")
measure(medlite, "MedLite-CRC")
measure(resnet, "ResNet50")
measure(efficientnet, "EfficientNet-B0")
measure(mobilenet, "MobileNetV2")
measure(shufflenet, "ShuffleNetV2")

