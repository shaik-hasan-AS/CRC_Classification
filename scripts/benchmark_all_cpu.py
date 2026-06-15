import time
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights, MobileNet_V2_Weights, ShuffleNet_V2_X1_0_Weights, EfficientNet_B0_Weights
from models.medlite_crc import build_model
import yaml

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")

medlite = build_model(cfg).to(device)
resnet = models.resnet50(weights=None).to(device)
mobilenet = models.mobilenet_v2(weights=None).to(device)
shufflenet = models.shufflenet_v2_x1_0(weights=None).to(device)
efficientnet = models.efficientnet_b0(weights=None).to(device)

def measure(model, name):
    model.eval()
    x = torch.randn(1, 3, 224, 224).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)
            
    # measure
    times = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    
    avg_latency = sum(times) / len(times)
    print(f"{name:15s}: {avg_latency:.2f} ms")

print("CPU Latency (Batch Size 1)")
measure(medlite, "MedLite-CRC")
measure(resnet, "ResNet50")
measure(efficientnet, "EfficientNet-B0")
measure(mobilenet, "MobileNetV2")
measure(shufflenet, "ShuffleNetV2")

