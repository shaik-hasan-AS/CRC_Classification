import os
import sys
import time
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.medlite_crc import build_model, LearnableStainNorm, LearnableHEDStainNorm
from data.dataset import get_train_val_loaders, get_crossval_loader
from utils.metrics import compute_metrics

import torch.multiprocessing as mp

def main():
    # Set multiprocessing start method to spawn to avoid CUDA fork deadlocks
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Load config from Ablation 3
    config_path = "configs/ablation/config_3_multiscale.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # We must use GPU for QAT training, then CPU for conversion/benchmarking
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training on device: {device}")
    
    # 1. Build Model
    print("\n[MODEL] Building MedLite-CRC (Ablation 3 configuration)...")
    model = build_model(cfg)
    
    # Load Best Checkpoint
    checkpoint_path = "outputs/checkpoints_ablation_multiscale/ckpt_epoch195_acc0.9946.pt"
    print(f"[MODEL] Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    
    model.train() # QAT requires training mode
    
    # 2. Prepare for QAT using FX Graph Mode
    print("\n[QAT] Preparing FX Graph for Quantization-Aware Training...")
    qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    
    # Exclude custom stain normalization layers from quantization to ensure stability
    qconfig_mapping.set_object_type(LearnableStainNorm, None)
    qconfig_mapping.set_object_type(LearnableHEDStainNorm, None)
    
    example_inputs = (torch.randn(1, 3, 224, 224),)
    
    # Trace and prepare QAT model
    model_prepared = prepare_qat_fx(copy.deepcopy(model), qconfig_mapping, example_inputs)
    model_prepared = model_prepared.to(device)
    
    # Override batch size to 32
    cfg["training"]["batch_size"] = 32
    cfg["training"]["val_batch_size"] = 32
    cfg["data"]["num_workers"] = 8
    print("\n[DATA] Loading NCT-100K train and val data loaders (batch_size=32, num_workers=8)...")
    raw_train_loader, raw_val_loader = get_train_val_loaders(cfg)
    
    # Subset for fast QAT calibration
    from torch.utils.data import Subset
    print("[DATA] Subsetting train loader to 10,000 images and val loader to 2,000 images...")
    train_subset = Subset(raw_train_loader.dataset, list(range(10000)))
    val_subset = Subset(raw_val_loader.dataset, list(range(2000)))
    
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=cfg["data"]["pin_memory"],
        persistent_workers=True
    )
    
    print("[DATA] Loading CRC-VAL-HE-7K test loader...")
    test_loader = get_crossval_loader(cfg)
    
    # 4. Training configuration
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 1
    print(f"\n[TRAIN] Fine-tuning for {epochs} epochs of QAT...")
    
    for epoch in range(epochs):
        model_prepared.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        t0 = time.time()
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model_prepared(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            if step % 200 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        epoch_time = time.time() - t0
        print(f"[EPOCH {epoch+1} DONE] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Time: {epoch_time:.1f}s")
        
        # Evaluate prepared model on val
        model_prepared.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model_prepared(imgs)
                preds = logits.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        print(f"  -> Validation Acc (Fake Quantized): {val_correct/val_total*100:.2f}%")
        
    # 5. Convert to fully quantized INT8 model on CPU
    print("\n[CONVERSION] Converting QAT model to quantized INT8 (CPU)...")
    model_prepared.eval()
    model_prepared = model_prepared.to("cpu")
    model_int8 = convert_fx(model_prepared)
    
    # Save quantized model
    os.makedirs("outputs/", exist_ok=True)
    quant_path = "outputs/medlite_qat_int8.pt"
    traced_int8 = torch.jit.trace(model_int8, example_inputs)
    torch.jit.save(traced_int8, quant_path)
    
    orig_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    quant_size = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"✓ Saved quantized model to {quant_path}")
    print(f"  - FP32 Size: {orig_size:.2f} MB")
    print(f"  - INT8 Size: {quant_size:.2f} MB")
    print(f"  - Compression: {orig_size/quant_size:.2f}x")
    
    # 6. Evaluate final INT8 model on CRC-VAL-HE-7K test set
    print("\n[EVALUATION] Evaluating INT8 model on CRC-VAL-HE-7K...")
    model_int8.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            # CPU evaluation for INT8
            logits = model_int8(imgs)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            
    metrics = compute_metrics(all_preds, all_labels, cfg["data"]["classes"])
    print(f"  - INT8 Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  - INT8 Test Macro-F1: {metrics['macro_f1']:.4f}")
    
    # 7. Benchmark Latency
    print("\n[BENCHMARK] Benchmarking CPU Latency (Batch Size 1)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # FP32 Latency
    model_fp32 = model.to("cpu")
    model_fp32.eval()
    with torch.no_grad():
        for _ in range(10): model_fp32(dummy_input)
        t_start = time.time()
        for _ in range(100): model_fp32(dummy_input)
        fp32_latency = (time.time() - t_start) * 1000 / 100
        
    # INT8 Latency
    with torch.no_grad():
        for _ in range(10): model_int8(dummy_input)
        t_start = time.time()
        for _ in range(100): model_int8(dummy_input)
        int8_latency = (time.time() - t_start) * 1000 / 100
        
    print(f"  - FP32 CPU Latency: {fp32_latency:.2f} ms")
    print(f"  - INT8 CPU Latency: {int8_latency:.2f} ms")
    print(f"  - Speedup: {fp32_latency/int8_latency:.2f}x")

if __name__ == "__main__":
    main()
