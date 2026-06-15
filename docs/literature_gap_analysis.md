# MedLite-CRC: Literature Review & Gap Analysis (Last 4-5 Years)

Based on a web search of recent literature concerning Colorectal Cancer (CRC) histopathology classification, lightweight CNNs, and edge deployment, here is a breakdown of what the field is currently doing and where **MedLite-CRC** might be lacking or vulnerable during peer review.

---

## 1. Edge Hardware & Deployment Benchmarks
* **What the field is doing:** Recent papers focusing on "edge deployment" rarely stop at measuring standard CPU/GPU inference time. They explicitly deploy and benchmark models on physical edge devices like the **NVIDIA Jetson Nano**, **Google Coral Edge TPU**, or **Raspberry Pi**. Furthermore, they almost universally apply **Post-Training Quantization (PTQ)** (e.g., FP32 to INT8) or **Pruning** to maximize hardware efficiency.
* **What MedLite-CRC is lacking:**
  * We currently only report CPU latency (`12.72 ms/image`) and model size (`2.05 MB`).
  * **Vulnerability:** Reviewers at edge-focused venues (like IEEE JBHI) may ask for INT8 quantized metrics or actual on-device latency/power consumption metrics (e.g., on a Jetson Nano). 
  * **Potential Fix:** It would be highly beneficial to add a section showing the model's metrics after INT8 quantization, even if tested on a simulated edge environment.

## 2. Handling Domain Shift & Stain Variations
* **What the field is doing:** To combat scanner domain shift, standard practices include explicit **Stain Normalization** (e.g., Macenko or Vahadane methods) before feeding images into the CNN, or using complex Domain Generalization/Transfer Learning schemes.
* **What MedLite-CRC is doing:** We are using a "Learnable Stain Norm" and arguing that *per-cohort training* with *Dataset Scale as a Regularizer* is the scientifically honest approach.
* **What MedLite-CRC is lacking:**
  * **Vulnerability:** We haven't compared our "Learnable Stain Norm" against a traditional pre-processing baseline (like applying Macenko normalization to MobileNetV2). Reviewers might ask: *"Is your custom architecture better, or would MobileNetV2 with Macenko preprocessing achieve the same thing?"*
  * **Potential Fix:** Add a quick ablation showing that standard networks + Macenko are either too slow (Macenko adds heavy preprocessing latency) or less accurate than MedLite-CRC's built-in learnable norm.

## 3. Knowledge Distillation & Modern Architectures
* **What the field is doing:** While MobileNetV2 and EfficientNet-B0 are great baselines, recent edge-AI papers often employ **Knowledge Distillation** (using a heavy teacher like ResNet-50 to train the lightweight student). Additionally, lightweight Vision Transformers (like **MobileViT**) are becoming the new baseline standard.
* **What MedLite-CRC is lacking:**
  * **Vulnerability:** We are comparing a custom CNN against older CNN baselines. Reviewers might ask why we didn't use Knowledge Distillation to boost our accuracy further, or how we compare to lightweight ViTs.
  * **Potential Fix:** We can either explicitly state in the paper that our methodology avoids Knowledge Distillation to maintain a true "train-from-scratch" lightweight paradigm, or quickly benchmark against a tiny ViT to prove CNNs are still superior for this specific local-texture task.

## 4. Interpretability (Grad-CAM)
* **What the field is doing:** Using Grad-CAM to explain predictions is an absolute requirement in medical AI papers today. 
* **What MedLite-CRC is lacking:**
  * As you already correctly identified in `target.md` (Blocker #4), we need Grad-CAM on **failure cases**. The literature heavily emphasizes understanding *why* AI makes mistakes to build clinical trust. Your current plan to include this is spot on and strictly necessary.

---

### Summary Recommendation for the Manuscript
The MedLite-CRC narrative ("Per-Cohort Training" and "Data Scale as Regularizer") is very strong and unique. However, to bulletproof the "Edge Deployment" claim for a journal like *Computers in Biology & Medicine*, the most critical missing piece is **INT8 Quantization** or **physical edge device metrics**. If you can show that MedLite-CRC retains its accuracy when quantized to INT8 and runs natively on an edge accelerator, the paper will be exceptionally strong.
