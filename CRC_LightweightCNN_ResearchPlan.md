# Lightweight CNN for Stain-Robust Colon Histopathology Classification
### Research Plan & Implementation Roadmap
> **VIT Chennai | B.Tech ECM | Target: IEEE JBHI / Computers in Biology & Medicine**

---

## 1. Problem Statement

Most high-performing CNNs for colon histopathology (ResNet, EfficientNet, Swin Transformer ensembles) achieve >99% accuracy on benchmark datasets but are:

- **Computationally heavy** — hundreds of MBs, billions of FLOPs
- **Stain-sensitive** — trained on one hospital's staining protocol, they fail on another's
- **Not deployed** — no paper measures actual latency, RAM, or energy on edge hardware

> **The core problem:** Accurate colon cancer screening models exist in research, but none are lightweight enough AND stain-robust enough to deploy in resource-constrained clinics.

---

## 2. Research Gap (Why This Is Publishable)

| What's Saturated | What's Open |
|---|---|
| >99% accuracy on NCT-CRC-HE-100K | Cross-dataset generalisation (different hospital staining) |
| Heavy models (ResNet, EfficientNet ensembles) | Lightweight architecture designed for medical texture |
| Binary cancer/non-cancer classification | 9-class multi-tissue classification at edge scale |
| Accuracy-only benchmarks | FLOPs + latency + RAM on real edge hardware |

**Key evidence of gap:** Lite-V2 (2025) — the most recent lightweight CRC model — still showed significant accuracy drop on independent test sets despite near-perfect validation performance. No paper has closed this cross-domain gap with a lightweight model.

---

## 3. Research Objective

> Design and validate a novel lightweight CNN architecture for colon histopathology classification that:
> 1. Achieves ≥98% accuracy on NCT-CRC-HE-100K (9-class)
> 2. Achieves ≥93% accuracy on CRC-VAL-HE-7K (cross-dataset, different patients)
> 3. Maintains competitive performance on UniToPatho (different hospital, polyp grading)
> 4. Operates under **<5M parameters**, **<50MB model size**, **<1 GFLOPs**
> 5. Demonstrates deployability with real inference metrics on CPU/edge hardware

---

## 4. Datasets

### Primary Training & Validation
| Dataset | Description | Access |
|---|---|---|
| **NCT-CRC-HE-100K** | 100,000 image patches, 224×224, 9 tissue classes, 86 patients | [zenodo.org/records/1214456](https://zenodo.org/records/1214456) |
| **CRC-VAL-HE-7K** | 7,180 patches from 50 *different* patients — same Zenodo record | Same link above |

### Cross-Domain Validation (Novelty Proof)
| Dataset | Description | Access |
|---|---|---|
| **UniToPatho** | 9,536 patches, colorectal polyp classification, different hospital protocol | [IEEE DataPort](https://ieee-dataport.org/open-access/unitopatho) |

### Class Labels (NCT-CRC-HE-100K)
1. ADI — Adipose
2. BACK — Background
3. DEB — Debris
4. LYM — Lymphocytes
5. MUC — Mucus
6. MUS — Smooth Muscle
7. NORM — Normal Colon Mucosa
8. STR — Cancer-Associated Stroma
9. TUM — Colorectal Adenocarcinoma Epithelium

> No WSI patching needed. All images are pre-tiled 224×224 JPEGs. Download and train immediately.

---

## 5. Proposed Architecture — MedLite-CRC

A novel lightweight CNN designed ground-up for medical histopathology texture, **not** repurposed from natural image models (unlike MobileNet/ShuffleNet).

### 5.1 Core Design Principles
- **Medical-texture-first:** Histopathology features (glandular structures, nuclear pleomorphism, stroma patterns) are spatially local and multi-scale → architecture must capture multi-scale texture efficiently
- **Stain invariance baked in:** Colour jitter + stain normalisation layer at input → model learns texture, not colour
- **Depthwise separable everywhere:** Reduce FLOPs without sacrificing receptive field
- **Lightweight attention:** Channel attention (SE-style) on critical feature maps only — no full transformer blocks

### 5.2 Architecture Overview

```
Input (224×224×3)
        │
┌───────▼────────┐
│  Stain-Norm    │  ← Learnable Macenko-style colour normalisation layer
│  Layer         │
└───────┬────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Stem Block                                    │
│  3×3 Conv (32) → BN → ReLU6                   │
│  3×3 DW Conv (32) → BN → ReLU6                │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Multi-Scale Feature Extraction (3 branches)  │
│  Branch A: 3×3 DW-Separable Conv (64)         │  ← Fine texture (nuclei)
│  Branch B: 5×5 DW-Separable Conv (64)         │  ← Mid-scale (glands)
│  Branch C: 7×7 DW-Separable Conv (64)         │  ← Coarse structure (stroma)
│  → Concatenate → 1×1 PW Conv (128)            │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Residual DW Blocks × 3                        │
│  (Depthwise Separable + Skip Connection)       │
│  Channels: 128 → 256 → 256                    │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Channel-Frequency Attention (SE-style)        │
│  Global Avg Pool → FC(r=16) → ReLU → FC → Sigmoid │
│  → Channel-wise scaling                        │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Global Average Pooling                        │
└───────┬────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────┐
│  Classifier Head                               │
│  FC(256) → Dropout(0.4) → FC(9) → Softmax     │
└────────────────────────────────────────────────┘
```

### 5.3 Target Efficiency Specs

| Metric | Target | Lite-V2 (2025, competitor) |
|---|---|---|
| Parameters | < 5M | 127,682 (~0.13M) |
| Model Size | < 50MB | 1.53MB |
| FLOPs | < 1 GFLOPs | ~0.05 GFLOPs |
| Accuracy (NCT-CRC) | ≥ 98% | ~99.9% (val only) |
| Accuracy (CRC-VAL) | ≥ 93% | Drops significantly |
| Inference (CPU) | < 50ms/image | Not reported |

> **Note:** Lite-V2 is extremely small but overfits to NCT-CRC. Our contribution is **cross-dataset robustness at comparable efficiency**, not just raw compression.

---

## 6. Implementation Plan

### Phase 1 — Data Pipeline (Week 1)
- [ ] Download NCT-CRC-HE-100K + CRC-VAL-HE-7K from Zenodo
- [ ] Download UniToPatho from IEEE DataPort
- [ ] Implement data loader with augmentation pipeline:
  - Random horizontal/vertical flip
  - Random rotation (90°, 180°, 270°)
  - Colour jitter (brightness, contrast, saturation)
  - **Stain augmentation** (Macenko / Vahadane random stain perturbation) ← key for stain robustness
- [ ] Class distribution analysis + confirm no severe imbalance
- [ ] Train/val split: 80/20 on NCT-CRC-HE-100K (stratified)

### Phase 2 — Baseline Experiments (Week 2–3)
Run all baselines on NCT-CRC-HE-100K → evaluate on CRC-VAL-HE-7K:

| Baseline | Purpose |
|---|---|
| MobileNetV2 | Standard lightweight baseline |
| EfficientNet-B0 | Strong lightweight baseline |
| ShuffleNetV2 | Ultra-lightweight baseline |
| Lite-V2 (2025) | Most recent direct competitor |
| ResNet-50 | Heavy upper-bound reference |

- Record: Accuracy, Macro-F1, AUC, Parameters, FLOPs, Inference time (CPU)
- **Expected finding:** All baselines drop ≥5-10% accuracy on CRC-VAL-HE-7K → this is your gap

### Phase 3 — MedLite-CRC Architecture Development (Week 3–5)
- [ ] Implement stem block + multi-scale branches
- [ ] Implement DW residual blocks
- [ ] Implement SE channel attention
- [ ] Implement learnable stain normalisation layer
- [ ] Initial training: 100 epochs, Adam, lr=1e-3, cosine decay
- [ ] Ablation study per component (each module ON/OFF)

### Phase 4 — Full Training & Cross-Dataset Validation (Week 5–7)
- [ ] Final training: 200 epochs on NCT-CRC-HE-100K
- [ ] Evaluate on CRC-VAL-HE-7K (cross-patient)
- [ ] Evaluate on UniToPatho (cross-hospital, cross-protocol)
- [ ] Generate confusion matrices per dataset
- [ ] Grad-CAM visualisations for explainability figures

### Phase 5 — Edge Deployment Benchmarking (Week 7–8)
- [ ] Export model to ONNX → TensorFlow Lite
- [ ] Measure on: Windows CPU (your machine), Raspberry Pi 4 (if available)
- [ ] Record: inference latency (ms), RAM usage (MB), model size (MB)
- [ ] Optional: INT8 quantisation → report accuracy vs. size trade-off

### Phase 6 — Paper Writing (Week 8–10)
- [ ] Abstract + Introduction
- [ ] Related Work (lightweight CNNs, stain normalisation, CRC classification)
- [ ] Methodology (architecture diagrams)
- [ ] Experiments & Results (tables + Grad-CAM figures)
- [ ] Conclusion + Future Work
- [ ] Submission

---

## 7. Paper Structure

```
1. Introduction
   - Clinical context (CRC, global burden)
   - Problem: heavy models, stain sensitivity, no deployment
   - Contribution: MedLite-CRC architecture + cross-dataset proof

2. Related Work
   - Lightweight CNNs for medical imaging
   - Stain normalisation methods
   - CRC histopathology classification survey

3. Proposed Architecture (MedLite-CRC)
   - Design rationale
   - Block-level description
   - Parameter count analysis

4. Experimental Setup
   - Datasets (NCT-CRC, CRC-VAL, UniToPatho)
   - Training configuration
   - Evaluation metrics

5. Results & Discussion
   - Accuracy comparison table (all baselines vs MedLite-CRC)
   - Cross-dataset performance table ← KEY TABLE
   - Efficiency comparison (FLOPs, params, latency)
   - Ablation study
   - Grad-CAM visualisations

6. Conclusion
```

---

## 8. Expected Contributions

1. **Novel architecture:** MedLite-CRC — first lightweight CNN designed specifically for medical histopathology texture with built-in stain normalisation
2. **Cross-dataset validation:** First to systematically benchmark generalisation across NCT-CRC → CRC-VAL → UniToPatho pipeline
3. **Deployment metrics:** First CRC lightweight paper to report CPU inference latency + RAM on edge hardware
4. **Ablation study:** Quantified contribution of each architectural component

---

## 9. Target Venues

| Venue | Type | Impact | Deadline Cadence |
|---|---|---|---|
| **Computers in Biology & Medicine** | Journal | IF ~7.7 | Rolling |
| **IEEE JBHI** | Journal | IF ~7.7 | Rolling |
| **Medical Image Analysis** | Journal | IF ~10.7 | Rolling |
| **IEEE Access** | Journal | IF ~3.9 | Rolling (fastest) |
| **MICCAI 2026 Workshops** | Conference | High visibility | ~March 2026 |

> **Recommendation:** Submit to *Computers in Biology & Medicine* first (rolling, good fit, reasonable review time). Prepare IEEE Access as backup.

---

## 10. Tech Stack

| Component | Tool |
|---|---|
| Framework | PyTorch |
| GPU | RTX 4080 (your machine) |
| Stain Augmentation | `staintools` / `torchstain` |
| Model Export | ONNX + TFLite |
| Visualisation | Grad-CAM via `pytorch-grad-cam` |
| Experiment Tracking | Weights & Biases (wandb) |
| Environment | Python 3.10, CUDA 12.x |

---

*Last updated: May 2026 | VIT Chennai*
