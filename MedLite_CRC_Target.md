# MedLite-CRC — Research Target
> Lightweight Stain-Robust CNN for Colon Histopathology Classification

---

## One-Line Goal
> Design a novel lightweight CNN that classifies colon histopathology tissue with high accuracy **across different hospitals' staining protocols**, deployable on edge hardware.

---

## The Problem We're Solving

Most CNN papers on colon histopathology:
- Hit 99%+ accuracy on benchmark data but **collapse on a different hospital's images**
- Use heavy models (ResNet, EfficientNet ensembles) — not deployable on low-resource hardware
- Never measure real deployment metrics (latency, RAM, FLOPs on actual hardware)

**Our claim:** A purpose-built lightweight architecture with stain-robustness baked in can match heavy models on benchmark data AND generalise where they fail.

---

## Datasets

| Role | Dataset | Size | Source |
|---|---|---|---|
| Train + Val | NCT-CRC-HE-100K | 100,000 patches, 9 classes | [zenodo.org/records/1214456](https://zenodo.org/records/1214456) |
| Cross-patient test | CRC-VAL-HE-7K | 7,180 patches, 50 different patients | Same Zenodo record |
| Cross-hospital test | UniToPatho | 9,536 patches, different staining protocol | [ieee-dataport.org/open-access/unitopatho](https://ieee-dataport.org/open-access/unitopatho) |

**No WSI patching needed** — all images are pre-tiled 224×224 JPEGs. Download and train immediately.

### The 9 Tissue Classes
`ADI` · `BACK` · `DEB` · `LYM` · `MUC` · `MUS` · `NORM` · `STR` · `TUM`

---

## Performance Targets

| Metric | Target | Why |
|---|---|---|
| Accuracy on NCT-CRC-HE-100K | ≥ 98% | Table stakes — baselines already hit 99% |
| Accuracy on CRC-VAL-HE-7K | ≥ 93% | **This is where we beat baselines** |
| Accuracy on UniToPatho | ≥ 88% | Cross-hospital domain shift proof |
| Parameters | < 5M | Lightweight claim |
| Model size | < 50 MB | Edge deployable |
| FLOPs | < 1 GFLOPs | Efficiency claim |
| CPU inference | < 50 ms/image | Real deployment metric |

---

## Our Architecture — MedLite-CRC

```
Input (224×224×3)
  │
  ├─ LearnableStainNorm         ← adapts to different staining protocols
  ├─ Stem Block                 ← Conv + Depthwise Conv
  ├─ MultiScale Branches        ← 3×3 + 5×5 + 7×7 parallel DW-Sep convs
  │    (fine nuclei · glands · coarse stroma)
  ├─ DWResBlock × 3             ← 128ch → 256ch → 256ch with skip connections
  ├─ SE Channel Attention       ← focus on informative feature channels
  ├─ Global Average Pooling
  └─ Classifier Head            ← FC → Dropout(0.4) → FC(9)
```

**Key design decisions:**
- Multi-scale branches capture histopathology at multiple spatial scales simultaneously
- Depthwise separable convolutions throughout → low FLOPs
- Learnable stain normalisation layer → domain shift robustness without external preprocessing
- SE attention → suppresses staining noise, emphasises structural features

---

## What Makes This Publishable

| Contribution | What it means |
|---|---|
| Novel architecture | First lightweight CNN designed ground-up for medical histopathology texture — not repurposed from natural images (unlike MobileNet, ShuffleNet) |
| Cross-dataset proof | Train on NCT-CRC → test on CRC-VAL + UniToPatho → prove generalisation where baselines fail |
| Edge deployment metrics | First CRC lightweight paper to report CPU latency + RAM on real hardware |
| Ablation study | Quantified contribution of each component (stain norm, multi-scale, SE block) |

---

## Baselines to Beat

| Model | Type | Known weakness |
|---|---|---|
| MobileNetV2 | Lightweight | Designed for natural images, drops on cross-dataset |
| EfficientNet-B0 | Lightweight | Cross-dataset generalisation gap |
| ShuffleNetV2 | Ultra-lightweight | Medical texture not captured well |
| Lite-V2 (Hanif, 2025) | Most recent competitor | Near-perfect val accuracy, significant drop on independent test set |
| ResNet-50 | Heavy upper bound | Reference only — not a competition |

---

## Timeline

| Phase | Task | Duration |
|---|---|---|
| 1 | Data pipeline + augmentation setup | Week 1 |
| 2 | Baseline experiments (all 4 baselines) | Week 2–3 |
| 3 | MedLite-CRC architecture development + ablations | Week 3–5 |
| 4 | Full training + cross-dataset validation | Week 5–7 |
| 5 | Edge deployment benchmarking | Week 7–8 |
| 6 | Paper writing + submission | Week 8–10 |

---

## Target Venues

| Venue | Type | Impact Factor |
|---|---|---|
| **Computers in Biology & Medicine** | Journal | ~7.7 |
| **IEEE JBHI** | Journal | ~7.7 |
| **Medical Image Analysis** | Journal | ~10.7 |
| **IEEE Access** | Journal | ~3.9 (fastest turnaround) |
| **MICCAI 2026 Workshops** | Conference | High visibility |

> First submission target: **Computers in Biology & Medicine** (rolling deadline).

---

*MedLite-CRC · VIT Chennai · B.Tech ECM*
