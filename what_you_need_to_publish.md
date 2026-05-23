# What You Actually Need to Publish

## The One Thing Reviewers Care About
> Does your model solve something existing models don't?

Your answer: **Cross-dataset generalisation at lightweight scale.**
Train on NCT-CRC-HE-100K. Test on CRC-VAL-HE-7K (different patients) and UniToPatho (different hospital). Show your model drops less than baselines.

---

## The Tables You Must Have

### Table 1 — Main Accuracy Comparison
Train on NCT-CRC-HE-100K, test on CRC-VAL-HE-7K.

| Model | Params (M) | Accuracy ↑ | Macro-F1 ↑ | Cross-Val Acc ↑ |
|---|---|---|---|---|
| MobileNetV2 | 3.4M | ~99% | ~0.99 | ~87% ← drops here |
| EfficientNet-B0 | 5.3M | ~99% | ~0.99 | ~89% |
| ShuffleNetV2 | 2.3M | ~97% | ~0.97 | ~85% |
| Lite-V2 (2025) | 0.13M | ~99.9% | ~0.99 | drops significantly |
| **MedLite-CRC (ours)** | **<5M** | **≥98%** | **≥0.98** | **≥93% ← your claim** |

### Table 2 — Efficiency Comparison
| Model | Params | FLOPs | Model Size | CPU Latency |
|---|---|---|---|---|
| ResNet-50 | 25.6M | 4.1G | 98MB | ~120ms |
| EfficientNet-B0 | 5.3M | 0.39G | 20MB | ~45ms |
| **MedLite-CRC** | **<5M** | **<1G** | **<50MB** | **<50ms** |

### Table 3 — Ablation Study
Each component ON vs OFF. Shows every piece of your architecture earns its place.

| Variant | Cross-Val Acc | Macro-F1 |
|---|---|---|
| Full model | ≥93% | best |
| w/o Stain Norm layer | drops | drops |
| w/o Multi-scale branches | drops | drops |
| w/o SE Attention | drops | drops |
| Single-scale (3×3 only) | drops | drops |

### Table 4 — Cross-Domain Generalisation
| Model | NCT-CRC (train domain) | CRC-VAL (cross-patient) | UniToPatho (cross-hospital) |
|---|---|---|---|
| EfficientNet-B0 | 99% | ~89% | ~82% |
| **MedLite-CRC** | ≥98% | ≥93% | ≥88% |

**This is your main novelty table. If you have this, you have a paper.**

---

## Metrics to Report Per Table

| Metric | Required | Why |
|---|---|---|
| Accuracy | ✅ | Standard |
| Macro-F1 | ✅ | Handles class imbalance — reviewers expect this |
| Weighted-F1 | ✅ | Secondary |
| Per-class F1 | ✅ | In appendix or supplementary |
| AUC-ROC | optional | Nice to have |
| Parameters (M) | ✅ | Lightweight claim |
| FLOPs (G) | ✅ | Efficiency claim |
| Model size (MB) | ✅ | Deployment claim |
| CPU inference (ms) | ✅ | Edge deployment claim |

---

## Figures You Must Have

1. **Architecture diagram** — one clean block diagram of MedLite-CRC
2. **Confusion matrix** — on CRC-VAL-HE-7K (the cross-patient set)
3. **GradCAM visualisation** — 3–4 images showing what the model focuses on
4. **Training curves** — loss + accuracy over epochs (train vs val)

---

## Minimum Experiments Needed

1. Train MedLite-CRC on NCT-CRC-HE-100K
2. Evaluate on CRC-VAL-HE-7K → get Table 1 + Table 4 numbers
3. Run all 4 baselines → same eval → fill comparison tables
4. Ablation: retrain 4 variants with one component removed each
5. Evaluate one model on UniToPatho → cross-hospital row in Table 4
6. Measure FLOPs + latency → Table 2

**Total training runs needed: ~10** (1 full model + 4 baselines + 4 ablation variants + 1 UniToPatho eval)

---

## What a Reviewer Will Reject You For

- ❌ Only testing on NCT-CRC-HE-100K (no cross-dataset proof)
- ❌ No ablation study
- ❌ No efficiency metrics (just accuracy)
- ❌ No comparison against Lite-V2 (2025) — it's the most recent paper in your exact space
- ❌ Accuracy < 98% on NCT-CRC (below existing baselines)

---

## Minimum Acceptable Results to Submit

| Checkpoint | Number needed |
|---|---|
| NCT-CRC accuracy | ≥ 98% |
| CRC-VAL accuracy | ≥ 93% |
| Macro-F1 (CRC-VAL) | ≥ 0.92 |
| Params | < 5M |
| Beat at least 3 baselines on cross-val | mandatory |
