# 📊 MedLite-CRC Experimental Logbook

This logbook serves as the single source of truth for all quantitative metrics, class-by-class classification statistics, and architectural benchmarks across all evaluated datasets and model configurations.

---

## 📂 Dataset Summary

1. **NCT-CRC-HE-100K** (In-Distribution Validation Set):
   - **Origin:** Germany (multicentric)
   - **Images:** 100,000 tiles (9 classes, balanced)
   - **Role:** Source training and internal validation baseline.

2. **CRC-VAL-HE-7K** (Out-of-Distribution Cross-Patient Test Set):
   - **Origin:** Germany (completely distinct patients, different scanners/hospitals)
   - **Images:** 7,180 tiles (9 classes, unbalanced)
   - **Role:** Standard OOD benchmark for clinical generalization.

3. **STARC-9** (Stanford Multi-Centric Cohort):
   - **Origin:** Stanford University Medical Center (highly variable scanners/protocols)
   - **Images:** ~630,000 tiles (9 classes, balanced)
   - **Role:** High-scale multi-centric dataset.

4. **CRC-5000** (Legacy/Noisy Cohort):
   - **Origin:** Kather et al. (2016)
   - **Images:** 5,000 tiles (8 classes, balanced)
   - **Role:** Challenging testbed for low-resolution, highly saturated legacy data.

---

## 📈 1. Overall Performance Matrix

| Model Setup | NCT-100K Val Acc | OOD 7K Test Acc | STARC-9 Val Acc | CRC-5000 Acc | Parameters | FLOPs (G) | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1** (Baseline CNN stem) | 99.02% | 94.05% | - | - | 0.453M | 0.349 | 1.89 MB |
| **Ablation 2** (+ Stain Adaptation) | 99.28% | 94.64% | - | - | 0.453M | 0.349 | 1.89 MB |
| **Ablation 3** (+ MultiScaleBranch) | 99.41% | 94.65% | - | - | 0.482M | 0.726 | 2.02 MB |
| **Ablation 4** (+ SEBlock attention) | **99.52%** | 93.82% | - | - | 0.490M | 0.726 | 2.05 MB |
| **KD (EfficientNet-B0 Teacher)** | 99.12% | 94.35% | - | - | 0.482M | 0.726 | 2.02 MB |
| **KD (MobileNetV2 Teacher) - V1** | 99.46% | **95.97%** | - | - | 0.482M | 0.726 | 2.02 MB |
| **KD (MobileNetV2 Teacher) - V2** | 99.35% | **95.84%** | - | - | 0.482M | 0.726 | 2.02 MB |
| **STARC-9 Baseline (V1)** | - | - | **99.79%** | - | 0.482M | 0.726 | 2.02 MB |
| **Hybrid 11-Class Model** | - | 93.50% | 99.76% | 45.23% | 0.482M | 0.726 | 2.02 MB |
| **KD on CRC-5000 (SOTA)** | - | - | - | **93.94%** | 0.482M | 0.726 | 2.02 MB |

---

## 🧪 2. Detailed Per-Class Breakdowns

### A. SOTA KD Student Model (MobileNetV2 Teacher) on OOD 7K Set (V1 Checkpoint)
* **Overall Accuracy:** 95.97%
* **Macro F1:** 0.9476
* **Weighted F1:** 0.9600

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **ADI** (Adipose) | 0.9977 | 0.9619 | 0.9795 | 1338 |
| **BACK** (Background) | 0.9988 | 1.0000 | 0.9994 | 847 |
| **DEB** (Debris) | 0.9631 | 1.0000 | 0.9812 | 339 |
| **LYM** (Lymphocytes) | 0.9769 | 1.0000 | 0.9883 | 634 |
| **MUC** (Mucus) | 0.9638 | 0.9787 | 0.9712 | 1035 |
| **MUS** (Smooth Muscle) | 0.8980 | 0.8176 | 0.8564 | 592 |
| **NORM** (Normal Mucosa) | 0.9767 | 0.9622 | 0.9694 | 741 |
| **STR** (Stroma) | 0.7567 | 0.8717 | 0.8084 | 421 |
| **TUM** (Tumor) | 0.9790 | 0.9813 | 0.9802 | 1233 |

---

### B. V2 Mitigated SOTA Model (Reflect Padding + 8px Border Mask) on OOD 7K Set
* **Overall Accuracy:** 95.84%
* **Background Noise Activation (Stroma):** 0.2524 (down from 0.3075, **17.9% reduction**)
* **Vanishing Gradient Rate:** 10.30% (down from 11.20%)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **ADI** (Adipose) | 0.9984 | 0.9574 | 0.9775 | 1338 |
| **BACK** (Background) | 0.9988 | 1.0000 | 0.9994 | 847 |
| **DEB** (Debris) | 0.9576 | 1.0000 | 0.9783 | 339 |
| **LYM** (Lymphocytes) | 0.9769 | 1.0000 | 0.9883 | 634 |
| **MUC** (Mucus) | 0.9647 | 0.9778 | 0.9712 | 1035 |
| **MUS** (Smooth Muscle) | 0.8942 | 0.8108 | 0.8505 | 592 |
| **NORM** (Normal Mucosa) | 0.9766 | 0.9595 | 0.9680 | 741 |
| **STR** (Stroma) | 0.7485 | 0.8717 | 0.8054 | 421 |
| **TUM** (Tumor) | 0.9766 | 0.9805 | 0.9786 | 1233 |

---

### C. KD Student Model (EfficientNet-B0 Teacher) on OOD 7K Set
* **Overall Accuracy:** 94.35%
* **Macro F1:** 0.9262
* **Weighted F1:** 0.9437

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **ADI** | 0.9929 | 0.9410 | 0.9662 | 1338 |
| **BACK** | 0.9369 | 1.0000 | 0.9674 | 847 |
| **DEB** | 0.9391 | 1.0000 | 0.9686 | 339 |
| **LYM** | 0.9709 | 0.9984 | 0.9844 | 634 |
| **MUC** | 0.9816 | 0.9768 | 0.9792 | 1035 |
| **MUS** | 0.8758 | 0.7264 | 0.7941 | 592 |
| **NORM** | 0.9887 | 0.9433 | 0.9655 | 741 |
| **STR** | 0.6680 | 0.8171 | 0.7350 | 421 |
| **TUM** | 0.9681 | 0.9830 | 0.9755 | 1233 |

---

### D. KD SOTA Student Model on CRC-5000 Legacy Cohort
* **Overall Accuracy:** 93.94%
* **Macro F1:** 0.9392
* **Weighted F1:** 0.9392

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **ADI** (Adipose) | 0.9672 | 0.9440 | 0.9555 | 125 |
| **BACK** (Background) | 0.9470 | 1.0000 | 0.9728 | 125 |
| **DEB** (Debris) | 0.8966 | 0.8320 | 0.8631 | 125 |
| **LYM** (Lymphocytes) | 0.9839 | 0.9760 | 0.9799 | 125 |
| **NORM** (Mucosa) | 0.9449 | 0.9600 | 0.9524 | 125 |
| **STR** (Stroma) | 0.8647 | 0.9200 | 0.8915 | 125 |
| **TUM** (Tumor) | 0.9752 | 0.9440 | 0.9593 | 125 |
| *MUC* | 0.0000 | 0.0000 | 0.0000 | 0 * |
| *MUS* | 0.0000 | 0.0000 | 0.0000 | 0 * |

*\*Note: Mucus (MUC) and Muscle (MUS) do not exist in the CRC-5000 taxonomy.*

---

## 📊 3. STARC-9 & Hybrid 11-Class Experiments

### A. 11-Class Hybrid Model (Unaugmented) on OOD 7K Set
* **Overall Accuracy:** 93.50%
* **Weighted F1:** 0.9339
* **LYM F1:** 0.9864
* **BACK F1:** 0.9554

### B. Extreme Augmentation Model (Gaussian Mask + Grayscale) trained on STARC-9
* **OOD 7K Test Accuracy:** 70.89% (catastrophic failure on fine-scale tissues)
* **Lymphocytes (LYM) Recall:** 2.80%
* **Stroma (STR) Recall:** 31.80%
* **Normal Mucosa (NORM) Recall:** 36.90%

### C. 11-Class Hybrid Model evaluated directly on CRC-5000
* **Accuracy:** 45.23%
* **Macro F1:** 0.2861
* **Reason for failure:** Overfitting to modern scanner pink/purple stain profiles due to Grayscale Dropout being completely disabled to preserve fine cell structure during high-scale training. Almost all tiles were incorrectly classified as Background (BACK).

---

## ⚙️ 4. Hardware and Computational Benchmarks

- **Profiling Platform:** Intel Core i7 CPU (16 threads) / NVIDIA RTX 4060 GPU
- **Input Image Dimension:** 3 channels, 224 x 224 pixels

### Latency and Throughput (Inference Time)
| Model Setup | CPU Latency (ms) | GPU Latency (ms) | Throughput (FPS - GPU) |
| :--- | :---: | :---: | :---: |
| **MedLite-CRC (V1)** | 1.84 ms | 0.72 ms | 1388.9 FPS |
| **MedLite-CRC (V2)** | 1.91 ms | 0.74 ms | 1351.4 FPS |
| **MobileNetV2 (Teacher)** | 3.52 ms | 1.25 ms | 800.0 FPS |
| **EfficientNet-B0 (Teacher)** | 6.84 ms | 2.15 ms | 465.1 FPS |

---

## 🔬 5. Statistical Significance Metrics (McNemar's Chi-Square)

*Validated on CRC-VAL-HE-7K test set (7,180 images)*

### A. Normal Conditions (MedLite-CRC SOTA vs. EfficientNet-B0)
- **Contingency Table:**
  - Both Correct: 6,673
  - Both Incorrect: 152
  - MedLite Correct / EffNet Incorrect: 221
  - MedLite Incorrect / EffNet Correct: 134
- **Chi-Square Statistic ($\chi^2$):** 20.830
- **P-Value:** $5.01 \times 10^{-6}$ (Extremely significant, rejecting the Null Hypothesis)

### B. Boundary Masked Conditions (Simulated Slide Edge Errors)
- **Contingency Table:**
  - Both Correct: 5,743
  - Both Incorrect: 225
  - MedLite Correct / EffNet Incorrect: 1,148
  - MedLite Incorrect / EffNet Correct: 64
- **Chi-Square Statistic ($\chi^2$):** 967.730
- **P-Value:** $1.86 \times 10^{-212}$ (Highly significant difference in favor of MedLite-CRC's architectural robustness to border artifacts)

---

## 🎲 6. Multi-Seed Robustness & Statistical Validation

*Comparing Baseline from-scratch models vs. V2 SOTA KD models across 3 random seeds (42, 123, 999) on the CRC-VAL-HE-7K test set.*

### A. Baseline Model (From-Scratch / Ablation 3 / No KD)
- **Seed 42:** Accuracy = 93.76%, Macro-F1 = 0.9214
- **Seed 123:** Accuracy = 94.71%, Macro-F1 = 0.9321
- **Seed 999:** Accuracy = 93.69%, Macro-F1 = 0.9179
- **Summary statistics:**
  - **Mean Accuracy:** **94.05%** (± 0.46%)
  - **Mean Macro-F1:** **0.9238** (± 0.0060)

### B. V2 SOTA Model (MobileNetV2 KD + Reflect Padding + 8px Border Mask)
- **Seed 42:** Accuracy = 95.78%, Macro-F1 = 0.9453
- **Seed 123:** Accuracy = 95.96%, Macro-F1 = 0.9472
- **Seed 999:** Accuracy = 95.46%, Macro-F1 = 0.9408
- **Summary statistics:**
  - **Mean Accuracy:** **95.73%** (± 0.21%)
  - **Mean Macro-F1:** **0.9444** (± 0.0027)

> [!NOTE]
> The V2 SOTA model not only increases out-of-distribution performance by **1.68% (absolute)**, but also halves the variance across random initializations (standard deviation drops from **0.46%** to **0.21%**), demonstrating superior convergence stability under knowledge distillation.

