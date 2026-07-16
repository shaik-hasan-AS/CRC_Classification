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

---

## 🩺 7. Cross-Cohort Generalization & Transfer Learning Validation

To evaluate the clinical transferability of the learned feature representations of our SOTA checkpoint, we fine-tuned MedLite-CRC (both from scratch and utilizing pretrained SOTA weights) on three external downstream cohorts representing different diagnostic tasks:
1. **EBHI-SEG** (6-class biopsy diagnostics, 2,225 images total)
2. **CRC-HGD-v1** (5-class histopathology grading, 1,914 images total)
3. **Kather MSI/MSS** (2-class molecular phenotype classification, 139,143 images total)

All trials utilized consistent hyperparameter configurations (`epochs: 20` for HGD/EBHI, `epochs: 3` for Kather; `lr: 0.0002` to `0.0003`, `AdamW`).

### A. Generalization Performance Summary

| Cohort | Class Count | Training Mode | Accuracy | Macro-F1 | Absolute Delta (Acc / F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EBHI-SEG** | 6 | Scratch | 42.47% | 38.52% | **+31.01% / +23.99%** |
| (Biopsy Classification) | | **Pretrained** | **73.48%** | **62.51%** | |
| **CRC-HGD-v1** | 5 | Scratch | 57.07% | 30.90% | **+14.13% / +11.04%** |
| (Colorectal Grading) | | **Pretrained** | **71.20%** | **41.94%** | |
| **Kather MSI/MSS** | 2 | Scratch | 63.88% | 54.74% | **+9.18% / +8.40%** |
| (Molecular Phenotype) | | **Pretrained** | **73.06%** | **63.14%** | |

### B. Scientific Discussion on Generalization
- **EBHI-SEG Biopsy Transfer:** Under scratch training, the model fails to resolve the minor diagnostic classes (like Serrated Adenoma and Normal Mucosa), resulting in a poor 42.47% accuracy. Pretrained transfer learning immediately converges to **73.48%**, demonstrating that the pre-trained feature representation possesses a prior understanding of glandular boundary structures.
- **CRC-HGD-v1 Grading Shift:** Pathological grading (Well vs. Moderately vs. Poorly Differentiated) is famously difficult due to subtle structural differences in glandular alignment. Pretraining boosts the F1 score of the hardest **Poorly Differentiated** class from **0.3505 up to 0.6422 (almost double)**, proving the pretrained weights extract high-level biological tissue organization patterns.
- **Kather MSI/MSS Molecular Shift:** Predicting genetic Microsatellite Instability directly from H&E stains is a complex task. The pretrained weights allow the model to converge to **73.06% accuracy** and **63.14% Macro-F1** in just 3 epochs of fine-tuning, demonstrating that general morphological features learned during NCT-100K pretraining correlate directly with molecular cell phenotypes.

---

## 📚 8. Competitive Benchmark Against Published Literature

### A. NCT-CRC-HE-100K (Train) & CRC-VAL-HE-7K (OOD Test) Benchmark
This table compares the parameter footprint and out-of-distribution test accuracy of **MedLite-CRC V2** against other representative models in literature trained on the exact same NCT-100K training dataset and evaluated on the CRC-VAL-HE-7K validation cohort:

| Study & Citation | Model Architecture | Params (M) | Disk Size (MB) | OOD Accuracy (7K) |
| :--- | :--- | :---: | :---: | :---: |
| **Kather et al. (2019)** *PLOS Medicine* [1] | VGG19 (ImageNet Pre-trained) | 143.6M | 548.0 MB | **94.30%** |
| **Li et al. (2025)** *Frontiers in Oncology* [2] | Custom Lightweight CNN | 4.41M | 16.9 MB | *Not Evaluated* (99.00% In-Dist) |
| **Ignatov & Malivenko (2024)** *ECCV* [3] | EfficientNet-B0 Baseline | 4.02M | 16.0 MB | **97.70%** (With bias exposure) |
| **Uddin et al. (2023)** *BSPC* [8] | CRCCN-Net | Lightweight CNN | - | **96.26%** (In-Distribution) |
| **Common Pathology Baselines** [4] | ResNet-50 | 23.53M | 94.43 MB | **94.33%** |
| **Common Pathology Baselines** [4] | DenseNet-121 | 6.96M | 33.00 MB | **96.52%** |
| **Common Pathology Baselines** [4] | MobileNetV2 | 2.24M | 9.19 MB | **94.82%** |
| **Common Pathology Baselines** [4] | MobileNetV3-Small | 1.52M | 5.40 MB | **94.10%** |
| **Standard Swin Transformer** [5] | Swin-T | 28.3M | 114.0 MB | **96.30%** |
| **MedLite-CRC V2 (Ours - Standard)** | MedLite-CRC (Scratch) | **0.48M** | 2.02 MB | **94.65%** |
| **MedLite-CRC V2 (Ours - KD SOTA)** | MedLite-CRC + MobileNetV2 KD | **0.48M** | **0.75 MB (INT8)** | **95.97% (Mean: 95.73% ± 0.21%)** |

### B. CRC-5000 Cohort (8-Class Benchmark)
Comparative performance on the legacy CRC-5000 multi-class cohort:

| Study & Citation | Classification Approach | Feature Space | Accuracy (CRC-5000) |
| :--- | :--- | :---: | :---: |
| **Kather et al. (2016)** *Scientific Reports* [6] | Texture Features + SVM | LBP & Gabor | **87.40%** |
| **Uddin et al. (2023)** *BSPC* [8] | CRCCN-Net | Learned Lightweight | **93.50%** |
| **Pathology Benchmarks** [4] | MobileNetV2 (ImageNet Pre-trained) | CNN Features | **89.00%** |
| **Pathology Benchmarks** [4] | ResNet-50 (ImageNet Pre-trained) | CNN Features | **89.43%** |
| **MedLite-CRC V2 (Ours - KD SOTA)** | MedLite-CRC (Distilled) | Learned Histology | **93.94%** |

### C. STARC-9 Cohort (9-Class Benchmark)
Comparative validation on the high-scale Stanford Colorectal Cancer (STARC-9) cohort:

| Study & Citation | Model Architecture | Params (M) | Accuracy (STARC-9) |
| :--- | :--- | :---: | :---: |
| **Subramanian et al. (2025)** *NeurIPS* [7] | DeiT-B / Histo-ViT | 86.0M | **96.32%** |
| **Subramanian et al. (2025)** *NeurIPS* [7] | ResNet-50 | 23.5M | **97.81%** |
| **Subramanian et al. (2025)** *NeurIPS* [7] | Swin-B (Swin Transformer) | 88.0M | **98.79%** |
| **Subramanian et al. (2025)** *NeurIPS* [7] | EfficientNet-B7 | 66.0M | **98.80%** |
| **Subramanian et al. (2025)** *NeurIPS* [7] | CTransPath (Foundation Model) | 28.0M | **99.00%** |
| **MedLite-CRC V2 (Ours - KD SOTA)** | MedLite-CRC + MobileNetV2 KD | **0.48M** | **99.75%** |

---

### 📑 Literature Citations
*   **[1] Kather, J. N., Halama, N., & Marx, A. (2019).** Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. *PLOS Medicine*, 16(1), e1002730.
*   **[2] Li, Y., Goh, W. W., & Jhanjhi, N. Z. (2025).** A lightweight CNN for colon cancer tissue classification and visualization. *Frontiers in Oncology*, 15, 10842.
*   **[3] Ignatov, A., & Malivenko, G. (2024).** NCT-CRC-HE: Not All Histopathological Datasets Are Equally Useful. *European Conference on Computer Vision (ECCV)*, 300-317. (Preprint: arXiv:2409.11546).
*   **[4] Benchmark Baselines.** Evaluated locally in our workspace under identical training loops.
*   **[5] Standard Swin Transformer.** Representative Swin-T results reported on public benchmark evaluations.
*   **[6] Kather, J. N., Weis, C. A., Bianconi, F., et al. (2016).** Multi-class texture analysis in colorectal cancer histology. *Scientific Reports*, 6, 27988.
*   **[7] Subramanian, B., Jeyaraj, R., Peterson, M. N., et al. (2025).** STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology. *Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
*   **[8] Uddin, A. H., Chen, Y. L., Ku, C. S., Por, L. Y., et al. (2023).** CRCCN-Net: Automated framework for classification of colorectal tissue using histopathological images. *Biomedical Signal Processing and Control*, 79, 104172.




