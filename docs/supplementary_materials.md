# Supplementary Materials: MedLite-CRC

This document consolidates the supplementary research analyses, experimental logs, competitive evaluations, and biological breakdowns for the MedLite-CRC publication.

---

## 1. The 9 Colorectal Cancer Tissue Classes: Biological Breakdown

To understand how the convolutional neural network processes tissue morphology, it is essential to map the biological characteristics of the 9 classes in the `NCT-CRC-HE-100K` and `CRC-VAL-HE-7K` cohorts:

1. **ADI (Adipose):** Fat tissue. Visually characterized by large, empty white vacuole bubbles with thin peripheral pink boundaries. High classification accuracy due to unique structural layout.
2. **BACK (Background):** Empty slide glass containing only white space.
3. **DEB (Debris):** Necrotic (dead) tissue and cellular debris. Appears as unstructured, dark red/purple necrotic fragments.
4. **LYM (Lymphocytes):** Immune cells. Highly distinct dense clusters of tiny, circular dark-purple nuclei.
5. **MUC (Mucus):** Secreted mucinous proteins. Characterized by pale, wispy blue/pink colloidal pools with low cell density.
6. **NORM (Normal colon mucosa):** Healthy mucosal lining. Shows highly structured, circular/oval glandular crypts lined by neatly arranged nuclei.
7. **TUM (Tumor):** Adenocarcinoma epithelium. Chaotic, disorganized sheets and overgrown, dark purple glands lacking structural symmetry.
8. **STR (Stroma) vs. MUS (Smooth Muscle):** The primary biological classification bottleneck:
   - Both tissue types stain the exact same shade of bright pink (eosinophilic) on standard H&E slides.
   - **Muscle fibers (MUS)** run in parallel, highly organized straight lines.
   - **Stroma connective fibers (STR)** are loose, disorganized, and slightly wavy.
   - Because of this deep structural similarity, standard models often confuse the two. MedLite-CRC utilizes its **MultiScaleBranch** (3x3, 5x5, and 7x7 parallel depthwise convolutions) to capture the wavelike macro-texture of stroma versus the parallel alignment of muscle, improving classification F1-scores.

---

## 2. Competitive Literature & Novelty Analysis

MedLite-CRC operates at the intersection of ultra-lightweight architecture design and domain-invariant generalization in computational pathology.

### 2.1 Parameter Efficiency Comparison
- **Li et al. (2025):** The primary custom lightweight CNN designed for CRC tissue classification requires **4.41 Million parameters** and **16.9 MB** of disk space to hit 99.00% accuracy.
- **MedLite-CRC (Ours):** Achieves **99.48%** in-distribution peak accuracy using only **0.48 Million parameters** (9.2× smaller) and occupies **0.72 MB** in INT8 format (23.5× smaller).

### 2.2 Train-from-Scratch Generalization
When trained strictly from scratch (without ImageNet pre-training) on NCT-100K and evaluated on the out-of-distribution (OOD) `CRC-VAL-HE-7K` cohort:
- Standard architectures drop in accuracy (EfficientNet-B0: 94.81%, ResNet-50: 94.33%).
- MedLite-CRC V1 achieves **94.71%** natively, and its MobileNetV2 KD-distilled counterpart hits a SOTA **95.96%** accuracy, outperforming models up to 48× larger.

### 2.3 Defense of Necessary Trade-offs
- **ImageNet Pre-trained Models:** While pre-trained Vision Transformers and heavy ResNets can exceed 97% OOD accuracy, they rely on massive parameter weights (86M+ parameters) pre-optimized on natural images (cars, dogs). This makes them unsuitable for local deployment on low-cost diagnostic edge terminals in resource-limited clinics.
- **ShuffleNetV2:** ShuffleNetV2 achieves 95.08% accuracy but requires **1.26 Million parameters** (2.6× larger than MedLite-CRC's 0.48M) and 5.23 MB disk space. For extreme edge microcontrollers with strict cache memory and RAM limits, MedLite-CRC provides a superior efficiency footprint.

### 2.4 Competitive Performance Matrix

| Model Architecture | Params (M) | In-Dist (100K) | OOD (7K) | STARC-9 | CRC-5000 | Deployed Footprint (INT8) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MedLite-CRC (Ours, KD)** | **0.48** | **99.46%** | **95.96%** | **99.75%** | **93.94%** | **2.02 MB** (FP32) |
| **MedLite-CRC (Ours, KD INT8)** | **0.48** | **99.46%** | **95.72%** | **—** | **—** | **0.72 MB** |
| **MedLite-CRC (Ours, standard)** | **0.48** | **99.48%** | **94.71%** | **99.79%** | **92.00%** | **2.02 MB** (FP32) |
| Li et al. (2025) CNN | 4.41 | 99.00% | - | - | - | 16.9 MB |
| MobileNetV2 | 2.24 | 99.18% | 94.82% | 99.63% | 89.00% | 9.19 MB |
| EfficientNet-B0 | 4.02 | 99.04% | 94.81% | 99.68% | 92.00% | 16.38 MB |
| ResNet-50 | 23.53 | 98.53% | 94.33% | 99.60% | 89.43% | 94.43 MB |

---

### 2.5 Cross-Cohort Generalization & Downstream Transfer Validation
To establish the clinical utility and semantic transferability of the learned MedLite-CRC feature representations, we benchmarked our pre-trained model against scratch training on three distinct external pathological cohorts:
- **EBHI-SEG** (6-class endoscopic biopsy, 2,225 tiles)
- **CRC-HGD-v1** (5-class differentiation grading, 1,914 tiles)
- **Kather MSI/MSS** (2-class Microsatellite Instability molecular classification, 139,143 tiles)

| Cohort | Downstream Diagnostic Task | Classes | Scratch Acc | Pre-trained Acc | Absolute Delta |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EBHI-SEG** | Biopsy tissue classification | 6 | 42.47% | **73.48%** | **+31.01%** |
| **CRC-HGD-v1** | Histopathology differentiation grading | 5 | 57.07% | **71.20%** | **+14.13%** |
| **Kather MSI/MSS**| Binary molecular classification | 2 | 63.88% | **73.06%** | **+9.18%** |

These results indicate that pre-training on NCT-100K under our MobileNetV2 knowledge distillation pipeline develops robust, scale-invariant priors for glandular margins, cellular arrangement, and chromatin density that transfer effectively across clinical tasks and scanner manufacturers.

---

## 3. Statistical Significance (McNemar's Test)

To prove that the performance gains of MedLite-CRC (KD Student) are statistically significant and not due to random initialization, we performed a McNemar's test against the 8× larger EfficientNet-B0 baseline on the 7,180-image `CRC-VAL-HE-7K` cohort.

### 3.1 Primary Analysis: Optimal Configurations
We compare the models under their respective configurations (MedLite-CRC KD at 95.96% accuracy vs. EfficientNet-B0 unmasked at 94.81% accuracy):

| | EfficientNet-B0 Correct | EfficientNet-B0 Incorrect |
| :--- | :---: | :---: |
| **MedLite-CRC KD Correct** | 6,673 | 221 |
| **MedLite-CRC KD Incorrect** | 134 | 152 |

- **Discordant Pairs:** MedLite-CRC KD correctly classified 221 images that the baseline failed on, while the baseline correctly classified 134 images that MedLite-CRC KD failed on.
- **Chi-Squared Statistic ($\chi^2$):** 20.830
- **P-Value:** **$5.01 \times 10^{-6}$**

Since $p < 0.05$, we decisively reject the null hypothesis, mathematically proving that our model's performance improvements are statistically significant.

### 3.2 Robustness Analysis: Masked Configurations
When both models are evaluated under test-time background noise masking (foreground-only tissue):

| | EfficientNet-B0 Masked Correct | EfficientNet-B0 Masked Incorrect |
| :--- | :---: | :---: |
| **MedLite-CRC KD Correct** | 5,743 | 1,148 |
| **MedLite-CRC KD Incorrect** | 64 | 225 |

- **Chi-Squared Statistic ($\chi^2$):** 967.730
- **P-Value:** **$1.86 \times 10^{-212}$**

Standard, unregularized CNN architectures collapse (EfficientNet-B0 drops to 80.88% accuracy) when background slide pixels are masked, while MedLite-CRC remains highly resilient, demonstrating superior morphological generalization.

---

## 4. Detailed Ablation Study: Failures & Negative Findings

To validate the specific architectural choices in MedLite-CRC, we document the negative results of five key experimental directions:

1. **CutMix Failure:** Replacing `MixUp` with `CutMix` (alpha=1.0) dropped cross-patient OOD accuracy from 94.50% to **91.09%** (Stroma F1 fell to 0.64). Histopathology slides represent continuous biological sheets. Introducing hard, square artificial boundaries via CutMix causes the network to learn these sharp artificial edges as shortcuts rather than the biological texture of the actual tissue.
2. **V2 Architectural Scaling:** Increasing the model's base channels from 32 (0.48M params, V1) to 48 (1.08M params, V2) with SiLU activations caused OOD accuracy to drop to **91.94%** (despite near-perfect training convergence of 99.98%). This confirms that over-parameterization triggers memorization of hospital-specific color and scanner profiles.
3. **Test-Time Augmentation (TTA) Degradation:** Applying 4-rotation TTA averaging during inference dropped accuracy to **92.70%** (specifically harming Muscle and Stroma F1-scores). The model learns directional heuristics relative to fibrous tissue orientations; averaging across arbitrary 90-degree rotations disrupts its confidence in these directional boundaries.
4. **Receptive Field Expansion (Large Kernels):** Replacing the $3\times3, 5\times5, 7\times7$ multi-scale branch with larger $7\times7, 9\times9, 11\times11$ depthwise convolutions dropped cross-patient accuracy to **93.93%**. Crucially, the F1-score for Lymphocytes dropped from 0.9921 to 0.9842. The massive 11x11 filters acted as a low-pass filter that smoothed over the critical high-frequency, crisp edge details required to identify tiny lymphocytic nuclei.
5. **Focal Loss & Pairwise Loss Overfitting:** Implementing a Focal Loss combined with a Pairwise Confusion Penalty specifically targeting Stroma vs. Smooth Muscle confusion eliminated training confusion (99.69% in-distribution validation accuracy), but cross-patient accuracy collapsed to **94.76%** (Stroma recall plummeted to 57.48%). Modifying loss functions to target hard cases causes the network to overfit to the specific stain and texture signatures of those hard cases within the training domain.
6. **HED-Space Stain Normalization:** Learnable stain normalization in biologically-grounded Hematoxylin-Eosin-DAB (HED) space achieved **94.18%** OOD accuracy. While robust, this was slightly lower than RGB-space learnable affine normalization (**94.71%**). The RGB affine layer has greater mathematical freedom to perform arbitrary linear rotations and shifts across channels, allowing it to adapt to non-linear color response differences across scanners that do not strictly conform to the linear Beer-Lambert deconvolution model.

---

## 5. Grad-CAM Spatial Interpretability Analysis

To verify that MedLite-CRC relies on true morphological features rather than exploiting background noise shortcuts or center biases, we developed a quantitative spatial analysis framework.

### 5.1 Quantitative Grad-CAM Tissue Alignment
We calculated the mathematical alignment score (overlap between the top-20% hottest pixels of the Grad-CAM activation map and the actual segmentations of the target tissues):
- **Lymphocytes (LYM):** 97.6% (perfect alignment, focusing on dense nuclei groups)
- **Stroma (STR):** 96.8% (high alignment, tracking fibrous collagen paths)
- **Tumor (TUM):** 96.2% (high alignment, focusing on epithelial sheets)
- **Normal Mucosa (NORM):** 96.0% (high alignment, tracking neat glandular walls)
- **Debris (DEB):** 85.2% (relaxed attention, diffusing into necrotic zones)

*Biological Interpretation:* The lower alignment score for Debris is biologically valid. Debris is unstructured necrotic scatter and mucus. The model correctly relaxes its spatial attention to mirror this biological reality, while maintaining a sharp 97.6% alignment on dense, structured classes like Lymphocytes.

### 5.2 V2 boundary Artifact Mitigation (Reflect Padding & Border Masking)
Standard zero-padding in convolutional networks creates a sharp artificial contrast (discontinuity) at the margins of histopathology patches, causing the model to learn "border ring" artifacts (especially on low-density tissue like adipose).

We evaluated the V2 model configuration (incorporating reflection padding and 8px border masking) on the `CRC-VAL-HE-7K` cohort:

| Metric / Configuration | V1 Baseline (Zero Pad, Unmasked) | V1 Masked (Zero Pad, 8px Mask) | V2 Mitigated (Reflect Pad, 8px Mask) |
| :--- | :---: | :---: | :---: |
| **OOD Accuracy** | 95.96% | 96.07% | **95.84%** |
| **Avg. Radial Distance from Center** | 21.49 px | 19.95 px | **20.32 px** |
| **Vanishing Gradient Rate** | 11.20% | 11.30% | **10.30%** |
| **Stroma Background Activation** | 0.3075 | 0.2500 | **0.2524** |
| **Stroma Tissue Activation** | 0.3272 | 0.2775 | **0.2741** |
| **Background Noise Reduction** | - | 18.7% | **17.9%** |

#### Insights & Impact
- **Successful Noise Elimination:** Applying the 8px border masking resulted in an **18% relative reduction** in background noise activation for stroma patches, confirming the removal of artificial border rings.
- **Improved Spatial Stability:** Switching to `reflect` padding reduced the **vanishing gradient rate from 11.2% to 10.30%** (a 1% absolute reduction), meaning the network maintains more robust, physically localizable representations.
- **Preserved Classification Capacity:** In only 3 epochs of fine-tuning, the V2 model recovered to **95.84% OOD accuracy** (virtually identical to the original SOTA), proving that reflect padding eliminates border artifacts without sacrificing generalization.
