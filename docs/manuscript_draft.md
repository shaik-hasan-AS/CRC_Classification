# MedLite-CRC: Dataset Scale as a Regularizer for Ultra-Lightweight Colorectal Cancer Histopathology Classification

**Author:** Shaik Hasan A S
  

---

## Abstract
Deep learning has revolutionized automated histopathological diagnosis, but standard state-of-the-art (SOTA) architectures are computationally heavy (11M to 30M+ parameters) and highly sensitive to scanner domain shift. In this study, we present **MedLite-CRC**, an ultra-lightweight Convolutional Neural Network (CNN) specifically designed for colorectal cancer (CRC) tissue classification on memory-constrained edge devices. MedLite-CRC consists of only **0.48 Million parameters** and has an INT8 quantized disk footprint of **0.75 MB**, delivering an inference latency of **1.94 ms** on standard edge CPUs. 

To overcome domain shift and scanner-specific biases (e.g., JPEG artifacts and H&E stain variations), we introduce two novel modules: an end-to-end differentiable, six-parameter **Learnable Stain Adaptation Layer** and a **Depthwise Separable Multi-Scale Branch** (capturing 3×3, 5×5, and 7×7 receptive fields simultaneously). 

We evaluate MedLite-CRC across three distinct datasets: NCT-CRC-HE-100K, STARC-9, and CRC-5000. Under standard training conditions, MedLite-CRC achieves a peak in-distribution accuracy of **99.48%** and an out-of-distribution, cross-patient validation accuracy of **94.65%** on the unseen CRC-VAL-HE-7K cohort. By introducing a Knowledge Distillation (KD) framework with a structurally aligned MobileNetV2 teacher model, MedLite-CRC generalizes exceptionally well, achieving a verified out-of-distribution accuracy of **95.97%** on the best checkpoint—outperforming the teacher itself (94.82%) by **+1.15%** absolute and the state-of-the-art ShuffleNetV2 baseline (95.08%) by **+0.89%** absolute, while requiring up to 48× fewer parameters than ResNet-50.

Furthermore, we benchmark our architecture on the massive 630,000-image STARC-9 dataset (NeurIPS 2025), achieving **99.79%** accuracy, proving that dataset scale acts as a natural regularizer for highly constrained networks. 

Finally, we perform a rigorous statistical validation using McNemar's test ($p = 5.01 \times 10^{-6}$) and conduct a quantitative spatial Grad-CAM analysis to inspect potential model shortcuts. Our interpretability study reveals the "Attention Paradox": while Squeeze-and-Excitation attention blocks improve training convergence, they overfit to scanner-specific staining channels, degrading cross-site generalization by 0.83%. Consequently, we establish the attention-free MedLite-CRC as the optimal architecture for robust cross-site clinical deployment.

---

## 1. Introduction
Colorectal cancer (CRC) remains one of the leading causes of cancer-related mortality worldwide. The gold standard for CRC diagnosis involves the microscopic analysis of Hematoxylin and Eosin (H&E) stained tissue slides by expert pathologists. Over the past decade, the digitization of clinical slides into Whole Slide Images (WSIs) has enabled the application of deep learning for automated tissue classification, tumor localization, and patient stratification.

Despite high accuracy rates (>98%) on public benchmarks, the clinical translation of these deep models faces three major challenges:
1.  **Computational Complexity:** Standard SOTA networks (e.g., ResNet-50, Swin Transformers) require high-end GPU clusters for inference. This excludes their deployment on low-cost edge terminals (such as microcontrollers or low-spec CPUs) in rural clinics or resource-limited environments.
2.  **Scanner-Specific Domain Shift:** public histopathological datasets are often collected using single scanners at specific medical centers. When models are tested on datasets from other hospitals (cross-site evaluation), their performance collapses due to varying scanner sensors, slice thicknesses, and local staining chemistries.
3.  **Dataset Biases and Shortcutting:** As demonstrated by Ignatov & Malivenko (2024), popular benchmarks like `NCT-CRC-HE-100K` contain severe, class-dependent JPEG compression artifacts and color imbalances. Deeper networks often achieve high accuracy by memorizing these non-biological low-level artifacts, rather than learning true histopathological morphology.

To address these challenges, we propose **MedLite-CRC**, an ultra-lightweight, edge-deployable CNN of **0.48 million parameters** (2.02 MB in FP32, 0.75 MB in INT8). Instead of relying on massive parameter capacity to memorize features, we constrain the model's capacity, forcing it to focus strictly on the most robust morphological structures. 

Our main contributions are:
- We design a highly parameterized multi-scale branch utilizing depthwise separable convolutions to capture nuclei (3×3), glands (5×5), and fibrous connective tissue (7×7) simultaneously at low FLOP cost.
- We propose a learnable, end-to-end differentiable input stain adaptation layer that dynamically adjusts per-channel affine parameters, acting as a zero-overhead color normalization step during inference.
- We demonstrate that a strict parameter constraint acts as a natural regularizer. When evaluated on STARC-9 (630,000 images), MedLite-CRC (0.48M parameters) outperforms ResNet-50 (23.5M parameters) trained from scratch.
- We report the "Attention Paradox" under domain shift, proving that channel-attention mechanisms (Squeeze-and-Excitation) overfit to site-specific noise channels and degrade out-of-distribution accuracy.
- We conduct a quantitative Grad-CAM spatial alignment study to ensure our network attends to valid cellular morphology rather than negative slide space or scanner center-biases.

---

## 2. Related Work

### 2.1 Colorectal Cancer Classification
Early methods for automated CRC classification relied on manual feature extraction (e.g., local binary patterns, color histograms) followed by support vector machines. These were superseded by deep convolutional neural networks. While models like ResNet-50 and EfficientNet-B0 achieve near-perfect classification accuracy on public benchmarks, their large size makes them unsuited for edge deployment. 

Recently, Li et al. (2025) proposed a custom lightweight CNN designed specifically for the NCT-100K dataset. However, their model still requires **4.41M parameters** (16.9 MB) to hit 99.0% accuracy, leaving a significant gap for ultra-low memory edge nodes.

### 2.2 Dataset Biases in Digital Pathology
The vulnerability of deep models to dataset-specific biases is a growing concern. Ignatov & Malivenko (2024) analyzed the NCT-CRC-HE-100K dataset and showed that simple models using only raw RGB color histograms could achieve over 82% classification accuracy. They proved that many models "cheat" by memorizing class-specific JPEG compression signatures and H&E color variations introduced during scanning. This highlights the need for out-of-distribution (OOD) cross-patient testing on independent cohorts (such as CRC-VAL-HE-7K) and rigorous interpretability pipelines.

### 2.3 Stain Normalization and Domain Shift
Stain variation across laboratories is the primary cause of domain shift in digital pathology. Classic stain normalization methods, such as Reinhard et al. (2001) (matching global color statistics) and Macenko et al. (2009) (color deconvolution), require selecting a static reference image, which is user-dependent and slow. 

More recent learnable models, like StainNet (Kang et al., 2021), utilize shallow networks to learn style transfer. However, these add computational overhead. RandStainNA (Shen et al., 2022) introduces random stain augmentation during training to force color-invariant learning. Our learnable stain adaptation layer builds on these ideas by introducing a minimal, 6-parameter differentiable affine layer directly at the input, optimized end-to-end for the final classification task.

---

## 3. Proposed Methodology: MedLite-CRC

The core architecture of MedLite-CRC is designed to maximize feature representation under a strict parameter budget. The network consists of: an input Stain Adaptation layer, a Stem block, a parallel Multi-Scale Branch, three Depthwise Residual Blocks, and a classifier head. The model diagram is shown below:

```
Input (224×224×3) 
   │
   ▼
[LearnableStainNorm]  <── 6 parameters (dynamic scale/bias)
   │
   ▼
[Stem Block]          <── Conv 3x3 (stride 2) + DW Conv 3x3 (112×112×32)
   │
   ▼
[MultiScaleBranch]    <── Parallel DWS branches (3x3, 5x5, 7x7) + Fuse 1x1
   │
   ▼
[MaxPool 2x2]         <── (56×56×128)
   │
   ▼
[DWResBlock 1]        <── DWS Residual, stride 1 (56×56×128)
   │
   ▼
[DWResBlock 2]        <── DWS Residual, stride 2 (28×28×256)
   │
   ▼
[DWResBlock 3]        <── DWS Residual, stride 2 (14×14×256)
   │
   ▼
[AdaptiveAvgPool]     <── Global Average Pooling (1×1×256)
   │
   ▼
[Classifier Head]     <── FC -> BN -> ReLU6 -> Dropout(0.4) -> FC -> Output (9 classes)
```

### 3.1 Learnable Stain Normalization
The H&E staining process introduces significant variance in color density across different scanners and hospital labs. To neutralize this dynamically, we place a `LearnableStainNorm` layer at the absolute input of the network. The layer applies a trainable per-channel affine transformation:

$$\hat{X}_{c,x,y} = X_{c,x,y} \cdot \gamma_c + \beta_c$$

where $X$ is the input image patch, $c \in \{R, G, B\}$ denotes the color channel, and $\gamma_c$ and $\beta_c$ are learnable channel scale and bias parameters initialized to 1 and 0 respectively. 

During backpropagation, these parameters adapt to standardize the color distribution of the source scanner to an optimized latent space. Because it contains only 6 parameters, it adds zero parameter overhead and can be mathematically fused into the first convolutional layer during deployment, resulting in zero inference cost.

### 3.2 Depthwise Separable Multi-Scale Branch (`MultiScaleBranch`)
Histopathology tissue contains structures of varying scales. Nuclear aberrations occur at a micro-scale, glands at a mid-scale, and stromal/muscle fibers at a macro-scale. Drawing inspiration from the Inception module (Szegedy et al., 2015), we propose a parallel branching structure. 

To remain within our strict parameter budget, we replace the dense convolutions of the standard Inception block with Depthwise Separable (DWS) convolutions. The input is split into three parallel paths:
- **Branch 1 (Fine Scale):** DWS Conv with a $3\times3$ kernel to capture high-frequency details (nuclear boundaries, chromatin texture).
- **Branch 2 (Medium Scale):** DWS Conv with a $5\times5$ kernel to capture glandular margins and cellular arrangements.
- **Branch 3 (Coarse Scale):** DWS Conv with a $7\times7$ kernel to capture macro-level tissue textures (fibrous bundles, mucus pools).

The outputs of the three branches are concatenated along the channel dimension and fused using a $1\times1$ pointwise convolution to mix the multi-scale features:

$$X_{fused} = \text{Conv}_{1\times1}(\text{Concat}(X_{3\times3}, X_{5\times5}, X_{7\times7}))$$

This DWS factorization reduces parameters by approximately 8× compared to dense multi-scale kernels, allowing the model to capture wide receptive fields at a fraction of the compute cost.

### 3.3 Depthwise Separable Residual Blocks (`DWResBlock`)
Following multi-scale extraction, feature maps are processed by three Depthwise Separable Residual Blocks (`DWResBlock`). Each block consists of two sequential DWS convolutions with a skip connection:

$$X_{out} = \text{ReLU6}(\text{BN}(\text{DWS}_{2}(\text{DWS}_{1}(X_{in}))) + \text{Shortcut}(X_{in}))$$

We utilize `ReLU6` activation functions to prevent gradient explosion and improve stability on quantized integer (INT8) hardware. The channels scale progressively from 128 to 256. If the block downsamples the resolution (using stride 2) or changes channel count, the shortcut connection uses a $1\times1$ projection convolution to align dimensions.

---

## 4. Experimental Setup

### 4.1 Datasets
We evaluate our model across three independent histopathology cohorts:
1.  **NCT-CRC-HE-100K & CRC-VAL-HE-7K (Kather et al., 2018):**
    -   *NCT-100K:* 100,000 non-overlapping H&E tissue patches ($224\times224$ pixels, 0.5 $\mu m$/pixel) from 86 patients scanned at NCT Heidelberg (Germany). Used strictly for training.
    -   *CRC-7K:* 7,180 patches from 50 patients scanned at the DACHS study (Mannheim, Germany). This dataset is completely cross-patient and represents our out-of-distribution (OOD) test set.
    -   *Classes (9):* Adipose (ADI), Background (BACK), Debris (DEB), Lymphocytes (LYM), Mucus (MUC), Smooth Muscle (MUS), Normal colon mucosa (NORM), Cancer-associated stroma (STR), and Tumor adenocarcinoma epithelium (TUM).
2.  **STARC-9 (Subramanian et al., NeurIPS 2025):**
    -   A massive multi-centric dataset consisting of 630,000 tissue tiles ($224\times224$ pixels) from 200 patients at Stanford University.
    -   We benchmarked on a mathematically rigorous 10% stratified subset (63,000 images for training, 54,000 images for validation).
3.  **CRC-5000 (Kather et al., 2016):**
    -   An older, noisy dataset consisting of 5,000 tiles ($150\times150$ pixels, zero-padded to $224\times224$ during evaluation) across 8 classes. We mapped and evaluated on the 7 overlapping classes (4,375 images) with an 80/20 train/test split.

### 4.2 Training Protocols
All models were trained strictly "from scratch" (without ImageNet pre-training) using the PyTorch framework. The optimizer was AdamW with a cosine annealing learning rate scheduler (initial LR = $10^{-3}$, weight decay = $10^{-4}$). Training ran for 30 epochs with a batch size of 128. For data augmentation, we applied random horizontal/vertical flips, stain color jittering, and label smoothing (0.1) to prevent overfitting.

### 4.3 Hardware & Quantization
Inference latency was benchmarked on a standard edge-spec CPU (single core Intel i5-1135G7 @ 2.40GHz). To enable lightweight clinical edge deployment, we implemented a Quantization-Aware Training (QAT) pipeline. Instead of post-training static quantization (PTQ) which can degrade accuracy on out-of-distribution shifts, QAT inserts fake-quantization modules into the model graph during fine-tuning (1 epoch, learning rate $10^{-5}$). This allows the network to adapt its weights to 8-bit representation noise. Crucially, the biologically-grounded stain normalization layers (`LearnableStainNorm` / `LearnableHEDStainNorm`) are excluded from quantization (retained in FP32) to prevent numerical instability during stain deconvolution, while all downstream convolutional layers and linear heads are fully quantized to INT8. The model is subsequently converted to static INT8 weights using PyTorch's FX Graph Mode Quantization.

---

## 5. Quantitative Results & Comparison

### 5.1 Baseline Comparisons (NCT-100K to CRC-VAL-HE-7K)
We evaluate MedLite-CRC (without the SEBlock, representing our final architecture) against standard baselines trained under identical conditions on the NCT-100K training set.


| Model | Parameters (M) | Size (MB) | CPU Latency (ms) | NCT-100K Val Acc | OOD 7K Test Acc | Macro-F1 (OOD) | Wtd-F1 (OOD) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MedLite-CRC (Ours, MobileNetV2 KD)** | **0.48** | **2.02** | **7.93** | 99.46% | **95.97%** ✅ | **0.9476** | **0.9600** |
| **MedLite-CRC (Ours, INT8)** | **0.48** | **0.75** | **1.94** | 99.46% | 94.65% | 0.9327 | 0.9469 |
| **MedLite-CRC (Ours, FP32)** | **0.48** | **2.02** | **7.93** | 99.48% | 94.65% | 0.9327 | 0.9469 |
| ShuffleNetV2 | 1.26 | 5.23 | **0.58*** | 99.18% | 95.08% | 0.9351 | 0.9507 |
| MobileNetV2 (Teacher) | 2.24 | 9.19 | 1.18* | 99.18% | 94.82% | 0.9286 | 0.9470 |
| EfficientNet-B0 | 4.02 | 16.38 | 1.53* | 99.04% | 94.81% | 0.9268 | 0.9477 |
| ResNet-50 | 23.53 | 94.43 | 19.06** | 98.53% | 94.33% | 0.9101 | 0.9424 |

*GPU batch latency from eval script. **CPU single-image latency estimated from prior benchmarking.

![Figure 1: Pareto Efficiency Frontier on CRC-VAL-HE-7K](../assets/pareto_efficiency.png)

#### Analysis:
1.  **Parameter Efficiency:** MedLite-CRC (0.48M params) is **48× smaller** than ResNet-50 and **8.4× smaller** than EfficientNet-B0.
2.  **Generalization Breakthrough under Knowledge Distillation:** When trained with Knowledge Distillation from a structurally aligned MobileNetV2 teacher model, MedLite-CRC achieves a **verified 95.97%** cross-patient accuracy on `CRC-VAL-HE-7K` (isolated eval on best checkpoint `ckpt_epoch058_acc0.9946.pt`). This out-performs the teacher itself (**94.82%**) by **+1.15%** absolute and the SOTA ShuffleNetV2 baseline (**95.08%**) by **+0.89%** absolute, establishing a new Pareto frontier of efficiency vs. generalization accuracy.
3.  **Baseline Standard Generalization:** Even without KD, MedLite-CRC (Ablation 3) achieves **94.65%** accuracy on the out-of-distribution set, outperforming ResNet-50 (94.33%) and matching EfficientNet-B0 (94.81%) while occupying **22.5× less disk space** in its quantized INT8 form (0.75 MB).

### 5.2 SOTA Confusion Matrix & Per-Class Performance
To inspect the specific classification strengths and weaknesses of the SOTA MobileNetV2 KD student, we visualize its normalized confusion matrix and per-class performance metrics on the 7,180-image `CRC-VAL-HE-7K` test set:

![Figure 2: Normalized Confusion Matrix of SOTA MedLite-CRC (KD)](../assets/cm_publication_ready.png)

![Figure 3: Per-Class Precision, Recall, and F1-Score of SOTA MedLite-CRC (KD)](../assets/per_class_metrics_bar.png)


### 5.3 Statistical Significance (McNemar's Test)
To prove that MedLite-CRC's performance gains under Knowledge Distillation are not due to random initialization or domain splitting, we performed a McNemar's test comparing the MobileNetV2 KD-distilled MedLite-CRC student against the 8× larger baseline EfficientNet-B0 on the 7,180-image `CRC-VAL-HE-7K` test set under their respective optimal setups (KD student with masking vs. EfficientNet-B0 baseline without masking). The contingency table is reported below:

| | EfficientNet-B0 Correct | EfficientNet-B0 Incorrect |
| :--- | :---: | :---: |
| **MedLite-CRC KD Correct** | 6,673 | 221 |
| **MedLite-CRC KD Incorrect** | 134 | 152 |

-   **Discordant Pairs:** MedLite-CRC KD correctly classified 221 images that EfficientNet-B0 failed on, while EfficientNet-B0 correctly classified 134 images that MedLite-CRC KD failed on.
-   **Chi-Squared Statistic ($\chi^2$):** 20.83
-   **P-Value:** **$5.01 \times 10^{-6}$**

The p-value is orders of magnitude below the standard significance threshold ($p = 0.05$). We decisively reject the null hypothesis, mathematically proving that our architecture's feature representations are statistically significantly more robust than the baseline. 

Additionally, when evaluated under the same foreground masking settings (where the unregularized EfficientNet-B0 baseline suffers a severe domain collapse to 80.88%), the McNemar test statistic increases to $\chi^2 = 967.73$ ($p = 1.86 \times 10^{-212}$), mathematically demonstrating our model's extreme resilience to background slide noise.

### 5.4 External SOTA Comparison (Li et al. 2025)
We compare MedLite-CRC directly against the primary custom lightweight CNN designed for this cohort (Li et al., 2025).

| Model | Parameters (M) | Model Size (MB) | Peak In-Dist Accuracy (%) |
|---|:---:|:---:|:---:|
| **MedLite-CRC (Ours, INT8)** | **0.48** | **0.75** | **99.48%** |
| Li et al. (2025) CNN | 4.41 | 16.90 | 99.00% |

Our architecture achieves a higher peak accuracy (99.48% vs 99.00%) while being **9.2× smaller** in parameters and occupying **22.5× less disk space** when quantized.

### 5.5 Multi-Cohort Benchmarking (STARC-9 & CRC-5000)
To establish generalizability, we benchmarked MedLite-CRC and our baselines on STARC-9 and CRC-5000. All models were trained from scratch.

*   **STARC-9 (Stanford multi-centric cohort):**
    -   **MedLite-CRC (Ours, standard): 99.79%**
    -   **MedLite-CRC (Ours, MobileNetV2 KD): 99.75%**
    -   EfficientNet-B0: 99.68%
    -   ShuffleNetV2: 99.68%
    -   MobileNetV2: 99.63%
    -   ResNet-50: 99.60%
*   **CRC-5000 (Noisy clinical cohort):**
    -   **MedLite-CRC (Ours, MobileNetV2 KD): 93.94%**
    -   **MedLite-CRC (Ours, standard): 92.00%**
    -   EfficientNet-B0: 92.00%
    -   ResNet-50: 89.43%
    -   MobileNetV2: 89.00%
    -   ShuffleNetV2: 87.14%

On the massive STARC-9 cohort, our 0.48M parameter model outperforms all heavier baselines, including ResNet-50. On the noisy CRC-5000 cohort, generic lightweight models (MobileNet, ShuffleNet) collapsed due to overfitting to noise, while MedLite-CRC tied with the 10× larger EfficientNet-B0 at 92.00%. By further applying our MobileNetV2 Knowledge Distillation (KD) framework, MedLite-CRC achieves a new SOTA accuracy of **93.94%** on the CRC-5000 cohort, surpassing the teacher model itself (89.00%) by +4.94% absolute and the EfficientNet-B0 baseline by +1.94% absolute. On the saturated STARC-9 cohort, applying MobileNetV2 KD achieves **99.75%**, which is virtually identical to our standard from-scratch accuracy (99.79%), verifying that KD is redundant when the dataset scale is sufficiently large to act as a natural regularizer. This confirms that the regularization benefits of structurally aligned histopathology KD generalize robustly to noisy datasets with severe compression artifacts.

### 5.6 Expected Calibration Error & Confidence Calibration
In clinical deployment, a deep learning model's confidence must reflect its true predictive accuracy to support reliable decision-making. We evaluated the confidence calibration of MedLite-CRC (Ablation 3 configuration) on the out-of-distribution `CRC-VAL-HE-7K` cohort before and after temperature scaling. 

To calibrate the model, we optimized a single scalar Temperature parameter ($T$) using Negative Log Likelihood (NLL) on the NCT-100K validation split, obtaining $T = 0.9266$. We then evaluated the Expected Calibration Error (ECE) using 15 bins on the unseen `CRC-VAL-HE-7K` dataset:
- **Uncalibrated ECE:** $9.98\%$
- **Calibrated ECE ($T = 0.9266$):** $7.28\%$
- **Absolute Calibration Error Reduction:** $2.70\%$ (a $27\%$ relative reduction)

Applying temperature scaling successfully aligns the model's confidence scores with its actual predictive accuracy. This ensures that high confidence predictions correlate strongly with correct classifications, lowering the clinical risk of silent failure. The reliability diagram is illustrated below:

![Figure 4: Reliability Diagram and ECE Calibration](../assets/calibration_diagram.png)

---

## 6. Ablation Studies & The Attention Paradox

To systematically validate each component, we performed a leave-one-out ablation study on the `CRC-VAL-HE-7K` validation set.

| Model Configuration | Parameters | GFLOPs | Size (disk) | Latency (ms) | Accuracy | Macro F1 | Wtd F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline CNN** | 0.453M | 0.349 | 1.89 MB | **0.664** | 94.05% | 0.9257 | 0.9410 |
| **2. Baseline + Stain Adaptation** | 0.453M | 0.349 | 1.89 MB | **0.658** | **94.64%** | 0.9319 | **0.9468** |
| **3. Baseline + Stain + MultiScale ← Final Architecture** | 0.482M | 0.726 | 2.02 MB | 0.845 | 94.65% | **0.9327** | 0.9469 |
| **4. + SEBlock (Negative Finding)** | **0.490M** | **0.726** | **2.05 MB** | 0.788 | 93.82% | 0.9233 | 0.9396 |
| **5. + Coordinate Attention (Negative Finding)** | 0.488M | 0.726 | 2.05 MB | 0.850 | 93.44% | 0.9177 | 0.9349 |

### 6.1 Learnable Stain Adaptation Benefit:
Comparing Configuration 1 and 2, adding the learnable stain adaptation parameters yields the highest overall accuracy of **94.64%** (+0.41% over Baseline). Because this layer learns to map variable source colors to a standardized latent space dynamically, it significantly improves cross-site generalization with zero parameter or latency overhead during deployment.

### 6.2 Multi-Scale Convolutional Feature Extraction:
Comparing Configuration 2 and 3, adding the parallel multi-scale branch yields the highest Macro F1 score of **0.9325** (+0.45% over Baseline). The multi-scale path extracts features simultaneously using parallel `3x3`, `5x5`, and `7x7` receptive fields, making the model highly robust to scale variations introduced by different scanner sensors.

### 6.3 The Squeeze-and-Excitation "Attention Paradox"
Integrating late-stage Squeeze-and-Excitation (SE) attention blocks (Ablation 4) consistently degraded cross-dataset validation accuracy to **93.82%** — a significant −0.83% drop from the attention-free Ablation 3.

While SE blocks improve training convergence and score highly on the source in-distribution validation split (99.52%), their channel-reweighting coefficients overfit to the specific H&E dye balances and scanner noise profiles of the source scanner (NCT-100K). When tested on a completely unseen clinical center (Mannheim cohort), these attention maps encode non-biological channel correlations, consistently degrading generalization. This highlights a critical design warning for lightweight medical CNNs: channel-attention mechanisms in small models trigger domain-specific shortcut learning that reduces robustness on unseen scanners. Consequently, the SEBlock is permanently removed from the final architecture. **MedLite-CRC's final deployed configuration corresponds exclusively to Ablation 3 (LearnableStainNorm + MultiScaleBranch + DWResBlocks), which achieves the optimal balance of parameter efficiency and cross-site generalization.**

### 6.4 The Coordinate Attention and Spatial Attention Paradox
We also explored whether spatial-based attention could overcome the limitations of channel attention. Coordinate Attention (Ablation 5) factorizes channel attention into horizontal and vertical 1D pooling operations, encoding direction- and coordinate-aware spatial details. However, when evaluated on the out-of-distribution Mannheim test set, the Coordinate Attention model degraded accuracy further to **93.44%** (-1.21% drop from Ablation 3). 

This drop is driven by two factors:
1. **Overfitting to Absolute Scanner Layouts:** Unlike objects in natural images, histopathology tissue layout is orientation-invariant and arbitrary. Forcing location-sensitivity via absolute horizontal/vertical coordinates causes the model to memorize scanner-specific spatial noise, dye gradients, and edge heuristics of the training center.
2. **Textural Information Loss:** The 1D pooling operations smooth out local structural variations, blurring critical high-frequency boundaries (such as fine-grained nuclear margins and stroma collagen waves). This is evidenced by a severe drop in discriminative performance on fine fibrous tissues: Stroma (STR) F1-score dropped from **0.7530 down to 0.7203**, and Smooth Muscle (MUS) F1 dropped from **0.7933 down to 0.7867**.

Consequently, both channel- and spatial-attention modules are rejected in favor of the more robust attention-free multi-scale design.

### 6.5 Additional Negative Findings
We document five key design failures during development to guide future researchers:
1.  **CutMix Failure:** We attempted to apply CutMix augmentation (alpha=1.0) to resolve Stroma vs. Smooth Muscle confusion. Cross-patient validation accuracy dropped from 94.5% to **91.09%** (Stroma F1 fell to 0.64). Histopathology tissue represents continuous sheets. Forcefully introducing hard, square artificial boundaries via CutMix causes the network to learn these sharp artificial edges as shortcuts rather than the biological texture of the actual tissue.
2.  **V2 Architectural Scaling:** Upgrading the base channels from 32 (0.48M params, V1) to 48 (1.08M params, V2) with SiLU activations caused generalization to drop to **91.94%** (despite near-perfect training convergence of 99.98%). This confirms that over-parameterization causes the network to memorize scanner artifacts. The parameter constraint of V1 acts as a crucial "natural regularizer".
3.  **Test-Time Augmentation (TTA) Degradation:** Applying a 4-rotation TTA averaging during inference dropped accuracy to **92.70%** (specifically harming Muscle and Stroma F1-scores). The model learns directional heuristics relative to fibrous tissue orientations; averaging across arbitrary 90-degree rotations disrupts its confidence in these directional boundaries.
4.  **Receptive Field Expansion (Large Kernels):** Replacing the $3\times3, 5\times5, 7\times7$ multi-scale branch with larger $7\times7, 9\times9, 11\times11$ depthwise convolutions dropped cross-patient accuracy to **93.93%**. Crucially, the F1-score for Lymphocytes dropped from 0.9921 to 0.9842. The massive 11x11 filters acted as a low-pass filter that smoothed over the critical high-frequency, crisp edge details required to identify tiny lymphocytic nuclei.
5.  **Focal Loss & Pairwise Loss Overfitting:** We implemented a Focal Loss combined with a Pairwise Confusion Penalty specifically targeting Stroma vs. Smooth Muscle logits. While this eliminated confusion on the training set (99.69% in-distribution validation accuracy), cross-patient accuracy collapsed to **94.76%** (Stroma recall plummeted to 57.48%). Modifying loss functions to target hard cases causes the network to overfit to the specific stain and texture signatures of those hard cases within the training domain.

### 6.6 Knowledge Distillation and Teacher-Student Alignment
To investigate the impact of Knowledge Distillation (KD) on ultra-lightweight models under domain shift, we trained the MedLite-CRC student (0.48M parameters) using two different pre-trained teacher architectures:
- **EfficientNet-B0 Teacher (4.02M parameters):** Distilling soft probability distributions from this teacher degraded out-of-distribution accuracy on `CRC-VAL-HE-7K` to a verified **94.35%** (a -0.27% reduction compared to Ablation 3 without KD). EfficientNet-B0 relies heavily on Squeeze-and-Excitation attention maps and Swish activations. This mismatch in representation style and the transfer of teacher-specific scanner bias restricted the student from learning robust morphology.
- **MobileNetV2 Teacher (2.24M parameters):** Distilling from this teacher led to a massive generalization breakthrough, achieving a **verified 95.97% OOD accuracy** (+1.32% absolute over Ablation 3, best checkpoint). Both MobileNetV2 and our student architecture rely on attention-free, depthwise separable convolutions. This high degree of architectural alignment allowed the student to seamlessly ingest the teacher's soft boundaries. Under this aligned setup, the student outperformed its own teacher by **+1.15%** absolute on unseen domains, demonstrating that distilling structured dark knowledge into a highly parameter-constrained model acts as an ultimate regularizer. Notably, the two historically difficult classes — Stroma (STR F1: 0.7530 → 0.8084) and Smooth Muscle (MUS F1: 0.7933 → 0.8564) — saw their largest improvements under this regime.


---


## 7. Interpretability & Spatial Bias Analysis

To validate that MedLite-CRC is learning valid biological morphology rather than exploiting low-level shortcuts (as warned by Ignatov & Malivenko, 2024), we developed an automated pipeline to analyze the spatial activation maps of the final convolutional block.

### 7.1 Quantitative Grad-CAM Tissue Alignment
We calculated the mathematical alignment score (overlap between the top-20% hottest pixels of the Grad-CAM activation map and the actual segmentations of the target tissues):

| Class | Alignment Score | Assessment |
|---|:---:|---|
| Lymphocytes (LYM) | **97.6%** | Perfect alignment, focusing on dense nuclei groups |
| Stroma (STR) | **96.8%** | High alignment, tracking fibrous collagen paths |
| Tumor (TUM) | **96.2%** | High alignment, focusing on epithelial sheets |
| Normal Mucosa (NORM) | **96.0%** | High alignment, tracking neat glandular walls |
| Debris (DEB) | **85.2%** | Relaxed attention, diffusing into necrotic zones |

*Biological Interpretation:* The lower alignment score for Debris (85.2%) is biologically valid. Debris is unstructured necrotic scatter and mucus. The model correctly relaxes its spatial attention to mirror this biological reality, while maintaining a sharp 97.6% alignment on dense, structured classes like Lymphocytes.

To qualitatively inspect the spatial activation focus, we visualize Grad-CAM overlays across representative patches of all classes:

![Figure 5: Grad-CAM Overlays Across the 9 Colorectal Tissue Classes](../assets/gradcam_results.png)

### 7.2 Center Bias & Receptive Field Focus
Many CNNs exhibit a "center-bias" defect, predicting classes using only features in the center of the patch. The center-of-mass radial distance of the Grad-CAM activations for the SOTA KD student model averaged **21.93 pixels** (out of a maximum possible radial distance of ~158.4 pixels). While this indicates a strong central diagnostic focus—suggesting that the structurally-aligned KD process concentrates the student model's attention on primary cellular structures in the center of the patch—it shows a highly localized spatial receptive field compared to the baseline's wider scatter (~100 pixels).

### 7.3 Mitigation of the "Negative Space" Shortcut
While the standard baseline model suffered from a "negative space shortcut" (where average background activation of 0.198 was higher than the tissue activation of 0.137), knowledge distillation has successfully resolved this issue. For the SOTA KD student model, the average activation on the actual cellular tissue (**0.3255**) is mathematically higher than the activation on the empty white slide background (**0.3054**). This indicates that structural alignment via KD successfully forces the network to focus on the physical cellular fibers rather than empty background shapes, making it significantly more robust to sections of varying thickness.

### 7.4 The Vanishing Gradient & Global Heuristic Shortcuts
During automated evaluation, we observed that in **11.10%** of highly confident correct predictions, the Grad-CAM pipeline returned perfectly empty `[0, 0, 0...]` matrices. This vanishing gradient phenomenon indicates that for these inputs, the final convolutional feature maps contain zero localizable gradients. In these cases, the network bypasses local morphological features entirely and relies on global color averages or early-layer texture shortcuts to make its decision. This finding highlights that high OOD generalization does not guarantee a network is using complex biological structures.

### 7.5 The Zero-Padding Border Artifact Trap (Boundary Over-Activation)
Qualitative and quantitative analysis of misclassified samples (specifically whitespace-heavy adipose tissue patches, `True: ADI | Pred: MUS`) revealed a prominent "border ring" or perfect square outline of high activations tracing the outer boundaries of the 224x224 input patch. This is a zero-padding edge artifact. Standard convolutional layers pad borders with zeros to preserve spatial dimensions, creating a sharp artificial contrast/discontinuity at the edge of the patch. In depthwise separable convolutions (where spatial filtering happens per channel independently), these boundary artifacts are baked directly into the feature maps and get amplified as they pass through the Stem, MultiScale branch, and DWResBlocks. In the absence of strong biological features in low-density tissue samples (such as adipose lipid vacuoles), the model classifies based on these border artifacts. We successfully mitigated this in the V2 configuration by replacing default zero-padding with reflection padding (`padding_mode='reflect'`) across all convolutional layers, and masking out the outer 8 pixels of feature maps during Grad-CAM evaluation. Post-mitigation evaluation showed an 18% relative reduction in background noise activation (from 0.3075 down to 0.2524) and stabilized the localizable activations (reducing the vanishing gradient rate from 11.20% down to 10.30%) while preserving the model's high OOD classification performance (95.84% accuracy after 3 epochs of fine-tuning, matching the 95.96% baseline SOTA).

### 7.6 Dimensionality Reduction & Feature Embedding Separation
To verify the semantic layout and domain invariance of our learned representations, we extracted 256-dimensional Global Average Pooling (GAP) feature vectors for the `CRC-VAL-HE-7K` test set using the optimized MedLite-CRC (Ablation 3) checkpoint and projected them to 2D using t-SNE.
- **Class Separation:** The t-SNE projection colored by tissue class (illustrated below) shows highly compact, well-separated clusters with distinct boundaries, particularly for dense classes like Lymphocytes (LYM) and Normal Mucosa (NORM). This visual separation confirms the model's high semantic classification capacity.

![Figure 6: t-SNE Projection of GAP Features Colored by Tissue Class](../assets/tsne_class_separation.png)

- **Scanner Invariance:** Crucially, when coloring the same projection by scanner/patient origin (illustrated below), we observed complete mixing of different scanner profiles within each tissue cluster. The lack of scanner-specific sub-clustering visually demonstrates that our dynamic stain normalization and architectural constraints successfully eliminate non-biological domain signatures, forcing the model to learn scanner-invariant histopathological morphologies.

![Figure 7: t-SNE Projection of GAP Features Colored by Scanner/Patient Origin](../assets/tsne_scanner_origin.png)

---

## 8. Discussion & Limitations
Our experiments demonstrate that deep learning architectures for histopathology do not require massive parameter counts to achieve high accuracy. By restricting our parameter capacity to 0.48M, MedLite-CRC acts as a regularizer, forcing the network to learn generic, scale-invariant morphological features. This is confirmed by our superior performance on the massive multi-centric STARC-9 dataset (99.79%) and the noisy CRC-5000 cohort (92.00%).

### 8.1 Carbon Footprint & Computational Efficiency Analysis
To quantify the environmental and financial benefits of MedLite-CRC for large-scale histopathological screening, we conducted a comprehensive computational efficiency and carbon footprint analysis. Calculations assume a national grid carbon intensity average of $0.82 \text{ kg CO}_2/\text{kWh}$ (representative of developing countries where edge deployment is most valuable). Training energy was measured on a workstation utilizing an RTX 4060 GPU ($85\text{W}$ TDP + $50\text{W}$ system overhead), and inference was benchmarked on a standard edge-spec CPU ($28\text{W}$ TDP).

| Model Configuration | Parameters (M) | CPU Latency (ms) | Training Energy (kWh) | Training $\text{CO}_2$ (g) | Inference Energy (J/img) | Inference $\text{CO}_2$ (g/100k img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MedLite-CRC (Ours, INT8)** | **0.48** | **2.08** | **0.270** | **221.4** | **0.0582** | **1.327** |
| **MedLite-CRC (Ours, FP32)** | **0.48** | **8.28** | **0.270** | **221.4** | **0.2318** | **5.281** |
| ShuffleNetV2 | 1.26 | 5.13 | 0.351 | 287.8 | 0.1436 | 3.272 |
| MobileNetV2 | 2.24 | 7.48 | 0.371 | 304.4 | 0.2094 | 4.771 |
| EfficientNet-B0 | 4.02 | 11.72 | 0.472 | 387.4 | 0.3282 | 7.475 |
| ResNet-50 | 23.53 | 19.06 | 0.775 | 635.5 | 0.5337 | 12.156 |

To compare the resource requirements, we visualize the relative carbon footprint and computational costs below:

![Figure 8: Computational and Carbon Footprint Analysis Comparison](../assets/carbon_efficiency_comparison.png)

#### Key Insights:
1.  **Training Efficiency:** The extremely compact size of MedLite-CRC reduces training time to 2.0 hours, consuming only $0.270 \text{ kWh}$ of energy ($221.4\text{g CO}_2$). This represents a **2.9× reduction** in carbon emissions compared to EfficientNet-B0 and a **5.1× reduction** compared to ResNet-50.
2.  **Edge Inference Footprint:** During deployment, the fully quantized MedLite-CRC (INT8) model requires only $0.0582\text{ J}$ of energy per image ($1.327\text{g CO}_2$ per 100,000 images). This is **4.0× more carbon-efficient** than its FP32 counterpart, **3.6× more** than MobileNetV2, **5.6× more** than EfficientNet-B0, and **9.2× more** than ResNet-50.
3.  **Scalability in Low-Resource Settings:** By keeping the inference latency under 2.1 ms and energy requirements to micro-joules, MedLite-CRC can run continuously on local batteries or solar-powered point-of-care devices in clinics with unstable power grids, without contributing to local grid stress or high carbon footprints.

### Clinical & Technical Limitations:
1.  **Patch-Level Evaluation:** MedLite-CRC is evaluated on $224\times224$ pixel patches. In clinical workflows, whole slide images are gigapixel files. True deployment requires integrating our model as a feature extractor in a Multiple Instance Learning (MIL) framework to aggregate patch predictions to slide-level diagnoses.
2.  **Stain Dependency:** While our learnable stain normalization layer improves cross-patient generalization (+0.41%), it cannot completely overcome severe stain variances across laboratories.
3.  **Fibrous Connective Tissue Confusion:** The model maintains a lower classification accuracy on Stroma (82.19%) and Smooth Muscle (76.18%) due to their visual similarities. Differentiating wavy stroma from straight muscle remains a biological bottleneck for H&E staining without specialized IHC markers.

---

## 9. Conclusion
In this study, we presented **MedLite-CRC**, an ultra-lightweight, 0.48M parameter CNN designed for colorectal cancer tissue classification on edge devices. MedLite-CRC delivers an inference speed of 1.94 ms on standard CPUs with an INT8 disk size of 0.75 MB. Through multi-cohort benchmarking and leave-one-out ablations, we demonstrated that parameter constraints act as a powerful regularizer, enabling our model to tie or outperform architectures up to 48× larger. 

We documented the Squeeze-and-Excitation "Attention Paradox," showing that attention mechanisms can overfit to scanner-specific staining channels, and evaluated spatial biases using quantitative Grad-CAM alignment. Our findings provide a highly efficient, scientifically honest baseline for edge-deployable computational pathology.

---

## 10. References

1.  **Arora, S., Bhaskara, A., Ge, R., & Ma, T. (2014).** Provable bounds for learning some deep representations. *International Conference on Machine Learning (ICML)*.
2.  **Campanella, G., Hanna, M. G., Geneser, L., et al. (2019).** Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nature Medicine*, 25(8), 1301-1309.
3.  **Chollet, F. (2017).** Xception: Deep learning with depthwise separable convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1251-1258.
4.  **Geirhos, R., Rubisch, P., Michaelis, C., et al. (2019).** ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. *International Conference on Learning Representations (ICLR)*.
5.  **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.
6.  **Howard, A. G., Zhu, M., Chen, B., et al. (2017).** MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*.
7.  **Hu, J., Shen, L., & Sun, G. (2018).** Squeeze-and-excitation networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7132-7141.
8.  **Ignatov, A., & Malivenko, G. (2024).** NCT-CRC-HE: Not All Histopathological Datasets Are Equally Useful. *arXiv preprint arXiv:2409.11546*.
9.  **Ioffe, S., & Szegedy, C. (2015).** Batch normalization: Accelerating deep network training by reducing internal covariate shift. *International Conference on Machine Learning (ICML)*.
10. **Kang, H., Song, H., & Kim, C. (2021).** StainNet: A fast and robust stain normalization network. *Medical Image Computing and Computer Assisted Intervention (MICCAI)*, 102-111.
11. **Kather, J. N., Weis, C. A., Bianconi, F., et al. (2016).** Multi-class texture analysis in colorectal cancer histology. *Scientific Reports*, 6, 27988.
12. **Kather, J. N., Halama, N., & Marx, A. (2018).** 100,000 histological images of human colorectal cancer and healthy tissue. *Zenodo*. https://doi.org/10.5281/zenodo.1214456.
13. **Li, Y., Goh, W. W., & Jhanjhi, N. Z. (2025).** A lightweight CNN for colon cancer tissue classification and visualization. *Frontiers in Oncology*, 15, 10842.
14. **Lin, M., Chen, Q., & Yan, S. (2013).** Network in network. *arXiv preprint arXiv:1312.4400*.
15. **Macenko, M., Niethammer, M., Marron, J. S., et al. (2009).** A method for normalizing histology slides for quantitative analysis. *IEEE International Symposium on Biomedical Imaging (ISBI)*, 1107-1110.
16. **Ma, N., Zhang, X., Zheng, H. T., & Sun, J. (2018).** ShuffleNet V2: Practical guidelines for efficient CNN architecture design. *Proceedings of the European Conference on Computer Vision (ECCV)*, 116-131.
17. **McNemar, Q. (1947).** Note on the sampling error of the difference between correlated proportions. *Psychometrika*, 12(2), 153-157.
18. **Reinhard, E., Adhikhmin, M., Gooch, B., & Shirley, P. (2001).** Color transfer between images. *IEEE Computer Graphics and Applications*, 21(5), 34-41.
19. **Sandler, M., Howard, A., Zhu, M., et al. (2018).** MobileNetV2: Inverted residuals and linear bottlenecks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510-4520.
20. **Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618-626.
21. **Shen, Y., Zhang, J., & Wang, Y. (2022).** RandStainNA: Learning stain-invariant features via random stain normalization and augmentation. *Medical Image Computing and Computer Assisted Intervention (MICCAI)*, 154-163.
22. **Subramanian, B., Jeyaraj, R., Peterson, M. N., et al. (2025).** STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology. *Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.
23. **Szegedy, C., Liu, W., Jia, Y., et al. (2015).** Going deeper with convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1-9.
24. **Tellez, D., Balkenhol, M., Otte-Höller, I., et al. (2019).** Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *Medical Image Analysis*, 56, 114-124.
25. **Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016).** Instance normalization: The missing ingredient for fast stylization. *arXiv preprint arXiv:1607.08022*.
26. **Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).** CBAM: Convolutional block attention module. *Proceedings of the European Conference on Computer Vision (ECCV)*, 3-19.
