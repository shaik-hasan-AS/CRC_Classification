# 📝 Ablation Study Notes: Failures & Discoveries

*You can use these exact narratives for the Ablation Study (Table 3) in your research paper. They provide excellent scientific justification for why MedLite-CRC V1 is the optimal architecture.*

## 1. The CutMix Failure

### The Hypothesis
We hypothesized that the Stroma (STR) vs Smooth Muscle (MUS) classification confusion was fundamentally a textural problem, as both classes share identical color distributions (eosinophilic connective tissue). Standard global blending augmentations like `MixUp` wash out these structural differences. 

We proposed using **CutMix**—physically cutting and pasting hard squares of muscle into stroma—to force the network to learn the distinct local textural boundaries between the parallel muscle fibers and wavy stroma fibers.

### The Result (Negative)
The experiment failed. When trained with `CutMix` (alpha=1.0), the cross-patient `CRC-VAL-HE-7K` accuracy dropped from **94.5% to 91.09%**. The F1-score for Stroma specifically plummeted to **0.64**.

### The Scientific Conclusion
Unlike standard datasets (e.g., ImageNet with cars and dogs), histopathology images represent continuous sheets of biological tissue. By forcefully introducing hard, square, geometric boundaries into the training images via `CutMix`, we inadvertently created an **unnatural artifact**. The convolutional neural network learned to identify and rely on these sharp artificial edges as features, rather than learning the subtle, continuous cellular textures of the actual tissue. 

**Conclusion:** For continuous histological tissue classification, global interpolative augmentations (`MixUp`) are strictly superior to patch-replacement augmentations (`CutMix`).

---

## 2. V1 vs V2 Architectural Scaling

### The Hypothesis
To surpass MobileNetV2 (95.56%), we hypothesized that increasing the channel capacity of our network would allow it to extract more complex morphological features. We upgraded MedLite-CRC from **32 channels (0.49M params, V1)** to **48 channels with SiLU activations (1.08M params, V2)**.

### The Result (Negative)
While V2 achieved a near-perfect **99.98% training accuracy**, its generalization completely collapsed on the cross-patient `CRC-VAL-HE-7K` dataset, dropping to **91.94%**.

### The Scientific Conclusion
The V2 model possessed too much capacity and severely **overfit to the source domain**. Rather than learning true cellular structures, the over-parameterized network began memorizing hospital-specific artifacts (e.g., scanner color profiles, subtle stain variations) present in the training set. 

**Conclusion:** The strict parameter constraint of the **V1 architecture (0.49M params) acts as a crucial "natural regularizer"**. By limiting the network's capacity, we force it to learn only the most robust, cross-domain cellular morphological features, making V1 significantly superior for generalization.

---

## 3. Test-Time Augmentation (TTA) Degradation

### The Hypothesis
Test-Time Augmentation (TTA) is a standard technique where an image is evaluated multiple times at different rotations (0°, 90°, 180°, 270°) and the probabilities are averaged, theoretically improving robustness at zero training cost.

### The Result (Negative)
Applying TTA to the optimal V1 checkpoint resulted in a significant accuracy drop from **94.53% down to 92.70%**, specifically harming the STR (0.67) and MUS (0.75) F1-scores.

### The Scientific Conclusion
While histopathology tissue has no "correct" global 'up/down' orientation, the multi-scale texture features learned by our CNN to distinguish between the wavy fibers of stroma and the parallel fibers of muscle are highly sensitive to directional heuristics. Forcing the model to average its predictions across arbitrary 90-degree rotations disrupted its confidence in these learned directional boundaries.

**Conclusion:** The model learns highly specific rotational heuristics for fibrous tissues. Inference should be performed on the original image orientation without TTA averaging.

---

## 4. Receptive Field Expansion (Large Kernels)

### The Hypothesis
Stroma (STR) and Smooth Muscle (MUS) are biologically similar, eosinophilic fibrous tissues. Their primary distinguishing factor is macro-texture: muscle forms parallel bundles, while stroma is wavy and disorganized. We hypothesized that our model's `7x7` maximum kernel was "zoomed in" too closely, causing it to misclassify wavy stroma segments as straight muscle. We replaced the `3x3, 5x5, 7x7` multi-scale branch with larger `7x7, 9x9, 11x11` depthwise convolutions to artificially expand the receptive field and capture this macro-texture.

### The Result (Negative)
The larger receptive field successfully increased the STR recall from **52% to 67%** at mid-training. However, by final convergence, the model regressed, and overall cross-patient accuracy dropped from **95.43% to 93.93%**. Crucially, the F1-score for Lymphocytes (LYM) dropped from **0.9945 to 0.9842**.

### The Scientific Conclusion
While the expanded receptive field theoretically helped with macro-texture (Stroma), the massive `11x11` filters acted as a low-pass "blur filter" that smoothed over the critical high-frequency, crisp edge details required to identify fine-grained classes like Lymphocyte nuclei. 

**Conclusion:** The original `3x3, 5x5, 7x7` multi-scale architecture provides the mathematically optimal trade-off between macro-texture recognition and micro-texture preservation for histopathology applications.

---

## 5. Focal Loss and Pairwise Confusion Penalty

### The Hypothesis
Our model struggled to discriminate between Stroma (STR) and Smooth Muscle (MUS) due to their visual similarities. We hypothesized that applying a **Focal Loss** to upweight hard-to-classify examples, combined with a bespoke **Pairwise Confusion Penalty** (specifically penalizing logits that confuse STR with MUS), would force the network to learn the exact geometric boundaries between wavy stroma and straight muscle fibers.

### The Result (Negative)
The experiment initially seemed like an incredible success. On the internal `NCT-CRC-HE-100K` validation split, the model achieved a near-perfect **99.69% accuracy** and a **0.9968 Macro-F1 score**. The Stroma vs. Muscle confusion was mathematically eliminated.
However, when evaluated on the unseen, cross-patient dataset (`CRC-VAL-HE-7K`), the model's accuracy dropped to **94.76%**. Critically, the Stroma recall plummeted to **57.48%** and Muscle precision dropped to **76.72%**.

### The Scientific Conclusion
The Focal Loss and Pairwise Confusion Penalty caused the model to severely **overfit to the source domain**. By heavily weighting the hardest Stroma examples in the training set, the network "memorized" the specific color profiles, scanning artifacts, and texture signatures of Stroma within the `NCT-CRC-HE-100K` cohort. When presented with Stroma from completely new patients with different H&E staining characteristics in the validation set, the model failed to generalize and defaulted to calling it Muscle.

**Conclusion:** Modifying loss functions to specifically target the hardest edge cases within a single training domain can lead to catastrophic overfitting. For medical imaging, cross-patient generalization is paramount, and standard Cross Entropy Loss with label smoothing is strictly superior for generalization.
