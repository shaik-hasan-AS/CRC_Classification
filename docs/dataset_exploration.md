# CRC Histopathology Datasets for Future Exploration

This document outlines alternative and follow-up datasets for our Colorectal Cancer (CRC) histopathology research, detailing their specifications and how useful they could be for the MedLite-CRC pipeline.

## 1. STARC-9 (Stanford Colorectal Cancer Dataset)
**The Ultimate Generalization Benchmark**

*   **Size:** 630,000 image tiles from 200 CRC patients.
*   **Classes:** 9 classes (exactly 70,000 tiles per class).
*   **Usefulness:** **Extremely High.** This is explicitly designed to solve the bias and diversity issues found in older datasets like `NCT-CRC-HE-100K`. It uses a framework to ensure morphological diversity and guarantees perfect class balance. If we want to prove our model isn't "cheating" and has truly learned biological morphology via our Structure-Forcing Pipeline, STARC-9 is the definitive test.

## 2. SurGen (Prognostic & Genetic Modeling)
**Moving from Classification to Predictive Oncology**

*   **Size:** 1,020 Whole-Slide Images (WSIs) from 843 CRC cases.
*   **Annotations:** Clinical and genetic annotations (e.g., KRAS, NRAS, BRAF mutations and mismatch repair status).
*   **Usefulness:** **High (for a new project phase).** This allows us to train a model to predict genetic mutations purely from H&E stained images. It represents a massive step up in clinical utility beyond just identifying tissue types.

## 3. CRC-HGD-v1 (Histological Grading)
**Assessing Cancer Severity**

*   **Size:** 1,899 images at multiple magnifications (4x, 10x, 20x, 40x).
*   **Classes:** Histological grading (Well Differentiated, Moderately Differentiated, and Poorly Differentiated).
*   **Usefulness:** **Medium-High.** Instead of asking *what* tissue this is, this dataset asks *how bad* the cancer is. It's a great follow-up task for a model that has already mastered basic tissue classification.

## 4. Colorectal_Cancer_IHC_CISH_HE_Epithelium_Segmentation
**Transitioning to Segmentation Tasks**

*   **Size/Scope:** High-resolution (10,000x10,000 pixel) images at 40x magnification.
*   **Features:** H&E alongside IHC/ISH markers, and includes pathologist-validated epithelium segmentation masks.
*   **Usefulness:** **Medium (Different Task).** Highly useful if we decide to pivot MedLite-CRC from an image classification model into a semantic segmentation model (e.g., using a U-Net architecture) to draw exact boundaries around tumors.

## 5. Kather Texture Dataset (2016) / Histology MNIST
**The Lightweight Benchmark**

*   **Size:** 5,000 images (150x150 pixels) representing 8 tissue categories.
*   **Usefulness:** **Low/Medium.** This is an older, much smaller predecessor to our current dataset. While good for quick local testing and basic benchmarking, our current model and dataset (`NCT-CRC-HE-100K`) have already far surpassed the complexity needed for this one.

## Summary & Recommendation
For our immediate goal of fixing domain shift and proving the model isn't just taking visual "shortcuts", **STARC-9** should be our next target. It provides the scale, diversity, and balance necessary to rigorously test the robustness of our new Structure-Forcing Pipeline.
