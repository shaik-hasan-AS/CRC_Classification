# 📂 Recommended Histopathology Classification Datasets for MedLite-CRC

To validate the stain-invariance, scale robustness, and out-of-distribution (OOD) generalization of the MedLite-CRC architecture, we recommend benchmarking against these four pure patch/image-level classification datasets. Each represents a verified, publicly accessible clinical dataset in the colorectal histopathology domain.

---

## 🎯 1. HunCRC (9-Class Colorectal Cohort)
* **Dataset Type:** Pure patch-level tissue classification (9 classes).
* **Scale:** 101,398 image patches ($512\times512$ pixels) extracted from 200 Whole Slide Images (WSIs).
* **Pathological Classes:**
  1. Adenocarcinoma
  2. High-Grade Dysplasia
  3. Low-Grade Dysplasia
  4. Inflammation
  5. Tumor Necrosis
  6. Suspicious for Invasion
  7. Resection Edge
  8. Technical Artifacts
  9. Normal Tissue
* **Access/Availability:** Publicly available on **The Cancer Imaging Archive (TCIA)** and **Figshare**.
* **Why it is useful for MedLite-CRC:**
  - **Zero-Shot Cross-Cohort Validation:** HunCRC shares high class overlap with STARC-9 and NCT-100K. Since MedLite-CRC is designed to be stain-invariant, we can run zero-shot evaluation (evaluating a model trained on STARC-9 directly on HunCRC without retraining) to prove that our learnable stain normalization layer and attention-free design generalize across distinct scanning sites.

---

## 🔬 2. CRC-HGD-v1 (Colorectal Cancer Histopathological Grading Dataset)
* **Dataset Type:** Multi-magnification histopathological grading classification (5 classes).
* **Scale:** 1,914 images ($800\times800$ pixels, JPEG, RGB).
* **Physical Domain:** Captured across multiple optical magnifications (**4x, 10x, 20x, and 40x**).
* **Pathological Classes:**
  1. Grade I — Well Differentiated (>95% gland formation)
  2. Grade II — Moderately Differentiated (50–95% gland formation)
  3. Grade III — Poorly Differentiated (<50% gland formation)
  4. Normal Colorectal Tissue
  5. Mixed Normal/Tumoral Tissue
* **Access/Availability:** Publicly hosted on **Mendeley Data** (Isfahan University of Medical Sciences, DOI: 10.17632/yfp5sfj47m.2).
* **Why it is useful for MedLite-CRC:**
  - **Magnification-Invariance Evaluation:** MedLite-CRC uses parallel depthwise separable convolutions with different kernel sizes (3x3, 5x5, 7x7) to capture multiscale context. Benchmarking on CRC-HGD-v1 allows us to test whether these multiscale branches provide robustness against physical magnification shifts (e.g., training on 20x slides and evaluating on 10x or 40x slides).
  - **High-Difficulty Clinical Grading:** Unlike distinguishing tissue types, grading cancer differentiation (Well, Moderately, or Poorly) requires capturing fine-grained nuclear chromatin configurations and glandular border distortions. This tests the limits of our lightweight encoder's representation capacity.

---

## 🩺 3. EBHI (Enteroscope Biopsy Histopathological Image Dataset)
* **Dataset Type:** Biopsy-level histopathological classification (6 classes).
* **Scale:** 5,170 high-resolution biopsy images ($224\times224$ pixels).
* **Pathological Classes:**
  1. Normal
  2. Polyp
  3. Adenoma
  4. Adenocarcinoma
  5. Low-Grade Intraepithelial Neoplasia
  6. High-Grade Intraepithelial Neoplasia
* **Access/Availability:** Publicly available on **Figshare** (DOI: 10.6084/m9.figshare.21540159).
* **Why it is useful for MedLite-CRC:**
  - **Acquisition Domain Generalization:** Surgical resection tissue (used in NCT-100K) maintains clean structural alignment. Endoscopic biopsy tissue (used in EBHI) is highly deformed, compressed, and fragmented due to extraction forces. Validating MedLite-CRC on EBHI tests whether our model is robust to mechanical acquisition artifacts, proving its clinical utility for real-world diagnostic biopsy processing.

---

## 🎗️ 4. Kather MSI/MSS Colorectal Patches
* **Dataset Type:** Binary molecular biomarker classification (Microsatellite Instability [MSI] vs. Microsatellite Stability [MSS]).
* **Scale:** 411,890 unique image patches ($224\times224$ pixels) from Formalin-Fixed Paraffin-Embedded (FFPE) slides, and 218,578 patches from snap-frozen (SF) slides from TCGA-COAD/READ cohorts.
* **Access/Availability:** Publicly available on **Zenodo** (Jakob Nikolas Kather research group).
* **Why it is useful for MedLite-CRC:**
  - **Complex Biomarker Prediction:** Predicting MSI vs. MSS directly from H&E morphology (instead of molecular PCR/IHC testing) is a landmark task in computational pathology. MSI-H tumors exhibit specific microscopic features (e.g., tumor-infiltrating lymphocytes, mucinous differentiation).
  - **Validating High-Level Representation Learning:** Because the morphological differences between MSI and MSS are highly subtle, this dataset evaluates whether our lightweight 0.48M model has the representational capacity to capture complex molecular features, going beyond basic tissue classification.

---

## 📂 Local Storage Mapping

The verified datasets have been downloaded, extracted, and structured in the local `data/` directory:

| Dataset | Local Workspace Path | Structure Details |
| :--- | :--- | :--- |
| **EBHI** | [`data/EBHI-SEG/`](file:///home/hasan/Desktop/codes/MedicalCNN_Research(intern)/medlite_crc/data/EBHI-SEG) | 6 class directories: `Adenocarcinoma`, `High-grade IN`, `Low-grade IN`, `Normal`, `Polyp`, `Serrated adenoma` (each containing an `image/` subfolder). |
| **CRC-HGD-v1** | [`data/CRC-HGD-v1/`](file:///home/hasan/Desktop/codes/MedicalCNN_Research(intern)/medlite_crc/data/CRC-HGD-v1) | 5 grade directories: `CRC_Grade__1__Well_Diff`, `CRC_Grade__2__Mod_Diff`, `CRC_Grade__3__Poorly_Diff`, `Mixed_Normal_Tumoral_Colon`, `Normal_Colon`. |
| **Kather MSI/MSS** | [`data/Kather_MSI_MSS/`](file:///home/hasan/Desktop/codes/MedicalCNN_Research(intern)/medlite_crc/data/Kather_MSI_MSS) | Separated into `train/` and `test/` to prevent data leakage. Subfolders: `train/MSS`, `train/MSIMUT`, `test/MSS`, `test/MSIMUT`. |

