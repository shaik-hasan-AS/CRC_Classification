# The 9 Tissue Classes of Colorectal Cancer

Here is a random sample taken directly from our `NCT-CRC-HE-100K` training dataset. You can see exactly what the CNN "sees" when it tries to classify these 224x224 histological crops.

![9 Tissue Classes](assets/9_classes_grid.png)

### The Biological Breakdown

1.  **ADI (Adipose):** Fat tissue. Very easy for the model to identify because it looks like large, empty white bubbles with thin pink borders.
2.  **BACK (Background):** Empty slide glass. Just white space.
3.  **DEB (Debris):** Necrotic (dead) tissue and cellular garbage. It looks like unstructured, dark red/purple mush.
4.  **LYM (Lymphocytes):** Immune cells. Extremely distinct because they look like dense clusters of tiny, dark purple dots (the nuclei).
5.  **MUC (Mucus):** Secreted proteins. Often looks like pale, wispy blue/pink clouds with very few cells.
6.  **NORM (Normal Mucosa):** Healthy colon lining. You can see the clear circular/oval "crypt" structures (the white holes surrounded by purple nuclei).
7.  **TUM (Tumor / Adenocarcinoma):** Cancerous epithelium. Looks like chaotic, overgrown, dark purple glands without the neat circular structure of NORM.

### The Problem Children (STR vs MUS)

Look very closely at **STR (Stroma)** and **MUS (Smooth Muscle)**:
*   **Color:** Both are stained the exact same shade of bright pink (eosinophilic).
*   **Texture:** The *only* difference is that Muscle fibers (**MUS**) run parallel in tight, straight lines. Stroma fibers (**STR**) are connective tissue, so they are looser, disorganized, and slightly wavy. 

This deep biological similarity is exactly why these are the lowest performing classes in our model (**MUS: 76.18%, STR: 82.19%**). Even expert human pathologists frequently confuse them on H&E slides without special trichrome stains.

To combat this, we explicitly designed the **MultiScaleBranch** (3x3, 5x5, 7x7 parallel depthwise convolutions). By simultaneously looking at fine textures (3x3) and macro-organization (7x7), the network learns to differentiate the "wavy" macro-structure of Stroma from the "straight" macro-structure of Muscle, pushing our cross-patient performance well beyond standard lightweight CNNs that rely solely on 3x3 kernels.

---

## MedLite-CRC Cross-Patient Accuracy Breakdown (Seed 123)

The following table demonstrates MedLite-CRC's per-class performance on the completely unseen `CRC-VAL-HE-7K` cohort (94.71% overall accuracy). Notice the near-perfect detection of clinically critical tissues like Tumor Epithelium and Lymphocytes.

| Tissue Class | Precision | Recall (Accuracy) | F1-Score | Support (Images) |
| :--- | :--- | :--- | :--- | :--- |
| **BACKGROUND (BACK)** | 99.65% | **100.00%** | 0.9982 | 847 |
| **LYMPHOCYTES (LYM)** | 98.90% | **99.53%** | 0.9921 | 634 |
| **DEBRIS (DEB)** | 96.84% | **99.41%** | 0.9811 | 339 |
| **TUMOR (TUM)** | 97.09% | **97.57%** | 0.9733 | 1233 |
| **NORMAL (NORM)** | 96.00% | **97.17%** | 0.9658 | 741 |
| **ADIPOSE (ADI)** | 98.92% | **95.96%** | 0.9742 | 1338 |
| **MUCUS (MUC)** | 96.75% | **94.78%** | 0.9575 | 1035 |
| **STROMA (STR)** | 69.48% | **82.19%** | 0.7530 | 421 |
| **SMOOTH MUSCLE (MUS)** | 82.75% | **76.18%** | 0.7933 | 592 |
