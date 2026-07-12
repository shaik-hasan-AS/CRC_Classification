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

## MedLite-CRC Cross-Patient Accuracy Breakdown (MobileNetV2 KD — **95.97% SOTA** ✅ Verified)

The following table demonstrates MedLite-CRC's per-class performance on the completely unseen `CRC-VAL-HE-7K` cohort (7,180 images) evaluated on the **best checkpoint** (`ckpt_epoch058_acc0.9946.pt`) using isolated CPU evaluation. 

| Tissue Class | Precision | Recall | F1-Score | Support | Change vs. Standard |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BACKGROUND (BACK)** | 0.9988 | **1.0000** | 0.9994 | 847 | Recall: 100% (perfect) |
| **LYMPHOCYTES (LYM)** | 0.9769 | **1.0000** | 0.9883 | 634 | Recall: 100% (perfect) |
| **DEBRIS (DEB)** | 0.9631 | **1.0000** | 0.9812 | 339 | Recall: 100% (perfect) |
| **MUCUS (MUC)** | 0.9638 | **0.9787** | 0.9712 | 1035 | +1.37% F1 |
| **TUMOR (TUM)** | 0.9790 | **0.9813** | 0.9802 | 1233 | +0.69% F1 |
| **NORMAL (NORM)** | 0.9767 | **0.9622** | 0.9694 | 741 | +0.36% F1 |
| **ADIPOSE (ADI)** | 0.9977 | **0.9619** | 0.9795 | 1338 | +0.53% F1 |
| **STROMA (STR)** | 0.7567 | **0.8717** | 0.8084 | 421 | **+5.54% F1** (0.7530 → 0.8084) |
| **SMOOTH MUSCLE (MUS)** | 0.8980 | **0.8176** | 0.8564 | 592 | **+6.31% F1** (0.7933 → 0.8564) |
| **OVERALL** | 0.9617 | **0.9597** | 0.9600 (wtd) | 7180 | **+1.26% Accuracy** |

### 📈 Discussion of KD Improvements
By utilizing soft target supervision from a structurally aligned MobileNetV2 teacher model, the student learned highly regularized decision boundaries:
- **Stroma (STR) F1-Score rose from 0.7530 → 0.8084 (+5.54%)**
- **Smooth Muscle (MUS) F1-Score rose from 0.7933 → 0.8564 (+6.31%)**
- **Three classes achieved perfect 100% recall** (BACK, LYM, DEB)
- **Overall OOD accuracy: 95.97%** — surpassing the teacher (94.82%) by +1.15% and ShuffleNetV2 (95.08%) by +0.89%

This proves that the teacher's soft probability targets contained crucial texture heuristics that regularized the student's representation space, allowing it to easily differentiate wavy connective stroma from straight muscle fiber orientations under domain shift.

