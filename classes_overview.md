# The 9 Tissue Classes of Colorectal Cancer

Here is a random sample taken directly from our `NCT-CRC-HE-100K` training dataset. You can see exactly what the CNN "sees" when it tries to classify these 224x224 histological crops.

![9 Tissue Classes](/home/hasan/.gemini/antigravity/brain/d5b61fe8-871c-4985-a5fd-4aa2ebe2db57/artifacts/9_classes_grid.png)

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

This is exactly why our 7x7 convolutions were failing! If you zoom in too far on a wavy Stroma fiber, it just looks like a straight pink line, making it identical to Muscle. That's why the **3x3 Dilated Convolution** trick (zooming out) is such a powerful architectural fix if the Knowledge Distillation doesn't solve it!
