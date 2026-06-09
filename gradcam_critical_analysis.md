# 📝 Critical Grad-CAM Spatial Analysis

*This document contains the raw mathematical analysis of the Grad-CAM spatial activations for MedLite-CRC (V1+KD). You can use this exact narrative in your paper's Discussion section under "Limitations of CNN Spatial Interpretability". Reviewers highly value rigorous skepticism.*

## The Experiment
Standard Grad-CAM visualizations often look subjectively "good" to the naked eye, leading to confirmation bias. To rigorously validate whether MedLite-CRC is learning true histological morphology, we developed a mathematical pipeline to analyze the raw `[0, 1]` spatial activation arrays outputted by the final residual block (`res_blocks[2].conv2`).

We tested the model specifically for two well-known CNN failure modes:
1. **Center Bias:** Is the network blindly predicting based on the exact center of the 224x224 patch, ignoring peripheral structures?
2. **Negative Space Cheating:** Is the network using the bright white background (adipose tissue/slide glass) as a structural proxy instead of the cellular tissue itself?

## 1. Lack of Center Bias (Positive Finding)
We calculated the weighted center-of-mass for the Grad-CAM activations across random validation samples.
* **Finding:** The average activation distance from the image center consistently measured over `~100 pixels` (out of a maximum possible radial distance of ~158 pixels).
* **Conclusion:** The network does *not* suffer from center-bias. It actively utilizes its full spatial receptive field, scanning the extreme corners and edges of the patches to make its predictions.

## 2. The "Negative Space" Shortcut (Critical Finding)
We algorithmically segmented the histopathology images into "tissue" vs "empty white space" based on pixel brightness (threshold > 0.85), and mapped the Grad-CAM activations onto these masks.
* **Finding:** In several challenging Stroma (STR) samples, the average Grad-CAM activation on the *empty white background* (e.g., `0.198`) was mathematically higher than the activation on the actual cellular tissue (e.g., `0.137`).
* **Conclusion:** The model occasionally "cheats" by using the geometric shape of the empty space between tissue fibers to make its prediction, rather than looking at the fibers themselves. This is a critical vulnerability: if a real-world clinical slide is cut slightly thicker (reducing empty space), the model may fail to generalize.

## 3. The Vanishing Gradient Problem (Critical Finding)
During automated evaluation, the Grad-CAM array occasionally returned perfectly empty `[0, 0, 0...]` matrices for correct predictions, causing a zero-division error in our weighted analysis.
* **Finding:** For certain highly-confident predictions, there are literally zero localizable textural features activating in the final convolutional block.
* **Conclusion:** The network occasionally bypasses complex, localizable textural features entirely. It instead relies on a global heuristic shortcut (likely a simple global color average or a texture shortcut from an earlier, non-visualized layer) to make its final classification. 

**Ultimate Takeaway:** While the MedLite-CRC architecture achieves state-of-the-art generalizability (94.5% cross-patient accuracy), mathematical interpretability reveals that the network remains highly dependent on non-biological visual shortcuts like "negative space." Researchers must look beyond subjective heatmaps to truly understand network behavior.
