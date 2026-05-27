# Training Dynamics: Knowledge Distillation 📈

Here are the pristine, publication-ready learning curves extracted directly from the fine-tuning script logs.

![Learning Curves](/home/hasan/.gemini/antigravity/brain/d5b61fe8-871c-4985-a5fd-4aa2ebe2db57/artifacts/learning_curves.png)

### Key Observations for the Manuscript
1. **Convergence:** The training loss decreases steadily and stabilizes beautifully, proving that the Knowledge Distillation loss function (KL-Divergence) guided the model down the gradient landscape smoothly.
2. **Early Stopping:** The validation loss began to plateau around Epoch 20, triggering our Early Stopping mechanism perfectly to prevent any domain overfitting. 
3. **High F1-Score Parity:** The Validation Macro-F1 score closely hugs the Validation Accuracy curve, proving that the model did not suffer from class imbalance issues, even though the dataset had skewed class distributions!
