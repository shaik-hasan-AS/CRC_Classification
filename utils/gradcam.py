"""
utils/gradcam.py
GradCAM visualisation for MedLite-CRC.
Hooks into the last residual block for heatmaps.

Run: python utils/gradcam.py --checkpoint outputs/checkpoints/best.pt --img path/to/img.tif
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from models.medlite_crc import build_model
from data.transforms import get_val_transforms


class GradCAM:
    """
    GradCAM implementation hooked onto the last DWResBlock output.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None, mask_border_width=0):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=input_tensor.shape[2:],
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        
        if mask_border_width > 0:
            cam[0:mask_border_width, :] = 0
            cam[-mask_border_width:, :] = 0
            cam[:, 0:mask_border_width] = 0
            cam[:, -mask_border_width:] = 0

        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, logits.softmax(dim=1)[0].detach().cpu().numpy()


def overlay_heatmap(img_pil, cam, alpha=0.45):
    """Overlay GradCAM heatmap on original image."""
    img_np   = np.array(img_pil.resize((224, 224))) / 255.0
    heatmap  = cm.jet(cam)[:, :, :3]
    overlay  = (1 - alpha) * img_np + alpha * heatmap
    overlay  = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def visualise(model, img_path, transform, class_names, save_path, cfg):
    """Generate and save GradCAM for a single image."""
    # Load image
    img_pil = Image.open(img_path).convert("RGB")
    tensor  = transform(img_pil)

    # Hook last res block
    target_layer = model.res_blocks[-1].conv2.pw[0]  # last PW conv in last DWResBlock
    gcam = GradCAM(model, target_layer)

    device = next(model.parameters()).device
    cam, pred_class, probs = gcam.generate(tensor.to(device))

    overlay = overlay_heatmap(img_pil, cam)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_pil.resize((224, 224)))
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("GradCAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    conf       = probs[pred_class] * 100
    axes[2].set_title(f"Overlay\nPred: {pred_label} ({conf:.1f}%)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("MedLite-CRC GradCAM Visualisation", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] GradCAM → {save_path}")
    print(f"  Prediction: {pred_label} (confidence: {conf:.1f}%)")

    return pred_class, probs


if __name__ == "__main__":
    import yaml
    from utils.metrics import load_checkpoint
    from data.dataset import CRC_CLASSES

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img",        required=True, help="Path to input image")
    parser.add_argument("--save",       default="outputs/gradcam/gradcam_out.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model)

    transform = get_val_transforms(cfg)
    visualise(model, args.img, transform, CRC_CLASSES, args.save, cfg)
