"""
data/transforms.py
Augmentation pipeline with stain augmentation for stain-robustness.
"""

import random
import numpy as np

import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


# ── Stain Augmentation ────────────────────────────────────────────────────────

class StainAugment:
    """
    Macenko-inspired random stain perturbation.
    Randomly shifts H&E stain vectors to simulate different lab protocols.
    Does NOT require torchstain — pure numpy/PIL implementation.
    """

    def __init__(self, sigma1=0.2, sigma2=0.2, p=0.5):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.p = p

        # Reference H&E stain matrix (Macenko standard)
        self.HE_ref = np.array([
            [0.5626, 0.7201, 0.4062],
            [0.2159, 0.8012, 0.5581],
        ])
        self.max_C_ref = np.array([1.9705, 1.0308])

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.clip(img_np, 1e-6, 1.0)

        # OD transform
        OD = -np.log(img_np).reshape(-1, 3)

        # Perturb stain concentrations
        alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1, 2)
        beta  = np.random.uniform(-self.sigma2, self.sigma2, 2)

        # Reconstruct with perturbed concentrations
        HE_aug = self.HE_ref.copy()
        maxC   = self.max_C_ref * alpha + beta

        try:
            C = np.dot(OD, np.linalg.pinv(HE_aug))
            C = np.clip(C, 0, None)
            C[:, 0] = np.clip(C[:, 0] * (maxC[0] / (self.max_C_ref[0] + 1e-6)), 0, maxC[0])
            C[:, 1] = np.clip(C[:, 1] * (maxC[1] / (self.max_C_ref[1] + 1e-6)), 0, maxC[1])
            OD_aug  = np.dot(C, HE_aug)
            img_aug = np.exp(-OD_aug).reshape(img_np.shape)
            img_aug = np.clip(img_aug * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(img_aug)
        except Exception:
            return img   # fallback to original on numerical issues


class ForegroundMasking:
    """
    Masks out the bright white background (negative space) in histopathology images.
    Replaces it with random Gaussian noise to prevent the CNN from memorizing
    hard geometric edges or using the empty space as a texture proxy.
    """
    def __init__(self, threshold=0.85, noise_mean=0.85, noise_std=0.05):
        self.threshold = threshold
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Calculate grayscale brightness
        brightness = np.mean(img_np, axis=-1)
        
        # Create mask for white background
        mask = brightness > self.threshold
        
        if not np.any(mask):
            return img # No background found
            
        # Generate Gaussian noise matching the general "bright but noisy" background of slides
        noise = np.random.normal(self.noise_mean, self.noise_std, img_np.shape).astype(np.float32)
        noise = np.clip(noise, 0.0, 1.0)
        
        # Apply mask
        img_np[mask] = noise[mask]
        
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)



class RandomRotation90:
    """Randomly rotate by 0, 90, 180, or 270 degrees."""

    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return F.rotate(img, angle)


# ── Transform Factories ───────────────────────────────────────────────────────

def get_train_transforms(cfg):
    """Full augmentation pipeline for training."""
    aug = cfg["data"]["augmentation"]
    mean = aug["normalize_mean"]
    std  = aug["normalize_std"]
    size = cfg["data"]["image_size"]

    # Resize first to avoid running expensive operations on high-res images
    ops = [T.Resize((size, size))]

    # Foreground Masking
    if aug.get("use_foreground_masking", True):
        ops.append(ForegroundMasking(
            threshold=aug.get("mask_threshold", 0.85)
        ))

    # Stain augmentation (before tensor conversion)
    if aug.get("use_stain_augment", True):
        ops.append(StainAugment(
            sigma1=aug.get("stain_sigma1", 0.2),
            sigma2=aug.get("stain_sigma2", 0.2),
            p=aug.get("stain_probability", 0.5),
        ))

    # Grayscale Augmentation (Color Dropout)
    grayscale_prob = aug.get("grayscale_prob", 0.0)
    if grayscale_prob > 0:
        ops.append(T.RandomGrayscale(p=grayscale_prob))

    # Spatial augmentations
    if aug.get("random_flip", True):
        ops += [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]

    # Colour augmentation
    if aug.get("color_jitter", True):
        ops.append(T.ColorJitter(
            brightness=aug.get("color_jitter_brightness", 0.2),
            contrast=aug.get("color_jitter_contrast", 0.2),
            saturation=aug.get("color_jitter_saturation", 0.1),
            hue=aug.get("color_jitter_hue", 0.05),
        ))

    # To tensor + normalise
    ops += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    # Random erasing (after tensor conversion — operates on tensors)
    if aug.get("random_erasing", False):
        ops.append(T.RandomErasing(
            p=aug.get("random_erasing_p", 0.15),
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
        ))

    return T.Compose(ops)


def get_val_transforms(cfg):
    """Minimal transforms for validation / testing (no augmentation)."""
    aug  = cfg["data"]["augmentation"]
    mean = aug["normalize_mean"]
    std  = aug["normalize_std"]
    size = cfg["data"]["image_size"]

    # Resize first
    ops = [T.Resize((size, size))]
    
    if aug.get("use_foreground_masking", True):
        ops.append(ForegroundMasking(
            threshold=aug.get("mask_threshold", 0.85)
        ))

    ops += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(ops)


def get_gradcam_transforms(cfg):
    """
    Same as val but returns both transformed tensor and original PIL image.
    Used by GradCAM visualisation.
    """
    return get_val_transforms(cfg)
