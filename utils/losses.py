"""
utils/losses.py
Focal Loss and auxiliary loss utilities for MedLite-CRC.

Focal Loss down-weights easy examples and focuses on hard ones —
critical for fixing the STR/MUS confusion where the model is confident but wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma > 0, the loss is reduced for well-classified examples (p_t >> 0.5),
    focusing the model's learning on hard, misclassified examples.

    Args:
        alpha      : Per-class weights tensor of shape (num_classes,) or None.
        gamma      : Focusing parameter (default: 2.0). Higher = more focus on hard examples.
        reduction  : 'mean', 'sum', or 'none'.
        label_smoothing : Optional label smoothing factor (0.0 = no smoothing).
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits  : (B, C) raw class scores
            targets : (B,) integer class labels
        """
        num_classes = logits.size(1)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Focal loss per class
        loss = -focal_weight * smooth_targets * log_probs  # (B, C)

        # Apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha.unsqueeze(0)  # (B, C) * (1, C)

        # Sum over classes, then reduce over batch
        loss = loss.sum(dim=1)  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
class PairwiseConfusionPenalty(nn.Module):
    """
    Extra penalty when the model predicts class_b for a true class_a (or vice versa).
    """
    def __init__(self, pairs, penalty_weight=0.5):
        super().__init__()
        self.pairs = pairs
        self.penalty_weight = penalty_weight

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        penalty = 0.0
        
        for (class_a, class_b) in self.pairs:
            # Mask for samples where true label is class_a
            mask_a = (targets == class_a)
            if mask_a.any():
                # Penalize predicting class_b
                penalty += probs[mask_a, class_b].mean()
                
            # Mask for samples where true label is class_b
            mask_b = (targets == class_b)
            if mask_b.any():
                # Penalize predicting class_a
                penalty += probs[mask_b, class_a].mean()
                
        return penalty * self.penalty_weight



def compute_class_weights(dataset, num_classes=9, method="inverse_freq"):
    """
    Compute per-class weights from a dataset for loss balancing.

    Args:
        dataset    : A PyTorch dataset or Subset (must have .targets or iterate)
        num_classes: Number of classes
        method     : 'inverse_freq' or 'effective_num'

    Returns:
        Tensor of shape (num_classes,) with normalised class weights.
    """
    # Count class frequencies
    counts = torch.zeros(num_classes)

    # Try to get targets directly (fast path)
    if hasattr(dataset, 'targets'):
        targets = torch.tensor(dataset.targets)
        for c in range(num_classes):
            counts[c] = (targets == c).sum().float()
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Subset case
        targets = torch.tensor(dataset.dataset.targets)
        indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
        subset_targets = targets[indices]
        for c in range(num_classes):
            counts[c] = (subset_targets == c).sum().float()
    else:
        # Slow path: iterate
        for _, label in dataset:
            counts[label] += 1

    # Avoid division by zero
    counts = counts.clamp(min=1.0)

    if method == "inverse_freq":
        # Inverse frequency, normalised so mean weight = 1.0
        weights = 1.0 / counts
        weights = weights / weights.mean()
    elif method == "effective_num":
        # Effective number of samples (Cui et al., 2019)
        beta = 0.9999
        effective = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights
