"""
models/medlite_crc.py

MedLite-CRC: Novel lightweight CNN for stain-robust colon histopathology classification.

Architecture:
  Input → StainNormLayer → Stem → MultiScaleBranches → DWResBlocks × 3
        → ChannelAttention (SE) → GAP → Classifier Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility Blocks ─────────────────────────────────────────────────────────────

def conv_bn_relu6(in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
    """Conv + BatchNorm + ReLU6 (standard mobile building block)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: DW conv + PW conv."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        pad = kernel // 2
        self.dw = conv_bn_relu6(in_ch, in_ch, kernel=kernel,
                                stride=stride, padding=pad, groups=in_ch)
        self.pw = conv_bn_relu6(in_ch, out_ch, kernel=1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


# ── Learnable Stain Normalisation Layer ───────────────────────────────────────

class LearnableStainNorm(nn.Module):
    """
    Learnable per-channel affine normalisation to handle stain variation.
    Equivalent to a trainable colour normalisation that adapts to the domain.
    Initialised to identity (no transformation) so training starts clean.
    """

    def __init__(self, num_channels=3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias


# ── Multi-Scale Branch ─────────────────────────────────────────────────────────

class MultiScaleBranch(nn.Module):
    """
    3 parallel depthwise separable branches with different kernel sizes:
      - 3×3: fine texture (nuclei, cell boundaries)
      - 5×5: mid-scale (glandular structures)
      - 7×7: coarse structure (stroma, tissue organisation)
    Concatenated and fused with a 1×1 pointwise conv.
    """

    def __init__(self, in_ch, branch_ch, out_ch):
        super().__init__()
        self.branch3 = DepthwiseSeparableConv(in_ch, branch_ch, kernel=3)
        self.branch5 = DepthwiseSeparableConv(in_ch, branch_ch, kernel=5)
        self.branch7 = DepthwiseSeparableConv(in_ch, branch_ch, kernel=7)
        self.fuse    = conv_bn_relu6(branch_ch * 3, out_ch, kernel=1, padding=0)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b3, b5, b7], dim=1)
        return self.fuse(out)


# ── Depthwise Residual Block ───────────────────────────────────────────────────

class DWResBlock(nn.Module):
    """
    Depthwise separable residual block.
    Skip connection with optional projection if channels change.
    Downsampling via stride=2 on first DW conv.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.bn    = nn.BatchNorm2d(out_ch)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out = out + self.shortcut(x)
        return F.relu6(out, inplace=True)


# ── Squeeze-and-Excitation Channel Attention ──────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Focuses the model on informative feature channels (e.g., nuclear features
    vs background noise in histopathology).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ── MedLite-CRC Main Architecture ─────────────────────────────────────────────

class MedLiteCRC(nn.Module):
    """
    MedLite-CRC: Lightweight stain-robust CNN for colon histopathology.

    Design goals:
      - <5M parameters
      - <1 GFLOPs
      - Cross-dataset generalisation via stain normalisation + augmentation
      - Multi-scale texture capture for histopathology patterns

    Args:
        num_classes   : number of output classes (default: 9 for NCT-CRC)
        base_channels : base channel width (default: 32)
        reduction     : SE block reduction ratio (default: 16)
        dropout       : classifier dropout rate (default: 0.4)
    """

    def __init__(self, num_classes=9, base_channels=32, reduction=16, dropout=0.4):
        super().__init__()

        C = base_channels  # 32

        # ── 1. Learnable Stain Normalisation
        self.stain_norm = LearnableStainNorm(num_channels=3)

        # ── 2. Stem Block
        self.stem = nn.Sequential(
            conv_bn_relu6(3, C, kernel=3, stride=2, padding=1),       # 224→112
            conv_bn_relu6(C, C, kernel=3, stride=1, padding=1,
                          groups=C),                                    # DW conv
        )

        # ── 3. Multi-Scale Feature Extraction
        self.multi_scale = MultiScaleBranch(
            in_ch=C, branch_ch=C * 2, out_ch=C * 4   # 32 → 64 per branch → 128 out
        )
        self.pool1 = nn.MaxPool2d(2, 2)   # 112→56

        # ── 4. Depthwise Residual Blocks
        self.res_blocks = nn.Sequential(
            DWResBlock(C * 4, C * 4, stride=1),         # 56×56, 128ch
            DWResBlock(C * 4, C * 8, stride=2),         # 56→28, 256ch
            DWResBlock(C * 8, C * 8, stride=2),         # 28→14, 256ch
        )

        # ── 5. Channel Attention
        self.se = SEBlock(C * 8, reduction=reduction)

        # ── 6. Final pooling
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # ── 7. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(C * 8, C * 8),
            nn.BatchNorm1d(C * 8),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(C * 8, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stain_norm(x)
        x = self.stem(x)
        x = self.multi_scale(x)
        x = self.pool1(x)
        x = self.res_blocks(x)
        x = self.se(x)
        x = self.pool2(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """Return intermediate features for GradCAM (after res_blocks)."""
        x = self.stain_norm(x)
        x = self.stem(x)
        x = self.multi_scale(x)
        x = self.pool1(x)
        x = self.res_blocks(x)
        return x


# ── Model Factory ──────────────────────────────────────────────────────────────

def build_model(cfg) -> MedLiteCRC:
    model_cfg = cfg.get("model", {})
    model = MedLiteCRC(
        num_classes   = cfg["data"]["num_classes"],
        base_channels = model_cfg.get("base_channels", 32),
        reduction     = model_cfg.get("attention_reduction", 16),
        dropout       = model_cfg.get("dropout", 0.4),
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_M": round(total / 1e6, 3),
        "trainable_M": round(trainable / 1e6, 3),
    }


if __name__ == "__main__":
    # Quick sanity check
    model = MedLiteCRC(num_classes=9)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape : {out.shape}")

    stats = count_parameters(model)
    print(f"Parameters   : {stats['total_M']}M total, {stats['trainable_M']}M trainable")

    try:
        from thop import profile
        macs, params = profile(model, inputs=(x[:1],), verbose=False)
        print(f"FLOPs        : {macs / 1e9:.3f} GFLOPs")
    except ImportError:
        print("Install thop for FLOPs: pip install thop")
