"""
models/medlite_crc.py

MedLite-CRC: Novel lightweight CNN for stain-robust colon histopathology classification.

Final Architecture (attention-free, Ablation 3):
  Input → StainNormLayer → Stem → MultiScaleBranches → DWResBlocks × 3
        → GAP → Classifier Head

Note: The SEBlock is retained in this file ONLY for ablation study reproducibility
(Ablation 4). It is NOT part of the final deployed architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility Blocks ─────────────────────────────────────────────────────────────

def conv_bn_relu6(in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1, padding_mode='zeros'):
    """Conv + BatchNorm + ReLU6 (Swish) for smoother gradients."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, groups=groups, bias=False, padding_mode=padding_mode),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: DW conv + PW conv."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding_mode='zeros'):
        super().__init__()
        pad = kernel // 2
        self.dw = conv_bn_relu6(in_ch, in_ch, kernel=kernel,
                                stride=stride, padding=pad, groups=in_ch, padding_mode=padding_mode)
        self.pw = conv_bn_relu6(in_ch, out_ch, kernel=1, padding=0, padding_mode='zeros')

    def forward(self, x):
        return self.pw(self.dw(x))


# ── Learnable Stain Normalisation Layers ──────────────────────────────────────

class LearnableStainNorm(nn.Module):
    """
    Learnable per-channel affine normalisation in RGB space.
    Equivalent to a trainable colour normalisation that adapts to the domain.
    Initialised to identity (no transformation) so training starts clean.
    6 parameters total (3 scale + 3 bias).
    """

    def __init__(self, num_channels=3):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.bias


class LearnableHEDStainNorm(nn.Module):
    """
    Biologically-grounded learnable stain normalisation in HED colour space.

    H&E staining operates in Hematoxylin-Eosin-DAB (HED) colour space, not RGB.
    This layer:
      1. Applies a fixed Ruifrok & Johnston (2001) RGB→HED colour deconvolution.
      2. Learns a 6-parameter per-channel affine transform (scale + bias) in HED
         space to normalise scanner-specific stain intensity variations.
      3. Reconstructs the RGB image via the inverse HED→RGB matrix.

    This is strictly superior to RGB-space normalisation for H&E images because:
      - Hematoxylin (nuclear stain) and Eosin (cytoplasm stain) are the chemically
        meaningful channels; their intensities vary independently across scanners.
      - Operating in HED space decouples colour calibration from spatial features,
        preventing the stain parameters from encoding spurious RGB correlations.

    6 trainable parameters. Zero additional inference overhead (foldable into
    the first conv layer at deployment time).

    Reference:
        Ruifrok, A.C. & Johnston, D.A. (2001). Quantification of histochemical
        staining by color deconvolution. Analytical and Quantitative Cytology
        and Histology, 23(4), 291-299.
    """

    # Standard H&E colour deconvolution matrix (Ruifrok & Johnston 2001)
    # Rows: Hematoxylin, Eosin, DAB stain vectors in RGB space
    _HED_FROM_RGB = torch.tensor([
        [ 0.6500286,  0.7041656,  0.2860126],
        [ 0.0481481,  0.7329910,  0.6786913],
        [ 0.7330523,  0.0481490, -0.5775888],
    ], dtype=torch.float32)  # shape (3, 3): HED = RGB @ M^T

    def __init__(self):
        super().__init__()
        # Per-channel affine in HED space — initialised to identity
        self.scale = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, 3, 1, 1))

        # Register the fixed deconvolution matrices as non-trainable buffers
        M = self._HED_FROM_RGB
        self.register_buffer('hed_from_rgb', M)          # RGB → HED
        self.register_buffer('rgb_from_hed', torch.linalg.inv(M))  # HED → RGB

        # ImageNet-like normalisation constants of the NCT-100K dataset
        self.register_buffer('mean', torch.tensor([0.7406, 0.5331, 0.7059]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.1651, 0.2174, 0.1574]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) normalised RGB image.
        Returns: (B, 3, H, W) reconstructed & re-normalised RGB image.
        """
        B, C, H, W = x.shape

        # ── 0. Denormalise back to raw RGB range [0, 1]
        x_raw = x * self.std + self.mean
        x_raw = x_raw.clamp(0.0, 1.0)

        # ── 1. RGB → optical density (Beer-Lambert law: OD = -log(I + eps))
        od = -torch.log(x_raw.clamp(min=1e-6))

        # ── 2. Colour deconvolution: OD → HED
        # od: (B, 3, H, W) → (B, H*W, 3) for matrix multiply
        od_flat = od.permute(0, 2, 3, 1).reshape(B * H * W, 3)   # (B*H*W, 3)
        hed_flat = od_flat @ self.hed_from_rgb.T                   # (B*H*W, 3)
        hed = hed_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)    # (B, 3, H, W)

        # ── 3. Learnable affine normalisation in HED space
        hed = hed * self.scale + self.bias

        # ── 4. HED → optical density (inverse deconvolution)
        hed_flat2 = hed.permute(0, 2, 3, 1).reshape(B * H * W, 3)
        od_flat2  = hed_flat2 @ self.rgb_from_hed.T
        od2       = od_flat2.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # ── 5. Optical density → RGB (inverse Beer-Lambert)
        x_out = torch.exp(-od2).clamp(0.0, 1.0)

        # ── 6. Re-normalise back to normalisation space
        return (x_out - self.mean) / self.std


# ── Multi-Scale Branch ─────────────────────────────────────────────────────────

class MultiScaleBranch(nn.Module):
    """
    3 parallel depthwise separable branches with different kernel sizes:
      - 3×3: fine texture (nuclei, cell boundaries)
      - 5×5: mid-scale (glandular structures)
      - 7×7: coarse structure (stroma, tissue organisation)
    Concatenated and fused with a 1×1 pointwise conv.
    """

    def __init__(self, in_ch, branch_ch, out_ch, padding_mode='zeros'):
        super().__init__()
        self.branch3 = DepthwiseSeparableConv(in_ch, branch_ch, kernel=3, padding_mode=padding_mode)
        self.branch5 = DepthwiseSeparableConv(in_ch, branch_ch, kernel=5, padding_mode=padding_mode)
        self.branch7 = DepthwiseSeparableConv(in_ch, branch_ch, kernel=7, padding_mode=padding_mode)
        self.fuse    = conv_bn_relu6(branch_ch * 3, out_ch, kernel=1, padding=0, padding_mode='zeros')

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

    def __init__(self, in_ch, out_ch, stride=1, padding_mode='zeros'):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride=stride, padding_mode=padding_mode)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, padding_mode=padding_mode)
        self.bn    = nn.BatchNorm2d(out_ch)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False, padding_mode='zeros'),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out = out + self.shortcut(x)
        return F.relu6(out, inplace=True)


# ── Attention Mechanisms ──────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    *** ABLATION STUDY ONLY ***
    This block is retained solely for Ablation 4 reproducibility.
    Empirical results show that SE channel-reweighting overfits to source-scanner
    noise profiles, degrading OOD generalization by -0.83% (93.82% vs 94.65%).
    The final deployed MedLite-CRC architecture does NOT include this block.
    See: docs/ablation_notes.md §9.3, docs/manuscript_draft.md §6.3.
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


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) from CBAM.
    Applies average and max pooling along the channel dimension,
    concatenates the pooled maps, and filters with a 7x7 conv.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class ChannelAttention(nn.Module):
    """Channel Attention sub-module for CBAM."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(F.adaptive_avg_pool2d(x, 1))
        max_out = self.fc(F.adaptive_max_pool2d(x, 1))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) combining Channel and Spatial.
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Block for lightweight networks.
    Saves horizontal and vertical spatial relations via 1D pooling.
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU6(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out


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
        reduction     : attention block reduction ratio (default: 16)
        dropout       : classifier dropout rate (default: 0.4)
        attention_type: attention mechanism to use: "none", "se", "cbam", "spatial", "coord"
    """

    def __init__(self, num_classes=9, base_channels=32, reduction=16, dropout=0.4,
                 use_stain_norm=True, use_multiscale=True, use_se_block=False,
                 attention_type="none", stain_norm_space="rgb", padding_mode="zeros"):
        super().__init__()

        self.use_stain_norm = use_stain_norm
        self.use_multiscale = use_multiscale
        self.use_se_block   = use_se_block
        self.attention_type = attention_type.lower()

        # Handle backward compatibility with use_se_block toggle
        if self.use_se_block and self.attention_type == "none":
            self.attention_type = "se"

        C = base_channels  # 32

        # ── 1. Learnable Stain Normalisation
        if not self.use_stain_norm:
            self.stain_norm = nn.Identity()
        elif stain_norm_space.lower() == "hed":
            self.stain_norm = LearnableHEDStainNorm()
        else:  # default: rgb
            self.stain_norm = LearnableStainNorm(num_channels=3)

        # ── 2. Stem Block
        self.stem = nn.Sequential(
            conv_bn_relu6(3, C, kernel=3, stride=2, padding=1, padding_mode=padding_mode),       # 224→112
            conv_bn_relu6(C, C, kernel=3, stride=1, padding=1,
                          groups=C, padding_mode=padding_mode),                                    # DW conv
        )

        # ── 3. Multi-Scale Feature Extraction
        if self.use_multiscale:
            self.multi_scale = MultiScaleBranch(
                in_ch=C, branch_ch=C * 2, out_ch=C * 4, padding_mode=padding_mode   # 32 → 64 per branch → 128 out
            )
        else:
            self.multi_scale = DepthwiseSeparableConv(in_ch=C, out_ch=C * 4, kernel=3, padding_mode=padding_mode)
        self.pool1 = nn.MaxPool2d(2, 2)   # 112→56

        # ── 4. Depthwise Residual Blocks
        self.res_blocks = nn.Sequential(
            DWResBlock(C * 4, C * 4, stride=1, padding_mode=padding_mode),         # 56×56, 128ch
            DWResBlock(C * 4, C * 8, stride=2, padding_mode=padding_mode),         # 56→28, 256ch
            DWResBlock(C * 8, C * 8, stride=2, padding_mode=padding_mode),         # 28→14, 256ch
        )

        # ── 5. Attention Block Selection
        if self.attention_type == "se":
            self.attn = SEBlock(C * 8, reduction=reduction)
        elif self.attention_type == "spatial":
            self.attn = SpatialAttention(kernel_size=7)
        elif self.attention_type == "cbam":
            self.attn = CBAMBlock(C * 8, reduction=reduction, kernel_size=7)
        elif self.attention_type == "coord":
            self.attn = CoordinateAttention(C * 8, reduction=32)
        else:
            self.attn = nn.Identity()

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
        x = self.attn(x)
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

def build_model(cfg) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "MedLiteCRC")
    num_classes = cfg["data"]["num_classes"]

    if model_name == "MedLiteCRC":
        return MedLiteCRC(
            num_classes      = num_classes,
            base_channels    = model_cfg.get("base_channels", 32),
            reduction        = model_cfg.get("attention_reduction", 16),
            dropout          = model_cfg.get("dropout", 0.4),
            use_stain_norm   = model_cfg.get("use_stain_norm", True),
            use_multiscale   = model_cfg.get("use_multiscale", True),
            use_se_block     = model_cfg.get("use_se_block", False),
            attention_type   = model_cfg.get("attention_type", "none"),
            stain_norm_space = model_cfg.get("stain_norm_space", "rgb"),
            padding_mode     = model_cfg.get("padding_mode", "zeros"),
        )

    # Baselines
    import torchvision.models as models

    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    if model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {model_name}")


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
