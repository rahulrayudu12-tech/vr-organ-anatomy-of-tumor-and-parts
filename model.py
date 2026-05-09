"""
model.py — MediVR AI Backend
3D U-Net for brain tumor segmentation (PyTorch + MONAI)
Also includes a fallback threshold model that runs without GPU.
"""

import numpy as np
from scipy import ndimage

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH = True
except ImportError:
    TORCH = False

try:
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss, DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose, ToTensord, NormalizeIntensityd, RandFlipd,
        RandRotate90d, RandGaussianNoised
    )
    MONAI = True
except ImportError:
    MONAI = False

try:
    from skimage.morphology import ball, dilation as ski_dilation
    SKIMAGE = True
except ImportError:
    SKIMAGE = False


# ══════════════════════════════════════════════════════════════════════════
#  OPTION A — Custom 3D U-Net (no MONAI dependency)
# ══════════════════════════════════════════════════════════════════════════

if TORCH:
    class DoubleConv3D(nn.Module):
        """Two 3×3×3 conv layers → BN → ReLU block."""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.block(x)

    class Down3D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.pool = nn.MaxPool3d(2)
            self.conv = DoubleConv3D(in_ch, out_ch)
        def forward(self, x): return self.conv(self.pool(x))

    class Up3D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv3D(in_ch, out_ch)
        def forward(self, x, skip):
            x = self.up(x)
            # Pad if needed
            diff = [skip.shape[i+2] - x.shape[i+2] for i in range(3)]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
            return self.conv(torch.cat([skip, x], dim=1))

    class UNet3D(nn.Module):
        """
        Lightweight 3D U-Net.
        in_channels: 1 (T1 only) or 4 (BraTS multi-modal)
        out_channels: 2 (binary) or 4 (BraTS: BG, NCR, ED, ET)
        """
        def __init__(self, in_channels: int = 1, out_channels: int = 2,
                     features: list = None):
            super().__init__()
            features = features or [32, 64, 128, 256]
            self.inc   = DoubleConv3D(in_channels, features[0])
            self.down1 = Down3D(features[0], features[1])
            self.down2 = Down3D(features[1], features[2])
            self.down3 = Down3D(features[2], features[3])
            self.bot   = DoubleConv3D(features[3], features[3]*2)
            self.up1   = Up3D(features[3]*2, features[3])
            self.up2   = Up3D(features[3],   features[2])
            self.up3   = Up3D(features[2],   features[1])
            self.up4   = Up3D(features[1],   features[0])
            self.out   = nn.Conv3d(features[0], out_channels, 1)

        def forward(self, x):
            s0 = self.inc(x)
            s1 = self.down1(s0)
            s2 = self.down2(s1)
            s3 = self.down3(s2)
            b  = self.bot(s3)
            x  = self.up1(b,  s3)
            x  = self.up2(x,  s2)
            x  = self.up3(x,  s1)
            x  = self.up4(x,  s0)
            return self.out(x)

        def count_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════
#  OPTION B — MONAI U-Net (preferred when available)
# ══════════════════════════════════════════════════════════════════════════

def build_monai_unet(in_channels: int = 1, out_channels: int = 2) -> "UNet":
    """Build the MONAI UNet — higher quality than the custom one."""
    if not MONAI:
        raise ImportError("pip install monai")
    return UNet(
        spatial_dims   = 3,
        in_channels    = in_channels,
        out_channels   = out_channels,
        channels       = (16, 32, 64, 128, 256),
        strides        = (2,  2,  2,  2),
        num_res_units  = 2,
        dropout        = 0.1,
    )


# ══════════════════════════════════════════════════════════════════════════
#  OPTION C — Threshold model (runs without GPU / PyTorch)
# ══════════════════════════════════════════════════════════════════════════

class ThresholdSegmentor:
    """
    Fast CPU-only segmenter using intensity thresholding + morphology.
    Dice ~0.5-0.65 on synthetic data — useful for prototyping.
    Replace with trained UNet3D for real performance.
    """
    def __init__(self, percentile: float = 97.0, min_voxels: int = 20):
        self.percentile = percentile
        self.min_voxels = min_voxels

    def predict(self, volume: np.ndarray) -> np.ndarray:
        """Returns binary mask (1 = tumor)."""
        threshold = np.percentile(volume, self.percentile)
        raw = (volume > threshold).astype(np.uint8)
        # Remove noise blobs
        labeled, n = ndimage.label(raw)
        sizes = ndimage.sum(raw, labeled, range(1, n+1))
        clean = np.zeros_like(raw)
        for i, sz in enumerate(sizes):
            if sz > self.min_voxels:
                clean[labeled == (i+1)] = 1
        # Dilate slightly
        if SKIMAGE:
            clean = ski_dilation(clean, ball(1)).astype(np.uint8)
        return clean

    def predict_proba(self, volume: np.ndarray) -> np.ndarray:
        """Returns soft probability map [0,1]."""
        lo = np.percentile(volume, 90)
        hi = np.percentile(volume, 100)
        proba = np.clip((volume - lo) / (hi - lo + 1e-8), 0, 1)
        return proba.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
#  Unified predict function
# ══════════════════════════════════════════════════════════════════════════

def predict_tumor_mask(
    volume: np.ndarray,
    model=None,
    device: str = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run segmentation on a preprocessed 3D volume.

    Args:
        volume:    shape (D,H,W), float32, normalized
        model:     UNet3D / MONAI UNet (optional). If None, uses ThresholdSegmentor.
        device:    'cpu' | 'cuda'
        threshold: probability threshold for binary mask

    Returns:
        binary mask uint8, same shape as input
    """
    if model is None or not TORCH:
        print("[predict] Using ThresholdSegmentor (no PyTorch model provided)")
        return ThresholdSegmentor().predict(volume)

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    # Add batch + channel dims: (1, 1, D, H, W)
    x = torch.tensor(volume[None, None], dtype=torch.float32).to(dev)

    with torch.no_grad():
        logits = model(x)                        # (1, C, D, H, W)
        proba  = torch.softmax(logits, dim=1)    # class probs
        pred   = (proba[0, 1] > threshold).cpu().numpy().astype(np.uint8)

    print(f"[predict] mask voxels: {pred.sum()}")
    return pred


# ══════════════════════════════════════════════════════════════════════════
#  Dice & metrics
# ══════════════════════════════════════════════════════════════════════════

def dice_coefficient(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    return float(2 * inter / (pred.sum() + truth.sum() + 1e-8))

def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(inter / (union + 1e-8))

def hausdorff_distance(pred: np.ndarray, truth: np.ndarray) -> float:
    """Approximate Hausdorff distance (slow, use only on small volumes)."""
    from scipy.spatial.distance import directed_hausdorff
    p_pts = np.argwhere(pred)
    t_pts = np.argwhere(truth)
    if len(p_pts) == 0 or len(t_pts) == 0:
        return float('inf')
    d1 = directed_hausdorff(p_pts, t_pts)[0]
    d2 = directed_hausdorff(t_pts, p_pts)[0]
    return float(max(d1, d2))

def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "dice":  dice_coefficient(pred, truth),
        "iou":   iou(pred, truth),
        "pred_voxels":  int(pred.sum()),
        "truth_voxels": int(truth.sum()),
    }


if __name__ == "__main__":
    import sys
    print("model.py — dependency check")
    print(f"  PyTorch:  {'✅ ' + torch.__version__ if TORCH else '❌  pip install torch'}")
    print(f"  MONAI:    {'✅' if MONAI else '❌  pip install monai'}")
    print(f"  scikit-image: {'✅' if SKIMAGE else '❌  pip install scikit-image'}")

    if TORCH:
        net = UNet3D(in_channels=1, out_channels=2)
        print(f"\n  UNet3D params: {net.count_params():,}")
        dummy = torch.randn(1, 1, 64, 64, 64)
        out = net(dummy)
        print(f"  Forward: {dummy.shape} → {out.shape}")

    print("\n  ThresholdSegmentor test:")
    vol = np.random.randn(64, 64, 64).astype(np.float32)
    seg = ThresholdSegmentor()
    mask = seg.predict(vol)
    print(f"    mask shape={mask.shape}  sum={mask.sum()}")
    print("\nmodel.py — OK")