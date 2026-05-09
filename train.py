"""
train.py — MediVR AI Backend
Train the 3D U-Net tumor segmentation model on BraTS or synthetic data.

Usage:
    python train.py --data_dir /path/to/brats --epochs 50 --batch_size 2
    python train.py --synthetic           # train on synthetic data (no dataset needed)
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH = True
except ImportError:
    TORCH = False
    print("❌ PyTorch not installed. pip install torch")
    exit(1)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model import UNet3D, dice_coefficient, evaluate # type: ignore
    import preprocessing
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Ensure 'model.py' and 'preprocessing.py' exist in: {Path(__file__).parent}")
    exit(1)


# ── Dataset ───────────────────────────────────────────────────────────────

class SyntheticBrainDataset(Dataset):
    """Infinite synthetic dataset for prototyping without real MRI data."""
    def __init__(self, n_samples: int = 200, size: int = 64):
        self.n = n_samples
        self.size = size

    def __len__(self): return self.n

    def __getitem__(self, idx):
        vol, gt = preprocessing.create_synthetic_mri(self.size, seed=idx)
        # Preprocess inline
        proc = preprocessing.normalize_mri(preprocessing.skull_strip(preprocessing.clip_intensity(vol)))
        # Tensors: (C, D, H, W)
        image = torch.tensor(proc[None], dtype=torch.float32)
        label = torch.tensor(gt[None].astype(np.int64), dtype=torch.long)
        return {"image": image, "label": label}


class BraTSDataset(Dataset):
    """BraTS 2021 NIfTI loader.
    Expects folder structure:
        data_dir/
          BraTS2021_00001/
            BraTS2021_00001_t1.nii.gz
            BraTS2021_00001_seg.nii.gz
          ...
    Download: https://www.synapse.org/#!Synapse:syn25829067
    """
    def __init__(self, data_dir: str, size: int = 64, modality: str = "t1"):
        self.paths = sorted(Path(data_dir).glob("BraTS2021_*"))
        self.size = size
        self.modality = modality
        print(f"[BraTSDataset] {len(self.paths)} subjects found")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        subj = self.paths[idx]
        mod_file = subj / f"{subj.name}_{self.modality}.nii.gz"
        seg_file = subj / f"{subj.name}_seg.nii.gz"

        result = preprocessing.full_pipeline(str(mod_file), target_size=self.size)
        proc = result["processed"]

        import nibabel as nib
        seg = nib.load(str(seg_file)).get_fdata().astype(np.int64)
        seg = preprocessing.resize_volume(seg, (self.size,)*3, order=0).astype(np.int64)
        seg = np.clip(seg, 0, 1)   # binary: tumor vs background

        image = torch.tensor(proc[None], dtype=torch.float32)
        label = torch.tensor(seg[None], dtype=torch.long)
        return {"image": image, "label": label}


# ── Loss ──────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        proba = torch.softmax(logits, dim=1)[:, 1]   # prob of class=1
        tgt   = (targets[:, 0] > 0).float()
        inter = (proba * tgt).sum()
        return 1 - (2 * inter + self.smooth) / (proba.sum() + tgt.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy, equal weight."""
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets[:, 0])
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss


# ── Training loop ─────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    dices = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            pred_np  = (torch.argmax(outputs, 1).cpu().numpy() > 0).astype(np.uint8)
            label_np = (labels[:, 0].cpu().numpy() > 0).astype(np.uint8)
            for p, l in zip(pred_np, label_np):
                dices.append(dice_coefficient(p, l))
    return total_loss / len(loader), np.mean(dices)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  MediVR — Training 3D U-Net")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  Size:    {args.size}³")
    print(f"{'='*55}\n")

    # Dataset
    if args.synthetic or not args.data_dir:
        print("Using synthetic dataset (no real MRI)")
        train_ds = SyntheticBrainDataset(n_samples=200, size=args.size)
        val_ds   = SyntheticBrainDataset(n_samples=40,  size=args.size)
    else:
        full_ds  = BraTSDataset(args.data_dir, size=args.size)
        split    = int(0.8 * len(full_ds))
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(full_ds, [split, len(full_ds)-split])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = UNet3D(in_channels=1, out_channels=2)
    model.to(device)
    print(f"  Params: {model.count_params():,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = CombinedLoss()

    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    history = []
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss            = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice    = val_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "val_dice": round(val_dice, 4),
            "lr": round(scheduler.get_last_lr()[0], 7),
        }
        history.append(record)

        marker = " ← best" if val_dice > best_dice else ""
        print(f"  [{epoch:3d}/{args.epochs}]  "
              f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"dice={val_dice:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{elapsed:.1f}s{marker}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
                "args": vars(args),
            }, ckpt_dir / "best_model.pth")

    # Save last checkpoint
    torch.save(model.state_dict(), ckpt_dir / "last_model.pth")

    # Save history
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ Training complete.  Best Dice: {best_dice:.4f}")
    print(f"  Checkpoints: {ckpt_dir}/")
    return model


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> "UNet3D":
    """Load a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet3D(in_channels=1, out_channels=2)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    print(f"[load_checkpoint] Loaded {ckpt_path}  epoch={ckpt.get('epoch','?')}  dice={ckpt.get('val_dice','?')}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediVR — Train 3D U-Net")
    parser.add_argument("--data_dir",   default=None,    help="BraTS dataset directory")
    parser.add_argument("--synthetic",  action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--size",       type=int, default=64, help="Volume size (cubic)")
    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()
    train(args)