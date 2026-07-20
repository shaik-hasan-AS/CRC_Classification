"""
data/dataset.py
Dataset classes and dataloader factories for NCT-CRC-HE-100K and CRC-VAL-HE-7K.
"""


from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from data.transforms import get_train_transforms, get_val_transforms


# ── Class Mapping ─────────────────────────────────────────────────────────────

CRC_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CRC_CLASSES)}

HYBRID_CLASSES = [
    "ADI", "BACK", "BLD", "DEB", "LYM", "MUC", "MUS", "NORM", "NOR_STANFORD", "STR", "TUM"
]
HYBRID_CLASS_TO_IDX = {cls: i for i, cls in enumerate(HYBRID_CLASSES)}


# ── Dataset Classes ───────────────────────────────────────────────────────────

class CRCDataset(ImageFolder):
    """
    Wrapper around ImageFolder for NCT-CRC-HE-100K and CRC-VAL-HE-7K.
    Handles .tif, .jpg, .png extensions.
    """

    def __init__(self, root: str, transform=None):
        super().__init__(root=root, transform=transform, allow_empty=True)
        self._verify_classes()

    def find_classes(self, directory: str) -> Tuple[list, dict]:
        # Enforce strict 9-class mapping regardless of which folders exist on disk
        return CRC_CLASSES, CLASS_TO_IDX

    def _verify_classes(self):
        found = set(self.classes)
        expected = set(CRC_CLASSES)
        missing = expected - found
        extra   = found - expected
        if missing:
            print(f"  [WARN] Missing classes in dataset: {missing}")
        if extra:
            print(f"  [INFO] Extra classes in dataset: {extra}")

    def __repr__(self):
        return (f"CRCDataset | root={self.root} | "
                f"classes={len(self.classes)} | samples={len(self.samples)}")


class HybridCRCDataset(torch.utils.data.Dataset):
    """
    Merges NCT-CRC-HE-100K and STARC-9 on the fly into 11 unified classes.
    """
    def __init__(self, root_100k: str, root_starc9: str, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load 100K
        ds_100k = ImageFolder(root_100k)
        for path, old_idx in ds_100k.samples:
            cls_name = ds_100k.classes[old_idx]
            self.samples.append((path, HYBRID_CLASS_TO_IDX[cls_name]))
            
        # Load STARC-9
        ds_starc9 = ImageFolder(root_starc9)
        for path, old_idx in ds_starc9.samples:
            cls_name = ds_starc9.classes[old_idx]
            # Un-corrupt the STARC-9 mappings dynamically
            if cls_name == "BACK":
                cls_name = "BLD"
            elif cls_name == "NORM":
                cls_name = "NOR_STANFORD"
            
            self.samples.append((path, HYBRID_CLASS_TO_IDX[cls_name]))
            
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        from torchvision.datasets.folder import default_loader
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class EvaluationHybridDataset(torch.utils.data.Dataset):
    """
    Wraps standard 9-class evaluation datasets (like CRC-VAL-HE-7K) 
    and maps them to the 11-class Hybrid indices.
    """
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples = []
        ds = ImageFolder(root)
        for path, old_idx in ds.samples:
            cls_name = ds.classes[old_idx]
            self.samples.append((path, HYBRID_CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        from torchvision.datasets.folder import default_loader
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# ── DataLoader Factories ──────────────────────────────────────────────────────

def get_train_val_loaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and validation DataLoaders from NCT-CRC-HE-100K.
    Uses 80/20 stratified split.
    """
    train_dir = cfg["data"].get("dataset_dir", cfg["data"].get("train_dir", cfg["data"].get("nct_crc_train_dir")))
    bs        = cfg["training"]["batch_size"]
    val_bs    = cfg["training"]["val_batch_size"]
    nw        = cfg["data"]["num_workers"]
    pin       = cfg["data"]["pin_memory"]

    if cfg["data"].get("is_hybrid", False):
        starc9_dir = cfg["data"]["starc9_train_dir"]
        train_dataset = HybridCRCDataset(train_dir, starc9_dir, transform=get_train_transforms(cfg))
        val_dataset   = HybridCRCDataset(train_dir, starc9_dir, transform=get_val_transforms(cfg))
    else:
        if "dataset_dir" in cfg["data"] or "train_dir" in cfg["data"]:
            from torchvision.datasets import ImageFolder
            train_dataset = ImageFolder(train_dir, transform=get_train_transforms(cfg))
            val_dataset   = ImageFolder(train_dir, transform=get_val_transforms(cfg))
        else:
            train_dataset = CRCDataset(train_dir, transform=get_train_transforms(cfg))
            val_dataset   = CRCDataset(train_dir, transform=get_val_transforms(cfg))

    # 80/20 split — same indices for both (different transforms)
    total    = len(train_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(cfg.get("project", {}).get("seed", 42))
    train_idx, val_idx = random_split(range(total), [train_size, val_size],
                                       generator=generator)

    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_idx.indices)
    val_subset   = Subset(val_dataset,   val_idx.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(nw > 0),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    print(f"[DATA] Train: {len(train_subset):,} | Val: {len(val_subset):,}")
    return train_loader, val_loader


def get_crossval_loader(cfg) -> Optional[DataLoader]:
    """
    Returns DataLoader for CRC-VAL-HE-7K (cross-patient test set).
    """
    val_dir = cfg["data"].get("nct_crc_val_dir", "")
    if not val_dir or not Path(val_dir).exists():
        print("[WARN] CRC-VAL-HE-7K path not found. Skipping cross-val loader.")
        return None

    if cfg["data"].get("is_hybrid", False):
        dataset = EvaluationHybridDataset(val_dir, transform=get_val_transforms(cfg))
    else:
        dataset = CRCDataset(val_dir, transform=get_val_transforms(cfg))
        
    loader  = DataLoader(
        dataset,
        batch_size=cfg["training"]["val_batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    print(f"[DATA] CRC-VAL-HE-7K (cross-patient): {len(dataset):,} images")
    return loader


def get_nonorm_loader(cfg) -> Optional[DataLoader]:
    """
    Returns DataLoader for NCT-CRC-HE-100K-NONORM (cross-stain test set).
    """
    nonorm_dir = cfg["data"].get("nct_crc_nonorm_dir", "")
    if not nonorm_dir or not Path(nonorm_dir).exists():
        print("[WARN] NCT-CRC-HE-100K-NONORM path not found. Skipping cross-stain loader.")
        return None

    if cfg["data"].get("is_hybrid", False):
        dataset = EvaluationHybridDataset(nonorm_dir, transform=get_val_transforms(cfg))
    else:
        dataset = CRCDataset(nonorm_dir, transform=get_val_transforms(cfg))
        
    loader  = DataLoader(
        dataset,
        batch_size=cfg["training"]["val_batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    print(f"[DATA] NCT-CRC-HE-100K-NONORM (cross-stain): {len(dataset):,} images")
    return loader
