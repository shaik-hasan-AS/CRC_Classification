#!/usr/bin/env python3
"""
Download NCT-CRC-HE-100K and CRC-VAL-HE-7K from Zenodo.
Run once before training: python scripts/download_data.py
"""


import sys
import zipfile
import urllib.request
from pathlib import Path


DATASETS = {
    "NCT-CRC-HE-100K": {
        "url": "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip",
        "dest": "data/NCT-CRC-HE-100K.zip",
        "extract_to": "data/",
    },
    "CRC-VAL-HE-7K": {
        "url": "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip",
        "dest": "data/CRC-VAL-HE-7K.zip",
        "extract_to": "data/",
    },
    "NCT-CRC-HE-100K-NONORM": {
        "url": "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K-NONORM.zip",
        "dest": "data/NCT-CRC-HE-100K-NONORM.zip",
        "extract_to": "data/",
    },
}


def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar = "#" * int(pct / 2)
    sys.stdout.write(f"\r  [{bar:<50}] {pct:.1f}%")
    sys.stdout.flush()


def download_and_extract(name, info):
    dest = Path(info["dest"])
    extract_to = Path(info["extract_to"])
    extract_to.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"[SKIP] {name} zip already exists at {dest}")
    else:
        print(f"\n[DOWNLOAD] {name}")
        print(f"  URL: {info['url']}")
        urllib.request.urlretrieve(info["url"], dest, reporthook=progress_bar)
        print()

    final_dir = extract_to / name
    if final_dir.exists():
        print(f"[SKIP] {name} already extracted at {final_dir}")
    else:
        print(f"[EXTRACT] {name} → {extract_to}")
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(extract_to)
        print(f"  Done.")

    return final_dir


def verify_structure(path: Path, expected_classes: list):
    missing = []
    for cls in expected_classes:
        if not (path / cls).exists():
            missing.append(cls)
    if missing:
        print(f"  [WARN] Missing class folders: {missing}")
    else:
        counts = {cls: len(list((path / cls).glob("*.tif"))) +
                           len(list((path / cls).glob("*.jpg"))) +
                           len(list((path / cls).glob("*.png")))
                  for cls in expected_classes}
        total = sum(counts.values())
        print(f"  [OK] {total:,} images across {len(expected_classes)} classes")
        for cls, n in counts.items():
            print(f"       {cls}: {n:,}")


if __name__ == "__main__":
    CLASSES = ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"]

    print("=" * 60)
    print("  MedLite-CRC Dataset Downloader")
    print("  Source: Zenodo record 1214456")
    print("=" * 60)

    for name, info in DATASETS.items():
        final_dir = download_and_extract(name, info)
        print(f"[VERIFY] {name}")
        verify_structure(final_dir, CLASSES)

    print("\n[DONE] Datasets ready. Update configs/config.yaml if needed.")
    print("  data/NCT-CRC-HE-100K/  ← training")
    print("  data/CRC-VAL-HE-7K/    ← cross-patient validation")
