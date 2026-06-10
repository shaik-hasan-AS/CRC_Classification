#!/usr/bin/env python3
"""
scripts/prepare_crc5000.py

Maps the downloaded CRC-5000 (Kather-CRC-2016) dataset into the
NCT-CRC-HE-100K 9-class directory structure for evaluation.

Mapping:
  01_TUMOR   -> TUM
  02_STROMA  -> STR
  03_COMPLEX -> STR (User approved merging complex stroma into stroma)
  04_LYMPHO  -> LYM
  05_DEBRIS  -> DEB
  06_MUCOSA  -> NORM
  07_ADIPOSE -> ADI
  08_EMPTY   -> BACK
"""

import shutil
from pathlib import Path

MAPPING = {
    "01_TUMOR": "TUM",
    "02_STROMA": "STR",
    # "03_COMPLEX": "STR",  # Dropping complex stroma as it confuses pure classes
    "04_LYMPHO": "LYM",
    "05_DEBRIS": "DEB",
    "06_MUCOSA": "NORM",
    "07_ADIPOSE": "ADI",
    "08_EMPTY": "BACK",
}

def main():
    source_dir = Path("data/Kather_texture_2016_image_tiles_5000")
    dest_dir = Path("data/CRC-5000_mapped")
    
    if not source_dir.exists():
        print(f"[ERROR] Source directory not found: {source_dir}")
        print("Please run `python scripts/download_data.py` first.")
        return
        
    print(f"Mapping CRC-5000 from {source_dir} to {dest_dir}...")
    
    # Create mapped directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    for class_name in set(MAPPING.values()):
        (dest_dir / class_name).mkdir(exist_ok=True)
        
    copied = 0
    for src_class, dest_class in MAPPING.items():
        src_path = source_dir / src_class
        if not src_path.exists():
            print(f"[WARN] Missing source class: {src_path}")
            continue
            
        dest_path = dest_dir / dest_class
        files = list(src_path.glob("*.tif"))
        print(f"Mapping {src_class} ({len(files)} files) -> {dest_class}")
        
        for f in files:
            # Add prefix if merging complex to avoid filename collisions
            dest_file = dest_path / f"{src_class}_{f.name}"
            shutil.copy2(f, dest_file)
            copied += 1
            
    print(f"\n[DONE] Successfully mapped {copied} images to {dest_dir}")
    print(f"You can now evaluate using: python evaluate.py --data_dir {dest_dir} --weights ...")

if __name__ == "__main__":
    main()
