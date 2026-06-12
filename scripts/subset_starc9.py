import os
import shutil
from pathlib import Path
import random

def subset_starc9():
    source_dir = Path("data/STARC-9-Unpacked/train")
    target_dir = Path("data/STARC-9-10percent/train")
    
    # Validation data is completely reused (not subsetted) to ensure rigorous testing
    val_source = Path("data/STARC-9-Unpacked/val")
    val_target = Path("data/STARC-9-10percent/val")
    
    print("=====================================================")
    print(" Creating 10% Stratified Subset of STARC-9")
    print("=====================================================")
    
    if target_dir.exists():
        print("Target directory already exists. Please delete it to re-run.")
        return

    random.seed(42)  # For deterministic splits
    
    # 1. Subset Training Data (10%) via symlinks
    print("\n--- Subsetting Training Data ---")
    total_copied = 0
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        images = list(class_dir.glob("*.*"))
        num_images = len(images)
        subset_size = int(num_images * 0.10)
        
        # Randomly sample 10%
        subset_images = random.sample(images, subset_size)
        
        # Create target class dir
        target_class_dir = target_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks
        for img in subset_images:
            # We use absolute paths for symlinks to be safe
            src_abs = img.resolve()
            dst_abs = target_class_dir / img.name
            os.symlink(src_abs, dst_abs)
            
        total_copied += subset_size
        print(f"[{class_dir.name}] Sampled {subset_size} / {num_images} images.")
        
    print(f"-> Total Training Images in Subset: {total_copied}")
    
    # 2. Symlink Validation Data (100%)
    print("\n--- Symlinking Validation Data ---")
    val_target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(val_source.resolve(), val_target)
    print("-> Validation data perfectly linked.")
    
    print("\n[DONE] 10% STARC-9 Subset created at data/STARC-9-10percent")

if __name__ == "__main__":
    subset_starc9()
