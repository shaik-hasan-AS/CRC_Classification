import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_crc5000():
    source_dir = Path("data/CRC-5000_mapped")
    train_dir = Path("data/CRC-5000/train")
    val_dir = Path("data/CRC-5000/val")
    
    print("=====================================================")
    print(" Splitting CRC-5000 (80/20 Train/Val)")
    print("=====================================================")
    
    if train_dir.exists() or val_dir.exists():
        print("Target directories already exist. Please delete data/CRC-5000 to re-run.")
        return

    # Collect all images and their labels
    all_images = []
    all_labels = []
    
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        images = list(class_dir.glob("*.*"))
        all_images.extend(images)
        all_labels.extend([class_dir.name] * len(images))
        
    print(f"Found {len(all_images)} total images.")
    
    # Stratified split 80/20
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        all_images, all_labels, test_size=0.20, stratify=all_labels, random_state=42
    )
    
    print(f"Split: {len(train_imgs)} Train | {len(val_imgs)} Val")
    
    # Helper to create symlinks
    def create_links(img_list, lbl_list, target_base):
        for img, lbl in zip(img_list, lbl_list):
            target_class_dir = target_base / lbl
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            src_abs = img.resolve()
            dst_abs = target_class_dir / img.name
            os.symlink(src_abs, dst_abs)
            
    # Create the split directories
    create_links(train_imgs, train_lbls, train_dir)
    create_links(val_imgs, val_lbls, val_dir)
    
    print("\n[DONE] CRC-5000 successfully split into data/CRC-5000/train and data/CRC-5000/val")

if __name__ == "__main__":
    split_crc5000()
