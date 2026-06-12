import os
import zipfile
import shutil
from pathlib import Path

TRAIN_ZIP_DIR = Path("data/STARC-9/Training_data_normalized")
VAL_ZIP_DIR = Path("data/STARC-9/Validation_data")
OUT_TRAIN_DIR = Path("data/STARC-9-Unpacked/train")
OUT_VAL_DIR = Path("data/STARC-9-Unpacked/val")

CLASS_MAPPING = {
    "FCT": "STR",
    "NCS": "DEB",
    "NOR": "NORM",
    "BLD": "BACK"
}

def extract_and_map(zip_path, out_dir):
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)
    print(f"  -> Extracted. Deleting original zip to save space...")
    os.remove(zip_path)

def rename_classes(base_dir):
    print(f"Mapping classes in {base_dir}...")
    for old_name, new_name in CLASS_MAPPING.items():
        old_path = base_dir / old_name
        new_path = base_dir / new_name
        if old_path.exists():
            print(f"  Renaming {old_name} -> {new_name}")
            # If the new path already exists (e.g. from merging validation sets), move the files
            if new_path.exists():
                for img in old_path.glob("*"):
                    shutil.move(str(img), str(new_path / img.name))
                old_path.rmdir()
            else:
                old_path.rename(new_path)

def main():
    print("============================================================")
    print(" Preparing STARC-9 Dataset")
    print("============================================================")
    
    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Process Training Zips
    if TRAIN_ZIP_DIR.exists():
        for zip_file in TRAIN_ZIP_DIR.glob("*.zip"):
            extract_and_map(zip_file, OUT_TRAIN_DIR)
        rename_classes(OUT_TRAIN_DIR)
        
    # 2. Process Validation Zips
    if VAL_ZIP_DIR.exists():
        for zip_file in VAL_ZIP_DIR.glob("*.zip"):
            extract_and_map(zip_file, OUT_VAL_DIR)
            
        # The validation zips might extract into top-level directories
        # We need to flatten them if so.
        for folder in OUT_VAL_DIR.iterdir():
            if folder.is_dir() and folder.name not in ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM", "FCT", "NCS", "NOR", "BLD"]:
                print(f"Flattening validation directory: {folder.name}")
                for class_dir in folder.iterdir():
                    if class_dir.is_dir():
                        target_class_dir = OUT_VAL_DIR / class_dir.name
                        target_class_dir.mkdir(parents=True, exist_ok=True)
                        for img_file in class_dir.glob("*"):
                            shutil.move(str(img_file), str(target_class_dir / img_file.name))
                shutil.rmtree(folder)
                
        rename_classes(OUT_VAL_DIR)
        
    print("\n[DONE] STARC-9 Dataset is fully extracted, mapped, and cleaned up!")

if __name__ == "__main__":
    main()
