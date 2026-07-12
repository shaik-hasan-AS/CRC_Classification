import os
import time
import socket
from huggingface_hub import hf_hub_download

# Set a global socket timeout to prevent connections from hanging indefinitely
socket.setdefaulttimeout(30.0)

def main():
    print("============================================================")
    print(" Robust Sequential STARC-9 Downloader")
    print("============================================================")
    
    repo_id = "Path2AI/STARC-9"
    repo_type = "dataset"
    local_dir = "data/STARC-9"
    
    files_to_download = [
        "Training_data_normalized/ADI.zip",
        "Training_data_normalized/BLD.zip",
        "Training_data_normalized/FCT.zip",
        "Training_data_normalized/LYM.zip",
        "Training_data_normalized/MUC.zip",
        "Training_data_normalized/MUS.zip",
        "Training_data_normalized/NCS.zip",
        "Training_data_normalized/NOR.zip",
        "Training_data_normalized/TUM.zip",
        "Validation_data/CURATED-TCGA-CRC-HE-VAL-20K-NORMALIZED.zip",
        "Validation_data/STANFORD-CRC-HE-VAL-LARGE-NORMALIZED.zip",
        "Validation_data/STANFORD-CRC-HE-VAL-SMALL-NORMALIZED.zip",
        "README.md",
        "metadata.json"
    ]
    
    os.makedirs(local_dir, exist_ok=True)
    
    for i, filename in enumerate(files_to_download, 1):
        target_path = os.path.join(local_dir, filename)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"[{i}/{len(files_to_download)}] Already exists: {filename} (skipped)")
            continue
            
        print(f"\n[{i}/{len(files_to_download)}] Downloading: {filename}")
        
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"  Successfully downloaded: {filename}")
                break
            except Exception as e:
                print(f"  Attempt {attempt}/{max_retries} failed with error: {e}")
                if attempt < max_retries:
                    print("  Waiting 10 seconds before retrying...")
                    time.sleep(10)
                else:
                    print(f"  Failed to download {filename} after {max_retries} attempts.")
                    exit(1)
                    
    print("\n[SUCCESS] All files downloaded successfully!")

if __name__ == "__main__":
    main()
