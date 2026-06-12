import os
from huggingface_hub import snapshot_download

def main():
    print("============================================================")
    print(" Downloading STARC-9 Dataset from Hugging Face")
    print("============================================================")
    print("This will download roughly 104GB of data into data/STARC-9.")
    print("It may take a while depending on your internet connection.")
    
    os.makedirs("data/STARC-9", exist_ok=True)
    
    # We only want the dataset images, not the 20GB of model weights from other researchers
    allow_patterns = [
        "Training_data_normalized/**",
        "Validation_data/**",
        "metadata.json",
        "README.md"
    ]
    
    try:
        download_path = snapshot_download(
            repo_id="Path2AI/STARC-9",
            repo_type="dataset",
            local_dir="data/STARC-9",
            allow_patterns=allow_patterns,
            max_workers=8
        )
        print(f"\n[DONE] Download complete! Dataset safely stored at: {download_path}")
    except KeyboardInterrupt:
        print("\n[PAUSED] Download paused. You can run this script again later to resume exactly where it left off!")

if __name__ == "__main__":
    main()
