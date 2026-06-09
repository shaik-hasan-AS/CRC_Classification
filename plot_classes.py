import os
import random
import matplotlib.pyplot as plt
from PIL import Image

data_dir = "data/NCT-CRC-HE-100K"
classes = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
plt.suptitle("The 9 Tissue Classes of Colorectal Cancer (H&E Stained)", fontsize=16)

for i, cls in enumerate(classes):
    cls_dir = os.path.join(data_dir, cls)
    if not os.path.exists(cls_dir):
        print(f"Directory {cls_dir} not found!")
        continue
    
    images = os.listdir(cls_dir)
    random_image = random.choice(images)
    img_path = os.path.join(cls_dir, random_image)
    
    img = Image.open(img_path)
    
    ax = axes[i // 3, i % 3]
    ax.imshow(img)
    ax.set_title(cls, fontsize=14, fontweight="bold")
    ax.axis('off')

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/9_classes_grid.png", dpi=150)
print("Saved 9_classes_grid.png")
