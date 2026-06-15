import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

log_file = "kd_train.txt"

epochs = []
train_loss = []
val_loss = []
val_acc = []
val_f1 = []

current_epoch = None

with open(log_file, "r") as f:
    for line in f:
        # Match Epoch line: Epoch [1/200]
        epoch_match = re.search(r"Epoch \[(\d+)/", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # Match TRAIN metrics: [TRAIN] loss=0.3041  acc=0.9865  F1=0.9864
        train_match = re.search(r"\[TRAIN\] loss=([\d.]+)\s+acc=([\d.]+)\s+F1=([\d.]+)", line)
        if train_match and current_epoch is not None:
            train_loss.append((current_epoch, float(train_match.group(1))))
        
        # Match VAL metrics: [VAL]   loss=0.5285  acc=0.9871  F1=0.9866
        val_match = re.search(r"\[VAL\]\s+loss=([\d.]+)\s+acc=([\d.]+)\s+F1=([\d.]+)", line)
        if val_match and current_epoch is not None:
            val_loss.append((current_epoch, float(val_match.group(1))))
            val_acc.append((current_epoch, float(val_match.group(2))))
            val_f1.append((current_epoch, float(val_match.group(3))))

train_df = pd.DataFrame(train_loss, columns=['epoch', 'train_loss'])
val_df = pd.DataFrame(val_loss, columns=['epoch', 'val_loss'])
acc_df = pd.DataFrame(val_acc, columns=['epoch', 'val_acc'])
f1_df = pd.DataFrame(val_f1, columns=['epoch', 'val_f1'])

# Merge
history = train_df.merge(val_df, on='epoch').merge(acc_df, on='epoch').merge(f1_df, on='epoch')

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("MedLite-CRC (V1+KD) Training Dynamics", fontweight='bold')

# Plot 1: Loss Curve
ax1.plot(history['epoch'], history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2.5)
ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5, linestyle='--')
ax1.set_title("Loss Convergence")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.7)

# Plot 2: Accuracy & F1 Curve
ax2.plot(history['epoch'], history['val_acc'], label='Val Accuracy', color='#2ca02c', linewidth=2.5)
ax2.plot(history['epoch'], history['val_f1'], label='Val Macro-F1', color='#d62728', linewidth=2.5, linestyle='-.')
ax2.set_title("Validation Metrics")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Score")
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()

os.makedirs("assets", exist_ok=True)
save_path = "assets/learning_curves.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved publication-ready curves to: {save_path}")
