"""
visualize.py — Plot predicted vs ground-truth surgical phases for one sequence.

FIX: model now returns 3 outputs (out1, out2, out3) — unpacked correctly.
Uses global normalisation stats (not per-video stats).

Usage:
    python visualize.py                   # default: features/06_1.npy
    python visualize.py features/21_1.npy
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from config import LABEL_NUM_ROOT, TRAIN_STATS_PATH, CLASS_NAMES, NUM_CLASSES
from train import MS_TCN

# ---- Pick sequence ----
feat_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("features", "06_1.npy")
label_path = feat_path.replace("features", LABEL_NUM_ROOT).replace(".npy", ".txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- Load normalisation stats ----
if not os.path.exists(TRAIN_STATS_PATH):
    raise FileNotFoundError(
        f"Stats file not found: {TRAIN_STATS_PATH}\n"
        "Run: python compute_train_stats.py"
    )
stats = np.load(TRAIN_STATS_PATH)
global_mean = stats["mean"]
global_std  = stats["std"]

# ---- Load model ----
model = MS_TCN().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ---- Load + normalise features ----
feat = np.load(feat_path).astype(np.float32)
feat = (feat - global_mean) / (global_std + 1e-8)
# FIX: apply per-sequence instance norm — identical to train.py:load_data()
seq_mean = feat.mean(axis=0, keepdims=True)
seq_std  = feat.std(axis=0, keepdims=True).clip(1e-5)
feat     = (feat - seq_mean) / seq_std

t_feat = torch.tensor(feat, dtype=torch.float32).to(device)
t_feat = t_feat.permute(1, 0).unsqueeze(0)

with torch.no_grad():
    # FIX: NUM_STAGES=2 → model() returns exactly 2 outputs, not 3.
    stage_outs = model(t_feat)
    pred = torch.argmax(stage_outs[-1], dim=1).squeeze().cpu().numpy()

# ---- Load ground truth ----
with open(label_path) as f:
    gt = np.array([int(x.strip()) for x in f if x.strip()])

min_len = min(len(pred), len(gt))
pred    = pred[:min_len]
gt      = gt[:min_len]
frames  = np.arange(min_len)

overall_acc = (pred == gt).mean()
print(f"Sequence  : {feat_path}")
print(f"Frames    : {min_len}")
print(f"Accuracy  : {overall_acc:.4f}")

# ---- Plot ----
cmap   = plt.cm.get_cmap("tab10", NUM_CLASSES)
fig, axes = plt.subplots(3, 1, figsize=(18, 6), sharex=True)

# Ground truth ribbon
axes[0].pcolormesh(frames.reshape(1, -1), np.array([[0, 1]]),
                   gt.reshape(1, -1), cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
axes[0].set_ylabel("Ground Truth", fontsize=10)
axes[0].set_yticks([])

# Prediction ribbon
axes[1].pcolormesh(frames.reshape(1, -1), np.array([[0, 1]]),
                   pred.reshape(1, -1), cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
axes[1].set_ylabel("Prediction", fontsize=10)
axes[1].set_yticks([])

# Difference
diff = (pred != gt).astype(float)
axes[2].fill_between(frames, diff, alpha=0.7, color="red", label="Wrong")
axes[2].fill_between(frames, 1 - diff, alpha=0.3, color="green", label="Correct")
axes[2].set_ylabel("Error", fontsize=10)
axes[2].set_xlabel("Frame Index", fontsize=10)
axes[2].legend(loc="upper right", fontsize=8)
axes[2].set_ylim(0, 1.1)

# Colour legend
patches = [
    mpatches.Patch(color=cmap(i), label=f"[{i}] {CLASS_NAMES[i]}")
    for i in range(NUM_CLASSES)
]
fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    f"Surgical Phase Prediction — {os.path.basename(feat_path)}  "
    f"(Acc={overall_acc:.3f})",
    fontsize=12
)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()