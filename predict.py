"""
predict.py — Quick prediction check on a set of videos.

FIX: model now returns 3 outputs (out1, out2, out3) — unpacked correctly.
Uses global normalisation stats from training set (not per-video stats).
"""

import os
import numpy as np
import torch

from config import FEATURE_ROOT, LABEL_NUM_ROOT, TRAIN_STATS_PATH
from train import MS_TCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

files = sorted(os.listdir(FEATURE_ROOT))[:5]   # preview first 5 sequences

for file in files:
    if not file.endswith(".npy"):
        continue
    print(f"\n===== {file} =====")

    feat = np.load(os.path.join(FEATURE_ROOT, file)).astype(np.float32)
    feat = (feat - global_mean) / (global_std + 1e-8)   # global normalisation

    t_feat = torch.tensor(feat, dtype=torch.float32).to(device)
    t_feat = t_feat.permute(1, 0).unsqueeze(0)

    with torch.no_grad():
        _, _, out3 = model(t_feat)   # ← FIXED: unpack 3 values, use final stage
        pred = torch.argmax(out3, dim=1).squeeze().cpu().numpy()

    label_file = file.replace(".npy", ".txt")
    label_path = os.path.join(LABEL_NUM_ROOT, label_file)
    if os.path.exists(label_path):
        with open(label_path) as f:
            gt = np.array([int(x.strip()) for x in f if x.strip()])
        gt   = gt[:len(pred)]
        acc  = (pred == gt).mean()
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Pred unique : {np.unique(pred)}")
        print(f"  GT unique   : {np.unique(gt)}")
    else:
        print(f"  Pred unique : {np.unique(pred)}  (no GT file)")