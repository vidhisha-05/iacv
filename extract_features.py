"""
extract_features.py — Extracts visual + gaze features per video sequence.

FIXES vs original:
  - Added proper ImageNet normalisation (mean/std) — CRITICAL fix
  - Gaze normalised to [0, 1] using video resolution
  - Added delta-gaze (Δx, Δy) as temporal gaze motion cue
  - INPUT_DIM now 2052 (2048 ResNet + 4 gaze features)
  - Replaced deprecated pretrained=True with weights=ResNet50_Weights.DEFAULT
  - Progress reported every 100 frames instead of 200

Output: features/<video_id>.npy  — shape: (T, 2052)

NOTE: Run this script ONCE. Then run compute_train_stats.py before training.
"""

import os

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import ResNet50_Weights

from config import (
    PROCESSED_ROOT, FEATURE_ROOT,
    VIDEO_WIDTH, VIDEO_HEIGHT,
    RESNET_DIM, GAZE_DIM, INPUT_DIM,
)

os.makedirs(FEATURE_ROOT, exist_ok=True)

# --------------------------------------------------------------------------- #
# MODEL — ResNet50 with proper ImageNet weights (no deprecation warning)
# --------------------------------------------------------------------------- #
print("Loading ResNet50 …")
_weights = ResNet50_Weights.DEFAULT
model    = models.resnet50(weights=_weights)
model.fc = torch.nn.Identity()   # strip classifier — keep 2048-d features
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"  Running on: {device}")

# --------------------------------------------------------------------------- #
# TRANSFORM — now includes correct ImageNet normalisation
# --------------------------------------------------------------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # CRITICAL: ResNet was trained on ImageNet-normalised images
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_image_feature(img_path: str) -> np.ndarray:
    """Return 2048-d ResNet50 feature vector for one image."""
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(inp)
    return feat.squeeze().cpu().numpy()   # (2048,)


# --------------------------------------------------------------------------- #
# MAIN LOOP
# --------------------------------------------------------------------------- #
csv_files = sorted(f for f in os.listdir(PROCESSED_ROOT) if f.endswith(".csv"))
print(f"\nTotal sequences to process: {len(csv_files)}\n")

for file in csv_files:

    save_path = os.path.join(FEATURE_ROOT, file.replace(".csv", ".npy"))
    if os.path.exists(save_path):
        print(f"  [skip] {file} — .npy already exists")
        continue

    print(f"── Processing {file}")
    df = pd.read_csv(os.path.join(PROCESSED_ROOT, file))
    T  = len(df)

    all_features = np.zeros((T, INPUT_DIM), dtype=np.float32)

    prev_gx, prev_gy = 0.0, 0.0   # for delta computation

    for i, row in df.iterrows():

        # ---- Visual feature ----
        img_feat = extract_image_feature(row["image_path"])   # (2048,)

        # ---- Gaze feature ---- normalised to [0, 1]
        gx = float(row["gaze_x"]) / VIDEO_WIDTH
        gy = float(row["gaze_y"]) / VIDEO_HEIGHT

        # ---- Delta gaze ---- motion cue between consecutive frames
        dgx = gx - prev_gx
        dgy = gy - prev_gy
        prev_gx, prev_gy = gx, gy

        gaze_feat = np.array([gx, gy, dgx, dgy], dtype=np.float32)   # (4,)

        all_features[i] = np.concatenate([img_feat, gaze_feat])

        if i % 100 == 0:
            print(f"  Frame {i:>5}/{T}")

    np.save(save_path, all_features)
    print(f"  ✅ Saved → {save_path}  shape={all_features.shape}\n")

print("Feature extraction complete.")
print(f"Each feature vector is {INPUT_DIM}-dimensional "
      f"({RESNET_DIM} ResNet + {GAZE_DIM} gaze).")