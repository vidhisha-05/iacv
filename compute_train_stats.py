"""
compute_train_stats.py — Compute and save global normalisation statistics
from the TRAINING SET only.

Run this ONCE after extract_features.py and before train.py.

Saves: features/train_stats.npz  (keys: mean, std)

WHY THIS MATTERS:
  Per-video normalisation (original code) means every video is centred at 0
  before it reaches the model — the test video statistics are therefore totally
  different from what the model saw during training, causing distribution shift.
  Using a FIXED global mean/std computed on the training set fixes this.
"""

import os
import numpy as np

from config import FEATURE_ROOT, LABEL_NUM_ROOT, TRAIN_VIDEOS, TRAIN_STATS_PATH, INPUT_DIM


def compute_train_stats() -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std over ALL training sequences."""

    # Two-pass algorithm for numerical stability:
    # Pass 1: accumulate sum and count
    # Pass 2: accumulate sum of squares

    print("Pass 1 — computing mean …")
    feat_sum   = np.zeros(INPUT_DIM, dtype=np.float64)
    total_frames = 0

    for fname in sorted(os.listdir(FEATURE_ROOT)):
        if not fname.endswith(".npy"):
            continue
        video_id = fname.split("_")[0]
        if video_id not in TRAIN_VIDEOS:
            continue

        feat = np.load(os.path.join(FEATURE_ROOT, fname)).astype(np.float64)
        feat_sum     += feat.sum(axis=0)
        total_frames += len(feat)
        print(f"  {fname}  — {len(feat)} frames")

    if total_frames == 0:
        raise RuntimeError("No training feature files found!")

    global_mean = feat_sum / total_frames
    print(f"\nTotal training frames: {total_frames}")

    print("\nPass 2 — computing std …")
    sq_sum = np.zeros(INPUT_DIM, dtype=np.float64)

    for fname in sorted(os.listdir(FEATURE_ROOT)):
        if not fname.endswith(".npy"):
            continue
        video_id = fname.split("_")[0]
        if video_id not in TRAIN_VIDEOS:
            continue

        feat    = np.load(os.path.join(FEATURE_ROOT, fname)).astype(np.float64)
        sq_sum += ((feat - global_mean) ** 2).sum(axis=0)

    global_std = np.sqrt(sq_sum / (total_frames - 1) + 1e-8)

    return global_mean.astype(np.float32), global_std.astype(np.float32)


if __name__ == "__main__":
    mean, std = compute_train_stats()

    np.savez(TRAIN_STATS_PATH, mean=mean, std=std)
    print(f"\n[OK] Saved normalisation stats -> {TRAIN_STATS_PATH}")
    print(f"   mean range : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   std  range : [{std.min():.4f},  {std.max():.4f}]")
    print("\nNext step: run  python train.py")
