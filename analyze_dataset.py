"""
analyze_dataset.py — Dataset statistics and class distribution analysis.

Uses config.py as the single source of truth for paths and class names.
"""

import os
import numpy as np
from collections import Counter, defaultdict

from config import (
    FEATURE_ROOT, LABEL_NUM_ROOT,
    CLASS_NAMES, NUM_CLASSES, UNKNOWN_ORIG_ID,
)

LABEL_ROOT = LABEL_NUM_ROOT
# Include "unknown" in display names for analysis (10 entries: 0-8 + unknown=9)
_CLASS_NAMES = list(CLASS_NAMES) + ["unknown"]
_TOTAL_CLASSES = NUM_CLASSES + 1   # 9 real + 1 unknown for display

seqs = []
for fname in sorted(os.listdir(FEATURE_ROOT)):
    if not fname.endswith('.npy'):
        continue
    feat = np.load(os.path.join(FEATURE_ROOT, fname))
    lpath = os.path.join(LABEL_ROOT, fname.replace('.npy', '.txt'))
    if not os.path.exists(lpath):
        continue
    with open(lpath) as f:
        labels = [int(x.strip()) for x in f if x.strip()]
    n = min(len(feat), len(labels))
    vid = fname.split('_')[0]
    seqs.append({'file': fname, 'vid': vid, 'T': n,
                 'feat_dim': feat.shape[1], 'labels': labels[:n]})

print(f"Total sequences: {len(seqs)}")
print(f"Feature dimension: {seqs[0]['feat_dim']}")
print()

vid_groups = defaultdict(list)
for s in seqs:
    vid_groups[s['vid']].append(s)

print("Per-video group stats:")
all_labels = []
for vid, group in sorted(vid_groups.items()):
    tot_frames = sum(s['T'] for s in group)
    lab_all = [l for s in group for l in s['labels']]
    dom = Counter(lab_all).most_common(1)[0]
    print(f"  Video {vid}: {len(group)} clips, {tot_frames} frames, "
          f"dom={_CLASS_NAMES[dom[0]]}({dom[1]/tot_frames*100:.0f}%)")
    all_labels.extend(lab_all)

print()
total = len(all_labels)
print(f"TOTAL frames across ALL sequences: {total}")
counts = Counter(all_labels)
print("Class distribution (ALL data):")
for c in range(_TOTAL_CLASSES):
    n = counts.get(c, 0)
    pct = n / total * 100
    bar = '#' * int(pct / 2)
    print(f"  [{c}] {_CLASS_NAMES[c]:<14} {n:>8} ({pct:5.1f}%) {bar}")

print()
lens = [s['T'] for s in seqs]
print("Sequence length stats:")
print(f"  min={min(lens)}, max={max(lens)}, mean={int(np.mean(lens))}, median={int(np.median(lens))}")
print(f"  Short  (<500): {sum(1 for l in lens if l < 500)}")
print(f"  Medium (500-2000): {sum(1 for l in lens if 500 <= l < 2000)}")
print(f"  Long   (>=2000): {sum(1 for l in lens if l >= 2000)}")

print()
print("Train videos (01-20) vs Test video (21):")
train_labels = [l for s in seqs if s['vid'] != '21' for l in s['labels']]
test_labels  = [l for s in seqs if s['vid'] == '21' for l in s['labels']]
print(f"  Train frames: {len(train_labels)}")
print(f"  Test  frames: {len(test_labels)}")

print("\nClass distribution in TRAIN only:")
tc = Counter(train_labels)
for c in range(_TOTAL_CLASSES):
    n = tc.get(c, 0)
    pct = n / max(len(train_labels), 1) * 100
    print(f"  [{c}] {_CLASS_NAMES[c]:<14} {n:>8} ({pct:5.1f}%)")

print("\nClass distribution in TEST only (video 21):")
ec = Counter(test_labels)
for c in range(_TOTAL_CLASSES):
    n = ec.get(c, 0)
    pct = n / max(len(test_labels), 1) * 100
    print(f"  [{c}] {_CLASS_NAMES[c]:<14} {n:>8} ({pct:5.1f}%)")

# Imbalance ratio
dominant = max(counts.values())
minority = min(v for v in counts.values() if v > 0)
print(f"\nImbalance ratio (max/min non-zero): {dominant}/{minority} = {dominant/minority:.1f}x")
