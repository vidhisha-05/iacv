"""
convert_labels.py — Converts string phase labels to numeric IDs.

FIXES vs original:
  - label2id imported from config.py (single source of truth)
  - Safe .get() lookup instead of KeyError crash on unknown label
  - Prints per-class frame count statistics for each video
  - Skips non-txt files safely

Output: labels_num/<video_id>.txt  — one integer per line
"""

import os
from collections import Counter

from config import LABEL_ROOT, LABEL_NUM_ROOT, LABEL2ID, CLASS_NAMES, NUM_CLASSES, UNKNOWN_ORIG_ID

os.makedirs(LABEL_NUM_ROOT, exist_ok=True)

global_counts: Counter = Counter()

txt_files = sorted(f for f in os.listdir(LABEL_ROOT) if f.endswith(".txt"))
print(f"Found {len(txt_files)} label files\n")

for file in txt_files:
    src = os.path.join(LABEL_ROOT, file)
    dst = os.path.join(LABEL_NUM_ROOT, file)

    with open(src) as f:
        raw_labels = [line.strip() for line in f if line.strip()]

    # Safe conversion — unknown labels fall back to class 9 ("unknown")
    num_labels = []
    bad = 0
    for lbl in raw_labels:
        idx = LABEL2ID.get(lbl, None)
        if idx is None:
            bad += 1
            idx = UNKNOWN_ORIG_ID   # graceful fallback
        num_labels.append(idx)

    # Per-file stats
    counts = Counter(num_labels)
    global_counts.update(counts)

    stats = "  ".join(
        f"{CLASS_NAMES[i]}={counts[i]}"
        for i in range(NUM_CLASSES)
        if counts[i] > 0
    )
    flag = f"  ⚠️  {bad} unknown labels mapped to class 9" if bad else ""
    print(f"[{file}]  {len(num_labels)} frames  |  {stats}{flag}")

    with open(dst, "w") as f:
        f.write("\n".join(map(str, num_labels)) + "\n")

# ---- Global distribution ----
print("\n" + "=" * 60)
print("GLOBAL CLASS DISTRIBUTION (training + test combined)")
print("=" * 60)
total = sum(global_counts.values())
for i in range(NUM_CLASSES):
    cnt  = global_counts[i]
    pct  = 100 * cnt / total if total > 0 else 0
    bar  = "█" * int(pct / 2)
    print(f"  [{i}] {CLASS_NAMES[i]:<15} {cnt:>6} frames  {pct:5.1f}%  {bar}")
print(f"\n  Total : {total} frames")