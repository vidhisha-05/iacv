"""
prepare_labels.py — Extract phase column from aligned CSV files into text files.

No logic changes needed — just aligned to use config paths.
Output: labels/<video_id>.txt  — one phase name per line
"""

import os
import pandas as pd

from config import PROCESSED_ROOT, LABEL_ROOT

os.makedirs(LABEL_ROOT, exist_ok=True)

csv_files = [f for f in os.listdir(PROCESSED_ROOT) if f.endswith(".csv")]
print(f"Found {len(csv_files)} processed sequences\n")

for file in sorted(csv_files):
    df         = pd.read_csv(os.path.join(PROCESSED_ROOT, file))
    labels     = df["phase"].astype(str).tolist()
    save_path  = os.path.join(LABEL_ROOT, file.replace(".csv", ".txt"))

    with open(save_path, "w") as f:
        f.write("\n".join(labels) + "\n")

    print(f"  Saved {save_path}  ({len(labels)} frames)")

print("\nDone.")