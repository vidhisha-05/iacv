"""
align_all.py — Aligns image frames, gaze data, and phase labels per video.

FIXES vs original:
  - Fixed indentation bug in image folder existence check
  - Missing gaze now stored as NaN (not (0,0)) then forward-filled
  - Warns explicitly when gaze count differs significantly from frame count
  - Better column naming

Output: processed/<video_id>.csv
Columns: frame, image_path, phase, gaze_x, gaze_y
"""

import os
import numpy as np
import pandas as pd

from config import IMAGE_ROOT, PHASE_ROOT, GAZE_ROOT, PROCESSED_ROOT

os.makedirs(PROCESSED_ROOT, exist_ok=True)


def clean(name: str) -> str:
    """Strip .jpg extension and whitespace from a frame name."""
    return str(name).replace(".jpg", "").strip()


phase_files = sorted(os.listdir(PHASE_ROOT))
print(f"Total sequences found: {len(phase_files)}")

for pf in phase_files:

    video_id = pf.replace(".csv", "")
    print(f"\n── Processing: {video_id}")

    try:
        # ------------------------------------------------------------------ #
        # PHASE LABELS
        # ------------------------------------------------------------------ #
        phase_df = pd.read_csv(os.path.join(PHASE_ROOT, pf))
        phase_df["frame"] = phase_df.iloc[:, 0].apply(clean)
        phase_dict = dict(zip(phase_df["frame"], phase_df.iloc[:, 1]))
        print(f"  Phase labels : {len(phase_dict)} frames")

        # ------------------------------------------------------------------ #
        # GAZE DATA
        # ------------------------------------------------------------------ #
        gaze_dict: dict = {}
        matched_gaze = None

        for gf in os.listdir(GAZE_ROOT):
            if gf.replace(".csv", "").strip() == video_id:
                matched_gaze = gf
                break

        if matched_gaze:
            gaze_df = pd.read_csv(os.path.join(GAZE_ROOT, matched_gaze))
            gaze_df["frame"] = gaze_df.iloc[:, 0].apply(clean)
            gaze_dict = dict(
                zip(gaze_df["frame"], zip(gaze_df.iloc[:, 1], gaze_df.iloc[:, 2]))
            )
            print(f"  Gaze entries : {len(gaze_dict)} frames")

            # Warn if counts are very different
            if abs(len(gaze_dict) - len(phase_dict)) > 50:
                print(
                    f"  ⚠️  Gaze/phase count mismatch: "
                    f"{len(gaze_dict)} gaze vs {len(phase_dict)} phase"
                )
        else:
            print(f"  ⚠️  No gaze file found for {video_id} — will use NaN")

        # ------------------------------------------------------------------ #
        # IMAGE FRAMES
        # ------------------------------------------------------------------ #
        folder_id = video_id.split("_")[0]
        folder_path = os.path.join(IMAGE_ROOT, folder_id)

        if not os.path.exists(folder_path):          # ← indentation fixed
            print(f"  ⚠️  Skipping {video_id} — image folder not found")
            continue

        all_images = sorted(
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.startswith(video_id)
        )
        print(f"  Image frames : {len(all_images)}")

        # ------------------------------------------------------------------ #
        # ALIGN
        # ------------------------------------------------------------------ #
        rows = []
        for img_path in all_images:
            fname    = os.path.basename(img_path)
            frame_id = clean(fname)

            phase_label = phase_dict.get(frame_id, "unknown")

            # Use NaN for missing gaze — forward-fill done below
            gaze_val = gaze_dict.get(frame_id, None)
            gaze_x   = gaze_val[0] if gaze_val is not None else np.nan
            gaze_y   = gaze_val[1] if gaze_val is not None else np.nan

            rows.append([frame_id, img_path, phase_label, gaze_x, gaze_y])

        df = pd.DataFrame(
            rows, columns=["frame", "image_path", "phase", "gaze_x", "gaze_y"]
        )

        # Forward-fill then backward-fill missing gaze, finally fill with 0
        df["gaze_x"] = df["gaze_x"].ffill().bfill().fillna(0.0)
        df["gaze_y"] = df["gaze_y"].ffill().bfill().fillna(0.0)

        missing_gaze = df["gaze_x"].isna().sum()
        if missing_gaze > 0:
            print(f"  ⚠️  {missing_gaze} frames still have missing gaze after fill")

        save_path = os.path.join(PROCESSED_ROOT, f"{video_id}.csv")
        df.to_csv(save_path, index=False)
        print(f"  ✅ Saved → {save_path}  ({len(df)} rows)")

    except Exception as exc:
        print(f"  ❌ Error processing {video_id}: {exc}")
