# IACV — MS-TCN++ for Surgical Phase Recognition

A multi-stage temporal convolutional network (MS-TCN++) pipeline for recognizing surgical workflow phases from video features (ResNet-50 + gaze data). Trained on 20 surgical videos, tested on video 21.

---

## Pipeline Overview

The training pipeline has **6 sequential steps** that must be run in order:

```
Raw Data (images/, gaze/, annotations/)
  │
  ├─ 1. align_all.py          →  processed/<video>.csv        (align frames, gaze, labels)
  ├─ 2. prepare_labels.py     →  labels/<video>.txt           (extract phase names)
  ├─ 3. convert_labels.py     →  labels_num/<video>.txt       (phase names → numeric IDs)
  ├─ 4. extract_features.py   →  features/<video>.npy         (ResNet50 + gaze features)
  ├─ 5. compute_train_stats.py→  features/train_stats.npz     (global mean/std)
  └─ 6. train.py              →  best_model.pth               (trained MS-TCN++ model)
                                  features/transition_matrix.npy
                                  features/init_dist.npy

Evaluation:
  └─ evaluate.py --video 21   →  confusion matrices, timelines, metrics
```

### Required Directory Layout

```
iacv/
├── images/          # Frame images per video (e.g., images/01/, images/02/, ...)
├── gaze/            # Gaze CSVs per video
├── annotations/
│   └── phase/       # Phase annotation CSVs per video
├── processed/       # (generated) Aligned CSVs
├── labels/          # (generated) Phase name text files
├── labels_num/      # (generated) Numeric label text files
├── features/        # (generated) .npy feature files + train_stats.npz
└── best_model.pth   # (generated) Trained model checkpoint
```

---

## Evaluation Results (Video 21 — Test Set)

The result images in this repository tell the story of what is going wrong. Below is the diagnosis derived from the confusion matrices and timeline plots.

### Confusion Matrices

| Matrix | File | Key Observation |
|--------|------|-----------------|
| Earlier model (10 classes) | `confusion_matrix_video21.png` | Predicts mostly closure + dissection. **Zero recall** on disinfection, dressing, hemostasis, irrigation. Predicts a 10th "unknown" class that shouldn't exist. |
| v7 Raw | `cm_raw_video21.png` | Still collapses: dissection is heavily over-predicted. **Zero recall** on anesthesia (4/18), dressing (0/196), hemostasis (0/16). |
| v7 Post-processed (Viterbi+TTA) | `cm_pp_video21.png` | Marginal improvement via Viterbi blending. Still **zero recall** on anesthesia, dressing, hemostasis. |

### Timeline Plots

- **`21_1_timeline.png`**: A 68-frame clip where GT is entirely "design" (green). The model predicts "dissection" (brown) for the entire clip — complete phase confusion.
- **`21_4_timeline.png`**: The longest clip (~1100 frames). The model broadly gets closure vs dissection transitions, but all minority phases (design, dressing, hemostasis, disinfection) are absorbed into the majority classes.
- **`training_2_84.png`**: Training prediction overlay shows the model learns to track majority phases but the jittery predictions at boundaries suggest overfitting to window-level statistics.

---

## Diagnosed Bugs

### Bug 1 — `visualize.py` Tuple Unpacking Crash (Still Present)

`visualize.py` (line 53) still has the **unfixed** 3-way unpack:

```python
_, _, out3 = model(t_feat)   # CRASH: model returns 2 outputs, not 3
```

`predict.py` was fixed in the latest commit, but `visualize.py` was not updated.

**Fix:** Change to `stage_outs = model(t_feat); out = stage_outs[-1]`

---

### Bug 2 — `visualize.py` Missing Per-Sequence Instance Norm

`visualize.py` applies global normalization but does **not** apply the per-sequence instance norm that `train.py`, `evaluate.py`, and `predict.py` all apply:

```python
# visualize.py (MISSING these 3 lines after global norm):
seq_mean = feat.mean(axis=0, keepdims=True)
seq_std  = feat.std(axis=0, keepdims=True).clip(1e-5)
feat     = (feat - seq_mean) / seq_std
```

This means `visualize.py` feeds differently-normalized features to the model, producing incorrect predictions.

---

### Bug 3 — `convert_labels.py` References Missing `"unknown"` Key

`config.py` defines `LABEL2ID` with **9 classes** (indices 0–8). There is no `"unknown"` key:

```python
LABEL2ID = {
    "anesthesia": 0, "closure": 1, "design": 2, "disinfection": 3,
    "dissection": 4, "dressing": 5, "hemostasis": 6, "incision": 7,
    "irrigation": 8,
}
```

But `convert_labels.py` (line 39) does:

```python
idx = LABEL2ID["unknown"]   # KeyError — "unknown" not in LABEL2ID
```

This will **crash** whenever an unrecognized label appears in the annotation files. It should use `UNKNOWN_ORIG_ID` (which is `9`) from config instead.

---

### Bug 4 — `InstanceNorm1d` Train-vs-Inference Behavior

The recent fix replaced `GroupNorm(1, C)` with `InstanceNorm1d(C, affine=True)`. However, `InstanceNorm1d` has **different default behavior in training vs eval mode**:

- **Training mode** (`model.train()`): computes per-instance statistics from the current input (correct).
- **Eval mode** (`model.eval()`): by default `track_running_stats=False`, so it still uses per-instance stats — this is fine with the current code.

However, if `track_running_stats` were ever set to `True`, it would accumulate running statistics from the 128-frame training windows and apply those at test time on full-length sequences — re-introducing the original distribution shift. The current code is safe, but this is a fragile design. Using explicit `LayerNorm` over channels would be more robust.

---

### Bug 5 — `fix_encoding.py` References Non-Existent File

`fix_encoding.py` (line 3) lists `'lovo_eval.py'` as a target file:

```python
files = ['train.py', 'evaluate.py', 'lovo_eval.py']
```

No file named `lovo_eval.py` exists in the repository. This will silently fail (caught by the try/except) but indicates incomplete cleanup.

---

### Bug 6 — `analyze_dataset.py` Hardcodes Constants

`analyze_dataset.py` defines its own `CLASS_NAMES` list (with 10 entries including `'unknown'`) and `FEATURE_ROOT`/`LABEL_ROOT` paths instead of importing from `config.py`. If the class mapping or paths ever change in `config.py`, this script will silently produce incorrect analysis.

---

## Summary Table

| # | File | Severity | Issue |
|---|------|----------|-------|
| 1 | `visualize.py:53` | **CRASH** | Unpacks 3 outputs from 2-stage model |
| 2 | `visualize.py:47` | **Wrong results** | Missing per-sequence instance norm |
| 3 | `convert_labels.py:39` | **CRASH** | `LABEL2ID["unknown"]` — key doesn't exist |
| 4 | `train.py` | **Fragile** | `InstanceNorm1d` safe only if `track_running_stats=False` |
| 5 | `fix_encoding.py:3` | Minor | References non-existent `lovo_eval.py` |
| 6 | `analyze_dataset.py` | Minor | Hardcoded constants diverge from `config.py` |

---

## Classes (9 surgical phases)

| ID | Phase | Typical Frequency |
|----|-------|-------------------|
| 0 | anesthesia | Low |
| 1 | closure | **High** (majority) |
| 2 | design | Low |
| 3 | disinfection | Very low |
| 4 | dissection | **High** (majority) |
| 5 | dressing | Low |
| 6 | hemostasis | Low |
| 7 | incision | Medium |
| 8 | irrigation | Medium |

The severe class imbalance (closure + dissection ≈ 80% of all frames) is the fundamental challenge this pipeline attempts to address through sqrt-frequency weighting, transition-aware sampling, and Viterbi decoding.
