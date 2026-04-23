"""
config.py — MS-TCN++ v7

ROOT CAUSE ANALYSIS (from v6 evaluation results):
  - val acc peaks at 84.58% but test acc is only 50.22%
  - Model predicts closure 79.2% of time (GT is 40.7%): majority class collapse
  - Zero recall on design (126 GT), dressing (196 GT), disinfection (9 GT) on test
  - tr_acc=0.999 by epoch 10: memorization complete before val even improves
  - Checkpoint selection used frame accuracy on videos 19/20; a model predicting only
    closure+dissection scores ~84% because those two classes ARE ~80% of frames.
    Every saved checkpoint was the most majority-biased one. This is the PRIMARY BUG.

v7 CHANGES:
  ┌───────────────────────────────────────────────────────────────────┐
  │ 1. CHECKPOINT CRITERION: frame accuracy → macro F1               │
  │    evaluate_validation() now returns macro-averaged F1 across    │
  │    all classes present in val GT. Majority-biased models that    │
  │    predict only closure+dissection score near-zero macro F1.     │
  │                                                                   │
  │ 2. WINDOW: 256 → 128                                             │
  │    Minority classes (design ~82 fr/vid, dressing ~21 fr/vid)     │
  │    are completely drowned in a 256-frame window. A 128-frame     │
  │    window centred on a minority frame has much higher label       │
  │    purity. MS-TCN with dilations [1,2,4,8,16,32] has RF=127     │
  │    frames per stage — WINDOW=128 is exactly the receptive field. │
  │                                                                   │
  │ 3. Pure-minority window sampler (Part 3 in build_sqrtfreq_windows)│
  │    For any class with < 600 frames, sample windows where ≥ 70%   │
  │    of frames belong to that class (unambiguous minority signal).  │
  │                                                                   │
  │ 4. Removed VideoNorm, MIXNORM_PROB, BN_MOMENTUM, EN_BETA,       │
  │    MIXUP_FRAC, RARE_CLASS_*, MIN_CLASS_WINDOWS, apply_mixup,    │
  │    temporal_cutmix, zero-recall rescue block.                     │
  │    Class weights restored to sqrt-frequency formula:             │
  │    w[c] = 1.0 / sqrt(max(n, 1)), capped at MAX_CLS_WEIGHT=4.0.  │
  │                                                                   │
  │ 5. Tighter regularization to slow memorization:                  │
  │    DROPOUT=0.5, WEIGHT_DECAY=5e-4, EPOCHS=120, PATIENCE=20,     │
  │    WARMUP_EPOCHS=8, TOTAL_WINDOWS_PER_EPOCH=500.                 │
  │                                                                   │
  │ 6. Per-sequence instance norm in load_data() + evaluate.py:      │
  │    After global norm, subtract per-sequence mean/std. Replaces   │
  │    VideoNorm with a deterministic preprocessing step identical    │
  │    at train and test time.                                        │
  │                                                                   │
  │ 7. FOCAL_GAMMA: 2.0 → 3.0                                        │
  │    Gamma=2 is insufficient with 9 classes and severe imbalance.  │
  │    Gamma=3 squares the focus factor, concentrating gradient on    │
  │    rare/hard frames more aggressively.                            │
  └───────────────────────────────────────────────────────────────────┘
"""

import os

# ===========================================================================
# PATHS
# ===========================================================================
IMAGE_ROOT       = "images"
GAZE_ROOT        = "gaze"
PHASE_ROOT       = "annotations/phase"
PROCESSED_ROOT   = "processed"
FEATURE_ROOT     = "features"
LABEL_ROOT       = "labels"
LABEL_NUM_ROOT   = "labels_num"
TRAIN_STATS_PATH = os.path.join(FEATURE_ROOT, "train_stats.npz")

MODEL_PATH       = "best_model.pth"

# ===========================================================================
# VIDEO SPLITS
# ===========================================================================
TRAIN_VIDEOS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "17", "18", "19", "20"
]
TEST_VIDEO   = "21"

# v7: only video 20 held out for validation (was ["19","20"]).
# Video 19 has 6 clips with diverse phases — too valuable to hold out.
VAL_VIDEOS  = ["20"]
TRAIN_CORE  = [v for v in TRAIN_VIDEOS if v not in set(VAL_VIDEOS)]   # 18 videos

# ===========================================================================
# FEATURE DIMENSIONS
# ===========================================================================
RESNET_DIM   = 2048
GAZE_DIM     = 4
INPUT_DIM    = RESNET_DIM + GAZE_DIM   # 2052
PROJ_DIM     = 128                     # 2052 -> 128 bottleneck projection

VIDEO_WIDTH  = 1920
VIDEO_HEIGHT = 1080

# ===========================================================================
# CLASS MAPPING — 9 real classes; unknown (id=9) dropped during training
# ===========================================================================
UNKNOWN_ORIG_ID = 9
NUM_CLASSES     = 9

LABEL2ID = {
    "anesthesia":   0,
    "closure":      1,
    "design":       2,
    "disinfection": 3,
    "dissection":   4,
    "dressing":     5,
    "hemostasis":   6,
    "incision":     7,
    "irrigation":   8,
}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]

# ===========================================================================
# MODEL
# ===========================================================================
HIDDEN_DIM  = 128
NUM_STAGES  = 2
DILATIONS   = [1, 2, 4, 8, 16, 32]   # RF per stage = 127 frames

# v7: DROPOUT raised 0.35 -> 0.5 to slow memorization
DROPOUT     = 0.5

# ===========================================================================
# TRAINING
# ===========================================================================
# v7: WINDOW reduced 256 -> 128 (matches MS-TCN receptive field; higher label
#     purity for minority classes centred in a 128-frame window)
WINDOW       = 128

EPOCHS       = 120        # reduced from 150; memorization is rapid
LR           = 2e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 5e-4       # raised from 1e-4 to tighten regularization
GRAD_CLIP    = 1.0

# ---- Loss ----
# v7: FOCAL_GAMMA raised 2.0 -> 3.0; gamma=2 insufficient for 9-class severe imbalance
FOCAL_GAMMA    = 3.0
# v7: sqrt-frequency formula: w[c] = 1/sqrt(n); capped at 4.0
MAX_CLS_WEIGHT = 4.0
SMOOTH_WEIGHT  = 0.08
LABEL_SMOOTH   = 0.05

# ---- Sampling ----
# v7: reduced 650 -> 500 (cleaner windows with WINDOW=128)
TOTAL_WINDOWS_PER_EPOCH = 500
TRANSITION_WIN_FRAC     = 0.25   # 125 transition + 375 sqrt-balanced windows

# ---- Feature augmentation ----
NOISE_STD    = 0.05
FEAT_DROP_P  = 0.12

# ---- Temporal speed augmentation ----
SPEED_AUG_PROB  = 0.30
SPEED_AUG_RANGE = (0.80, 1.25)

# ---- Post-processing ----
VITERBI_TRANS_SMOOTH = 1.0
SMOOTH_KERNEL        = 11
MIN_DURATION         = 25
MIN_SEG_FRAMES       = 15

# Viterbi blend: 0.15 keeps raw model signal dominant
VITERBI_STRENGTH     = 0.15

# ---- Validation + Early stopping ----
VAL_CHECK_EVERY       = 5
# v7: PATIENCE reduced 45 -> 20; stop faster when macro F1 plateaus
PATIENCE              = 20
EARLY_STOP_MIN_EPOCHS = 60

# ---- Legacy / backward-compat stubs (kept for evaluate.py imports) ----
VITERBI_BACKWARD_PENALTY = 0.0   # disabled — phases are non-monotonic
