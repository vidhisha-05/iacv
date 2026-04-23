"""
evaluate.py — v7: Final optimization pass

v7 CHANGES:
  1. Empirical initial state prior in Viterbi.
     v5 used uniform π = 1/C for all phases. Surgical videos start with
     predictable phases (e.g. anesthesia/design). The empirical prior from
     training corrects the first N frames, which propagates through the
     Viterbi chain improving overall alignment.

  2. Post-Viterbi minimum segment duration filter (MIN_SEG_FRAMES=15).
     Viterbi can still produce 1-3 frame "blip" predictions at uncertain
     transitions. These are nearly always wrong. A lightweight filter after
     Viterbi removes any segment shorter than MIN_SEG_FRAMES frames by
     absorbing it into the surrounding segment.

  3. Test-time augmentation (TTA, default n=3).
     Run inference 3× (1 clean + 2 mildly perturbed) and average softmax
     probabilities before Viterbi. This reduces variance from stochastic
     feature noise in the test sequence, giving Viterbi a smoother, more
     confident probability signal to work with.
     Cost: 3× inference time (still < 1 second for most sequences).
     Flag: --no-tta to disable.

Usage:
    python evaluate.py              # evaluates video 21
    python evaluate.py --video 21
    python evaluate.py --no-tta     # disable test-time augmentation
    python evaluate.py --no-viterbi # fallback mode filter
    python evaluate.py --video 06   # sanity-check on a training video

Requires:
    features/train_stats.npz         (from compute_train_stats.py)
    features/transition_matrix.npy   (from train.py)
    features/init_dist.npy           (from train.py  — v6)
    best_model.pth                   (from train.py)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    FEATURE_ROOT, LABEL_NUM_ROOT, TRAIN_STATS_PATH,
    MODEL_PATH, NUM_CLASSES, CLASS_NAMES, TEST_VIDEO,
    SMOOTH_KERNEL, MIN_DURATION, MIN_SEG_FRAMES, UNKNOWN_ORIG_ID,
    VITERBI_BACKWARD_PENALTY,   # kept in import (currently 0.0 — disabled)
    VITERBI_STRENGTH,           # CHANGE 5: soft-blend weight
)
from train import MS_TCN


# ════════════════════════════════════════════════════════════════════ #
# PATHS FOR v6 ARTEFACTS
# ════════════════════════════════════════════════════════════════════ #
TRANSITION_PATH = os.path.join(FEATURE_ROOT, "transition_matrix.npy")
INIT_DIST_PATH  = os.path.join(FEATURE_ROOT, "init_dist.npy")


# ════════════════════════════════════════════════════════════════════ #
# LOAD HELPERS
# ════════════════════════════════════════════════════════════════════ #

def load_transition_matrix() -> np.ndarray:
    """Load transition matrix saved by train.py."""
    if os.path.exists(TRANSITION_PATH):
        T = np.load(TRANSITION_PATH)
        if T.shape == (NUM_CLASSES, NUM_CLASSES):
            return T
    return None


def load_init_distribution() -> np.ndarray:
    """
    v6: Load empirical initial state distribution saved by train.py.
    Returns None if the file doesn't exist (old checkpoint) — Viterbi
    falls back to uniform prior in that case.
    """
    if os.path.exists(INIT_DIST_PATH):
        d = np.load(INIT_DIST_PATH)
        if d.shape == (NUM_CLASSES,):
            return d.astype(np.float64)
    return None


# ════════════════════════════════════════════════════════════════════ #
# v6: TEST-TIME AUGMENTATION
# ════════════════════════════════════════════════════════════════════ #

def inference_tta(model, feat: np.ndarray, dev, n_aug: int = 3,
                  noise_std: float = 0.008) -> np.ndarray:
    """
    v6: Test-time augmentation — average softmax probabilities over n_aug runs.

    Run 1 is always clean. Runs 2..n_aug add mild Gaussian noise (std=0.008,
    much smaller than training noise of 0.04) to the normalised features.

    WHY THIS HELPS:
      The test sequence has a single deterministic feature vector per frame.
      Small perturbations in the feature space cause the model to produce
      slightly different logit scores, which when averaged give a smoother
      probability distribution. This is particularly valuable for frames
      near class boundaries where the model is uncertain: the clean run
      and perturbed runs may disagree on the exact boundary frame, but their
      averaged probability is more calibrated, giving Viterbi a better signal.

    Returns:
      softmax_probs: (T, NUM_CLASSES) average probability array
    """
    all_probs = []

    for i in range(n_aug):
        feat_aug = feat.copy()
        if i > 0:   # run 0 is clean
            feat_aug += np.random.normal(0, noise_std, feat_aug.shape).astype(np.float32)

        t = torch.tensor(feat_aug, dtype=torch.float32).to(dev)
        t = t.permute(1, 0).unsqueeze(0)   # (1, D, T)

        with torch.no_grad():
            stage_outs = model(t)
            probs = (torch.softmax(stage_outs[-1].squeeze(0), dim=0)
                     .permute(1, 0)
                     .cpu()
                     .numpy())   # (T, C)

        all_probs.append(probs)

    return np.mean(all_probs, axis=0)   # (T, C)


# ════════════════════════════════════════════════════════════════════ #
# v6: VITERBI SMOOTHING (with empirical initial state prior)
# ════════════════════════════════════════════════════════════════════ #

def viterbi_smooth(logits_softmax: np.ndarray,
                   trans: np.ndarray,
                   init_dist: np.ndarray = None) -> np.ndarray:
    """
    Viterbi decoding with empirical initial state prior (v6).

    Args:
      logits_softmax : (T, C) float — model softmax probabilities
      trans          : (C, C) float — T[i,j] = P(phase j | phase i)
      init_dist      : (C,)   float — P(phase c at frame 0); uniform if None

    WHY INIT DIST MATTERS:
      v5 set log_pi = uniform (all classes equally likely at t=0).
      But surgical videos follow a strict ordering: they virtually always
      begin with high-level phases like anesthesia or design, never with
      dressing or closure. A wrong initial state in Viterbi means the first
      few frames are misclassified, and this error propagates through the
      entire sequence via the chain structure.

      Using the empirical distribution (e.g. P(anesthesia)=0.6, P(design)=0.3,
      P(others)<0.05) anchors the decoder to the correct start of the workflow.

    CHANGE 6 — Backward-jump penalty REMOVED:
      A previous iteration applied log(5.0) penalty to any i→j with j < i-1.
      Analysis of the confusion matrix showed this caused design recall to
      collapse from 11% → 0% because the surgical phase order in this dataset
      is NOT strictly monotonic (dressing can follow dissection non-linearly).
      The empirical transition matrix already encodes valid transitions; any
      hardcoded ordering constraint is therefore harmful and has been removed.

    Returns:
      pred: (T,) int — Viterbi-decoded labels
    """
    T, C = logits_softmax.shape

    log_emit  = np.log(np.clip(logits_softmax, 1e-10, 1.0))   # (T, C)
    log_trans = np.log(np.clip(trans,          1e-10, 1.0))   # (C, C)
    # CHANGE 6: NO backward-jump penalty — removed because surgical phases
    # in this dataset are not monotonically ordered and the penalty destroyed
    # design-class recall. The learned transition matrix encodes all constraints.

    # v6: use empirical initial distribution instead of uniform
    if init_dist is not None:
        log_pi = np.log(np.clip(init_dist, 1e-10, 1.0))
    else:
        log_pi = np.zeros(C) - np.log(C)   # uniform fallback

    dp  = np.full((T, C), -np.inf)
    ptr = np.zeros((T, C), dtype=np.int32)

    dp[0] = log_pi + log_emit[0]

    for t in range(1, T):
        trans_dp = dp[t - 1:t, :].T + log_trans   # (C, C): [prev, next]
        best_prev = trans_dp.max(axis=0)            # (C,)
        ptr[t]    = trans_dp.argmax(axis=0)
        dp[t]     = best_prev + log_emit[t]

    pred = np.zeros(T, dtype=np.int32)
    pred[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        pred[t] = ptr[t + 1, pred[t + 1]]

    return pred


# ════════════════════════════════════════════════════════════════════ #
# FALLBACK POST-PROCESSING (mode + min-duration filter)
# ════════════════════════════════════════════════════════════════════ #

def mode_filter(pred: np.ndarray, kernel: int = SMOOTH_KERNEL) -> np.ndarray:
    """Sliding-window mode filter on class predictions."""
    half   = kernel // 2
    padded = np.pad(pred, half, mode="edge")
    result = np.zeros_like(pred)
    for i in range(len(pred)):
        window    = padded[i:i + kernel].astype(np.int32)
        counts    = np.bincount(window, minlength=NUM_CLASSES)
        result[i] = counts.argmax()
    return result


def min_duration_filter(pred: np.ndarray, min_dur: int = MIN_DURATION) -> np.ndarray:
    """
    Remove segments shorter than min_dur frames by absorbing them into
    the adjacent segment.

    v6: applied AFTER Viterbi (not just in fallback) using MIN_SEG_FRAMES=15.
    Also still available as part of postprocess_fallback with MIN_DURATION=25.
    """
    smoothed = pred.copy()
    i = 0
    while i < len(smoothed):
        cls = smoothed[i]
        j   = i
        while j < len(smoothed) and smoothed[j] == cls:
            j += 1
        seg_len = j - i
        if seg_len < min_dur:
            fill = smoothed[i - 1] if i > 0 else (smoothed[j] if j < len(smoothed) else cls)
            smoothed[i:j] = fill
        i = max(j, i + 1)
    return smoothed


def postprocess_fallback(pred: np.ndarray) -> np.ndarray:
    """Fallback: mode filter then min-duration filter."""
    return min_duration_filter(mode_filter(pred))


# ════════════════════════════════════════════════════════════════════ #
# METRICS
# ════════════════════════════════════════════════════════════════════ #

def confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    C  = NUM_CLASSES
    cm = np.zeros((C, C), dtype=np.int64)
    for p, g in zip(pred, gt):
        if 0 <= int(g) < C and 0 <= int(p) < C:
            cm[int(g), int(p)] += 1
    return cm


def recall_per_class(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1).astype(float)
    return np.where(row_sum > 0, np.diag(cm) / row_sum, 0.0)


def compute_f1(cm: np.ndarray):
    """Return (per_class_f1, macro_f1, weighted_f1)."""
    precision = np.zeros(NUM_CLASSES)
    recall    = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp + 1e-8)
        recall[c]    = tp / (tp + fn + 1e-8)

    f1          = 2 * precision * recall / (precision + recall + 1e-8)
    support     = cm.sum(axis=1)
    macro_f1    = f1.mean()
    weighted_f1 = (f1 * support).sum() / (support.sum() + 1e-8)
    return f1, macro_f1, weighted_f1


def edit_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Normalised edit distance score ∈ [0, 100] between segment sequences."""
    def segments(seq):
        s, prev = [], seq[0]
        for v in seq[1:]:
            if v != prev:
                s.append(prev)
                prev = v
        s.append(prev)
        return s

    ps, gs = segments(pred), segments(gt)
    m, n   = len(ps), len(gs)
    dp     = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ps[i - 1] == gs[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return max(0.0, (1.0 - dp[m, n] / max(m, n)) * 100.0)


def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).mean())


def zero_recall_diagnostic(pred: np.ndarray, gt: np.ndarray):
    """Print class-level frequency table; flag zero-recall classes."""
    total = len(gt)

    valid_mask = (gt >= 0) & (gt < NUM_CLASSES)
    pred_v = pred[valid_mask]
    gt_v   = gt[valid_mask]

    pred_counts = np.bincount(pred_v, minlength=NUM_CLASSES)
    gt_counts   = np.bincount(gt_v,   minlength=NUM_CLASSES)

    print(f"\n  ── Zero-Recall Diagnostic ──")
    print(f"  {'Class':<16} {'GT':>7} {'GT%':>6}  {'Pred':>7} {'Pred%':>6}  Status")
    print(f"  {'-' * 62}")

    zero_recall_classes = []
    for c in range(NUM_CLASSES):
        gt_n   = int(gt_counts[c])
        pred_n = int(pred_counts[c])
        gt_pct   = 100.0 * gt_n   / max(total, 1)
        pred_pct = 100.0 * pred_n / max(total, 1)

        if gt_n == 0:
            status = "--  (not in video)"
        elif pred_n == 0:
            status = "[X] ZERO RECALL"
            zero_recall_classes.append(c)
        elif pred_n < gt_n * 0.1:
            status = "[!]  under-predicted"
        elif pred_n > gt_n * 3.0:
            status = "[!]  over-predicted"
        else:
            status = "OK"

        print(
            f"  [{c}] {CLASS_NAMES[c]:<13} "
            f"{gt_n:>7}  {gt_pct:5.1f}%  "
            f"{pred_n:>7}  {pred_pct:5.1f}%  "
            f"{status}"
        )

    if zero_recall_classes:
        print(f"\n  [X] {len(zero_recall_classes)} zero-recall class(es): "
              f"{[CLASS_NAMES[c] for c in zero_recall_classes]}")
    else:
        print(f"\n  [OK] No zero-recall classes!")


# ════════════════════════════════════════════════════════════════════ #
# REPORTING
# ════════════════════════════════════════════════════════════════════ #

def print_results_table(label: str, pred: np.ndarray, gt: np.ndarray):
    cm              = confusion_matrix(pred, gt)
    cls_recall      = recall_per_class(cm)
    f1, macro, wf1  = compute_f1(cm)
    acc             = accuracy(pred, gt)
    es              = edit_score(pred, gt)

    print(f"\n  ── {label} ──")
    print(f"  Full Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1      : {macro:.4f}")
    print(f"  Weighted F1   : {wf1:.4f}")
    print(f"  Edit Score    : {es:.2f} / 100")
    print()
    print(f"  {'Class':<16} {'Recall':>7}  {'F1':>7}  {'Support':>8}")
    print(f"  {'-' * 48}")
    for c in range(NUM_CLASSES):
        sup = int(cm[c, :].sum())
        if sup == 0:
            continue
        print(
            f"  [{c}] {CLASS_NAMES[c]:<13} "
            f"{cls_recall[c]:>6.4f}  "
            f"{f1[c]:>6.4f}  "
            f"{sup:>8}"
        )

    active_cls = [c for c in range(NUM_CLASSES) if cm[c, :].sum() > 0]
    worst = sorted(active_cls, key=lambda c: cls_recall[c])[:3]
    print(f"\n  [!]  Worst classes: " +
          ", ".join(f"[{c}]{CLASS_NAMES[c]}({cls_recall[c]:.3f})" for c in worst))
    return cm, acc, macro


def print_summary(raw_acc, pp_acc, raw_macro, pp_macro):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  ACCURACY SUMMARY")
    print(f"{SEP}")
    print(f"  {'Metric':<30} {'Raw':>8}  {'Post-Proc':>10}")
    print(f"  {'-' * 52}")
    print(f"  {'Full Accuracy':<30} {raw_acc*100:>7.2f}%  {pp_acc*100:>9.2f}%")
    print(f"  {'Macro F1':<30} {raw_macro:>8.4f}  {pp_macro:>10.4f}")
    print(f"\n  Target: 60-65%")
    if pp_acc >= 0.65:
        print(f"  [>] TARGET MET! Post-processed accuracy = {pp_acc*100:.2f}%")
    elif pp_acc >= 0.60:
        print(f"  [^] Good progress! {pp_acc*100:.2f}%  (gap to 65%: {(0.65-pp_acc)*100:.2f}%)")
    else:
        print(f"  [^] Gap to 60%: {(0.60 - pp_acc) * 100:.2f}% remaining")
    print(f"{SEP}\n")


# ════════════════════════════════════════════════════════════════════ #
# PLOTS
# ════════════════════════════════════════════════════════════════════ #

def plot_confusion_matrix(cm: np.ndarray, save_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(11, 9))
    im      = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title or "Confusion Matrix", fontsize=13)
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [graph] Confusion matrix → {save_path}")


def plot_timeline(pred_raw: np.ndarray, pred_pp: np.ndarray,
                  gt: np.ndarray, video_id: str, save_path: str):
    """Three-row timeline: Ground Truth / Raw / Post-processed."""
    cmap      = matplotlib.colormaps.get_cmap("tab10").resampled(NUM_CLASSES)
    fig, axes = plt.subplots(3, 1, figsize=(20, 5), sharex=True)

    for ax, seq, title in zip(
        axes,
        [gt, pred_raw, pred_pp],
        ["Ground Truth", "Raw Predictions", "Post-Processed (Viterbi+TTA v6)"]
    ):
        ax.imshow([seq], aspect="auto", cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=9)

    axes[-1].set_xlabel("Frame Index", fontsize=10)

    patches = [
        mpatches.Patch(color=cmap(i / max(NUM_CLASSES - 1, 1)),
                       label=f"[{i}] {CLASS_NAMES[i]}")
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Phase Timeline — Video {video_id} (v6)", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [graph] Timeline → {save_path}")


# ════════════════════════════════════════════════════════════════════ #
# MAIN
# ════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MS-TCN++ v7 surgical phase model"
    )
    parser.add_argument("--video",       default=TEST_VIDEO,
                        help=f"Video ID to evaluate (default: {TEST_VIDEO})")
    parser.add_argument("--no-viterbi",  action="store_true",
                        help="Use simple mode filter instead of Viterbi")
    parser.add_argument("--no-tta",      action="store_true",
                        help="Disable test-time augmentation (TTA)")
    parser.add_argument("--tta-runs",    type=int, default=3,
                        help="Number of TTA inference runs (default: 3)")
    parser.add_argument("--kernel",      type=int, default=SMOOTH_KERNEL,
                        help=f"Fallback mode filter kernel (default: {SMOOTH_KERNEL})")
    args = parser.parse_args()

    test_video = args.video
    use_tta    = not args.no_tta
    SEP = "=" * 68
    print(f"\n{SEP}\n  EVALUATION v7 — Video {test_video}\n{SEP}\n")

    # ── Normalisation stats ─────────────────────────────────────────
    if not os.path.exists(TRAIN_STATS_PATH):
        raise FileNotFoundError(
            f"Stats not found: {TRAIN_STATS_PATH}\n"
            "Run: python compute_train_stats.py"
        )
    stats       = np.load(TRAIN_STATS_PATH)
    global_mean = stats["mean"]
    global_std  = stats["std"]

    # ── v6: Transition matrix ───────────────────────────────────────
    trans = None
    if not args.no_viterbi:
        trans = load_transition_matrix()
        if trans is not None:
            print(f"  [OK] Viterbi enabled — transition matrix loaded")
        else:
            print(f"  [!]  Transition matrix not found — using mode filter")
            print(f"       Run train.py first to generate features/transition_matrix.npy")

    # ── v6: Initial state distribution ─────────────────────────────
    init_dist = load_init_distribution()
    if init_dist is not None:
        print(f"  [OK] Empirical initial state prior loaded (v6)")
    else:
        print(f"  [!]  init_dist.npy not found — using uniform Viterbi prior")
        print(f"       Run train.py v6 to generate features/init_dist.npy")

    # ── TTA info ───────────────────────────────────────────────────
    if use_tta:
        print(f"  [OK] TTA enabled ({args.tta_runs} inference runs per sequence)")
    else:
        print(f"  [--] TTA disabled")

    # ── Model ───────────────────────────────────────────────────────
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MS_TCN().to(dev)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=dev, weights_only=True)
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {MODEL_PATH}  |  {n_params:,} params  |  device={dev}\n")

    # ── Feature files for test video ────────────────────────────────
    feat_files = sorted(
        f for f in os.listdir(FEATURE_ROOT)
        if f.endswith(".npy") and f.split("_")[0] == test_video
    )
    if not feat_files:
        print(f"No feature files found for video {test_video}")
        return

    # ── Aggregate accumulators ──────────────────────────────────────
    cm_raw = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_pp  = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    all_pred_raw, all_pred_pp, all_gt = [], [], []

    for fname in feat_files:
        print(f"→ {fname}")

        feat = np.load(os.path.join(FEATURE_ROOT, fname)).astype(np.float32)
        feat = (feat - global_mean) / (global_std + 1e-8)
        # v7: per-sequence instance norm — identical to train-time preprocessing
        seq_mean = feat.mean(axis=0, keepdims=True)
        seq_std  = feat.std(axis=0, keepdims=True).clip(1e-5)
        feat     = (feat - seq_mean) / seq_std

        # ── Inference: TTA or single pass ──────────────────────────
        if use_tta and args.tta_runs > 1:
            softmax_probs = inference_tta(model, feat, dev, n_aug=args.tta_runs)
            pred_raw      = softmax_probs.argmax(axis=1).astype(np.int32)
        else:
            feat_t = torch.tensor(feat, dtype=torch.float32).to(dev)
            feat_t = feat_t.permute(1, 0).unsqueeze(0)
            with torch.no_grad():
                stage_outs    = model(feat_t)
                pred_raw      = stage_outs[-1].squeeze(0).argmax(dim=0).cpu().numpy()
                softmax_probs = (torch.softmax(stage_outs[-1].squeeze(0), dim=0)
                                 .permute(1, 0).cpu().numpy())

        # ── Load GT ────────────────────────────────────────────────
        label_path = os.path.join(LABEL_NUM_ROOT, fname.replace(".npy", ".txt"))
        if not os.path.exists(label_path):
            print(f"  [!]  No label file — skipping")
            continue

        with open(label_path) as f:
            gt_raw = np.array([int(x.strip()) for x in f if x.strip()])

        min_len       = min(len(pred_raw), len(gt_raw))
        pred_raw      = pred_raw[:min_len]
        gt_raw        = gt_raw[:min_len]
        softmax_probs = softmax_probs[:min_len]

        # ── Mask unknown GT frames ──────────────────────────────────
        valid_mask = gt_raw != UNKNOWN_ORIG_ID
        gt         = gt_raw[valid_mask]
        pred_raw_v = pred_raw[valid_mask]
        probs_v    = softmax_probs[valid_mask]

        if len(gt) == 0:
            print(f"  [!]  All frames are unknown — skipping")
            continue

        # ── Post-process ───────────────────────────────────────────
        if trans is not None:
            # CHANGE 5 (v6.2): soft-blended Viterbi.
            # Hard Viterbi destroyed design recall (11% → 0%) because it fully
            # committed to majority-class paths and had no way to recover
            # minority-class probabilities once they were marginalised out.
            # Soft blend: weighted average of Viterbi one-hot and raw probs.
            #   VITERBI_STRENGTH=0.4 → 40% Viterbi structure, 60% raw signal.
            # This retains temporal smoothness while preserving the raw model's
            # weak-but-nonzero predictions for design/dressing/irrigation.
            pred_viterbi = viterbi_smooth(probs_v, trans, init_dist=init_dist)
            pred_viterbi = min_duration_filter(pred_viterbi,
                                               min_dur=MIN_SEG_FRAMES)
            # Build one-hot encoding of Viterbi path → (T, C)
            viterbi_onehot = np.eye(NUM_CLASSES)[pred_viterbi]   # (T, C)
            # Blend: weighted average of structured Viterbi and raw probs
            blended_probs  = (VITERBI_STRENGTH * viterbi_onehot
                              + (1.0 - VITERBI_STRENGTH) * probs_v)
            pred_pp = blended_probs.argmax(axis=1).astype(np.int32)
        else:
            pred_pp = postprocess_fallback(pred_raw_v)

        # ── Safety clip — ensure no out-of-range predictions ───────────
        # Important: model output is NUM_CLASSES=9 logits (0-8).
        # Unknown class (id=9) is only in GT; model never trained on it.
        # Clip both raw and pp predictions to valid range [0, NUM_CLASSES-1].
        pred_raw_v = np.clip(pred_raw_v, 0, NUM_CLASSES - 1)
        pred_pp    = np.clip(pred_pp,    0, NUM_CLASSES - 1)

        from collections import Counter
        print(f"  GT dist  : {dict(Counter(gt.tolist()))}")
        print(f"  Raw dist : {dict(Counter(pred_raw_v.tolist()))}")
        print(f"  PP  dist : {dict(Counter(pred_pp.tolist()))}")
        print(
            f"  Raw acc={accuracy(pred_raw_v, gt):.4f}  "
            f"PP acc={accuracy(pred_pp, gt):.4f}  "
            f"ES_raw={edit_score(pred_raw_v, gt):.1f}  "
            f"ES_pp={edit_score(pred_pp, gt):.1f}"
        )

        cm_raw  += confusion_matrix(pred_raw_v, gt)
        cm_pp   += confusion_matrix(pred_pp,    gt)
        all_pred_raw.extend(pred_raw_v.tolist())
        all_pred_pp.extend(pred_pp.tolist())
        all_gt.extend(gt.tolist())

        # Timeline plot
        plot_timeline(pred_raw_v, pred_pp, gt, test_video,
                      fname.replace(".npy", "_timeline.png"))

    if not all_gt:
        print("No results to report.")
        return

    all_pred_raw = np.array(all_pred_raw)
    all_pred_pp  = np.array(all_pred_pp)
    all_gt       = np.array(all_gt)

    # ── Full aggregate report ───────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  AGGREGATE RESULTS — Video {test_video}  ({len(all_gt)} frames total)")
    print(f"{SEP}")

    zero_recall_diagnostic(all_pred_pp, all_gt)

    _, raw_acc, raw_macro = print_results_table("RAW MODEL OUTPUT", all_pred_raw, all_gt)
    _, pp_acc,  pp_macro  = print_results_table(
        "POST-PROCESSED (Viterbi + TTA v7)", all_pred_pp, all_gt
    )
    print_summary(raw_acc, pp_acc, raw_macro, pp_macro)

    # Confusion matrices
    plot_confusion_matrix(cm_raw, f"cm_raw_video{test_video}.png",
                          f"Confusion — Raw — Video {test_video}")
    plot_confusion_matrix(cm_pp,  f"cm_pp_video{test_video}.png",
                          f"Confusion — Viterbi+TTA v7 — Video {test_video}")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()