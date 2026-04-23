"""
train.py — MS-TCN++ v7

v7 KEY CHANGES:
  1. Checkpoint criterion: frame accuracy → macro F1 (PRIMARY BUG FIX)
  2. WINDOW: 256 → 128 (matches TCN receptive field; higher minority purity)
  3. Pure-minority window sampler: Part 3 in build_sqrtfreq_windows
  4. Removed VideoNorm, apply_mixup, temporal_cutmix, CutMix, rescue block
  5. Class weights: sqrt-frequency formula w[c]=1/sqrt(n), cap=4.0
  6. Per-sequence instance norm in load_data() (deterministic, train=test)
  7. WARMUP_EPOCHS=8, DROPOUT=0.5, WEIGHT_DECAY=5e-4, FOCAL_GAMMA=3.0
  8. best_val_acc renamed best_val_score throughout train()
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (
    FEATURE_ROOT, LABEL_NUM_ROOT, TRAIN_STATS_PATH,
    INPUT_DIM, PROJ_DIM, NUM_CLASSES, HIDDEN_DIM, DILATIONS, DROPOUT, NUM_STAGES,
    EPOCHS, WINDOW, LR, LR_MIN, WEIGHT_DECAY, GRAD_CLIP,
    SMOOTH_WEIGHT, LABEL_SMOOTH, FOCAL_GAMMA, MAX_CLS_WEIGHT,
    TOTAL_WINDOWS_PER_EPOCH, TRANSITION_WIN_FRAC,
    TRAIN_VIDEOS, TRAIN_CORE, VAL_VIDEOS,
    NOISE_STD, FEAT_DROP_P,
    SPEED_AUG_PROB, SPEED_AUG_RANGE,
    MODEL_PATH, CLASS_NAMES,
    UNKNOWN_ORIG_ID,
    VAL_CHECK_EVERY, PATIENCE, EARLY_STOP_MIN_EPOCHS,
)

# ════════════════════════════════════════════════════════════════════ #
# DEVICE
# ════════════════════════════════════════════════════════════════════ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ════════════════════════════════════════════════════════════════════ #
# MODEL: FeatureProjection + MS-TCN++
# ════════════════════════════════════════════════════════════════════ #



class FeatureProjection(nn.Module):
    """
    v6: Learned projection from raw 2052-dim input to compact 256-dim space.

    WHY THIS IS THE MOST IMPORTANT v6 CHANGE:
      The original first GatedTemporalBlock(2052→128) had 1.57M parameters
      just to map raw ResNet+gaze features into the hidden space. With only
      17 training videos (~25K frames), this layer could easily memorize
      patient-specific visual signatures rather than learning surgical
      phase structure.

      This projection forces a single global compression step that learns
      *which* of the 2052 raw dimensions carry surgical phase information.
      The downstream TCN then operates on 256 clean features instead of
      2052 noisy ones.

      Parameter count effect:
        First TCN block (2052→128): 2052*128*3*2 = 1,572,864 params
        Projection (2052→256):               2052*256 =   524,800 params
        New first TCN block (256→128):   256*128*3*2 =   196,608 params
        Net savings: ~851K params (54% reduction in first layer alone)

      Total model: ~3.1M → ~1.8M (42% fewer params, same architecture depth)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(INPUT_DIM, PROJ_DIM, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, PROJ_DIM)   # equivalent to LayerNorm
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, INPUT_DIM, T)  →  (1, PROJ_DIM, T)
        return self.act(self.norm(self.conv(x)))


class GatedTemporalBlock(nn.Module):
    """
    Gated Dilated Temporal Conv block.
    output = tanh(filter_conv(x)) × sigmoid(gate_conv(x)) + residual
    GroupNorm(1, C) ≡ LayerNorm for 1-D — stable at any sequence length.
    """
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.filter_conv = nn.Conv1d(in_ch, out_ch, 3,
                                     padding=dilation, dilation=dilation)
        self.gate_conv   = nn.Conv1d(in_ch, out_ch, 3,
                                     padding=dilation, dilation=dilation)
        self.norm        = nn.GroupNorm(1, out_ch)
        self.dropout     = nn.Dropout(DROPOUT)
        self.residual    = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        out = self.norm(out)
        out = self.dropout(out)
        res = x if self.residual is None else self.residual(x)
        return out + res


class TCNStage(nn.Module):
    """Stack of GatedTemporalBlocks + 1×1 output projection."""
    def __init__(self, in_ch: int, hid_ch: int, out_ch: int, dilations: list):
        super().__init__()
        layers, cur = [], in_ch
        for d in dilations:
            layers.append(GatedTemporalBlock(cur, hid_ch, d))
            cur = hid_ch
        layers.append(nn.Conv1d(hid_ch, out_ch, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MS_TCN(nn.Module):
    """
    v7: FeatureProjection → 2-stage MS-TCN++.
    VideoNorm removed; per-sequence instance norm is done in load_data()
    and evaluate.py before features reach the model.

    Architecture: INPUT_DIM(2052) → FeatureProjection(128) → Stage1 → Stage2
    Returns tuple (out1, out2) — each (1, NUM_CLASSES, T).
    """
    def __init__(self):
        super().__init__()
        self.proj   = FeatureProjection()
        self.stages = nn.ModuleList([
            TCNStage(PROJ_DIM, HIDDEN_DIM, NUM_CLASSES, DILATIONS),
        ])
        for _ in range(NUM_STAGES - 1):
            self.stages.append(
                TCNStage(NUM_CLASSES, HIDDEN_DIM, NUM_CLASSES, DILATIONS)
            )

    @staticmethod
    def _soft_input(logits):
        return torch.softmax(logits, dim=1).clamp(1e-4, 1.0 - 1e-4)

    def forward(self, x):
        h = self.proj(x)   # (1, INPUT_DIM, T) → (1, PROJ_DIM, T)
        outputs = []
        for i, stage in enumerate(self.stages):
            logits = stage(h)
            outputs.append(logits)
            if i < len(self.stages) - 1:
                h = self._soft_input(logits)
        return tuple(outputs)


# ════════════════════════════════════════════════════════════════════ #
# LOSS FUNCTION
# ════════════════════════════════════════════════════════════════════ #

class FocalLoss(nn.Module):
    """
    Focal Loss with per-class weights and label smoothing.
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
    Unchanged from v5 — the loss itself was not the bottleneck.
    """
    def __init__(self, class_weights: torch.Tensor, gamma: float = FOCAL_GAMMA,
                 smoothing: float = LABEL_SMOOTH):
        super().__init__()
        self.register_buffer('weight', class_weights)
        self.gamma     = gamma
        self.smoothing = smoothing
        self.C         = NUM_CLASSES

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        probs     = log_probs.exp()

        # Label smoothing
        with torch.no_grad():
            smooth_val = self.smoothing / (self.C - 1)
            smooth_tgt = torch.full_like(log_probs, smooth_val)
            smooth_tgt.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        ce_hard      = -(log_probs * smooth_tgt).sum(dim=1)
        p_t          = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        alpha        = self.weight[targets]

        return (alpha * focal_factor * ce_hard).mean()


def tmse_loss(logits: torch.Tensor, threshold: float = 4.0) -> torch.Tensor:
    """TMSE temporal smoothing loss. logits: (1, C, T)."""
    log_p = torch.log_softmax(logits, dim=1)
    diff  = log_p[:, :, 1:] - log_p[:, :, :-1]
    return torch.clamp(diff ** 2, max=threshold).mean()


# ════════════════════════════════════════════════════════════════════ #
# DATA LOADING
# ════════════════════════════════════════════════════════════════════ #

def load_global_stats():
    if not os.path.exists(TRAIN_STATS_PATH):
        raise FileNotFoundError(
            f"Stats not found: '{TRAIN_STATS_PATH}'\n"
            "Run: python compute_train_stats.py"
        )
    d = np.load(TRAIN_STATS_PATH)
    return d["mean"], d["std"]


def drop_unknown(labels: list) -> np.ndarray:
    """Return boolean mask of frames where label != UNKNOWN_ORIG_ID."""
    return np.array([l != UNKNOWN_ORIG_ID for l in labels])


def load_data(global_mean, global_std, video_list=None):
    """
    Load sequences for the given video list.
    Defaults to TRAIN_CORE (17 training videos).
    Pass video_list=VAL_VIDEOS to load validation data.

    CHANGE 8 (v6.3): sequence duplication removed. load_data() returns
    sequences exactly as loaded from disk. Domain invariance is handled
    by VideoNorm at the model level instead of data-level duplication.
    """
    if video_list is None:
        video_list = TRAIN_CORE

    X, Y = [], []
    total_kept = 0
    total_dropped = 0

    for fname in sorted(os.listdir(FEATURE_ROOT)):
        if not fname.endswith(".npy"):
            continue
        if fname.split("_")[0] not in video_list:
            continue

        feat  = np.load(os.path.join(FEATURE_ROOT, fname)).astype(np.float32)
        feat  = (feat - global_mean) / (global_std + 1e-8)
        # v7: per-sequence instance norm — removes video-specific mean/std offset.
        # Identical at train and test time (deterministic). Replaces VideoNorm.
        seq_mean = feat.mean(axis=0, keepdims=True)
        seq_std  = feat.std(axis=0, keepdims=True).clip(1e-5)
        feat     = (feat - seq_mean) / seq_std

        lpath = os.path.join(LABEL_NUM_ROOT, fname.replace(".npy", ".txt"))
        if not os.path.exists(lpath):
            print(f"  [!]  No label for {fname}")
            continue

        with open(lpath) as f:
            labels = [int(x.strip()) for x in f if x.strip()]

        n    = min(len(feat), len(labels))
        feat = feat[:n]
        lab  = labels[:n]

        mask      = drop_unknown(lab)
        feat_kept = feat[mask]
        lab_kept  = [l for l, m in zip(lab, mask) if m]

        total_dropped += n - mask.sum()
        total_kept    += mask.sum()

        if len(feat_kept) < 10:
            continue

        X.append(feat_kept)
        Y.append(lab_kept)

    print(f"\nLoaded {len(X)} sequences from {video_list}")
    print(f"  Frames kept    : {total_kept:,}")
    print(f"  Frames dropped : {total_dropped:,}  (unknown class)")
    return X, Y


# ════════════════════════════════════════════════════════════════════ #
# CLASS WEIGHTS
# ════════════════════════════════════════════════════════════════════ #

def compute_class_weights(Y: list) -> torch.Tensor:
    """
    v7: Sqrt-frequency inverse weights, capped at MAX_CLS_WEIGHT=4.0.
    Formula: w[c] = 1.0 / sqrt(max(n[c], 1)), then normalize by median.
    Sqrt (not linear) inverse-freq avoids extreme weight ratios while still
    boosting minority classes substantially over majority.
    """
    all_labels = [l for seq in Y for l in seq]
    counts     = Counter(all_labels)
    weights    = np.zeros(NUM_CLASSES, dtype=np.float32)

    for c in range(NUM_CLASSES):
        n = counts.get(c, 0)
        weights[c] = 1.0 / np.sqrt(max(n, 1))

    # Normalize by median of nonzero weights so majority class ~ 1.0
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / (np.median(nonzero) + 1e-8)

    weights = np.clip(weights, 0.0, MAX_CLS_WEIGHT)

    # Zero out truly absent classes
    for c in range(NUM_CLASSES):
        if counts.get(c, 0) == 0:
            weights[c] = 0.0

    print(f"\nClass weights (sqrt-freq, cap={MAX_CLS_WEIGHT}):")
    for c in range(NUM_CLASSES):
        n = counts.get(c, 0)
        print(f"  [{c}] {CLASS_NAMES[c]:<14} n={n:>7}  weight={weights[c]:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════════ #
# v6: TRANSITION-AWARE SAMPLING
# ════════════════════════════════════════════════════════════════════ #

def build_frame_level_pool(Y: list) -> dict:
    """Build frame-level index per class: {class_id: [(seq_idx, frame_idx)]}."""
    pool: dict = defaultdict(list)
    for si, labels in enumerate(Y):
        for fi, cls in enumerate(labels):
            pool[cls].append((si, fi))

    print("\nFrame pool sizes:")
    for c in range(NUM_CLASSES):
        if pool[c]:
            print(f"  [{c}] {CLASS_NAMES[c]:<14} : {len(pool[c]):>8} frames")
        else:
            print(f"  [{c}] {CLASS_NAMES[c]:<14} : NO FRAMES [!]")
    return pool


def find_transition_frames(Y: list) -> list:
    """
    v6: Find all (seq_idx, frame_idx) pairs where a class TRANSITION occurs.

    WHY THIS MATTERS:
      Class boundaries are the hardest frames to classify — the model needs
      to decide exactly when one phase ends and the next begins. With random
      sampling, a 256-frame window centred on a random frame in a 500-frame
      dissection segment will NEVER hit a transition.

      In a typical training video with 9 phases, there are only ~8-16
      transition points. Each transition point covers at most a 256-frame
      window. With random sampling, the probability of hitting a transition
      window is roughly 16*256 / 4000 ≈ 10%. With transition-targeted
      sampling, 100% of these windows hit boundaries.

      This directly trains the model on the confusion zone between adjacent
      phases — the primary source of test-time errors.
    """
    transitions = []
    for si, labels in enumerate(Y):
        for fi in range(1, len(labels)):
            if labels[fi] != labels[fi - 1]:
                transitions.append((si, fi))

    print(f"\nTransition frames found: {len(transitions)}")
    return transitions


# ════════════════════════════════════════════════════════════════════ #
# v6: TEMPORAL SPEED AUGMENTATION
# ════════════════════════════════════════════════════════════════════ #

def speed_augment(feat: np.ndarray, labels: list) -> tuple:
    """
    v6: Temporal resampling to simulate surgeon speed variation.

    With probability SPEED_AUG_PROB, resample the sequence at a random
    factor ∈ [0.8, 1.25]:
      factor > 1  →  sequence gets longer (slow surgeon, phases stretched)
      factor < 1  →  sequence gets shorter (fast surgeon, phases compressed)

    Features are linearly interpolated; labels use nearest-neighbor.
    The TCN handles variable-length sequences natively.

    WHY NOT TEMPORAL JITTER (rolling):
      Rolling shifts phase order and creates invalid wrap-arounds.
      Resampling preserves the phase order while changing density — this
      is the correct temporal augmentation for surgical workflow data.
    """
    if random.random() >= SPEED_AUG_PROB:
        return feat, labels

    T      = len(feat)
    factor = random.uniform(SPEED_AUG_RANGE[0], SPEED_AUG_RANGE[1])
    new_T  = max(20, int(T * factor))

    new_idx_f = np.linspace(0, T - 1, new_T)

    # Vectorised linear interpolation for features
    idx_lo = np.floor(new_idx_f).astype(np.int32).clip(0, T - 1)
    idx_hi = np.ceil(new_idx_f).astype(np.int32).clip(0, T - 1)
    alpha  = (new_idx_f - idx_lo)[:, np.newaxis]   # (new_T, 1)

    new_feat = ((1.0 - alpha) * feat[idx_lo] + alpha * feat[idx_hi]).astype(feat.dtype)

    # Nearest-neighbor for labels
    label_idx  = np.round(new_idx_f).astype(np.int32).clip(0, T - 1)
    new_labels = [labels[i] for i in label_idx]

    return new_feat, new_labels


# ════════════════════════════════════════════════════════════════════ #
# FEATURE AUGMENTATION
# ════════════════════════════════════════════════════════════════════ #

def augment_features(feat: np.ndarray) -> np.ndarray:
    """
    Mild feature-level augmentation (unchanged from v5).
    Does NOT touch temporal structure — speed_augment handles that separately.
    """
    feat = feat.copy()

    if random.random() < 0.5:
        feat += np.random.normal(0, NOISE_STD, feat.shape).astype(np.float32)

    if random.random() < 0.4:
        mask  = np.random.binomial(1, 1.0 - FEAT_DROP_P, feat.shape[1]).astype(np.float32)
        feat *= mask[np.newaxis, :]

    if random.random() < 0.3:
        scale = np.random.uniform(0.9, 1.1, feat.shape[1]).astype(np.float32)
        feat *= scale[np.newaxis, :]

    return feat


# ════════════════════════════════════════════════════════════════════ #
# WINDOW SAMPLING
# ════════════════════════════════════════════════════════════════════ #

def sample_window(feat, labels, center=None):
    """Extract WINDOW-length slice, optionally centred on a frame."""
    T = len(feat)
    if T <= WINDOW:
        return feat, labels
    if center is not None:
        start = max(0, min(center - WINDOW // 2, T - WINDOW))
    else:
        start = random.randint(0, T - WINDOW)
    return feat[start:start + WINDOW], labels[start:start + WINDOW]


def build_pure_minority_windows(X, Y, frame_pool, threshold=600,
                                purity=0.7, n_per_class=20):
    """
    v7 Part 3: For minority classes (< threshold frames), sample windows where
    >= purity fraction of frames belong to that class.

    Provides unambiguous minority-class signal: the model sees a window that is
    overwhelmingly one rare class, not just a few isolated frames surrounded by
    majority context. This is essential when dressing/design/disinfection have
    < 600 frames total in the training pool.
    """
    windows = []
    for cls in range(NUM_CLASSES):
        pool = frame_pool[cls]
        if len(pool) == 0 or len(pool) >= threshold:
            continue
        attempts  = 0
        collected = 0
        while collected < n_per_class and attempts < n_per_class * 10:
            attempts += 1
            si, fi = random.choice(pool)
            fw, lw = sample_window(X[si], Y[si], center=fi)
            lw_arr = lw if isinstance(lw, list) else list(lw)
            if len(lw_arr) == 0:
                continue
            purity_actual = lw_arr.count(cls) / len(lw_arr)
            if purity_actual >= purity:
                fw = augment_features(fw)
                fw, lw = speed_augment(fw, lw_arr)
                windows.append((fw, lw))
                collected += 1
    return windows


def build_sqrtfreq_windows(X: list, Y: list,
                           frame_pool: dict,
                           transitions: list,
                           augment: bool = True) -> list:
    """
    v7: sqrt-balanced + transition sampling + pure-minority windows.

    Split:
      Part 1: (1-TRANSITION_WIN_FRAC) x TOTAL = sqrt-balanced class windows
      Part 2: TRANSITION_WIN_FRAC x TOTAL      = transition-centred windows
      Part 3: pure-minority windows for classes with < 600 frames
    """
    n_base  = round(TOTAL_WINDOWS_PER_EPOCH * (1.0 - TRANSITION_WIN_FRAC))
    n_trans = TOTAL_WINDOWS_PER_EPOCH - n_base

    windows = []

    # ── Part 1: sqrt-balanced class windows ──────────────────────────
    counts = {c: len(frame_pool[c]) for c in range(NUM_CLASSES)}
    total_inv_sqrt = sum(1.0 / np.sqrt(max(counts[c], 1))
                         for c in range(NUM_CLASSES) if counts[c] > 0)

    for cls in range(NUM_CLASSES):
        pool = frame_pool[cls]
        if not pool:
            continue
        proportion = (1.0 / np.sqrt(len(pool))) / total_inv_sqrt
        n_windows  = max(1, round(proportion * n_base))

        chosen = random.choices(pool, k=n_windows)
        for (si, fi) in chosen:
            fw, lw = sample_window(X[si], Y[si], center=fi)
            if augment:
                fw = augment_features(fw)
                fw, lw = speed_augment(fw, lw)
            windows.append((fw, lw))

    # ── Part 2: transition-centred windows ───────────────────────────
    if transitions and n_trans > 0:
        chosen_t = random.choices(transitions, k=n_trans)
        for (si, fi) in chosen_t:
            fw, lw = sample_window(X[si], Y[si], center=fi)
            if augment:
                fw = augment_features(fw)
                fw, lw = speed_augment(fw, lw)
            windows.append((fw, lw))

    # ── Part 3: pure-minority windows (v7) ───────────────────────────
    if augment:
        minority_wins = build_pure_minority_windows(
            X, Y, frame_pool, threshold=600, purity=0.7, n_per_class=20
        )
        windows.extend(minority_wins)

    random.shuffle(windows)
    return windows


# ════════════════════════════════════════════════════════════════════ #
# LEARNING RATE — warmup + cosine decay
# ════════════════════════════════════════════════════════════════════ #

WARMUP_EPOCHS = 8   # v7: longer warmup slows early memorization

def get_lr(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return LR / 10.0 + (LR - LR / 10.0) * (epoch - 1) / max(WARMUP_EPOCHS - 1, 1)
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + np.cos(np.pi * progress))


# ════════════════════════════════════════════════════════════════════ #
# TRAINING EPOCH
# ════════════════════════════════════════════════════════════════════ #

def run_epoch(model, windows, criterion, optimizer, epoch_frac: float = 0.0):
    """
    Run one training epoch. Returns avg_loss, train_acc, pred_distribution.

    Adaptive TMSE smoothing weight ramps from SMOOTH_WEIGHT → 1.5×SMOOTH_WEIGHT
    linearly over training (epoch_frac 0→1).

    CHANGE 4 (v6.3): Before the forward pass for each window, call
    model.vidnorm.update_bank(f) to populate the VideoNorm style bank.
    Bank is capped to 200 entries inside update_bank() itself.
    """
    model.train()
    total_loss  = 0.0
    total_right = 0
    total_n     = 0
    pred_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)

    # Adaptive smoothing weight ramps up with epoch progress
    smooth_w = SMOOTH_WEIGHT * (1.0 + 0.5 * epoch_frac)

    for feat_np, labels_list in windows:
        f = torch.tensor(feat_np,     dtype=torch.float32).to(device)
        l = torch.tensor(labels_list, dtype=torch.long).to(device)
        f = f.permute(1, 0).unsqueeze(0)   # (1, D, T)

        stage_outs = model(f)

        cls_loss = sum(
            criterion(so.squeeze(0).permute(1, 0), l)
            for so in stage_outs
        )
        smo_loss = smooth_w * sum(tmse_loss(so) for so in stage_outs)
        loss     = cls_loss + smo_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            pred_flat = stage_outs[-1].squeeze(0).argmax(dim=0)
            total_right += (pred_flat == l).sum().item()
            total_n     += len(l)
            for c in pred_flat.cpu():
                pred_counts[c] += 1

    avg_loss  = total_loss / max(len(windows), 1)
    train_acc = total_right / max(total_n, 1)
    return avg_loss, train_acc, pred_counts


# ════════════════════════════════════════════════════════════════════ #
# v6: VALIDATION EVALUATION
# ════════════════════════════════════════════════════════════════════ #

def evaluate_validation(model, val_X: list, val_Y: list) -> float:
    """
    Run inference on full validation sequences. Returns macro-averaged F1
    across all classes present in val GT.

    v7 PRIMARY BUG FIX: Previous criterion was frame accuracy, which scores
    ~84% when predicting only closure+dissection (80% of val frames).
    Macro F1 scores near zero for that behaviour, so saved checkpoints
    will be those that actually recognise minority classes.
    """
    model.eval()
    all_pred = []
    all_gt   = []

    with torch.no_grad():
        for feat, labels in zip(val_X, val_Y):
            feat_t = torch.tensor(feat, dtype=torch.float32).to(device)
            feat_t = feat_t.permute(1, 0).unsqueeze(0)   # (1, D, T)
            stage_outs = model(feat_t)
            pred = stage_outs[-1].squeeze(0).argmax(dim=0).cpu().numpy()
            n = min(len(pred), len(labels))
            all_pred.extend(pred[:n].tolist())
            all_gt.extend(labels[:n])

    all_pred = np.array(all_pred)
    all_gt   = np.array(all_gt)

    # Macro F1 over classes present in GT
    f1s = []
    for c in range(NUM_CLASSES):
        gt_c   = (all_gt == c)
        pred_c = (all_pred == c)
        if gt_c.sum() == 0:
            continue
        tp = int((pred_c & gt_c).sum())
        fp = int((pred_c & ~gt_c).sum())
        fn = int((~pred_c & gt_c).sum())
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Also report frame accuracy for monitoring
    frame_acc = float((all_pred == all_gt).mean())
    print(f"    val frame_acc={frame_acc*100:.2f}%  val_macro_f1={macro_f1:.4f}")
    return macro_f1   # checkpoint on macro F1, not frame accuracy


# ════════════════════════════════════════════════════════════════════ #
# TRANSITION MATRIX (for Viterbi post-processing)
# ════════════════════════════════════════════════════════════════════ #

def learn_transition_matrix(Y: list) -> np.ndarray:
    """
    Empirical phase transition probabilities from training labels.
    T[i, j] = P(next phase is j | current phase is i).
    Saved to features/transition_matrix.npy.
    """
    from config import VITERBI_TRANS_SMOOTH
    C     = NUM_CLASSES
    trans = np.zeros((C, C), dtype=np.float64)

    for labels in Y:
        for t in range(len(labels) - 1):
            i, j = labels[t], labels[t + 1]
            if 0 <= i < C and 0 <= j < C:
                trans[i, j] += 1

    trans    += VITERBI_TRANS_SMOOTH
    row_sums  = trans.sum(axis=1, keepdims=True)
    trans     = trans / (row_sums + 1e-12)

    save_path = os.path.join(FEATURE_ROOT, "transition_matrix.npy")
    np.save(save_path, trans)
    print(f"\n  Transition matrix saved → {save_path}")
    return trans


# ════════════════════════════════════════════════════════════════════ #
# v6: INITIAL STATE DISTRIBUTION (for Viterbi)
# ════════════════════════════════════════════════════════════════════ #

def learn_init_distribution(Y: list) -> np.ndarray:
    """
    v6: Compute empirical initial state distribution from training sequences.

    The Viterbi decoder in v5 used a uniform prior π = 1/C for all classes.
    In practice, surgical videos almost always start with a small number of
    phases (e.g. anesthesia, design). Using the empirical distribution as the
    initial prior helps Viterbi assign correct labels to the first N frames,
    which propagates through the rest of the sequence.

    Laplace smoothing (+1) prevents zero-probability states.
    Saved to features/init_dist.npy.
    """
    init_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for labels in Y:
        if labels:
            c = labels[0]
            if 0 <= c < NUM_CLASSES:
                init_counts[c] += 1

    # Laplace smoothing
    init_dist = (init_counts + 1.0) / (init_counts.sum() + NUM_CLASSES)

    save_path = os.path.join(FEATURE_ROOT, "init_dist.npy")
    np.save(save_path, init_dist.astype(np.float32))
    print(f"  Initial state distribution saved → {save_path}")
    print(f"    {[(CLASS_NAMES[c], f'{init_dist[c]:.3f}') for c in np.argsort(init_dist)[::-1][:5]]}")
    return init_dist


# ════════════════════════════════════════════════════════════════════ #
# MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════ #

def train():
    SEP = "=" * 68
    print(f"\n{SEP}")
    print(f"  MS-TCN++ v7 - Final Optimization Pass")
    print(f"  ProjLayer | TransitionSampling | SpeedAug | ValCheckpoint")
    print(f"  TRAIN_CORE={TRAIN_CORE}  VAL={VAL_VIDEOS}")
    print(f"{SEP}")

    # ── Load stats ──────────────────────────────────────────────────
    global_mean, global_std = load_global_stats()

    # ── Load training data (TRAIN_CORE = 17 videos) ─────────────────
    print("\n--- Training data -------------------------------------------")
    X_train, Y_train = load_data(global_mean, global_std, video_list=TRAIN_CORE)

    # ── Load validation data (VAL_VIDEOS = 2 videos) ─────────────────
    print("\n--- Validation data -----------------------------------------")
    val_X, val_Y = load_data(global_mean, global_std, video_list=VAL_VIDEOS)
    use_val = bool(val_X)
    if use_val:
        val_frames = sum(len(y) for y in val_Y)
        print(f"  -> {len(val_X)} seqs, {val_frames} frames  (model selection)")
    else:
        print("  [!] No val data found - saving on training loss as fallback")

    # ── Class weights ────────────────────────────────────────────────
    class_weights = compute_class_weights(Y_train).to(device)
    criterion     = FocalLoss(class_weights, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH)

    # ── Transition matrix + init distribution ────────────────────────
    print("\nLearning transition matrix from training data...")
    learn_transition_matrix(Y_train)
    learn_init_distribution(Y_train)

    # ── Model ────────────────────────────────────────────────────────
    model    = MS_TCN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {NUM_STAGES}-stage MS-TCN++ v6 | {n_params:,} params")
    print(f"  PROJ_DIM={PROJ_DIM}  HIDDEN_DIM={HIDDEN_DIM}  "
          f"DROPOUT={DROPOUT}  DILATIONS={DILATIONS}")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Frame pool + transition frames ───────────────────────────────
    print("\nBuilding frame-level class pool...")
    frame_pool  = build_frame_level_pool(Y_train)
    transitions = find_transition_frames(Y_train)

    empty_cls = [c for c in range(NUM_CLASSES) if not frame_pool[c]]
    if empty_cls:
        print(f"  [!]  Classes with NO frames: "
              f"{[CLASS_NAMES[c] for c in empty_cls]}")

    print(f"\nTraining config:")
    print(f"  Epochs (max)   : {EPOCHS}  (early stop after {PATIENCE} eps no val improvement)")
    print(f"  Windows/epoch  : {TOTAL_WINDOWS_PER_EPOCH} "
          f"({int(TOTAL_WINDOWS_PER_EPOCH*(1-TRANSITION_WIN_FRAC))} sqrt-balanced + "
          f"{int(TOTAL_WINDOWS_PER_EPOCH*TRANSITION_WIN_FRAC)} transition)")
    print(f"  Speed aug      : p={SPEED_AUG_PROB}, factor in {SPEED_AUG_RANGE}")
    print(f"  Val check every: {VAL_CHECK_EVERY} epochs")
    print()

    best_val_score         = -1.0
    last_improvement_epoch = 0
    best_train_loss        = float("inf")   # fallback if no val data

    for epoch in range(1, EPOCHS + 1):
        # LR schedule
        current_lr = get_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # CHANGE 4: epoch_frac runs 0.0 (epoch 1) → 1.0 (epoch EPOCHS)
        epoch_frac = (epoch - 1) / max(EPOCHS - 1, 1)

        # Build windows fresh each epoch
        windows = build_sqrtfreq_windows(
            X_train, Y_train, frame_pool, transitions, augment=True
        )
        avg_loss, train_acc, pred_dist = run_epoch(
            model, windows, criterion, optimizer, epoch_frac=epoch_frac
        )

        # ── Console log ─────────────────────────────────────────────
        n_nonzero = int((pred_dist > 0).sum())
        print(
            f"Ep {epoch:>3}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  "
            f"tr_acc={train_acc:.3f}  "
            f"lr={current_lr:.2e}  "
            f"active={n_nonzero}/{NUM_CLASSES}"
        )

        # ── Validation check ─────────────────────────────────────────
        if epoch % VAL_CHECK_EVERY == 0:
            if use_val:
                val_score = evaluate_validation(model, val_X, val_Y)
                marker    = ""
                if val_score > best_val_score:
                    best_val_score         = val_score
                    last_improvement_epoch = epoch
                    torch.save(model.state_dict(), MODEL_PATH)
                    marker = "  << SAVED"
                print(
                    f"  [val] macro_f1={val_score:.4f}  "
                    f"best={best_val_score:.4f}"
                    f"{marker}"
                )
                # Early stopping
                if (epoch >= EARLY_STOP_MIN_EPOCHS and
                        (epoch - last_improvement_epoch) >= PATIENCE):
                    print(f"\n  [stop] No val improvement for {PATIENCE} "
                          f"epochs - stopping at epoch {epoch}")
                    break
            else:
                # Fallback: save on training loss
                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"  [OK] Saved (loss={best_train_loss:.4f})")

    print(f"\n[>] Training complete.")
    if use_val:
        print(f"   Best val macro_f1 : {best_val_score:.4f}  (epoch {last_improvement_epoch})")
    print(f"   Model             -> {MODEL_PATH}")
    print(f"\n[*] Run:  python evaluate.py --video 21")


if __name__ == "__main__":
    train()