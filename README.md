# IACV: MS-TCN++ v7 for Surgical Phase Recognition

This repository contains an implementation of MS-TCN++ (Multi-Stage Temporal Convolutional Network) for surgical workflow phase recognition, specifically tailored to address class imbalance and temporal smoothness.

## Model Architecture and Strategy

The project relies on extracting features (e.g., using a ResNet + Gaze model) and passing them through a multi-stage temporal convolutional network. 
Key strategies include:
- Sqrt-frequency inverse class weighting to handle severe imbalance.
- Transition-aware sampling to learn difficult class boundaries.
- Temporal speed augmentations.
- Test-time augmentation and Viterbi-based smoothing for decoding phase sequences.

## Known Critical Issues

During a recent diagnostic pass, two critical bugs were identified in this implementation:

1. **Test Accuracy Collapse (`GroupNorm` Misuse)**
   In `FeatureProjection` and `GatedTemporalBlock`, the model uses `nn.GroupNorm(1, C)` to approximate LayerNorm for 1D sequences. However, in PyTorch, `GroupNorm(1, C)` on shape `(B, C, T)` computes the mean and variance across both the channel and the temporal dimensions simultaneously (`C * T`). 
   - During training, normalization is computed over a `128`-frame window.
   - During inference, normalization is computed over the entire video (e.g., `~4000` frames).
   This causes a massive distribution shift, completely corrupting test-time representations and collapsing test accuracy despite high training accuracy.

2. **Prediction Pipeline Crash (`predict.py`)**
   In `config.py`, the number of network stages is defined as `NUM_STAGES = 2`. The model returns exactly two outputs (one for each stage). However, `predict.py` attempts to unpack three outputs:
   ```python
   _, _, out3 = model(t_feat)
   ```
   This will result in an immediate `ValueError: not enough values to unpack (expected 3, got 2)` during runtime.

## Usage

*Note: You must resolve the `GroupNorm` and tuple unpacking bugs before running the full pipeline.*

- **Train:** `python train.py`
- **Evaluate:** `python evaluate.py --video 21`
- **Predict:** `python predict.py`
