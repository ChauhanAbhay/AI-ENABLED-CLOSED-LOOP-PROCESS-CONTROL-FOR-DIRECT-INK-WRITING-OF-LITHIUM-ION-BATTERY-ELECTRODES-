"""
Sensor pre-processing pipeline  (Section 5.3, Figures 8-9).

Stages, in order:
    1. outlier removal   -> clip to +/-4 std of the training mean
    2. min-max scaling   -> per channel, scaler fitted on TRAIN data only
    3. sliding windows    -> 30 s look-back (300 samples) -> width 30 s ahead
    4. chronological split -> 80/20, NO shuffle (preserves temporal causality)
"""
from __future__ import annotations
import numpy as np

from .config import BILSTM


class MinMaxScaler:
    """Per-channel min-max scaler, fitted on training data only."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray):
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        # guard against zero-range channels
        self._range = np.where(
            (self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_) / self._range

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def clip_outliers(x: np.ndarray, n_std: float = 4.0,
                  stats: tuple | None = None):
    """Clip each channel to +/- n_std of the (training) mean. Returns clipped
    array and the (mean, std) used so the same stats apply to validation."""
    if stats is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    else:
        mean, std = stats
    lo, hi = mean - n_std * std, mean + n_std * std
    return np.clip(x, lo, hi), (mean, std)


def make_windows(features: np.ndarray, target: np.ndarray,
                 window: int, horizon: int, stride: int = 1):
    """
    Build sliding windows.

    features : (T, n_feat) scaled input channels
    target   : (T,) filament width (unscaled, microns)
    window   : look-back length in samples (300)
    horizon  : prediction horizon in samples (300 = 30 s ahead)
    stride   : step between consecutive window starts. The report uses
               step=1 (maximal overlap); a larger stride reduces redundancy
               and training cost while preserving temporal coverage.

    Returns X (N, window, n_feat), y (N,) where y is the width `horizon`
    samples after the end of each window.
    """
    T = len(features)
    starts = range(0, T - window - horizon + 1, stride)
    starts = list(starts)
    if len(starts) == 0:
        raise ValueError("Series too short for given window + horizon.")
    X = np.empty((len(starts), window, features.shape[1]), dtype=np.float32)
    y = np.empty(len(starts), dtype=np.float32)
    for j, i in enumerate(starts):
        X[j] = features[i: i + window]
        y[j] = target[i + window + horizon - 1]
    return X, y


def build_supervised(dataset: dict, cfg: BILSTM = BILSTM, stride: int = 10):
    """
    Full pipeline: dict from synthetic.make_dataset -> train/val tensors.

    Chronological 80/20 split is applied BEFORE windowing so no validation
    window ever overlaps training samples (prevents leakage). `stride` controls
    window density (step between window starts); the report's step=1 produces
    heavily overlapping windows, stride>1 trades a little coverage for speed.
    """
    feats = np.column_stack([
        dataset["pressure"],
        dataset["nozzle_temp"],
        dataset["humidity"],
    ]).astype(np.float32)
    width = dataset["width"].astype(np.float32)

    # chronological split point
    split = int(len(feats) * cfg.train_frac)
    f_tr, f_val = feats[:split], feats[split:]
    w_tr, w_val = width[:split], width[split:]

    # outlier clip — stats from train only
    f_tr, stats = clip_outliers(f_tr, n_std=4.0)
    f_val, _ = clip_outliers(f_val, n_std=4.0, stats=stats)

    # scale — fit on train only
    scaler = MinMaxScaler().fit(f_tr)
    f_tr_s = scaler.transform(f_tr)
    f_val_s = scaler.transform(f_val)

    X_tr, y_tr = make_windows(f_tr_s, w_tr, cfg.window_len, cfg.horizon_len, stride)
    X_val, y_val = make_windows(f_val_s, w_val, cfg.window_len, cfg.horizon_len, stride)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "scaler": scaler,
    }
