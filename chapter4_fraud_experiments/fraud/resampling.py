"""Rebalancing of the classes, used only by the perceptron at fit time.

``sklearn.MLPClassifier`` exposes neither ``class_weight`` nor ``sample_weight``, and at a low share of
frauds it collapses towards the majority class. The training sample is rebalanced to 50/50, and the
calibration downstream, fitted at the natural share of frauds, restores the probability scale.
"""
from __future__ import annotations

import numpy as np


def oversample_to_balanced(y: np.ndarray, *, seed: int) -> np.ndarray:
    """Indices that rebalance ``y`` to 50/50.

    The majority class is kept whole; the minority class is drawn **with replacement** until it equals
    it. The indices are then shuffled.
    """
    y = np.asarray(y).astype(int)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    if idx_0.size == 0 or idx_1.size == 0:
        raise ValueError("oversample_to_balanced needs both classes in y.")
    idx_major, idx_minor = (idx_0, idx_1) if idx_0.size >= idx_1.size else (idx_1, idx_0)
    rng = np.random.default_rng(seed)
    extra = rng.choice(idx_minor, size=idx_major.size, replace=True)
    out = np.concatenate([idx_major, extra])
    rng.shuffle(out)
    return out.astype(np.int64)
