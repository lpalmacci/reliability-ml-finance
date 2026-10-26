"""The two thresholds of the ensemble decision (Chapter 4, Ensembles, thresholds and errors).

- tau_F1: maximises F1 of the ensemble mean on the validation part, over 19 values between 0.05 and
  0.95 and 39 quantiles of pbar; it is selected again for every ensemble. pbar is the mean after clipping.
- tau_cost = 1/(r + 1), with r the ratio between the cost of a missed fraud and that of a false alarm:
  r = 9 on carclaims (tau = 0.1) and r = 100 on ULB.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

#: fixed grid of candidate thresholds
TAU_GRID = np.linspace(0.05, 0.95, 19)

#: declared cost ratio by dataset
COST_RATIO = {"carclaims": 9, "ulb": 100}


def tau_candidates(pb: np.ndarray, n_quantiles: int = 39, base: np.ndarray = TAU_GRID) -> np.ndarray:
    """Candidate thresholds: the fixed grid **plus** the quantiles of the ensemble mean."""
    q = np.quantile(pb, np.linspace(0.01, 0.99, n_quantiles))
    return np.unique(np.clip(np.concatenate([base, q]), 1e-6, 1.0 - 1e-6))


def select_tau(P: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Argmax-F1 of the **mean** of the ensemble (P already clipped, so that pbar = post.mu)."""
    pb = P.mean(axis=1)
    grid = tau_candidates(pb) if grid is None else np.asarray(grid)
    scores = [f1_score(y, (pb >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def cost_tau(dataset: str) -> float:
    return 1.0 / (COST_RATIO[dataset] + 1.0)


def implied_ratio(tau: float) -> float:
    """(1 - tau) / tau: the relative cost of a missed fraud that the threshold implies."""
    return (1.0 - tau) / tau
