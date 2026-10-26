"""Readings of the members' probabilities as scores of doubt (higher = more doubtful), and budget metrics.

The readings (Chapter 4, The procedure and the simpler readings):

- "confidex": the signal, -u with the predictive law and the directional criterion; u = +1/2 (last in the
  order of doubt) for the degenerate cases;
- the four simpler readings of Chapter 3: the distance |pbar - tau|, the studentised margin
  |pbar - tau| / s, and the same two on the logit scale;
- "s", the standard deviation of the members, and "c mean", the posterior mean of the concentration;
- "confidex plug-in": the signal with the law at the posterior mean of c (robustness row).

``baselines`` is the one of Chapter 3 with one adaptation: where s = 0 the studentised margins are not
defined and the case goes last in the order of doubt (score -inf). "s = 0" is decided with the tie
criterion of the library (max - min < ``tie_eps`` on the clipped probabilities), because in floating
point the standard deviation of identical members is not exactly zero.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata

SIMPLE: tuple[str, ...] = ("distance", "studentised", "logit distance", "logit studentised")
READINGS: tuple[str, ...] = ("confidex", *SIMPLE, "s", "c mean", "confidex plug-in")


def auroc(score, y, decimals=10):
    """AUROC with which a score of doubt separates the errors (y = 1) from the correct decisions, with
    ties counted as one half after rounding the scores to ``decimals`` places (as in Chapter 3)."""
    y = np.asarray(y, dtype=int)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.round(np.asarray(score, dtype=float), decimals), method="average")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def baselines(P, pbar, tau, tie_eps=1e-9):
    tie = (P.max(axis=1) - P.min(axis=1)) < tie_eps
    s = P.std(axis=1, ddof=1)
    L = logit(P); sl = L.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"distance": -np.abs(pbar - tau),
               "studentised": -np.abs(pbar - tau) / s,
               "logit distance": -np.abs(logit(pbar) - logit(tau)),
               "logit studentised": -np.abs(L.mean(axis=1) - logit(tau)) / sl}
    out["studentised"] = np.where(tie, -np.inf, out["studentised"])
    out["logit studentised"] = np.where(tie, -np.inf, out["logit studentised"])
    return out


def order(doubt: np.ndarray) -> np.ndarray:
    """From the most doubtful case; ties keep the order of the rows (mergesort, as in Chapter 3)."""
    return np.argsort(-doubt, kind="mergesort")


def captured(order_: np.ndarray, err: np.ndarray, k: int) -> float:
    """Share of the errors among the first k cases of the order."""
    n_err = err.sum()
    return float(err[order_[:k]].sum() / n_err) if n_err else float("nan")
