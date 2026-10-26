"""Functionals of induced metric laws, their standard errors and the equivalence (TOST) rule.

Standard errors are asymptotic and distribution-free:
- mean: sd / sqrt(n);
- quantile q_p: sqrt(p (1 - p) / n) / f(q_p), with 1 / f estimated by the
  spacing (q_{p+h} - q_{p-h}) / (2 h), h = min(n^{-1/3}, p / 2, (1 - p) / 2);
- expected shortfall (lower tail, sign flipped): sd((q_p - y) 1{y <= q_p}) / (p sqrt(n)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float


def mean_est(y: np.ndarray) -> Estimate:
    return Estimate(float(np.mean(y)), float(np.std(y, ddof=1) / np.sqrt(len(y))))


def quantile_est(y: np.ndarray, p: float) -> Estimate:
    n = len(y)
    h = min(n ** (-1.0 / 3.0), p / 2.0, (1.0 - p) / 2.0)
    q, lo, hi = np.quantile(y, [p, p - h, p + h])
    inv_density = (hi - lo) / (2.0 * h)
    return Estimate(float(q), float(inv_density * np.sqrt(p * (1.0 - p) / n)))


def var_est(y: np.ndarray, p: float = 0.05) -> Estimate:
    """Value-at-Risk of the lower tail at level 1 - p, reported as a positive loss."""
    q = quantile_est(y, p)
    return Estimate(-q.value, q.se)


def es_est(y: np.ndarray, p: float = 0.05) -> Estimate:
    """Expected shortfall of the lower tail at level 1 - p, reported as a positive loss."""
    q = np.quantile(y, p)
    tail = y <= q
    es = -float(np.mean(y[tail]))
    excess = (q - y) * tail
    return Estimate(es, float(np.std(excess, ddof=1) / (p * np.sqrt(len(y)))))


FUNCTIONALS = {
    "mean": mean_est,
    "q05": lambda y: quantile_est(y, 0.05),
    "q50": lambda y: quantile_est(y, 0.50),
    "q95": lambda y: quantile_est(y, 0.95),
    "var95": lambda y: var_est(y, 0.05),
    "es95": lambda y: es_est(y, 0.05),
}


@dataclass(frozen=True)
class Equivalence:
    diff: float
    lo: float
    hi: float
    eta: float
    equivalent: bool


def tost(est_g: Estimate, est_o: Estimate, eta: float, level: float = 0.05) -> Equivalence:
    """Two one-sided tests: equivalent if the (1 - 2 level) interval of the difference lies in [-eta, eta]."""
    z = norm.ppf(1.0 - level)
    d = est_g.value - est_o.value
    half = z * np.hypot(est_g.se, est_o.se)
    return Equivalence(d, d - half, d + half, eta, bool(d - half >= -eta and d + half <= eta))
