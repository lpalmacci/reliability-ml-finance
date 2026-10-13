"""Planted defects P1-P6 used to validate the protocol (known-answer tests).

P1, P5 and P6 act on simulated paths and apply to every process; P2, P3 and P4
change the parameters of the process (see `DGP.perturbed`). Severity delta = 0
always returns the unperturbed oracle.
"""

from __future__ import annotations

import numpy as np

from tavp.dgp.base import DGP

PATH_DEFECTS = ("P1", "P5", "P6")
PARAM_DEFECTS = ("P2", "P3", "P4")

# Which defects are defined for which process (Table P4 of the chapter).
APPLICABILITY = {
    "P1": ("gbm", "heston", "gjr_garch_t", "regime"),
    "P2": ("gjr_garch_t",),
    "P3": ("heston", "gjr_garch_t"),
    "P4": ("heston", "gjr_garch_t", "regime"),
    "P5": ("gbm", "heston", "gjr_garch_t", "regime"),
    "P6": ("gbm", "heston", "gjr_garch_t", "regime"),
}

VOL_SCALE_STEP = 0.20  # P5: volatility multiplied by 1 + 0.20 delta
DRIFT_STEP_ANNUAL = 0.05  # P6: annual drift shifted by 0.05 delta


def shuffle_fraction(returns: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    """P1: permute at random a fraction delta of the positions within each path.

    The marginal law of each return is preserved; serial dependence (volatility
    clustering, leverage, regimes) is destroyed on the permuted positions.
    """
    n, T = returns.shape
    k = int(round(delta * T))
    if k < 2:
        return returns.copy()
    positions = np.argsort(rng.random((n, T)), axis=1)[:, :k]
    order = np.argsort(rng.random((n, k)), axis=1)
    source = np.take_along_axis(positions, order, axis=1)
    out = returns.copy()
    np.put_along_axis(out, positions, np.take_along_axis(returns, source, axis=1), axis=1)
    return out


def scale_volatility(returns: np.ndarray, delta: float, drift_per_step: float) -> np.ndarray:
    """P5: r' = m + (1 + 0.2 delta)(r - m), with m the stationary drift per step."""
    return drift_per_step + (1.0 + VOL_SCALE_STEP * delta) * (returns - drift_per_step)


def shift_drift(returns: np.ndarray, delta: float, dt: float) -> np.ndarray:
    """P6: r' = r + 0.05 delta dt (annual drift error of 5 percentage points at delta = 1)."""
    return returns + DRIFT_STEP_ANNUAL * delta * dt


def apply_path_defect(kind: str, returns: np.ndarray, delta: float, dgp: DGP,
                      rng: np.random.Generator) -> np.ndarray:
    if delta == 0.0:
        return returns
    if kind == "P1":
        return shuffle_fraction(returns, delta, rng)
    if kind == "P5":
        return scale_volatility(returns, delta, dgp.drift_per_step())
    if kind == "P6":
        return shift_drift(returns, delta, dgp.dt)
    raise ValueError(f"{kind} is not a path defect")


def is_applicable(kind: str, dgp_name: str) -> bool:
    return dgp_name in APPLICABILITY[kind]
