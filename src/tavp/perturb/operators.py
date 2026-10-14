"""Planted defects P1-P7 used to validate the protocol (known-answer tests).

P1, P5, P6 and P7 act on simulated paths and apply to every process; P2, P3 and P4
change the parameters of the process (see `DGP.perturbed`). Severity delta = 0
always returns the unperturbed oracle. P1-P6 are pre-registered; P7 (experiment E9,
deviation D6) was added after the case study to test the explanation of its failures.
"""

from __future__ import annotations

import numpy as np

from tavp.dgp.base import DGP

PATH_DEFECTS = ("P1", "P5", "P6", "P7")
PARAM_DEFECTS = ("P2", "P3", "P4")

# Which defects are defined for which process (Table P4 of the chapter).
APPLICABILITY = {
    "P1": ("gbm", "heston", "gjr_garch_t", "regime"),
    "P2": ("gjr_garch_t",),
    "P3": ("heston", "gjr_garch_t"),
    "P4": ("heston", "gjr_garch_t", "regime"),
    "P5": ("gbm", "heston", "gjr_garch_t", "regime"),
    "P6": ("gbm", "heston", "gjr_garch_t", "regime"),
    "P7": ("gbm", "heston", "gjr_garch_t", "regime"),
}

VOL_SCALE_STEP = 0.20  # P5: volatility multiplied by 1 + 0.20 delta
DRIFT_STEP_ANNUAL = 0.05  # P6: annual drift shifted by 0.05 delta
ACF1_STEP = -0.10  # P7: first-order autocorrelation of returns equal to -0.10 delta


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


def ma1_coefficient(rho: float) -> float:
    """Coefficient theta of an MA(1) whose first-order autocorrelation theta / (1 + theta^2) is rho."""
    if rho == 0.0:
        return 0.0
    if abs(rho) >= 0.5:
        raise ValueError("an MA(1) has first-order autocorrelation in (-0.5, 0.5)")
    return (1.0 - np.sqrt(1.0 - 4.0 * rho**2)) / (2.0 * rho)


def negative_autocorrelation(returns: np.ndarray, delta: float, drift_per_step: float,
                             rng: np.random.Generator) -> np.ndarray:
    """P7: r'_t = m + (u_t + theta u_{t-1}) / sqrt(1 + theta^2), u = r - m, autocorrelation -0.10 delta.

    The variance of each return is preserved and its first-order autocorrelation becomes
    -0.10 delta, so that the paths mean-revert and the variance of the return over the
    horizon shrinks by about 1 + 2 rho. u_{-1} is the last centred return of another path,
    drawn at random, so that the first return of each path has the same law as the others.
    """
    theta = ma1_coefficient(ACF1_STEP * delta)
    u = returns - drift_per_step
    prev = np.concatenate([u[rng.permutation(u.shape[0]), -1:], u[:, :-1]], axis=1)
    return drift_per_step + (u + theta * prev) / np.sqrt(1.0 + theta**2)


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
    if kind == "P7":
        return negative_autocorrelation(returns, delta, dgp.drift_per_step(), rng)
    raise ValueError(f"{kind} is not a path defect")


def is_applicable(kind: str, dgp_name: str) -> bool:
    return dgp_name in APPLICABILITY[kind]
