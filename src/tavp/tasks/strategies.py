"""Vectorised decision rules (the declared use of the case study).

All rules take an array of log-prices of shape (n, L) and return positions of
the same shape. The position at index t uses information up to t only and is
applied to the return from t to t+1 (no look-ahead; see tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from tavp.utils import TRADING_DAYS


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean along axis 1; NaN until the window is full."""
    n, L = x.shape
    out = np.full((n, L), np.nan)
    if window > L:
        return out
    c = np.cumsum(np.pad(x, ((0, 0), (1, 0))), axis=1)
    out[:, window - 1:] = (c[:, window:] - c[:, :-window]) / window
    return out


def rolling_vol(logp: np.ndarray, window: int) -> np.ndarray:
    """Annualised trailing standard deviation (ddof = 1) of the last `window` log-returns.

    The value at index t uses the returns ending at t, i.e. log-prices up to t.
    """
    n, L = logp.shape
    r = np.diff(logp, axis=1)  # r[:, t-1] = return from t-1 to t
    out = np.full((n, L), np.nan)
    if window < 2 or window > L - 1:
        return out
    c1 = np.cumsum(np.pad(r, ((0, 0), (1, 0))), axis=1)
    c2 = np.cumsum(np.pad(r**2, ((0, 0), (1, 0))), axis=1)
    s1 = c1[:, window:] - c1[:, :-window]
    s2 = c2[:, window:] - c2[:, :-window]
    var = np.maximum((s2 - s1**2 / window) / (window - 1), 0.0)
    out[:, window:] = np.sqrt(var * TRADING_DAYS)
    return out


def buy_and_hold(logp: np.ndarray) -> np.ndarray:
    return np.ones_like(logp)


def trend_following(logp: np.ndarray, fast: int, slow: int, dz: float) -> np.ndarray:
    """Sign of MA_fast - MA_slow of log-prices with a dead zone dz; positions in {-1, 0, 1}."""
    s = rolling_mean(logp, fast) - rolling_mean(logp, slow)
    pos = np.where(s > dz, 1.0, np.where(s < -dz, -1.0, 0.0))
    pos[np.isnan(s)] = 0.0
    return pos


def vol_target(logp: np.ndarray, lb: int, target: float, lmax: float) -> np.ndarray:
    """Long-only exposure clip(target / sigma_hat, 0, lmax)."""
    vol = rolling_vol(logp, lb)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.clip(target / vol, 0.0, lmax)
    pos[~np.isfinite(vol) | (vol < 1e-8)] = 0.0
    return pos


def trend_vol_filter(logp: np.ndarray, fast: int, slow: int, dz: float, lb: int, threshold: float) -> np.ndarray:
    """Trend-following switched off when sigma_hat exceeds a threshold fixed in-sample."""
    pos = trend_following(logp, fast, slow, dz)
    vol = rolling_vol(logp, lb)
    pos[~(vol <= threshold)] = 0.0
    return pos


@dataclass(frozen=True)
class Strategy:
    name: str
    rule: Callable[..., np.ndarray]
    grid: tuple[dict, ...]  # empty grid = no calibration
    # Parameters fixed in-sample from the context rather than chosen on the grid.
    needs_threshold: bool = False

    def positions(self, logp: np.ndarray, params: dict) -> np.ndarray:
        return self.rule(logp, **params)


def _grid(**axes) -> tuple[dict, ...]:
    keys = list(axes)
    return tuple(dict(zip(keys, values)) for values in product(*axes.values()))


def build_strategies(cfg: dict) -> dict[str, Strategy]:
    """Strategies and grids from the `strategies` section of the pre-registration."""
    tf = cfg["TF"]
    tf_grid = tuple(g for g in _grid(fast=tf["fast"], slow=tf["slow"], dz=tf["dz"]) if g["fast"] < g["slow"])
    vt = cfg["VT"]
    tvf = cfg["TVF"]
    tvf_grid = tuple(g for g in _grid(fast=tvf["fast"], slow=tvf["slow"], dz=tvf["dz"], lb=tvf["lb"], q=tvf["q"])
                     if g["fast"] < g["slow"])
    return {
        "BH": Strategy("BH", lambda logp: buy_and_hold(logp), ()),
        "TF": Strategy("TF", trend_following, tf_grid),
        "VT": Strategy("VT", vol_target, _grid(lb=vt["lb"], target=vt["target"], lmax=vt["lmax"])),
        "TVF": Strategy("TVF", trend_vol_filter, tvf_grid, needs_threshold=True),
    }


class IndicatorCache:
    """Rolling indicators of a set of paths, computed once per window and kept on the last `tail` columns.

    Used by L3, where every grid point is evaluated on the same scenarios: the moving
    averages and volatilities are shared across grid points.
    """

    def __init__(self, logp: np.ndarray, tail: int):
        self.logp, self.tail = logp, tail
        self._ma: dict[int, np.ndarray] = {}
        self._vol: dict[int, np.ndarray] = {}

    def ma(self, w: int) -> np.ndarray:
        if w not in self._ma:
            self._ma[w] = rolling_mean(self.logp, w)[:, -self.tail:]
        return self._ma[w]

    def vol(self, lb: int) -> np.ndarray:
        if lb not in self._vol:
            self._vol[lb] = rolling_vol(self.logp, lb)[:, -self.tail:]
        return self._vol[lb]


def cached_positions(name: str, params: dict, cache: IndicatorCache) -> np.ndarray:
    """Positions on the last `cache.tail` columns, identical to the rule applied to the full paths."""
    if name == "BH":
        return np.ones((cache.logp.shape[0], cache.tail))
    if name in ("TF", "TVF"):
        s = cache.ma(params["fast"]) - cache.ma(params["slow"])
        pos = np.where(s > params["dz"], 1.0, np.where(s < -params["dz"], -1.0, 0.0))
        pos[np.isnan(s)] = 0.0
        if name == "TVF":
            pos[~(cache.vol(params["lb"]) <= params["threshold"])] = 0.0
        return pos
    if name == "VT":
        vol = cache.vol(params["lb"])
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = np.clip(params["target"] / vol, 0.0, params["lmax"])
        pos[~np.isfinite(vol) | (vol < 1e-8)] = 0.0
        return pos
    raise ValueError(name)
