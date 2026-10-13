"""Backtest of positions on log-price paths, scenario assembly and in-sample calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tavp.tasks.strategies import Strategy, rolling_vol
from tavp.utils import TRADING_DAYS


@dataclass(frozen=True)
class Objective:
    """Calibration objective J = Sharpe - turnover_weight * turnover - dd_weight * MaxDD."""

    turnover_weight: float = 0.01
    dd_weight: float = 1.0

    def __call__(self, m: dict[str, np.ndarray]) -> np.ndarray:
        return m["sharpe"] - self.turnover_weight * m["turnover"] - self.dd_weight * m["maxdd"]


def assemble(context_logp: np.ndarray, scen_returns: np.ndarray, carry: int) -> np.ndarray:
    """Log-price paths (n, carry + T): the last `carry` context log-prices followed by the scenario.

    Scenarios are anchored at the last observed log-price, so only their
    dynamics matter.
    """
    tail = context_logp[-carry:]
    n = scen_returns.shape[0]
    future = tail[-1] + np.cumsum(scen_returns, axis=1)
    return np.concatenate([np.broadcast_to(tail, (n, carry)), future], axis=1)


def evaluate(logp: np.ndarray, pos: np.ndarray, n_eval: int, cost: float = 0.0,
             periods: int = TRADING_DAYS) -> dict[str, np.ndarray]:
    """Metrics on the last `n_eval` returns of each path.

    pnl_t = pos_{t-1} r_t - cost |pos_{t-1} - pos_{t-2}|. Returns per path:
    annualised Sharpe ratio (0 if the P&L is constant), maximum drawdown of the
    equity curve started at 1, total log-return and annualised turnover.
    """
    r = np.diff(logp, axis=1)[:, -n_eval:]
    held = pos[:, -n_eval - 1:-1]
    prev = pos[:, -n_eval - 2:-2]
    trades = np.abs(held - prev)
    pnl = held * r - cost * trades
    mean = pnl.mean(axis=1)
    sd = pnl.std(axis=1, ddof=1)
    sharpe = np.where(sd > 1e-12, mean / np.where(sd > 1e-12, sd, 1.0), 0.0) * np.sqrt(periods)
    equity = np.exp(np.cumsum(pnl, axis=1))
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(pnl), 1)), equity], axis=1), axis=1)[:, 1:]
    maxdd = np.maximum(np.max(1.0 - equity / peak, axis=1), 0.0)
    return {
        "sharpe": sharpe,
        "maxdd": maxdd,
        "ret": pnl.sum(axis=1),
        "turnover": trades.sum(axis=1) * periods / n_eval,
    }


def vol_threshold(context_logp: np.ndarray, lb: int, q: float, carry: int) -> float:
    """TVF threshold: quantile q of sigma_hat over the in-sample part of the context."""
    vol = rolling_vol(context_logp[None, :], lb)[0, carry:]
    return float(np.quantile(vol[np.isfinite(vol)], q))


def resolve_params(strategy: Strategy, grid_point: dict, context_logp: np.ndarray, carry: int) -> dict:
    """Turn a grid point into rule parameters (TVF: replace q by the in-sample threshold)."""
    if not strategy.needs_threshold:
        return dict(grid_point)
    p = dict(grid_point)
    q = p.pop("q")
    p["threshold"] = vol_threshold(context_logp, p["lb"], q, carry)
    return p


def calibrate(strategy: Strategy, context_logp: np.ndarray, carry: int, objective: Objective,
              cost: float = 0.0) -> tuple[dict, dict]:
    """Choose the grid point that maximises J on the context (in-sample backtest after warm-up).

    Returns (grid_point, rule_parameters). Ties are broken by grid order.
    """
    if not strategy.grid:
        return {}, {}
    logp = context_logp[None, :]
    n_eval = len(context_logp) - 1 - carry
    best, best_val = None, -np.inf
    for g in strategy.grid:
        params = resolve_params(strategy, g, context_logp, carry)
        m = evaluate(logp, strategy.positions(logp, params), n_eval, cost)
        val = float(objective(m)[0])
        if val > best_val:
            best, best_val = g, val
    return best, resolve_params(strategy, best, context_logp, carry)


def scenario_metrics(strategy: Strategy, params: dict, context_logp: np.ndarray, scen_returns: np.ndarray,
                     carry: int, cost: float = 0.0) -> dict[str, np.ndarray]:
    """Metrics of a calibrated rule on N scenarios that continue the context."""
    logp = assemble(context_logp, scen_returns, carry)
    return evaluate(logp, strategy.positions(logp, params), scen_returns.shape[1], cost)
