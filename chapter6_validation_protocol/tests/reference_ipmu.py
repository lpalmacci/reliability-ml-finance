"""Reference implementations copied verbatim (comments translated) from the author's earlier
backtesting notebook, used only to test that the vectorised port reproduces them."""

import numpy as np


def log_returns(prices):
    prices = np.clip(np.asarray(prices, dtype=float), 1e-12, None)
    return np.log(prices[1:] / prices[:-1])


def rolling_mean(x, window):
    out = np.full_like(x, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1:] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def trend_following_ma(prices, fast=20, slow=60, threshold=0.0):
    T = len(prices)
    diff = rolling_mean(prices, fast) - rolling_mean(prices, slow)
    pos = np.zeros(T, dtype=float)
    pos[diff > threshold] = 1.0
    pos[diff < -threshold] = -1.0
    pos[np.isnan(diff)] = 0.0
    return pos


def vol_target_long_only(prices, lookback=20, target_vol=0.10, periods_per_year=12, max_leverage=2.0):
    T = len(prices)
    r = log_returns(prices)
    rv = np.full(T, np.nan, dtype=float)
    for t in range(lookback, T):
        rv[t] = np.std(r[t - lookback:t], ddof=1) * np.sqrt(periods_per_year)
    pos = np.zeros(T, dtype=float)
    for t in range(T):
        if not np.isfinite(rv[t]) or rv[t] < 1e-8:
            pos[t] = 0.0
        else:
            pos[t] = float(np.clip(target_vol / rv[t], 0.0, max_leverage))
    return pos


def backtest_positions(prices, pos, cost_bps=0.0):
    T = len(prices)
    r = log_returns(prices)
    pos_exec = pos[:-1].copy()
    turnover = np.abs(np.diff(pos, prepend=pos[0]))
    pnl_log = pos_exec * r - (cost_bps * 1e-4) * turnover[:-1]
    equity = np.ones(T, dtype=float)
    equity[1:] = np.exp(np.cumsum(pnl_log))
    return equity, pnl_log


def sharpe_and_maxdd(equity, pnl_log, periods_per_year=12):
    sharpe = (np.mean(pnl_log) / (np.std(pnl_log, ddof=1) + 1e-12)) * np.sqrt(periods_per_year)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(1.0 - equity / np.where(peak > 0, peak, 1.0)))
    return float(sharpe), max_dd
