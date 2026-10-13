"""Levels L2 (evaluation fidelity) and L3 (optimisation fidelity) for one context.

L2: the rule calibrated on the context is run on N generated and N oracle scenarios; each
functional of each metric is compared by a TOST equivalence test with tolerance eta.
L3: every grid point is evaluated on generated scenarios and on two independent oracle
samples with common random numbers; the regret of choosing the rule on generated
scenarios is measured on the second oracle sample.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, norm

from tavp.tasks.backtest import Objective, assemble, calibrate, evaluate, resolve_params
from tavp.tasks.functionals import FUNCTIONALS, tost
from tavp.tasks.strategies import IndicatorCache, Strategy, cached_positions

METRICS = ("sharpe", "maxdd", "ret")


def calibrate_all(strategies: dict[str, Strategy], ctx_logp: np.ndarray, carry: int, objective: Objective,
                  cost: float) -> dict[str, tuple[dict, dict]]:
    return {name: calibrate(s, ctx_logp, carry, objective, cost) for name, s in strategies.items()}


CHUNK = 10_000  # scenarios processed at a time: bounds memory without changing any result


def _chunks(n: int):
    return [(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]


def _metrics_by_strategy(strategies, params: dict, ctx_logp, r, carry: int, cost: float) -> dict:
    """Metrics of every calibrated rule on the scenarios r, computed chunk by chunk."""
    T = r.shape[1]
    parts = {name: [] for name in strategies}
    for a, b in _chunks(len(r)):
        cache = IndicatorCache(assemble(ctx_logp, r[a:b], carry), T + 2)
        for name in strategies:
            parts[name].append(evaluate(cache.logp[:, -cache.tail:], cached_positions(name, params[name], cache), T, cost))
        del cache
    return {name: {m: np.concatenate([p[m] for p in ps]) for m in ps[0]} for name, ps in parts.items()}


def l2_context(strategies, calibrated, ctx_logp, rg, ro, use_cfg: dict, carry: int, level: float,
               resolvability_ratio: float) -> list[dict]:
    rows = []
    cost = use_cfg["cost"]
    params = {name: calibrated[name][1] for name in strategies}
    MG = _metrics_by_strategy(strategies, params, ctx_logp, rg, carry, cost)
    MO = _metrics_by_strategy(strategies, params, ctx_logp, ro, carry, cost)
    for name in strategies:
        mg, mo = MG[name], MO[name]
        for metric, funcs in use_cfg["functionals"].items():
            w1 = float(np.mean(np.abs(np.sort(mg[metric]) - np.sort(mo[metric])))) if len(mg[metric]) == len(mo[metric]) else np.nan
            for f in funcs:
                eg, eo = FUNCTIONALS[f](mg[metric]), FUNCTIONALS[f](mo[metric])
                eta = use_cfg["tolerances"][metric][f]
                t = tost(eg, eo, eta, level)
                rows.append({
                    "strategy": name, "metric": metric, "functional": f, "est_g": eg.value, "se_g": eg.se,
                    "est_o": eo.value, "se_o": eo.se, "diff": t.diff, "lo": t.lo, "hi": t.hi, "eta": eta,
                    "equivalent": t.equivalent, "half_width": (t.hi - t.lo) / 2.0,
                    "resolvable": (t.hi - t.lo) / 2.0 <= resolvability_ratio * eta, "w1": w1,
                    "grid_point": str(calibrated[name][0]),
                })
    return rows


def l3_context(strategies, ctx_logp, rg, ro1, ro2, use_cfg: dict, carry: int, level: float) -> list[dict]:
    objective = Objective(**use_cfg["objective"])
    cost = use_cfg["cost"]
    T = rg.shape[1]
    eta = use_cfg["tolerances"]["regret"]
    z = norm.ppf(1.0 - level)
    grids = {name: [resolve_params(s, g, ctx_logp, carry) for g in s.grid] for name, s in strategies.items() if s.grid}
    sum_g = {name: np.zeros(len(p)) for name, p in grids.items()}
    sum_1 = {name: np.zeros(len(p)) for name, p in grids.items()}
    j2 = {name: [[] for _ in p] for name, p in grids.items()}
    n = len(rg)
    for a, b in _chunks(n):
        caches = [IndicatorCache(assemble(ctx_logp, r[a:b], carry), T + 2) for r in (rg, ro1, ro2)]
        for name, plist in grids.items():
            for i, p in enumerate(plist):
                jv = [objective(evaluate(c.logp[:, -c.tail:], cached_positions(name, p, c), T, cost)) for c in caches]
                sum_g[name][i] += jv[0].sum()
                sum_1[name][i] += jv[1].sum()
                j2[name][i].append(jv[2])
        del caches
    rows = []
    for name, s in strategies.items():
        if not s.grid:
            continue
        jg, j1 = sum_g[name] / n, sum_1[name] / n
        ig, io = int(np.argmax(jg)), int(np.argmax(j1))
        d = np.concatenate(j2[name][io]) - np.concatenate(j2[name][ig])  # paired regret on the independent sample
        reg, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        spread = float(np.max(j1) - np.median(j1))
        tau = kendalltau(jg, j1).statistic
        rows.append({
            "strategy": name, "regret": reg, "se": se, "upper": reg + z * se, "eta": eta,
            "pass": bool(reg + z * se <= eta), "regret_norm": reg / spread if spread > 0 else np.nan,
            "kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
            "lambda_g": str(s.grid[ig]), "lambda_o": str(s.grid[io]), "same_choice": ig == io,
        })
    return rows
