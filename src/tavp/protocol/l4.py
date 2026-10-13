"""Level L4: calibration of induced laws without an oracle (here in pseudo-real mode).

A long path plays the role of observed history. At non-overlapping origins the rule is
calibrated on the preceding T0 returns, the generator produces the induced law of each
metric, and the PIT of the realised metric is recorded. Calibrated induced laws give
uniform PITs; the tests are Kolmogorov-Smirnov, Berkowitz and the Kupiec coverage test.
"""

from __future__ import annotations

import numpy as np

from tavp.experiments.common import to_logp
from tavp.protocol.stats import berkowitz, holm, ks_uniform, kupiec_pof, randomized_pit
from tavp.protocol.task_levels import METRICS, calibrate_all
from tavp.tasks.backtest import Objective, assemble, evaluate


def pits_on_history(generator, history: np.ndarray, start: int, strategies, cfg: dict,
                    rng: np.random.Generator) -> list[dict]:
    """PITs at origins start, start + T, ... of a history of returns (one row per origin, strategy, metric)."""
    T = cfg["data"]["scenario_prices"] - 1
    T0, carry = cfg["data"]["context_returns"], cfg["data"]["carry"]
    use = cfg["use"]
    objective = Objective(**use["objective"])
    n_origins, N = cfg["protocol"]["L4"]["n_origins"], cfg["protocol"]["L4"]["N"]
    rows = []
    for j in range(n_origins):
        t0 = start + j * T
        if t0 + T > len(history):
            break
        ctx = to_logp(history[t0 - T0:t0])
        calibrated = calibrate_all(strategies, ctx, carry, objective, use["cost"])
        realised = assemble(ctx, history[None, t0:t0 + T], carry)
        logp_g = assemble(ctx, generator.sample_returns(N, T, rng), carry)
        for name, s in strategies.items():
            p = calibrated[name][1]
            mr = evaluate(realised, s.positions(realised, p), T, use["cost"])
            mg = evaluate(logp_g, s.positions(logp_g, p), T, use["cost"])
            for m in METRICS:
                rows.append({"origin": j, "strategy": name, "metric": m,
                             "pit": randomized_pit(mg[m], float(mr[m][0]), rng)})
    return rows


def calibration_tests(df, cfg: dict) -> list[dict]:
    """Tests per (strategy, metric) on PITs pooled over histories; Holm over the whole family."""
    L4 = cfg["protocol"]["L4"]
    rows = []
    for (s, m), g in df.groupby(["strategy", "metric"]):
        seqs = [h.sort_values("origin")["pit"].to_numpy() for _, h in g.groupby("history")]
        u = np.concatenate(seqs)
        p_b3, p_b2 = berkowitz(seqs)
        rows.append({"strategy": s, "metric": m, "n": len(u), "p_ks": ks_uniform(u),
                     "p_kupiec": kupiec_pof(u, tuple(L4["interval"])), "p_berkowitz": p_b3,
                     "p_berkowitz_marginal": p_b2,
                     "coverage": float(np.mean((u >= L4["interval"][0]) & (u <= L4["interval"][1])))})
    pv = np.array([[r["p_ks"], r["p_kupiec"], r["p_berkowitz"]] for r in rows])
    rej = holm(pv.ravel(), L4["level"]).reshape(pv.shape)
    for r, rr in zip(rows, rej):
        r["reject_ks"], r["reject_kupiec"], r["reject_berkowitz"] = map(bool, rr)
        r["reject_any"] = bool(rr.any())
    return rows
