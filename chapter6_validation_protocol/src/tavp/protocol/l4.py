"""Level L4: calibration of induced laws without an oracle (here in pseudo-real mode).

A long path plays the role of observed history. At non-overlapping origins the rule is
calibrated on the preceding T0 returns, the generator produces the induced law of each
metric, and the PIT of the realised metric is recorded. Calibrated induced laws give
uniform PITs; the tests are Kolmogorov-Smirnov, Berkowitz and the Kupiec coverage test.
"""

from __future__ import annotations

import numpy as np

from tavp.experiments.common import to_logp
from tavp.protocol.stats import berkowitz, holm, ks_uniform, kupiec_pof, moment_test_hac, randomized_pit
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


DENSITY_TESTS = {"berkowitz_joint": "p_berkowitz", "berkowitz_marginal": "p_berkowitz_marginal",
                 "moments_hac": "p_moments_hac"}


def calibration_tests(df, cfg: dict) -> list[dict]:
    """Tests per (strategy, metric) on PITs pooled over histories; Holm over the whole family.

    The family is KS, Kupiec and one density test chosen by `protocol.L4.density_test`:
    "berkowitz_joint" (pre-registered: mean, variance and AR(1) dependence of the normal
    scores), "berkowitz_marginal" (mean and variance) or "moments_hac" (raw moments with a
    long-run covariance). In mode S the PITs of consecutive origins are dependent even under
    the true law, and only the last test keeps its size (deviation D4 of docs/DEVIATIONS.md).
    The decisions of every variant are reported, with the suffix of the variant.
    """
    L4 = cfg["protocol"]["L4"]
    rows = []
    for (s, m), g in df.groupby(["strategy", "metric"]):
        seqs = [h.sort_values("origin")["pit"].to_numpy() for _, h in g.groupby("history")]
        u = np.concatenate(seqs)
        p_b3, p_b2 = berkowitz(seqs)
        rows.append({"strategy": s, "metric": m, "n": len(u), "p_ks": ks_uniform(u),
                     "p_kupiec": kupiec_pof(u, tuple(L4["interval"])), "p_berkowitz": p_b3,
                     "p_berkowitz_marginal": p_b2, "p_moments_hac": moment_test_hac(seqs),
                     "coverage": float(np.mean((u >= L4["interval"][0]) & (u <= L4["interval"][1])))})
    for variant, col in DENSITY_TESTS.items():
        pv = np.array([[r["p_ks"], r["p_kupiec"], r[col]] for r in rows])
        rej = holm(pv.ravel(), L4["level"]).reshape(pv.shape)
        for r, rr in zip(rows, rej):
            r[f"reject_ks_{variant}"], r[f"reject_kupiec_{variant}"], r[f"reject_density_{variant}"] = map(bool, rr)
            r[f"reject_any_{variant}"] = bool(rr.any())
    applied = L4.get("density_test", "berkowitz_joint")
    for r in rows:
        r["density_test"] = applied
        for k in ("reject_ks", "reject_kupiec", "reject_density", "reject_any"):
            r[k] = r[f"{k}_{applied}"]
    return rows
