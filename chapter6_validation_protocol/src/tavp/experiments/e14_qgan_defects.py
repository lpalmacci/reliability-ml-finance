"""E14 (added after experiment E13, not pre-registered; deviation D11): the properties of the Quant GAN as
planted defects, as E9 does for the TC-VAE.

The stylized facts of E13 show two departures of the Quant GAN (G3) from the oracles. Its returns exceed an
extreme quantile of the oracle more often than the oracle does, and the variance of its return over the
horizon is smaller than the sum of the daily variances (variance ratio below one), the sign of a negative
dependence spread over several lags. E14 plants each property alone in the oracles, with the size observed in
G3 (median over its runs per process and regime), and records which levels and cells detect it:

- P8, extreme returns (`tavp.perturb.operators.extreme_returns`), at the pre-registered severities and at the
  severity whose frequency of returns beyond the 99.99% quantile of the absolute centred return of the oracle
  equals that of G3; no matched severity where G3 exceeds that quantile less often than the oracle;
- P7, negative first-order autocorrelation (E9), at the severity whose variance ratio over the horizon,
  1 + 2 rho (T - 1) / T, equals that of G3; none where the variance ratio of G3 is at least one.

Sizes as in E2 and E9 (K = 50 contexts, N = 40,000 scenarios). The facts of G3 are computed on the scenarios
of the stylized facts of E13 (same streams), so the variance ratios equal those reported there.

    python -m tavp.experiments.e14_qgan_defects --config configs/full.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.perturb.operators import ACF1_STEP, EXTREME_FACTOR, EXTREME_PROB_STEP
from tavp.utils import rng

N_PATHS = 20_000  # as the stylized facts (tavp.experiments.e4_facts)
QUANTILE = 0.9999
MAX_DELTA = 5.0


def _oracle_abs(cfg: dict, dgp_key: str) -> np.ndarray:
    dgp = common.dgps(cfg)[dgp_key]
    T = cfg["data"]["scenario_prices"] - 1
    r = dgp.simulate_returns(N_PATHS, T, rng(cfg["master_seed"], "facts", dgp_key, "oracle"))
    return np.abs(r - dgp.drift_per_step()).ravel()


def generator_facts(cfg: dict, dgp_key: str, spec: dict, threshold: float) -> dict:
    """Frequency of returns beyond the threshold, relative to 1 - QUANTILE, and variance ratio of one run of G3."""
    import torch

    torch.set_num_threads(1)
    dgp = common.dgps(cfg)[dgp_key]
    T = cfg["data"]["scenario_prices"] - 1
    gen = runner.make_generator(cfg, dgp_key, spec)
    r = gen.sample_returns(N_PATHS, T, rng(cfg["master_seed"], "facts", dgp_key, runner.gen_id(spec)))
    ok = bool(np.isfinite(r).all())
    exceed = float(np.mean(np.abs(r - dgp.drift_per_step()) > threshold)) / (1.0 - QUANTILE) if ok else np.inf
    vr = float(r.sum(axis=1).var() / (T * r.var())) if ok else np.nan
    return {"dgp": dgp_key, "regime": spec["regime"], "seed": spec["seed"], "exceed_ratio": exceed, "vr": vr}


def p8_rate(u: np.ndarray, threshold: float, delta: float) -> float:
    """Expected frequency of |r - m| > threshold under P8 of severity delta, from oracle draws u = |r - m|."""
    pi = EXTREME_PROB_STEP * delta
    c = np.sqrt(1.0 + pi * (EXTREME_FACTOR**2 - 1.0))
    return float((1.0 - pi) * np.mean(u / c > threshold) + pi * np.mean(EXTREME_FACTOR * u / c > threshold))


def p8_severity(u: np.ndarray, threshold: float, target_rate: float) -> float:
    """Severity of P8 whose frequency of returns beyond the threshold equals target_rate (bisection), or 0."""
    if target_rate <= p8_rate(u, threshold, 0.0):
        return 0.0
    lo, hi = 0.0, MAX_DELTA
    if p8_rate(u, threshold, hi) < target_rate:
        return MAX_DELTA
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if p8_rate(u, threshold, mid) < target_rate else (lo, mid)
    return 0.5 * (lo + hi)


def p7_severity(vr: float, T: int) -> float:
    """Severity of P7 whose variance ratio 1 + 2 rho (T - 1) / T equals vr, or 0 if vr >= 1."""
    if not np.isfinite(vr) or vr >= 1.0:
        return 0.0
    rho = (vr - 1.0) * T / (2.0 * (T - 1))
    return rho / ACF1_STEP


def matched(cfg: dict, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Facts of every run of G3 and the matched severities of P8 and P7 per process and regime."""
    from tavp.experiments import e13_qgan

    T = cfg["data"]["scenario_prices"] - 1
    u = {d: _oracle_abs(cfg, d) for d in cfg["dgps"]}
    thr = {d: float(np.quantile(u[d], QUANTILE)) for d in cfg["dgps"]}
    runs = pd.DataFrame(Parallel(n_jobs=jobs, verbose=5)(
        delayed(generator_facts)(cfg, d, sp, thr[d]) for d, sp in e13_qgan.settings(cfg)))
    rows = []
    for (d, regime), g in runs.groupby(["dgp", "regime"]):
        e, vr = float(g.exceed_ratio.median()), float(g.vr.median())
        rate = e * (1.0 - QUANTILE)
        rows.append({"dgp": d, "regime": regime, "threshold": thr[d], "exceed_ratio": e, "vr": vr,
                     "oracle_rate": p8_rate(u[d], thr[d], 0.0) / (1.0 - QUANTILE),
                     "delta_p8": round(p8_severity(u[d], thr[d], rate), 2),
                     "delta_p7": round(p7_severity(vr, T), 2)})
    return runs, pd.DataFrame(rows)


def settings(cfg: dict, m: pd.DataFrame | None) -> list[tuple[str, dict]]:
    grid = [float(d) for d in cfg["meta_validation"]["severities"] if d > 0]
    out = []
    for key in cfg["dgps"]:
        p8 = set(grid)
        p7 = set()
        if m is not None:
            mk = m[m.dgp == key]
            p8 |= {float(x) for x in mk.delta_p8 if x > 0}
            p7 |= {float(x) for x in mk.delta_p7 if x > 0}
        out += [(key, {"kind": "oracle", "defect": "P8", "delta": d}) for d in sorted(p8)]
        out += [(key, {"kind": "oracle", "defect": "P7", "delta": d}) for d in sorted(p7)]
    return out


def _args(p):
    p.add_argument("--stage", choices=["grid", "all"], default="all",
                   help="grid: P8 at the pre-registered severities only (needs no trained generator)")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    jobs = common.n_jobs(cfg, args.jobs)
    m = None
    if args.stage == "all":
        runs, m = matched(cfg, min(jobs, 4))
        common.save_table(runs, "e14", "qgan_facts", cfg)
        common.save_table(m, "e14", "matched", cfg)
        print(m.to_string())
    mv = cfg["meta_validation"]
    cfg["protocol"]["L2"]["K"], cfg["protocol"]["L2"]["N"] = mv["K"], mv["N"]  # sizes of E2
    runner.run(cfg, "e14", settings(cfg, m), jobs)


if __name__ == "__main__":
    main()
