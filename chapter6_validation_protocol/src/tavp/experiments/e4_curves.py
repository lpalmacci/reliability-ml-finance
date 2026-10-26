"""Descriptive curves for the figures of the chapter (not a level of the protocol and not a test).

For every generator of the case study and for the oracle, on independent scenarios drawn from a stream
of their own, this module records: the autocorrelation of returns and of absolute returns and the
leverage correlation at lags 1-20 (pooled estimators of level L1), the quantiles of daily returns and
of the horizon return, and a few sample paths. For the oracle, the 5% and 95% quantiles of each curve
over independent oracle samples of the same size give the band of sampling variability. For one context
of process D3 it also records the induced laws of the metrics of the calibrated rules, the object that
level L2 compares. The Quant GAN (E13) is included when its trained runs exist.

    python -m tavp.experiments.e4_curves --config configs/full.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.protocol.task_levels import _metrics_by_strategy, calibrate_all
from tavp.tasks.backtest import Objective
from tavp.tasks.strategies import build_strategies
from tavp.utils import rng

N_PATHS = 20_000
LAGS = 20
BAND_SAMPLES = 50
N_PLOT_PATHS = 30
QLEVELS = np.r_[0.001, 0.0025, 0.005, np.linspace(0.01, 0.99, 99), 0.995, 0.9975, 0.999]
INDUCED = {"dgp": "gjr_garch_t", "k": 0, "N": 100_000, "bins": 120}


def specs(cfg: dict) -> list[dict]:
    G = cfg["generators"]
    out = [{"kind": "oracle"}]
    out += [{"kind": "tcvae", "regime": r, "seed": s} for r in G["tcvae"]["regimes"] for s in range(G["tcvae"]["seeds"])]
    out += [{"kind": k, "seed": s} for k in ("bootstrap", "garch") for s in range(G[k]["seeds"])]
    out += qgan_specs(cfg)
    return out


def qgan_specs(cfg: dict) -> list[dict]:
    from tavp.experiments.e13_qgan import REGIMES, SEEDS, settings

    ok = all((common.run_dir(cfg, "qgan", common.dgps(cfg)[d].name, r, s) / "meta.json").exists()
             for d in cfg["dgps"] for r in REGIMES for s in range(SEEDS))
    return [sp for d, sp in settings(cfg) if d == next(iter(cfg["dgps"]))] if ok else []


def curves(r: np.ndarray) -> dict[str, np.ndarray]:
    rc = r - r.mean()
    a = np.abs(r)
    ac = a - a.mean()
    r2 = r**2
    r2c = r2 - r2.mean()
    out = {
        "acf_r": np.array([np.mean(rc[:, :-k] * rc[:, k:]) for k in range(1, LAGS + 1)]) / r.var(),
        "acf_abs": np.array([np.mean(ac[:, :-k] * ac[:, k:]) for k in range(1, LAGS + 1)]) / a.var(),
        "leverage": np.array([np.mean(rc[:, :-k] * r2c[:, k:]) for k in range(1, LAGS + 1)]) / (r.std() * r2.std()),
        "q_daily": np.quantile((r - r.mean()) / r.std(), QLEVELS),
        "q_horizon": np.quantile(r.sum(axis=1), QLEVELS),
    }
    return out


def _rows(meta: dict, c: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, v in c.items():
        xs = QLEVELS if name.startswith("q_") else np.arange(1, LAGS + 1)
        rows += [{**meta, "curve": name, "x": float(x), "value": float(y)} for x, y in zip(xs, v)]
    return rows


def task(cfg: dict, dgp_key: str, spec: dict) -> tuple[list[dict], list[dict]]:
    import torch

    torch.set_num_threads(1)
    T = cfg["data"]["scenario_prices"] - 1
    gen = runner.make_generator(cfg, dgp_key, {k: v for k, v in spec.items()})
    r = gen.sample_returns(N_PATHS, T, rng(cfg["master_seed"], "curves", dgp_key, runner.gen_id(spec)))
    meta = {"dgp": dgp_key, "kind": spec["kind"], "regime": spec.get("regime"), "seed": spec.get("seed")}
    paths = []
    if spec.get("seed") in (None, 0):
        lp = np.concatenate([np.zeros((N_PLOT_PATHS, 1)), np.cumsum(r[:N_PLOT_PATHS], axis=1)], axis=1)
        paths = [{**meta, "path": i, "t": t, "logprice": float(lp[i, t])} for i in range(N_PLOT_PATHS)
                 for t in range(T + 1)]
    return _rows(meta, curves(r)), paths


def band_task(cfg: dict, dgp_key: str, b: int) -> list[dict]:
    T = cfg["data"]["scenario_prices"] - 1
    dgp = common.dgps(cfg)[dgp_key]
    r = dgp.simulate_returns(N_PATHS, T, rng(cfg["master_seed"], "curves_band", dgp_key, b))
    return _rows({"dgp": dgp_key, "b": b}, curves(r))


def induced_task(cfg: dict, spec: dict) -> list[dict]:
    """Induced laws of the metrics of the calibrated rules for one context, by generator."""
    import torch

    torch.set_num_threads(1)
    d, k, N = INDUCED["dgp"], INDUCED["k"], INDUCED["N"]
    dgp = common.dgps(cfg)[d]
    T = cfg["data"]["scenario_prices"] - 1
    carry = cfg["data"]["carry"]
    strategies = build_strategies(cfg["use"]["strategies"])
    ctx = common.context(cfg, dgp, k)
    cal = calibrate_all(strategies, ctx, carry, Objective(**cfg["use"]["objective"]), cfg["use"]["cost"])
    params = {name: cal[name][1] for name in strategies}
    gen = runner.make_generator(cfg, d, spec)
    r = gen.sample_returns(N, T, rng(cfg["master_seed"], "curves_induced", d, runner.gen_id(spec), k))
    M = _metrics_by_strategy(strategies, params, ctx, r, carry, cfg["use"]["cost"])
    rows = []
    for name, ms in M.items():
        for metric, v in ms.items():
            rows.append({"kind": spec["kind"], "regime": spec.get("regime"), "seed": spec.get("seed"),
                         "strategy": name, "metric": metric, "values": np.asarray(v, dtype=np.float32)})
    return rows


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    jobs = common.n_jobs(cfg, args.jobs)
    sp = specs(cfg)
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(task)(cfg, d, s) for d in cfg["dgps"] for s in sp)
    common.save_table(pd.DataFrame([r for c, _ in res for r in c]), "e4_curves", "curves", cfg)
    common.save_table(pd.DataFrame([r for _, p in res for r in p]), "e4_curves", "paths", cfg)
    band = Parallel(n_jobs=jobs, verbose=5)(delayed(band_task)(cfg, d, b) for d in cfg["dgps"]
                                            for b in range(BAND_SAMPLES))
    common.save_table(pd.DataFrame([r for rows in band for r in rows]), "e4_curves", "band", cfg)
    ind_specs = [{"kind": "oracle"}, {"kind": "tcvae", "regime": "rich", "seed": 0}, {"kind": "bootstrap", "seed": 0},
                 {"kind": "garch", "seed": 0}] + [s for s in qgan_specs(cfg) if s["regime"] == "rich" and s["seed"] == 0]
    ind = Parallel(n_jobs=jobs, verbose=5)(delayed(induced_task)(cfg, s) for s in ind_specs)
    rows = [r for rows in ind for r in rows]
    # store the laws as quantiles (the figure needs densities, the file stays small)
    q = np.linspace(0.0005, 0.9995, 1999)
    out = [{k: v for k, v in r.items() if k != "values"} | {"q": float(a), "value": float(b)}
           for r in rows for a, b in zip(q, np.quantile(r["values"], q))]
    common.save_table(pd.DataFrame(out), "e4_curves", "induced", cfg)


if __name__ == "__main__":
    main()
