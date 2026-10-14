"""E11 (additional analysis on market data, not pre-registered; deviation D8): level L4 on the market.

The daily US market log-returns 1926-2026 are cut into windows of 20 years sliding by 10 years and
aligned to the end of the sample, so that each held-out decade is used once. In each window the
TC-VAE (three runs), the stationary bootstrap and the GJR-GARCH-t model are trained on the first 10
years, and level L4 is applied at the 42 non-overlapping origins of the last 10 years: the rules are
calibrated on the 500 preceding days of market data, the generator gives the induced law of each
metric, and the PIT of the realised metric is recorded. The PITs are pooled over windows and tested
as in E6, for each generator and each run of the TC-VAE separately: the three runs of a window share
the realised metrics, so their PITs are not independent and are not pooled.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common
from tavp.generators.base import GarchBaseline, StationaryBootstrap
from tavp.market import VINTAGE_SHA256, market_log_returns
from tavp.protocol.l4 import calibration_tests, pits_on_history
from tavp.tasks.strategies import build_strategies
from tavp.utils import REPO_ROOT, load_yaml, rng, stable_hash

TCVAE_SEEDS = 3


def windows(cfg: dict, max_windows: int | None = None) -> list[dict]:
    """Windows of 2 x train_returns days sliding by train_returns, aligned to the end of the sample."""
    d, r = market_log_returns()
    L = cfg["data"]["train_returns"]
    n = len(r) // L - 1
    out = []
    for w in range(n):  # w = 0 is the most recent window
        end = len(r) - w * L
        start = end - 2 * L
        out.append({"window": w, "start": int(start), "end": int(end), "train_first": int(d[start]),
                    "test_first": int(d[start + L]), "test_last": int(d[end - 1])})
    out = sorted(out, key=lambda x: x["start"])
    return out[-max_windows:] if max_windows else out


def target_hash(win: dict) -> str:
    return stable_hash({"market": VINTAGE_SHA256, "start": win["start"], "end": win["end"]})


def run_dir(cfg: dict, win: dict, seed: int, kind: str = "tcvae"):
    return common.results_root(cfg) / "runs_market" / kind / f"w{win['window']}" / f"seed{seed}"


def train_qgan(cfg: dict, win: dict, seed: int) -> str:
    """G3 (E13, addendum): Quant GAN trained on the first half of the window, on the Apple GPU when available."""
    from tavp.experiments.e13_qgan import qgan_cfg
    from tavp.generators.qgan import QuantGANGenerator

    out = run_dir(cfg, win, seed, "qgan")
    if (out / "meta.json").exists():
        return str(out)
    _, r = market_log_returns()
    train = r[win["start"]: win["start"] + cfg["data"]["train_returns"]]
    QuantGANGenerator(qgan_cfg(), target_hash(win), name="qgan").fit(
        train, seed=common.stream_seed(cfg, "train_market", "qgan", win["window"], seed), out_dir=out)
    return str(out)


def train_tcvae(cfg: dict, win: dict, seed: int) -> str:
    import json

    import torch

    from tavp.generators.tcvae import TCVAEGenerator, windows_from_history

    torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_threads", 2)))
    out = run_dir(cfg, win, seed)
    if (out / "meta.json").exists():
        return str(out)
    _, r = market_log_returns()
    train = r[win["start"]: win["start"] + cfg["data"]["train_returns"]]
    power_k = json.loads((REPO_ROOT / "results" / "train" / "scaling_decision.json").read_text())["power_k"]
    model_cfg = load_yaml(REPO_ROOT / "configs" / cfg["generators"]["tcvae"]["config"])
    TCVAEGenerator(model_cfg, target_hash(win), name="tcvae", power_k=power_k).fit(
        windows_from_history(train, cfg["data"]["scenario_prices"]),
        seed=common.stream_seed(cfg, "train_market", "tcvae", win["window"], seed), out_dir=out,
        epochs=cfg.get("runtime", {}).get("epochs"))
    return str(out)


def task(cfg: dict, win: dict, kind: str, seed: int) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    _, r = market_log_returns()
    h = r[win["start"]: win["end"]]
    train = h[: cfg["data"]["train_returns"]]
    if kind == "tcvae":
        from tavp.generators.tcvae import TCVAEGenerator
        gen = TCVAEGenerator.load(run_dir(cfg, win, seed))
    elif kind == "qgan":
        from tavp.generators.qgan import QuantGANGenerator
        gen = QuantGANGenerator.load(run_dir(cfg, win, seed, "qgan"))
    elif kind == "bootstrap":
        gen = StationaryBootstrap(train, target_hash(win))
    else:
        gen = GarchBaseline(train, target_hash(win), burn=cfg["generators"]["garch"]["burn"])
    hash_ok = gen.target_hash == target_hash(win)
    rows = pits_on_history(gen, h, cfg["data"]["train_returns"], build_strategies(cfg["use"]["strategies"]), cfg,
                           rng(cfg["master_seed"], "L4market", kind, seed, win["window"]))
    meta = {"generator": kind, "kind": kind, "seed": seed, "window": win["window"], "history": f"{win['window']}_{seed}",
            "test_first": win["test_first"], "test_last": win["test_last"], "hash_ok": hash_ok}
    return [{**meta, **x} for x in rows]


def tests_table(pits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calibration tests by generator and run, pooled over windows (window -1) and per window."""
    tests = []
    for (g, s), grp in pits.groupby(["generator", "seed"]):
        tests += [{"generator": g, "seed": int(s), "window": -1, **x} for x in calibration_tests(grp, cfg)]
        for w, gw in grp.groupby("window"):  # per held-out decade, for the description only
            tests += [{"generator": g, "seed": int(s), "window": int(w), **x} for x in calibration_tests(gw, cfg)]
    return pd.DataFrame(tests)


def garch_table(cfg: dict) -> pd.DataFrame:
    """Drift and volatility of benchmark B2 in each window, against the sample of its training and held-out halves."""
    import numpy as np

    _, r = market_log_returns()
    L, rows = cfg["data"]["train_returns"], []
    for w in windows(cfg):
        tr, te = r[w["start"]: w["start"] + L], r[w["start"] + L: w["end"]]
        g = GarchBaseline(tr, target_hash(w), burn=cfg["generators"]["garch"]["burn"])
        rows.append({"window": w["window"], "test_first": w["test_first"], "drift_garch": g.sim.drift_per_step() * 252,
                     "mean_train": float(tr.mean()) * 252, "mean_test": float(te.mean()) * 252,
                     "vol_garch": float(np.sqrt(g.sim.unconditional_variance * 252)),
                     "vol_train": float(tr.std()) * np.sqrt(252), "vol_test": float(te.std()) * np.sqrt(252)})
    return pd.DataFrame(rows)


def _args(p):
    p.add_argument("--max-windows", type=int, default=None, help="smoke runs: only the most recent windows")
    p.add_argument("--tests-only", action="store_true", help="recompute the tests from the stored PITs")
    p.add_argument("--qgan", action="store_true",
                   help="E13: add the Quant GAN (three runs per window) to the stored PITs and recompute the tests")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    if args.qgan:
        wins = windows(cfg)
        seeds = min(TCVAE_SEEDS, 3)
        for w in wins:  # training on the GPU, one window at a time
            Parallel(n_jobs=3)(delayed(train_qgan)(cfg, w, s) for s in range(seeds))
        res = Parallel(n_jobs=common.n_jobs(cfg, args.jobs), verbose=5)(
            delayed(task)(cfg, w, "qgan", s) for w in wins for s in range(seeds))
        path = common.results_root(cfg) / "e11" / "pits.parquet"
        old = pd.read_parquet(path)
        pits = pd.concat([old[old.generator != "qgan"], pd.DataFrame([x for rows in res for x in rows])],
                         ignore_index=True)
        common.save_table(pits, "e11", "pits", cfg)
        common.save_table(tests_table(pits, cfg), "e11", "tests", cfg)
        return
    if args.tests_only:
        pits = pd.read_parquet(common.results_root(cfg) / "e11" / "pits.parquet")
        common.save_table(tests_table(pits, cfg), "e11", "tests", cfg)
        common.save_table(garch_table(cfg), "e11", "garch", cfg)
        return
    jobs = common.n_jobs(cfg, args.jobs)
    wins = windows(cfg, args.max_windows)
    seeds = min(TCVAE_SEEDS, cfg["generators"]["tcvae"]["seeds"])
    Parallel(n_jobs=jobs, verbose=5)(delayed(train_tcvae)(cfg, w, s) for w in wins for s in range(seeds))
    todo = [(w, "tcvae", s) for w in wins for s in range(seeds)] + [(w, k, 0) for w in wins for k in ("bootstrap", "garch")]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(task)(cfg, *t) for t in todo)
    pits = pd.DataFrame([x for rows in res for x in rows])
    common.save_table(pits, "e11", "pits", cfg)
    common.save_table(pd.DataFrame(wins), "e11", "windows", cfg)
    common.save_table(tests_table(pits, cfg), "e11", "tests", cfg)
    common.save_table(garch_table(cfg), "e11", "garch", cfg)


if __name__ == "__main__":
    main()
