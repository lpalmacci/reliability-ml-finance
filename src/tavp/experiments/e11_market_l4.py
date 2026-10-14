"""E11 (additional analysis on market data, not pre-registered; deviation D8): level L4 on the market.

The daily US market log-returns 1926-2026 are cut into windows of 20 years sliding by 10 years and
aligned to the end of the sample, so that each held-out decade is used once. In each window the
TC-VAE (three runs), the stationary bootstrap and the GJR-GARCH-t model are trained on the first 10
years, and level L4 is applied at the 42 non-overlapping origins of the last 10 years: the rules are
calibrated on the 500 preceding days of market data, the generator gives the induced law of each
metric, and the PIT of the realised metric is recorded. The PITs are pooled over windows (and runs)
and tested as in E6.
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


def run_dir(cfg: dict, win: dict, seed: int):
    return common.results_root(cfg) / "runs_market" / "tcvae" / f"w{win['window']}" / f"seed{seed}"


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


def _args(p):
    p.add_argument("--max-windows", type=int, default=None, help="smoke runs: only the most recent windows")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    jobs = common.n_jobs(cfg, args.jobs)
    wins = windows(cfg, args.max_windows)
    seeds = min(TCVAE_SEEDS, cfg["generators"]["tcvae"]["seeds"])
    Parallel(n_jobs=jobs, verbose=5)(delayed(train_tcvae)(cfg, w, s) for w in wins for s in range(seeds))
    todo = [(w, "tcvae", s) for w in wins for s in range(seeds)] + [(w, k, 0) for w in wins for k in ("bootstrap", "garch")]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(task)(cfg, *t) for t in todo)
    pits = pd.DataFrame([x for rows in res for x in rows])
    common.save_table(pits, "e11", "pits", cfg)
    common.save_table(pd.DataFrame(wins), "e11", "windows", cfg)
    tests = []
    for g, grp in pits.groupby("generator"):
        tests += [{"generator": g, "window": -1, **x} for x in calibration_tests(grp, cfg)]  # -1: pooled
        for w, gw in grp.groupby("window"):  # per held-out decade, for the description only
            tests += [{"generator": g, "window": int(w), **x} for x in calibration_tests(gw, cfg)]
    common.save_table(pd.DataFrame(tests), "e11", "tests", cfg)


if __name__ == "__main__":
    main()
