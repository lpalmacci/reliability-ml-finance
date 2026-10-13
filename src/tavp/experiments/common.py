"""Shared infrastructure of the experiments: configuration, data streams, generators, I/O."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tavp import __version__
from tavp.dgp.processes import make_dgp
from tavp.utils import REPO_ROOT, int_seed, load_config, rng, stable_hash

RESULTS = REPO_ROOT / "results"


def parse_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/full.yaml")
    p.add_argument("--jobs", type=int, default=None, help="parallel workers (default from config)")
    if extra:
        extra(p)
    return p.parse_args()


def load(cfg_path: str) -> dict:
    cfg = load_config(REPO_ROOT / cfg_path)
    cfg["_config_path"] = cfg_path
    return cfg


def dgps(cfg: dict) -> dict:
    return {k: make_dgp(v) for k, v in cfg["dgps"].items()}


def to_logp(returns: np.ndarray) -> np.ndarray:
    """Log-prices starting at 0 from a vector of log-returns."""
    return np.concatenate([[0.0], np.cumsum(returns)])


def context(cfg: dict, dgp, k: int) -> np.ndarray:
    """Context k (calibration history) of a process: T0 returns, as log-prices. Shared by all experiments."""
    r = dgp.simulate_returns(1, cfg["data"]["context_returns"], rng(cfg["master_seed"], "context", dgp.name, k))[0]
    return to_logp(r)


def history(cfg: dict, dgp, s: int) -> np.ndarray:
    """Single long history s of a process (training part followed by the held-out part)."""
    return dgp.history(cfg["data"]["history_returns"], rng(cfg["master_seed"], "history", dgp.name, s))


def rich_paths(cfg: dict, dgp, s: int) -> np.ndarray:
    """Rich training set s: independent stationary paths of returns."""
    T = cfg["data"]["scenario_prices"] - 1
    return dgp.simulate_returns(cfg["data"]["rich_paths"], T, rng(cfg["master_seed"], "rich", dgp.name, s))


def results_root(cfg: dict) -> Path:
    """results/ for the full run, results/<suffix>/ for smoke runs, so they never mix."""
    suffix = cfg.get("runtime", {}).get("results_suffix")
    return RESULTS / suffix if suffix else RESULTS


def run_dir(cfg: dict, generator: str, dgp_name: str, regime: str, seed: int) -> Path:
    return results_root(cfg) / "runs" / generator / dgp_name / regime / f"seed{seed}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def provenance(cfg: dict) -> dict:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return {"package_version": __version__, "git_commit": git_commit(), "config": cfg["_config_path"],
            "config_hash": stable_hash(clean), "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_table(df: pd.DataFrame, experiment: str, name: str, cfg: dict) -> Path:
    out = results_root(cfg) / experiment
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    df.to_parquet(path, index=False)
    (out / f"{name}.provenance.json").write_text(json.dumps(provenance(cfg), indent=2))
    return path


def n_jobs(cfg: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(cfg.get("runtime", {}).get("jobs", max(1, (os.cpu_count() or 2) - 2)))


def stream_seed(cfg: dict, *keys) -> int:
    return int_seed(cfg["master_seed"], *keys)
