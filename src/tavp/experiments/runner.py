"""Run levels L0-L3 of the protocol for a set of (process, generator) settings.

A task is one (setting, context index k). Random streams are keyed so that oracle samples
are shared across settings (common random numbers) and generated samples are shared
across severities of the same defect, which makes power curves smooth in the severity.
"""

from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common
from tavp.generators.base import GarchBaseline, OracleGenerator, StationaryBootstrap
from tavp.protocol import l1
from tavp.protocol.task_levels import calibrate_all, l2_context, l3_context
from tavp.tasks.backtest import Objective
from tavp.tasks.strategies import build_strategies
from tavp.utils import rng, stable_hash


def gen_id(spec: dict) -> str:
    """Stream identifier of a generator; excludes the severity so that severities share draws."""
    parts = [spec["kind"]] + [str(spec[k]) for k in ("defect", "regime", "seed") if spec.get(k) is not None]
    return "_".join(parts)


def label(spec: dict) -> str:
    return gen_id(spec) + (f"_d{spec['delta']:g}" if spec.get("delta") is not None else "")


@lru_cache(maxsize=64)
def _generator(cfg_path: str, dgp_key: str, spec_json: str):
    cfg = common.load(cfg_path)
    spec = json.loads(spec_json)
    dgp = common.dgps(cfg)[dgp_key]
    kind = spec["kind"]
    if kind == "oracle":
        return OracleGenerator(dgp, spec.get("defect"), spec.get("delta", 0.0))
    if kind in ("tcvae", "vae_mlp", "vae_mlp_wide"):
        from tavp.generators.tcvae import TCVAEGenerator
        return TCVAEGenerator.load(common.run_dir(cfg, kind, dgp.name, spec["regime"], spec["seed"]))
    train = common.history(cfg, dgp, spec["seed"])[: cfg["data"]["train_returns"]]
    if kind == "bootstrap":
        return StationaryBootstrap(train, dgp.hash())
    if kind == "garch":
        return GarchBaseline(train, dgp.hash(), burn=cfg["generators"]["garch"]["burn"])
    raise ValueError(kind)


def make_generator(cfg: dict, dgp_key: str, spec: dict):
    return _generator(cfg["_config_path"], dgp_key, json.dumps(spec, sort_keys=True))


def load_l1_null(cfg: dict) -> dict:
    df = pd.read_parquet(common.results_root(cfg) / "e1" / "l1_null.parquet")
    return {d: {c: g[c].to_numpy() for c in g.columns if c not in ("dgp", "b")} for d, g in df.groupby("dgp")}


def l1_null_task(cfg: dict, dgp_key: str, b: int) -> dict:
    dgp = common.dgps(cfg)[dgp_key]
    c = cfg["protocol"]["L1"]
    T = cfg["data"]["scenario_prices"] - 1
    m = cfg["master_seed"]
    a = dgp.simulate_returns(c["n_paths"], T, rng(m, "L1null", dgp_key, b, "a"))
    bb = dgp.simulate_returns(c["n_paths"], T, rng(m, "L1null", dgp_key, b, "b"))
    return {"dgp": dgp_key, "b": b, **l1.discrepancies(a, bb, c)}


def task(cfg: dict, experiment: str, dgp_key: str, spec: dict, k: int, null: dict | None,
         parts: tuple[str, ...] = ("L1", "L2", "L3")) -> dict[str, list]:
    import torch

    torch.set_num_threads(1)
    m = cfg["master_seed"]
    dgp = common.dgps(cfg)[dgp_key]
    gen = make_generator(cfg, dgp_key, spec)
    gid = gen_id(spec)
    P = cfg["protocol"]
    T = cfg["data"]["scenario_prices"] - 1
    carry = cfg["data"]["carry"]
    meta = {"experiment": experiment, "dgp": dgp_key, "generator": label(spec), "kind": spec["kind"],
            "defect": spec.get("defect"), "delta": spec.get("delta", 0.0), "regime": spec.get("regime"),
            "seed": spec.get("seed"), "k": k}
    out = {"l0": [], "l1": [], "l2": [], "l3": []}

    hash_ok = gen.target_hash == dgp.hash()
    K1, K2, K3 = P["L2"]["K"], P["L2"]["K"], P["L3"]["K"]

    def valid(x):
        return bool(np.isfinite(x).all())

    if "L1" in parts and k < K1 and null is not None:
        rg = gen.sample_returns(P["L1"]["n_paths"], T, rng(m, "L1G", dgp_key, gid, k))
        ro = dgp.simulate_returns(P["L1"]["n_paths"], T, rng(m, "L1O", dgp_key, k))
        if valid(rg):
            res = l1.test_against_null(l1.discrepancies(rg, ro, P["L1"]), null[dgp_key], P["L1"]["level"])
            out["l1"] += [{**meta, "statistic": s, **v} for s, v in res.items()]
        out["l0"].append({**meta, "level": "L1", "hash_ok": hash_ok, "valid": valid(rg)})

    if "L2" in parts and k < K2:
        strategies = build_strategies(cfg["use"]["strategies"])
        ctx = common.context(cfg, dgp, k)
        calibrated = calibrate_all(strategies, ctx, carry, Objective(**cfg["use"]["objective"]), cfg["use"]["cost"])
        N = P["L2"]["N"]
        rg = gen.sample_returns(N, T, rng(m, "L2G", dgp_key, gid, k))
        ro = dgp.simulate_returns(N, T, rng(m, "L2O", dgp_key, k))
        ok = valid(rg)
        out["l0"].append({**meta, "level": "L2", "hash_ok": hash_ok, "valid": ok})
        if ok:
            rows = l2_context(strategies, calibrated, ctx, rg, ro, cfg["use"], carry, P["L2"]["tost_level"],
                              P["L0"]["resolvability_ratio"])
            out["l2"] += [{**meta, **r} for r in rows]
            if "L3" in parts and k < K3:
                ro2 = dgp.simulate_returns(P["L3"]["N"], T, rng(m, "L3O2", dgp_key, k))
                n3 = P["L3"]["N"]
                rows = l3_context(strategies, ctx, rg[:n3], ro[:n3], ro2, cfg["use"], carry, P["L3"]["level"])
                out["l3"] += [{**meta, **r} for r in rows]
    return out


def cached_task(cfg: dict, experiment: str, dgp_key: str, spec: dict, k: int, null: dict | None,
                parts: tuple[str, ...]) -> dict[str, list]:
    """Run a task, or return its stored result if it was already computed with the same settings.

    Results are stored per task under results/<experiment>/tasks/, so an interrupted
    experiment resumes where it stopped and produces exactly the same output.
    """
    science = {k_: v for k_, v in cfg.items() if k_ != "runtime" and not k_.startswith("_")}
    key = stable_hash({"cfg": science, "experiment": experiment, "dgp": dgp_key, "spec": spec, "k": k,
                       "parts": list(parts)})
    path = common.results_root(cfg) / experiment / "tasks" / dgp_key / f"{label(spec)}__k{k}__{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    out = task(cfg, experiment, dgp_key, spec, k, null, parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return out


def run(cfg: dict, experiment: str, settings: list[tuple[str, dict]], jobs: int,
        parts: tuple[str, ...] = ("L1", "L2", "L3")) -> dict:
    null = load_l1_null(cfg) if "L1" in parts else None
    K = max(cfg["protocol"]["L2"]["K"], cfg["protocol"]["L3"]["K"] if "L3" in parts else 0)
    tasks = [(d, s, k) for d, s in settings for k in range(K)]
    results = Parallel(n_jobs=jobs, verbose=5, max_nbytes=None)(
        delayed(cached_task)(cfg, experiment, d, s, k, null, parts) for d, s, k in tasks)
    tables = {}
    for level in ("l0", "l1", "l2", "l3"):
        rows = [r for res in results for r in res[level]]
        if rows:
            tables[level] = pd.DataFrame(rows)
            common.save_table(tables[level], experiment, level, cfg)
    return tables
