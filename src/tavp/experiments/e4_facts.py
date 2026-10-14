"""Stylized facts of the scenarios of every generator of the case study (diagnostic of L1).

Level L1 stores discrepancies and their p-values; to explain the verdicts, this module
records the value of each statistic for the oracle and for every trained generator, on
independent scenarios, so that the chapter can show which properties are misreproduced.

    python -m tavp.experiments.e4_facts --config configs/full.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.protocol.l1 import stylized_facts
from tavp.utils import TRADING_DAYS, rng

N_PATHS = 20_000


def specs(cfg: dict) -> list[dict]:
    G = cfg["generators"]
    out = [{"kind": "oracle"}]
    out += [{"kind": "tcvae", "regime": r, "seed": s} for r in G["tcvae"]["regimes"] for s in range(G["tcvae"]["seeds"])]
    out += [{"kind": "vae_mlp", "regime": "single", "seed": s} for s in range(G["vae_mlp"]["seeds"])]
    out += [{"kind": k, "seed": s} for k in ("bootstrap", "garch") for s in range(G[k]["seeds"])]
    if "vae_mlp_wide" in G:  # exploratory configuration (deviation D2)
        out += [{"kind": "vae_mlp_wide", "regime": "single", "seed": s} for s in range(G["vae_mlp_wide"]["seeds"])]
    return out


def task(cfg: dict, dgp_key: str, spec: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    T = cfg["data"]["scenario_prices"] - 1
    gen = runner.make_generator(cfg, dgp_key, spec)
    r = gen.sample_returns(N_PATHS, T, rng(cfg["master_seed"], "facts", dgp_key, runner.gen_id(spec)))
    f = stylized_facts(r, cfg["protocol"]["L1"])
    row = {k: float(np.mean(v)) for k, v in f.items()}
    row["acf_r1"] = float(np.atleast_1d(f["acf_r"])[0])
    # variance ratio at the horizon: Var(sum of T returns) / (T Var(daily return)); 1 without autocorrelation
    row["vr"] = float(r.sum(axis=1).var() / (T * r.var()))
    row["mean"] *= TRADING_DAYS
    row["std"] *= np.sqrt(TRADING_DAYS)
    return {"dgp": dgp_key, "generator": runner.label(spec), "kind": spec["kind"], "regime": spec.get("regime"),
            "seed": spec.get("seed"), **row}


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    todo = [(d, s) for d in cfg["dgps"] for s in specs(cfg)]
    rows = Parallel(n_jobs=common.n_jobs(cfg, args.jobs), verbose=5)(delayed(task)(cfg, d, s) for d, s in todo)
    common.save_table(pd.DataFrame(rows), "e4", "facts", cfg)


if __name__ == "__main__":
    main()
