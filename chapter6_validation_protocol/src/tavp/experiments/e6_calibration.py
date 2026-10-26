"""E6: level L4 (calibration without oracle) in pseudo-real mode.

Part `meta` validates L4 itself: the oracle (which must pass) and defective oracles (which
should fail) on 20 independent histories per process. Part `case` applies L4 to the trained
generators and baselines, each on the held-out half of its own training history.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.protocol.l4 import calibration_tests, pits_on_history
from tavp.tasks.strategies import build_strategies
from tavp.utils import rng


def task(cfg: dict, dgp_key: str, spec: dict, s: int) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    dgp = common.dgps(cfg)[dgp_key]
    gen = runner.make_generator(cfg, dgp_key, spec)
    h = common.history(cfg, dgp, s)
    strategies = build_strategies(cfg["use"]["strategies"])
    rows = pits_on_history(gen, h, cfg["data"]["train_returns"], strategies, cfg,
                           rng(cfg["master_seed"], "L4", dgp_key, runner.gen_id(spec), s))
    meta = {"dgp": dgp_key, "generator": runner.label(spec), "kind": spec["kind"], "defect": spec.get("defect"),
            "delta": spec.get("delta", 0.0), "history": s}
    return [{**meta, **r} for r in rows]


def jobs_meta(cfg: dict) -> list[tuple[str, dict, int]]:
    out = []
    dgps = common.dgps(cfg)
    for key, dgp in dgps.items():
        specs = [{"kind": "oracle"}]
        for defect, delta in cfg["meta_validation"]["l4_defects"]:
            if defect == "P2" and dgp.name != "gjr_garch_t":
                continue
            specs.append({"kind": "oracle", "defect": defect, "delta": float(delta)})
        out += [(key, sp, s) for sp in specs for s in range(cfg["meta_validation"]["l4_histories"])]
    return out


def jobs_case(cfg: dict) -> list[tuple[str, dict, int]]:
    out = []
    for key in cfg["dgps"]:
        for kind in ("tcvae", "vae_mlp", "bootstrap", "garch"):
            g = cfg["generators"][kind]
            if "single" not in g.get("regimes", ["single"]):
                continue
            for s in range(g["seeds"]):
                spec = {"kind": kind, "regime": "single", "seed": s} if kind in ("tcvae", "vae_mlp") \
                    else {"kind": kind, "seed": s}
                out.append((key, spec, s))
    return out


def _args(p):
    p.add_argument("--part", choices=["meta", "case"], required=True)
    p.add_argument("--tests-only", action="store_true", help="recompute the tests from the stored PITs")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    name = f"e6_{args.part}"
    if args.tests_only:
        pits = pd.read_parquet(common.results_root(cfg) / name / "pits.parquet")
    else:
        todo = jobs_meta(cfg) if args.part == "meta" else jobs_case(cfg)
        res = Parallel(n_jobs=common.n_jobs(cfg, args.jobs), verbose=5)(delayed(task)(cfg, *j) for j in todo)
        pits = pd.DataFrame([r for rows in res for r in rows])
        common.save_table(pits, name, "pits", cfg)
    tests = []
    for (d, g), grp in pits.groupby(["dgp", "generator"]):
        tests += [{"dgp": d, "generator": g, **r} for r in calibration_tests(grp, cfg)]
    common.save_table(pd.DataFrame(tests), name, "tests", cfg)


if __name__ == "__main__":
    main()
