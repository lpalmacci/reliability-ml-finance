"""E13: a second deep generator, the Quant GAN (G3), pre-registered by addendum (docs/ADDENDUM_QGAN.md).

The settings of G3 are frozen in configs/generators/qgan.yaml after the fidelity check Q0 and before this
experiment. G3 is trained on the same data as the other generators: the first ten years of each of the five
histories of each process (regime "single") and the five long histories of E8 with as many returns as the
rich training set of the TC-VAE (regime "rich"; the Quant GAN needs one history longer than its receptive
field of 127 days, so it cannot use the independent paths of 60 prices of that set). It is then validated
under exactly the protocol of E4, E5 and E8 (levels L0-L3, same contexts, oracle scenarios and sizes) and,
in the single regime, at level L4 on the held-out half of each history as in E6.

Stages: `train` (Apple GPU when available), `protocol`, `l4`, `facts` (stylized facts as in E4).
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.protocol.l4 import calibration_tests, pits_on_history
from tavp.tasks.strategies import build_strategies
from tavp.utils import REPO_ROOT, load_yaml, rng, stable_hash

SEEDS = 5
REGIMES = ("single", "rich")


def qgan_cfg() -> dict:
    return load_yaml(REPO_ROOT / "configs" / "generators" / "qgan.yaml")


def train_history(cfg: dict, dgp, regime: str, seed: int):
    if regime == "rich":
        return common.long_history(cfg, dgp, seed)
    return common.history(cfg, dgp, seed)[: cfg["data"]["train_returns"]]


def train_one(cfg: dict, dgp_key: str, regime: str, seed: int) -> str:
    from tavp.generators.qgan import QuantGANGenerator

    dgp = common.dgps(cfg)[dgp_key]
    out = common.run_dir(cfg, "qgan", dgp.name, regime, seed)
    if (out / "meta.json").exists():
        return str(out)
    QuantGANGenerator(qgan_cfg(), dgp.hash(), name="qgan").fit(
        train_history(cfg, dgp, regime, seed), seed=common.stream_seed(cfg, "train", "qgan", dgp.name, regime, seed),
        out_dir=out)
    return str(out)


def settings(cfg: dict) -> list[tuple[str, dict]]:
    # the hash of the settings of G3 enters the cache key of each task (it is not part of the configuration)
    h = stable_hash(qgan_cfg())
    return [(d, {"kind": "qgan", "regime": r, "seed": s, "qgan_cfg": h})
            for d in cfg["dgps"] for r in REGIMES for s in range(SEEDS)]


def l4_task(cfg: dict, dgp_key: str, spec: dict, s: int) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    dgp = common.dgps(cfg)[dgp_key]
    gen = runner.make_generator(cfg, dgp_key, spec)
    rows = pits_on_history(gen, common.history(cfg, dgp, s), cfg["data"]["train_returns"],
                           build_strategies(cfg["use"]["strategies"]), cfg,
                           rng(cfg["master_seed"], "L4", dgp_key, runner.gen_id(spec), s))
    meta = {"dgp": dgp_key, "generator": runner.label(spec), "kind": "qgan", "defect": None, "delta": 0.0, "history": s}
    return [{**meta, **r} for r in rows]


def _args(p):
    p.add_argument("--stage", choices=["train", "protocol", "l4", "facts"], required=True)
    p.add_argument("--train-jobs", type=int, default=3, help="concurrent training processes on the GPU")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    if args.stage == "train":
        todo = [(d, r, s) for d in cfg["dgps"] for r in REGIMES for s in range(SEEDS)]
        Parallel(n_jobs=args.train_jobs, verbose=5)(delayed(train_one)(cfg, *t) for t in todo)
    elif args.stage == "facts":
        from tavp.experiments import e4_facts

        todo = [(d, sp) for d, sp in settings(cfg)]
        rows = Parallel(n_jobs=common.n_jobs(cfg, args.jobs), verbose=5)(delayed(e4_facts.task)(cfg, d, sp) for d, sp in todo)
        common.save_table(pd.DataFrame(rows), "e13", "facts", cfg)
    elif args.stage == "protocol":
        runner.run(cfg, "e13", settings(cfg), common.n_jobs(cfg, args.jobs))
    else:
        todo = [(d, sp, sp["seed"]) for d, sp in settings(cfg) if sp["regime"] == "single"]
        res = Parallel(n_jobs=common.n_jobs(cfg, args.jobs), verbose=5)(delayed(l4_task)(cfg, *t) for t in todo)
        pits = pd.DataFrame([r for rows in res for r in rows])
        common.save_table(pits, "e13", "l4_pits", cfg)
        tests = []
        for (d, g), grp in pits.groupby(["dgp", "generator"]):
            tests += [{"dgp": d, "generator": g, **r} for r in calibration_tests(grp, cfg)]
        common.save_table(pd.DataFrame(tests), "e13", "l4_tests", cfg)


if __name__ == "__main__":
    main()
