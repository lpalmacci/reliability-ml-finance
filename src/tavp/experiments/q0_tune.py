"""Q0b: one bounded round of tuning of the Quant GAN on the data of the paper, after Q0 failed its criterion.

Q0 reproduced the advantage of the TCN over the GARCH(1,1) on the return distribution and on the leverage
effect, but not on volatility clustering (ACF scores worse in all three runs), and in one of the three runs the
dependence scores were still improving at the end of training. This round, fixed before it was run, is the only
one allowed:

- grid: 12,000 steps (instead of 6,000), generator learning rate in {1e-4, 3e-5}, R1 weight in {1, 10}; all
  other settings, the checkpoint rule and the metrics as in Q0; one new training seed (10) per configuration;
- choice: the configuration whose run beats the GARCH(1,1) of Q0 on the largest number of the 12 metrics
  (ties: more EMDs beaten, then the smaller generator learning rate, then the smaller R1 weight);
- confirmation: the chosen configuration is trained with three new seeds (20, 21, 22) and judged with the
  criterion of Q0, unchanged (at least two runs better than the GARCH(1,1) on at least 10 of the 12 metrics,
  including the four EMDs).

Only the data of the paper (May 2009 - December 2018) are used, before any experiment of the protocol with
the Quant GAN. If the confirmation fails, the Quant GAN is not used in the case study.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common
from tavp.experiments.q0_fidelity import LENGTH, METRICS, PERIOD, M, garch_paths
from tavp.generators.qgan import QuantGANGenerator, paper_scores
from tavp.market import market_log_returns
from tavp.utils import REPO_ROOT, load_yaml, rng

GRID = [{"steps": 12000, "lr_g": lg, "r1_gamma": g} for lg in (1e-4, 3e-5) for g in (1.0, 10.0)]
TUNE_SEED = 10
CONFIRM_SEEDS = (20, 21, 22)


def tag(c: dict) -> str:
    return f"lrg{c['lr_g']:g}_r1{c['r1_gamma']:g}"


def run(cfg: dict, h, qcfg: dict, change: dict, seed: int) -> dict:
    out = common.results_root(cfg) / "q0" / "tune" / tag(change) / f"seed{seed}"
    if (out / "meta.json").exists():
        g = QuantGANGenerator.load(out)
    else:
        g = QuantGANGenerator({**qcfg, **change}, "q0", name="qgan").fit(
            h, seed=common.stream_seed(cfg, "q0", "tune", tag(change), seed), out_dir=out)
    paths = g.sample_returns(M, LENGTH, rng(cfg["master_seed"], "q0", "tune", tag(change), seed))
    return {"config": tag(change), **change, "seed": seed, "selected_step": g.meta["selected_step"],
            **paper_scores(h, paths)}


def beats(row, garch) -> tuple[int, int]:
    better = [m for m in METRICS if row[m] < garch[m]]
    return len(better), sum(m.startswith("emd") for m in better)


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    d, r = market_log_returns()
    h = r[(d >= PERIOD[0]) & (d <= PERIOD[1])]
    qcfg = load_yaml(REPO_ROOT / "configs" / "generators" / "qgan.yaml")
    garch = paper_scores(h, garch_paths(h, M, LENGTH, rng(cfg["master_seed"], "q0", "garch")))
    tune = pd.DataFrame(Parallel(n_jobs=4)(delayed(run)(cfg, h, qcfg, c, TUNE_SEED) for c in GRID))
    tune[["better", "emd_better"]] = [beats(x, garch) for _, x in tune.iterrows()]
    best = tune.sort_values(["better", "emd_better", "lr_g", "r1_gamma"], ascending=[False, False, True, True]).iloc[0]
    change = {"steps": int(best.steps), "lr_g": float(best.lr_g), "r1_gamma": float(best.r1_gamma)}
    conf = pd.DataFrame(Parallel(n_jobs=3)(delayed(run)(cfg, h, qcfg, change, s) for s in CONFIRM_SEEDS))
    conf[["better", "emd_better"]] = [beats(x, garch) for _, x in conf.iterrows()]
    conf["pass"] = (conf.better >= 10) & (conf.emd_better == 4)
    common.save_table(tune, "q0", "tune", cfg)
    common.save_table(conf, "q0", "confirm", cfg)
    common.save_table(pd.DataFrame([{"model": "GARCH(1,1)", **garch}]), "q0", "tune_garch", cfg)
    print(tune[["config", "selected_step", "better", "emd_better"] + METRICS].to_string())
    print("chosen:", change)
    print(conf[["seed", "selected_step", "better", "emd_better", "pass"] + METRICS].to_string())
    print("ACCEPTED" if int(conf["pass"].sum()) >= 2 else "NOT ACCEPTED")


if __name__ == "__main__":
    main()
