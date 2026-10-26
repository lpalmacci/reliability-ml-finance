"""Q0: fidelity check of the Quant GAN implementation (G3) on the experiment of the paper.

Wiese et al. (2020, Section 7.4, Table 2) train the pure TCN on daily S&P 500 log returns from May 2009 to
December 2018 and find that it beats a GARCH(1,1) with constant drift and Gaussian innovations on all twelve
metrics of their Table 2 (EMD and DY metric of 1-, 5-, 20- and 100-day log returns, ACF scores of returns,
absolute and squared returns, leverage score), often by a factor of 2 to 10. Q0 repeats the experiment with
the implementation of tavp.generators.qgan, the settings of configs/generators/qgan.yaml and three training
runs, on the closest series available in the replication package: the daily value-weighted US market return
of the Ken French Data Library over the same period (tavp.market). The metrics are computed as in the paper,
with M = 500 generated paths of 4,000 returns and lags up to 250. The absolute values of two of them are not
comparable with Table 2: the partition of the DY metric is only loosely specified ("about five historical
observations per bin" gives much larger sums than those reported) and the ACF scores of the paper appear to be
normalised by the number of lags (our norm divided by the square root of 250 is close to the reported values).
The check therefore compares the TCN with the GARCH(1,1) under the same implementation, as the paper does.

Acceptance criterion, fixed before the check was run: the implementation is accepted if, for at least two of
the three runs, the TCN scores better than the GARCH(1,1) on at least 10 of the 12 metrics, including the four
EMDs. The settings are then frozen for every experiment of the protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tavp.experiments import common
from tavp.generators.qgan import QuantGANGenerator, paper_scores
from tavp.market import market_log_returns
from tavp.utils import REPO_ROOT, load_yaml, rng

PERIOD = (20090501, 20181231)
PAPER = {  # Table 2 of the paper
    "TCN": {"emd_1": 0.0039, "emd_5": 0.0039, "emd_20": 0.0040, "emd_100": 0.0154, "dy_1": 19.1199, "dy_5": 21.1167,
            "dy_20": 26.3294, "dy_100": 28.1315, "acf_id": 0.0212, "acf_abs": 0.0248, "acf_sq": 0.0214,
            "leverage": 0.3291},
    "GARCH(1,1)": {"emd_1": 0.0199, "emd_5": 0.0145, "emd_20": 0.0276, "emd_100": 0.0935, "dy_1": 32.7090,
                   "dy_5": 27.4760, "dy_20": 39.3796, "dy_100": 46.4779, "acf_id": 0.0223, "acf_abs": 0.0291,
                   "acf_sq": 0.0253, "leverage": 0.4636},
}
METRICS = list(PAPER["TCN"])
SEEDS = (0, 1, 2)
# Q0 was run with the starting values of the settings that the paper does not report; configs/generators/qgan.yaml
# now holds the values frozen after the tuning round Q0b (tavp.experiments.q0_tune)
START = {"lr_g": 1e-4, "r1_gamma": 1.0, "steps": 6000}
M, LENGTH = 500, 4000


def garch_paths(r: np.ndarray, n: int, T: int, gen: np.random.Generator) -> np.ndarray:
    """GARCH(1,1) with constant drift and Gaussian innovations, fitted by ML and simulated from stationarity."""
    from arch import arch_model

    p = arch_model(100.0 * r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off").params
    mu, om, a, b = p["mu"] / 100, p["omega"] / 1e4, p["alpha[1]"], p["beta[1]"]
    burn = 1000
    s2 = np.full(n, om / (1 - a - b))
    eps = np.zeros(n)
    out = np.empty((n, T))
    for t in range(burn + T):
        s2 = om + a * eps**2 + b * s2
        eps = np.sqrt(s2) * gen.standard_normal(n)
        if t >= burn:
            out[:, t - burn] = mu + eps
    return out


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    d, r = market_log_returns()
    h = r[(d >= PERIOD[0]) & (d <= PERIOD[1])]
    qcfg = {**load_yaml(REPO_ROOT / "configs" / "generators" / "qgan.yaml"), **START}
    root = common.results_root(cfg) / "q0"
    rows = []
    for s in SEEDS:
        out = root / "runs" / f"seed{s}"
        if (out / "meta.json").exists():
            g = QuantGANGenerator.load(out)
        else:
            g = QuantGANGenerator(qcfg, "q0", name="qgan").fit(h, seed=common.stream_seed(cfg, "q0", s), out_dir=out)
        paths = g.sample_returns(M, LENGTH, rng(cfg["master_seed"], "q0", "tcn", s))
        rows.append({"model": "TCN", "seed": s, "selected_step": g.meta["selected_step"], **paper_scores(h, paths)})
    gp = garch_paths(h, M, LENGTH, rng(cfg["master_seed"], "q0", "garch"))
    rows.append({"model": "GARCH(1,1)", "seed": -1, **paper_scores(h, gp)})
    for k, v in PAPER.items():
        rows.append({"model": f"paper {k}", "seed": -1, **v})
    t = pd.DataFrame(rows)
    garch = t[t.model == "GARCH(1,1)"].iloc[0]
    wins = []
    for _, x in t[t.model == "TCN"].iterrows():
        better = [m for m in METRICS if x[m] < garch[m]]
        wins.append({"seed": int(x.seed), "better": len(better), "all_emd": all(f"emd_{k}" in better for k in (1, 5, 20, 100))})
    w = pd.DataFrame(wins)
    w["pass"] = (w.better >= 10) & w.all_emd
    common.save_table(t, "q0", "scores", cfg)
    common.save_table(w, "q0", "criterion", cfg)
    print(t.to_string())
    print(w.to_string())
    print("ACCEPTED" if int(w["pass"].sum()) >= 2 else "NOT ACCEPTED")


if __name__ == "__main__":
    main()
