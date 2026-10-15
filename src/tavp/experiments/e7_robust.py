"""E7: robustness of the L2 verdicts of the case study.

Re-runs level L2 for the TC-VAE (single-history regime), the Quant GAN (single-history regime, with
`--with-qgan`, deviation D11) and the two classical benchmarks (first three histories or runs each) under four variants: transaction cost of 10 basis points, two
alternative calibration objectives, and N = 10,000 scenarios. It also recomputes the L1 flags
of the negative control against an independent null of the same size B, to check that sharing
one null sample across the comparisons does not distort the false-alarm rates. The robustness to the
tolerances (x0.5, x2) is computed from stored results by tavp.report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.protocol.stats import empirical_pvalue, holm
from tavp.utils import deep_update

N_HISTORIES = 3


def settings(cfg: dict, with_qgan: bool = False) -> list[tuple[str, dict]]:
    G = cfg["generators"]
    out = []
    for d in cfg["dgps"]:
        out += [(d, {"kind": "tcvae", "regime": "single", "seed": s}) for s in range(min(N_HISTORIES, G["tcvae"]["seeds"]))]
        for kind in ("bootstrap", "garch"):
            out += [(d, {"kind": kind, "seed": s}) for s in range(min(N_HISTORIES, G[kind]["seeds"]))]
    if with_qgan:  # D11: same histories as the TC-VAE
        from tavp.experiments import e13_qgan

        out += [(d, sp) for d, sp in e13_qgan.settings(cfg) if sp["regime"] == "single" and sp["seed"] < N_HISTORIES]
    return out


def variants(cfg: dict) -> dict[str, dict]:
    rob = cfg["robustness"]
    obj_low, obj_high = rob["objectives"]
    return {"cost": {"use": {"cost": rob["cost"]}},
            "obj_low": {"use": {"objective": obj_low}},
            "obj_high": {"use": {"objective": obj_high}},
            "n_alt": {"protocol": {"L2": {"N": rob["N_alt"]}}}}


def l1_null_sensitivity(cfg: dict, jobs: int) -> pd.DataFrame:
    """Recompute the L1 flags of the negative control against an independent null of B pairs."""
    B = cfg["protocol"]["L1"]["b_null"]
    rows = Parallel(n_jobs=jobs)(delayed(runner.l1_null_task)(cfg, d, 100_000 + b) for d in cfg["dgps"] for b in range(B))
    null = pd.DataFrame(rows)
    l1 = pd.read_parquet(common.results_root(cfg) / "e1" / "l1.parquet")
    level = cfg["protocol"]["L1"]["level"]
    out = []
    for (d, k), g in l1.groupby(["dgp", "k"]):
        nd = null[null.dgp == d]
        p = np.array([empirical_pvalue(r.D, nd[r.statistic].to_numpy()) for r in g.itertuples()])
        rej = holm(p, level)
        out += [{"dgp": d, "k": k, "statistic": s, "p_indep": pk, "flag_raw_indep": pk < level, "flag_holm_indep": rk}
                for s, pk, rk in zip(g.statistic, p, rej)]
    df = pd.DataFrame(out)
    common.save_table(df, "e7_l1null", "l1_indep", cfg)
    return df


def _args(p):
    p.add_argument("--with-qgan", action="store_true",
                   help="D11: add the Quant GAN (run by `make qgan`; the check of the L1 null is not repeated)")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    if not args.with_qgan:
        l1_null_sensitivity(cfg, common.n_jobs(cfg, args.jobs))
    for name, override in variants(cfg).items():
        cfg_v = deep_update(cfg, override)
        runner.run(cfg_v, f"e7_{name}", settings(cfg, args.with_qgan), common.n_jobs(cfg, args.jobs), parts=("L2",))


if __name__ == "__main__":
    main()
