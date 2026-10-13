"""E7: robustness of the L2 verdicts of the case study.

Re-runs level L2 for the TC-VAE (single-history regime) and the two classical benchmarks
(first three histories each) under four variants: transaction cost of 10 basis points, two
alternative calibration objectives, and N = 10,000 scenarios. It also recomputes the L1 flags
of the negative control against an independent null of B = 500 pairs. The robustness to the
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


def settings(cfg: dict) -> list[tuple[str, dict]]:
    out = []
    for d in cfg["dgps"]:
        for s in range(N_HISTORIES):
            out += [(d, {"kind": "tcvae", "regime": "single", "seed": s}), (d, {"kind": "bootstrap", "seed": s}),
                    (d, {"kind": "garch", "seed": s})]
    return out


def variants(cfg: dict) -> dict[str, dict]:
    rob = cfg["robustness"]
    obj_low, obj_high = rob["objectives"]
    return {"cost": {"use": {"cost": rob["cost"]}},
            "obj_low": {"use": {"objective": obj_low}},
            "obj_high": {"use": {"objective": obj_high}},
            "n_alt": {"protocol": {"L2": {"N": rob["N_alt"]}}}}


def l1_null_sensitivity(cfg: dict, jobs: int, B: int = 500) -> pd.DataFrame:
    """Recompute the L1 flags of the negative control against an independent null of B pairs."""
    rows = Parallel(n_jobs=jobs)(delayed(runner.l1_null_task)(cfg, d, 100_000 + b) for d in cfg["dgps"] for b in range(B))
    null = pd.DataFrame(rows)
    l1 = pd.read_parquet(common.results_root(cfg) / "e1" / "l1.parquet")
    level = cfg["protocol"]["L1"]["level"]
    out = []
    for (d, k), g in l1.groupby(["dgp", "k"]):
        nd = null[null.dgp == d]
        p = np.array([empirical_pvalue(r.D, nd[r.statistic].to_numpy()) for r in g.itertuples()])
        rej = holm(p, level)
        out += [{"dgp": d, "k": k, "statistic": s, "p_b500": pk, "flag_raw_b500": pk < level, "flag_holm_b500": rk}
                for s, pk, rk in zip(g.statistic, p, rej)]
    df = pd.DataFrame(out)
    common.save_table(df, "e7_l1null", "l1_b500", cfg)
    return df


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    l1_null_sensitivity(cfg, common.n_jobs(cfg, args.jobs))
    for name, override in variants(cfg).items():
        cfg_v = deep_update(cfg, override)
        runner.run(cfg_v, f"e7_{name}", settings(cfg), common.n_jobs(cfg, args.jobs), parts=("L2",))


if __name__ == "__main__":
    main()
