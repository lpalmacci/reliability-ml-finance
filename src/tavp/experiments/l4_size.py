"""Size of the L4 tests when the PITs are uniform but serially dependent (deviation D4).

In mode S the PITs of consecutive origins of one history are dependent even under the true
law. This simulation draws uniform PITs from a Gaussian AR(1) copula with the first-order
correlation observed in E6 and records how often each test rejects at level 0.05, with the
layout of the meta-validation of L4 (20 histories of 42 origins) and of one history.

    python -m tavp.experiments.l4_size --config configs/full.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tavp.experiments import common
from tavp.protocol.stats import berkowitz, ks_uniform, kupiec_pof, moment_test_hac
from tavp.utils import rng

PHIS = (0.0, 0.15, 0.3)
LAYOUTS = ((20, 42), (1, 42))
REPS = 1000


def ar1_pits(r: np.random.Generator, phi: float, T: int) -> np.ndarray:
    z = np.empty(T)
    z[0] = r.standard_normal()
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi**2) * r.standard_normal()
    return stats.norm.cdf(z)


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    level, interval = cfg["protocol"]["L4"]["level"], tuple(cfg["protocol"]["L4"]["interval"])
    rows = []
    for H, T in LAYOUTS:
        for phi in PHIS:
            r = rng(cfg["master_seed"], "L4size", H, T, phi)
            rej = {k: 0 for k in ("ks", "kupiec", "berkowitz_joint", "berkowitz_marginal", "moments_hac")}
            for _ in range(REPS):
                seqs = [ar1_pits(r, phi, T) for _ in range(H)]
                u = np.concatenate(seqs)
                p_joint, p_marg = berkowitz(seqs)
                p = {"ks": ks_uniform(u), "kupiec": kupiec_pof(u, interval), "berkowitz_joint": p_joint,
                     "berkowitz_marginal": p_marg, "moments_hac": moment_test_hac(seqs)}
                for k, v in p.items():
                    rej[k] += v < level
            rows += [{"histories": H, "origins": T, "phi": phi, "test": k, "rejection": v / REPS} for k, v in rej.items()]
    df = pd.DataFrame(rows)
    common.save_table(df, "e6_size", "size", cfg)
    print(df.pivot_table(index=["histories", "phi"], columns="test", values="rejection").round(3))


if __name__ == "__main__":
    main()
