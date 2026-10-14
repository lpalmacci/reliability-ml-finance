"""E10, sensitivity of the positive control to the fit of benchmark B2 (market-data study, deviation D8).

The optimiser of the arch package can stop at a local optimum of the GJR-GARCH-t likelihood and
report success. This script refits every benchmark B2 of a configuration (single and long
histories) with the multistart fit of `tavp.generators.base.fit_gjr_garch_t` and records the gain
in log-likelihood (`fits`). With `--protocol` it also validates the multistart fits on the long
histories with levels L0-L3, under the same contexts, oracle scenarios and random streams as B2 in
E8, so that the two differ only by the fit. The pre-registered results of B2 are not changed.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.generators.base import fit_gjr_garch_t

PARAMS = ["mu", "omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]


def refit(cfg: dict, dgp_key: str, regime: str, seed: int) -> dict:
    dgp = common.dgps(cfg)[dgp_key]
    if regime == "rich":
        train = common.long_history(cfg, dgp, seed)
    else:
        train = common.history(cfg, dgp, seed)[: cfg["data"]["train_returns"]]
    one, many = fit_gjr_garch_t(100.0 * train), fit_gjr_garch_t(100.0 * train, multistart=True)
    row = {"dgp": dgp_key, "regime": regime, "seed": seed, "n": len(train), "loglik": float(one.loglikelihood),
           "loglik_ms": float(many.loglikelihood), "gain": float(many.loglikelihood - one.loglikelihood)}
    row.update({p: float(one.params[p]) for p in PARAMS})
    row.update({p + "_ms": float(many.params[p]) for p in PARAMS})
    return row


def _args(p):
    p.add_argument("--protocol", action="store_true", help="also validate the multistart fits of the long histories")


def main():
    args = common.parse_args(__doc__, _args)
    cfg = common.load(args.config)
    jobs = common.n_jobs(cfg, args.jobs)
    todo = [(d, r, s) for d in cfg["dgps"] for r in ("single", "rich") for s in range(cfg["generators"]["garch"]["seeds"])]
    rows = Parallel(n_jobs=jobs, verbose=5)(delayed(refit)(cfg, *t) for t in todo)
    common.save_table(pd.DataFrame(rows), "e10ms", "fits", cfg)
    if args.protocol:
        settings = [(d, {"kind": "garch_ms", "regime": "rich", "seed": s})
                    for d in cfg["dgps"] for s in range(cfg["generators"]["garch"]["seeds"])]
        runner.run(cfg, "e10ms", settings, jobs)


if __name__ == "__main__":
    main()
