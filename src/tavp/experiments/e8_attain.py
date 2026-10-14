"""E8 (added after the results, not pre-registered; deviation D5): attainability and positive control.

Two kinds of generators are validated with levels L0-L3 under the same contexts and oracle
scenarios as E4 and E5:

- the oracle with estimated drift (tavp.generators.base.EstimatedDriftOracle), on the single
  histories of E5 and on long histories with as many returns as the rich training set of the
  TC-VAE. It defines which cells of level L2 a generator can pass with histories of that length;
- the stationary bootstrap and the GJR-GARCH-t benchmark trained on the long histories, the
  benchmarks of the rich regime, which test whether a trained generator passes the attainable cells.
"""

from __future__ import annotations

import pandas as pd

from tavp.experiments import common, runner


def generator_table(cfg: dict) -> pd.DataFrame:
    """Drift error and fitted parameters of every generator of E8 and of the single-history benchmarks."""
    rows = []
    specs = settings(cfg) + [(d, {"kind": k, "seed": s}) for d in cfg["dgps"] for k in ("bootstrap", "garch")
                             for s in range(cfg["generators"][k]["seeds"])]
    for d, spec in specs:
        g = runner.make_generator(cfg, d, spec)
        dgp = common.dgps(cfg)[d]
        if spec["kind"] == "oracle_drift":
            drift = dgp.drift_per_step() + g.shift
        elif spec["kind"] == "bootstrap":
            drift = float(g.history.mean())
        else:
            drift = g.sim.drift_per_step()
        row = {"dgp": d, "kind": spec["kind"], "regime": spec.get("regime", "single"), "seed": spec["seed"],
               "drift_error_annual": (drift - dgp.drift_per_step()) / dgp.dt}
        if spec["kind"] == "bootstrap":
            row["block_mean"] = g.block_mean
        if spec["kind"] == "garch":
            row.update({k: v for k, v in g.sim.params().items() if isinstance(v, (int, float))})
        rows.append(row)
    df = pd.DataFrame(rows)
    common.save_table(df, "e8", "generators", cfg)
    return df


def settings(cfg: dict) -> list[tuple[str, dict]]:
    G = cfg["generators"]
    out = []
    for d in cfg["dgps"]:
        out += [(d, {"kind": "oracle_drift", "regime": r, "seed": s})
                for r in ("single", "rich") for s in range(G["bootstrap"]["seeds"])]
        for kind in ("bootstrap", "garch"):
            out += [(d, {"kind": kind, "regime": "rich", "seed": s}) for s in range(G[kind]["seeds"])]
    return out


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    runner.run(cfg, "e8", settings(cfg), common.n_jobs(cfg, args.jobs))
    generator_table(cfg)


if __name__ == "__main__":
    main()
