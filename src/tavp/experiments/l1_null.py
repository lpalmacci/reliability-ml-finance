"""Null law of the L1 discrepancies and the L1 p-values of every experiment.

The discrepancy D of a comparison does not depend on the null: tasks store D, and the
p-values and flags of the final tables are computed here against the null of
`protocol.L1.b_null` pairs. For the record of deviation D3 they are also computed against
the first `protocol.L1.b_null_prereg` pairs, which are the pre-registered null (the pair b
uses the same random stream whatever B is).

    python -m tavp.experiments.l1_null --config configs/full.yaml
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tavp.experiments import common, runner
from tavp.protocol.stats import empirical_pvalue, holm


def null_path(cfg: dict):
    return common.results_root(cfg) / "e1" / "l1_null.parquet"


def ensure(cfg: dict, jobs: int) -> pd.DataFrame:
    """Extend the stored null to B pairs per process, computing only the missing pairs."""
    B = cfg["protocol"]["L1"]["b_null"]
    path = null_path(cfg)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["dgp", "b"])
    have = set(zip(old.dgp, old.b))
    todo = [(d, b) for d in cfg["dgps"] for b in range(B) if (d, b) not in have]
    if not todo:
        return old
    rows = Parallel(n_jobs=jobs, verbose=5)(delayed(runner.l1_null_task)(cfg, d, b) for d, b in todo)
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if len(old) else pd.DataFrame(rows)
    df = df.sort_values(["dgp", "b"]).reset_index(drop=True)
    common.save_table(df, "e1", "l1_null", cfg)
    return df


def load(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(null_path(cfg))
    B = cfg["protocol"]["L1"]["b_null"]
    n = df.groupby("dgp").b.nunique()
    if (n < B).any():
        raise RuntimeError(f"L1 null has fewer than {B} pairs: run python -m tavp.experiments.l1_null")
    return df[df.b < B]


def as_dict(null: pd.DataFrame) -> dict:
    return {d: {c: g[c].to_numpy() for c in g.columns if c not in ("dgp", "b")} for d, g in null.groupby("dgp")}


def _flags(g: pd.DataFrame, null: dict, level: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([empirical_pvalue(r.D, null[r.statistic]) for r in g.itertuples()])
    return p, holm(p, level)


def pvalues(l1: pd.DataFrame, null: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """p-values and flags of every comparison (one comparison = one group of statistics)."""
    c = cfg["protocol"]["L1"]
    level, b0 = c["level"], c["b_null_prereg"]
    full, prereg = as_dict(null), as_dict(null[null.b < b0])
    by = [k for k in ("experiment", "dgp", "generator", "regime", "seed", "k") if k in l1.columns]
    out = []
    for _, g in l1.groupby(by, dropna=False, sort=False):
        g = g.copy()
        nd, n0 = full[g.dgp.iloc[0]], prereg[g.dgp.iloc[0]]
        med = np.array([np.median(nd[s]) for s in g.statistic])
        g["D_null_median"], g["ratio"] = med, g.D.to_numpy() / med
        g["p"], rej = _flags(g, nd, level)
        g["flag_raw"], g["flag_holm"] = g.p < level, rej
        g["p_prereg"], rej0 = _flags(g, n0, level)
        g["flag_raw_prereg"], g["flag_holm_prereg"] = g.p_prereg < level, rej0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def update(cfg: dict, null: pd.DataFrame | None = None) -> list[str]:
    """Recompute the L1 table of every experiment found under the results directory."""
    null = load(cfg) if null is None else null
    done = []
    for path in sorted(common.results_root(cfg).glob("*/l1.parquet")):
        experiment = path.parent.name
        l1 = pd.read_parquet(path)
        common.save_table(pvalues(l1, null, cfg), experiment, "l1", cfg)
        done.append(experiment)
    return done


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    null = ensure(cfg, common.n_jobs(cfg, args.jobs))
    print("L1 tables updated:", ", ".join(update(cfg, null[null.b < cfg["protocol"]["L1"]["b_null"]])))


if __name__ == "__main__":
    main()
