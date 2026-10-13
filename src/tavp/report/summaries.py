"""Summaries of the protocol outputs: size (E1), power and detection (E2), dissociation (E3), verdicts.

Definitions used throughout the chapter:
- L1 power of a statistic: share of the K independent comparisons in which it is flagged
  after the Holm correction; L1 flags a generator when any statistic is flagged.
- L2 verdict of a cell (strategy, metric, functional): pass if the share of contexts (and
  seeds) in which the TOST declares equivalence is at least PR_min.
- L3 verdict of a strategy: pass if the share of contexts in which the upper bound of the
  regret is within the tolerance is at least PR_min.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["dgp", "generator", "kind", "defect", "delta"]


def _keys(df: pd.DataFrame) -> list[str]:
    return [k for k in KEYS + ["regime", "seed"] if k in df.columns]


def l1_rates(l1: pd.DataFrame) -> pd.DataFrame:
    """Per setting and statistic: raw and Holm flag rates; plus the 'any statistic' row."""
    keys = [k for k in KEYS if k in l1.columns]
    stat = l1.groupby(keys + ["statistic"], dropna=False).agg(
        flag_raw=("flag_raw", "mean"), flag_holm=("flag_holm", "mean"), ratio=("ratio", "median"),
        n=("flag_raw", "size")).reset_index()
    anyf = l1.groupby(keys + ["k"], dropna=False).flag_holm.any().reset_index()
    anyf = anyf.groupby(keys, dropna=False).flag_holm.mean().reset_index()
    anyf["statistic"] = "any"
    return pd.concat([stat, anyf], ignore_index=True)


def l2_cells(l2: pd.DataFrame, pr_min: float) -> pd.DataFrame:
    keys = [k for k in KEYS if k in l2.columns]
    g = l2.groupby(keys + ["strategy", "metric", "functional"], dropna=False)
    out = g.agg(pr=("equivalent", "mean"), resolvable=("resolvable", "mean"), diff=("diff", "mean"),
                abs_diff=("diff", lambda x: float(np.abs(x).mean())), half_width=("half_width", "median"),
                eta=("eta", "first"), n=("equivalent", "size")).reset_index()
    out["pass"] = out["pr"] >= pr_min
    return out


def l3_strategies(l3: pd.DataFrame, pr_min: float) -> pd.DataFrame:
    keys = [k for k in KEYS if k in l3.columns]
    out = l3.groupby(keys + ["strategy"], dropna=False).agg(
        pr=("pass", "mean"), regret=("regret", "mean"), regret_norm=("regret_norm", "mean"),
        kendall_tau=("kendall_tau", "mean"), same_choice=("same_choice", "mean"), n=("pass", "size")).reset_index()
    out["pass"] = out["pr"] >= pr_min
    return out


def verdicts(l1: pd.DataFrame | None, l2: pd.DataFrame, l3: pd.DataFrame | None, pr_min: float,
             l1_threshold: float = 0.5) -> pd.DataFrame:
    """One row per setting: L1 power (any statistic), number of failed L2 cells, L3 failures, overall verdict."""
    keys = [k for k in KEYS if k in l2.columns]
    cells = l2_cells(l2, pr_min)
    v = cells.groupby(keys, dropna=False).agg(l2_cells=("pass", "size"),
                                              l2_failed=("pass", lambda x: int((~x).sum()))).reset_index()
    if l3 is not None and len(l3):
        s3 = l3_strategies(l3, pr_min).groupby(keys, dropna=False).agg(
            l3_failed=("pass", lambda x: int((~x).sum()))).reset_index()
        v = v.merge(s3, on=keys, how="left")
    if l1 is not None and len(l1):
        r1 = l1_rates(l1)
        r1 = r1[r1.statistic == "any"][keys + ["flag_holm"]].rename(columns={"flag_holm": "l1_power"})
        v = v.merge(r1, on=keys, how="left")
        v["l1_flag"] = v["l1_power"] >= l1_threshold
    v["fit_for_use"] = (v["l2_failed"] == 0) & (v.get("l3_failed", 0) == 0)
    return v


def dissociation(l1: pd.DataFrame, l2: pd.DataFrame, pr_min: float, l1_threshold: float = 0.5) -> pd.DataFrame:
    """E3: classify each (setting, strategy) by the L1 flag and the L2 verdict of that strategy."""
    keys = [k for k in KEYS if k in l2.columns]
    cells = l2_cells(l2, pr_min)
    strat = cells.groupby(keys + ["strategy"], dropna=False).agg(l2_fail=("pass", lambda x: bool((~x).any()))).reset_index()
    r1 = l1_rates(l1)
    r1 = r1[r1.statistic == "any"][keys + ["flag_holm"]].rename(columns={"flag_holm": "l1_power"})
    out = strat.merge(r1, on=keys, how="left")
    out["l1_flag"] = out["l1_power"] >= l1_threshold
    out["category"] = np.select(
        [out.l1_flag & out.l2_fail, out.l1_flag & ~out.l2_fail, ~out.l1_flag & out.l2_fail],
        ["both flag", "L1 only (irrelevant defect)", "L2 only (hidden defect)"], "both pass")
    return out


def load_partial(root, experiment: str) -> dict[str, pd.DataFrame]:
    """Tables assembled from the cached tasks of an experiment that is still running."""
    import pickle
    from pathlib import Path

    rows = {"l0": [], "l1": [], "l2": [], "l3": []}
    for f in sorted(Path(root, experiment, "tasks").rglob("*.pkl")):
        with open(f, "rb") as fh:
            out = pickle.load(fh)
        for k in rows:
            rows[k] += out[k]
    return {k: pd.DataFrame(v) for k, v in rows.items() if v}
