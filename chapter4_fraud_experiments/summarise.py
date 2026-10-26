"""Step 5: summary of an evaluation across partitions.

    python summarise.py                    # results/evaluation/        -> summary/*.parquet, results/chapter4_results.json
    python summarise.py --unconstrained    # results/evaluation_unconstrained_lightgbm/
                                           #                            -> summary/*.parquet, results/chapter4_results_unconstrained_lightgbm.json

It evaluates nothing: it reads the results saved by each job. For every comparison it reports, across
the partitions, the mean of the differences, the number of partitions whose bootstrap interval lies above
and below zero, the minimum and the maximum; the other quantities (shares captured, room of the budget,
false positives and missed frauds, share of the budget spent on the predicted positives) as means across
partitions. On ULB the main analysis uses the partitions whose TabPFN members are complete (all ten).

The law at the posterior mean of c ("confidex plug-in", a robustness row of the main analysis) is also
reported among the ablations, with the same fields as the other variants and on the same partitions: its
bootstrap interval is computed from the bootstrap means saved by each job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fraud.data import N_PARTITIONS, PARTITION_DIR
from fraud.members import member_path
from fraud.plan import plan
from fraud.readings import SIMPLE
from fraud.registry import RESULTS

KEYS = ["dataset", "strategy", "threshold", "K"]
PLUGIN = "confidex plug-in"
ABL_COLS = ["captured", "k", "FP_captured", "FN_captured", "share_budget_positive", "random", "perfect"]
RESULT_FILES = {"final": "chapter4_results.json", "unconstrained": "chapter4_results_unconstrained_lightgbm.json"}
EVALUATIONS = {"final": "evaluation", "unconstrained": "evaluation_unconstrained_lightgbm"}


def complete_tabpfn_partitions(dataset: str) -> list[int]:
    out = []
    for p in range(N_PARTITIONS[dataset]):
        members = [m for m in plan(dataset, p, "s2") if m.family == "TabPFN"]
        if all(member_path(m).exists() for m in members):
            out.append(p)
    return out


class Reader:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def table(self, name: str, sub: str = "jobs") -> pd.DataFrame:
        parts = []
        if not (self.dir / sub).exists():
            return pd.DataFrame()
        for d in sorted((self.dir / sub).iterdir()):
            f, info = d / f"{name}.parquet", d / "done.json"
            if f.exists() and info.exists():
                i = json.loads(info.read_text())
                parts.append(pd.read_parquet(f).assign(dataset=i["dataset"], partition=i["partition"]))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def keep_main(df: pd.DataFrame, complete: dict[str, list[int]]) -> pd.DataFrame:
    m = np.zeros(len(df), dtype=bool)
    for d, ps in complete.items():
        m |= (df.dataset == d).to_numpy() & df.partition.isin(ps).to_numpy()
    return df[m]


def summary(df: pd.DataFrame, groups: list[str], diff: str, lo: str, hi: str, means: list[str] = ()) -> pd.DataFrame:
    g = df.groupby(groups)
    out = g.agg(partitions=(diff, "size"), mean=(diff, "mean"), min=(diff, "min"), max=(diff, "max"))
    out["above_zero"] = g.apply(lambda x: int((x[lo] > 0).sum()), include_groups=False)
    out["below_zero"] = g.apply(lambda x: int((x[hi] < 0).sum()), include_groups=False)
    for c in means:
        out[c] = g[c].mean()
    return out.reset_index()


def _saved_interval(boot: dict[str, np.ndarray], a: str, b: str) -> tuple[float, float]:
    """As ``fraud.evaluate._interval``, on the bootstrap means saved in bootstrap.npz."""
    d = boot[a] - boot[b]
    return float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))


def plugin_comparison(directory: Path) -> pd.DataFrame:
    """The law at the posterior mean of c as a variant of the ablations: against Confidex on the same errors,
    at the budgets k_rho at the same k, at k_alpha each verdict at its own k. Same fields as
    comparison_ablations, on the jobs with ablations only. For every job it also checks that the same
    computation, applied to a variant already saved, reproduces its interval."""
    out = []
    for d in sorted((directory / "jobs").iterdir()):
        info, f_abl = d / "done.json", d / "comparison_ablations.parquet"
        if not (info.exists() and f_abl.exists()):
            continue
        i = json.loads(info.read_text())
        abl = pd.read_parquet(f_abl)
        z = np.load(d / "bootstrap.npz")
        boot = dict(zip(z["keys"], z["means"].astype(np.float64)))
        x = abl[abl.variant == "non-directional"].iloc[0]
        lo, hi = _saved_interval(boot, f"{x.strategy}|{x.threshold}|{x.K}|{x.budget}|non-directional|ablation",
                                 f"{x.strategy}|{x.threshold}|{x.K}|{x.budget}|confidex")
        assert abs(lo - x.lo) < 1e-5 and abs(hi - x.hi) < 1e-5, (d.name, lo, x.lo, hi, x.hi)
        r = pd.read_parquet(d / "rows.parquet")
        m = r[r.budget.isin(set(abl.budget))].groupby(["strategy", "threshold", "K", "budget", "reading"])[ABL_COLS].mean()
        for (s, t, K, nb, reading), v in m.iterrows():
            if reading != PLUGIN:
                continue
            c = m.loc[(s, t, K, nb, "confidex")]
            lo, hi = _saved_interval(boot, f"{s}|{t}|{K}|{nb}|{PLUGIN}", f"{s}|{t}|{K}|{nb}|confidex")
            out.append({"strategy": s, "threshold": t, "K": K, "budget": nb, "variant": PLUGIN, **v.to_dict(),
                        "confidex": c.captured, "k_confidex": c.k, "diff": v.captured - c.captured, "lo": lo, "hi": hi,
                        "dataset": i["dataset"], "partition": i["partition"]})
    return pd.DataFrame(out)


def member_summary(pm: pd.DataFrame) -> pd.DataFrame:
    """Performance of the members on the test part by dataset, construction and family: median, minimum and
    maximum over all the members of all the partitions."""
    g = pm.groupby(["dataset", "strategy", "family"])
    out = g.agg(members=("AUROC", "size"), partitions=("partition", "nunique"), test_prevalence=("test_prevalence", "mean"))
    for c in ("AUROC", "AUPRC", "Brier"):
        out[f"{c}_median"], out[f"{c}_min"], out[f"{c}_max"] = g[c].median(), g[c].min(), g[c].max()
    return out.reset_index()


def ensemble_summary(pe: pd.DataFrame) -> pd.DataFrame:
    """Ensembles of 36 members on the test part by dataset, construction and threshold: mean, minimum and
    maximum across partitions."""
    g = pe[pe.K == 36].groupby(["dataset", "strategy", "threshold"])
    out = g.agg(partitions=("partition", "nunique"))
    for c in ("test_prevalence", "AUROC", "AUPRC", "Brier", "tau", "F1", "precision", "recall", "FP", "FN",
              "share_positive"):
        out[f"{c}_mean"], out[f"{c}_min"], out[f"{c}_max"] = g[c].mean(), g[c].min(), g[c].max()
    return out.reset_index()


def data_summary() -> list[dict]:
    """Rows, frauds, variables and the four parts (rows and frauds, minimum and maximum across
    partitions), from the manifests of the partitions."""
    out = []
    for d in N_PARTITIONS:
        m = yaml.safe_load((PARTITION_DIR / f"{d}_manifest.yaml").read_text())
        parts = {}
        for part, share in m["proportions"].items():
            r = [x["rows"][part] for x in m["partitions"].values()]
            f = [x["frauds"][part] for x in m["partitions"].values()]
            parts[part] = {"share": share, "rows_min": min(r), "rows_max": max(r), "frauds_min": min(f), "frauds_max": max(f)}
        out.append({"dataset": d, "file": m["file"], "sha256": m["sha256_dataset"], "rows": m["rows"],
                    "frauds": m["frauds"], "variables": m["variables"], "partitions": len(m["partitions"]),
                    "groups": m["groups"], "parts": parts})
    return out


def most_chosen(df: pd.DataFrame, groups: list[str], col: str) -> pd.Series:
    return df.groupby(groups)[col].agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))


def main(version: str = "final") -> None:
    directory = RESULTS / EVALUATIONS[version]
    out_dir = directory / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = Reader(directory)
    complete = {d: complete_tabpfn_partitions(d) for d in ("carclaims", "ulb")}
    rows = rd.table("rows")
    comp = keep_main(rd.table("comparison"), complete)
    comp_c = keep_main(rd.table("comparison_within_class"), complete)
    comp_a = keep_main(rd.table("comparison_allocation"), complete)
    comp_abl = rd.table("comparison_ablations")
    if not comp_abl.empty:
        comp_abl = pd.concat([comp_abl, plugin_comparison(directory)], ignore_index=True)
    draws = keep_main(rd.table("draws"), complete)

    # share captured, FP, FN and share of budget on positives of the chosen reading, per partition
    rm = (rows.groupby(["dataset", "partition", "strategy", "threshold", "K", "budget", "reading"])
          [["captured", "FP_captured", "FN_captured", "share_budget_positive"]].mean())
    sel = comp.apply(lambda r: rm.loc[(r.dataset, r.partition, r.strategy, r.threshold, r.K, r.budget,
                                       r.validation_choice)], axis=1)
    comp = comp.assign(**{f"choice_{c}": sel[c] for c in sel.columns})

    main_ = summary(comp, KEYS + ["budget"], "diff_val", "lo_val", "hi_val",
                    means=["confidex", *SIMPLE, "choice_captured", "k", "random", "perfect", "FP_captured",
                           "FN_captured", "share_budget_positive", "choice_FP_captured", "choice_FN_captured",
                           "choice_share_budget_positive"])
    main_ = main_.merge(most_chosen(comp, KEYS + ["budget"], "validation_choice").rename("choices").reset_index(),
                        on=KEYS + ["budget"])
    conservative = summary(comp, KEYS + ["budget"], "diff_test", "lo_test", "hi_test")
    within = summary(comp_c, KEYS + ["class", "budget"], "diff_val", "lo_val", "hi_val",
                     means=["confidex", "k", "n_class", "errors_class", "random", "perfect"])
    within = within.merge(most_chosen(comp_c, KEYS + ["class", "budget"], "validation_choice")
                          .rename("choices").reset_index(), on=KEYS + ["class", "budget"])
    alloc = summary(comp_a, KEYS + ["budget"], "diff", "lo", "hi",
                    means=["confidex", "allocation", "k_positive", "k_negative", "random", "perfect"])
    abl = (summary(comp_abl, KEYS + ["budget", "variant"], "diff", "lo", "hi",
                   means=["captured", "confidex", "k", "k_confidex", "random", "perfect"])
           if not comp_abl.empty else pd.DataFrame())
    gate = rd.table("comparison_gate")
    gate_sum = (summary(gate, KEYS + ["budget"], "diff", "lo", "hi",
                        means=["captured", "allocation", "k", "k_allocation", "share_abstention",
                               "share_abstained_positive", "share_positive", "errors_among_abstained"])
                if not gate.empty else pd.DataFrame())
    diag_test = (draws.groupby(KEYS)[["share_degenerate", "errors_degenerate", "share_at_border",
                                      "share_opposite_side", "regime_unimodal", "regime_J", "regime_U",
                                      "e_val", "share_positive_test", "errors", "FP", "FN", "k_alpha",
                                      "k_rho=1", "tau"]].mean().reset_index())

    # performance, and diagnostics on the validation part
    perf = {}
    for name in ("members", "ensemble", "reliability"):
        parts = [pd.read_parquet(f) for f in sorted((directory / "performance").glob(f"*/{name}.parquet"))]
        perf[name] = keep_main(pd.concat(parts, ignore_index=True), complete) if parts else pd.DataFrame()
    from fraud.members import partition_dir
    diag = []
    for d, ps in complete.items():
        for p in ps:
            f = partition_dir(d, p, version) / "diagnostics.yaml"
            if not f.exists():
                continue
            y = yaml.safe_load(f.read_text())
            for s, x in y["validation_diagnostics"].items():
                diag.append({"dataset": d, "partition": p, "strategy": s,
                             **{k: v for k, v in x.items() if not isinstance(v, dict)},
                             **{f"clip_{k}": v for k, v in x["share_at_clip_by_family"].items()},
                             "tau_F1_K36": x["validation_thresholds"]["tau_F1"]["tau"],
                             "grid_bound": y["grid"]["grid_bound"]})
    diag = pd.DataFrame(diag)

    tables = {"main": main_, "conservative": conservative, "within_class": within, "allocation": alloc,
              "ablations": abl, "diagnostics_test": diag_test, "diagnostics_validation": diag,
              "performance_members": perf["members"], "performance_ensemble": perf["ensemble"],
              "reliability": perf["reliability"], "comparison_by_partition": comp,
              "within_class_by_partition": comp_c, "allocation_by_partition": comp_a,
              "ablations_by_partition": comp_abl, "gate_allocation": gate_sum, "gate_by_partition": gate}
    for name, df in tables.items():
        if not df.empty:
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
    done = [json.loads(f.read_text()) for f in (directory / "jobs").glob("*/done.json")]
    result = {
        "members": version,
        "complete_tabpfn_partitions": complete,
        "jobs_completed": len(done),
        "main": main_.to_dict(orient="records"),
        "within_class": within.to_dict(orient="records"),
        "allocation": alloc.to_dict(orient="records"),
        "ablations": abl.to_dict(orient="records") if not abl.empty else [],
        "gate_allocation": gate_sum.to_dict(orient="records") if not gate_sum.empty else [],
        "performance_members": member_summary(perf["members"]).to_dict(orient="records"),
        "performance_ensemble": ensemble_summary(perf["ensemble"]).to_dict(orient="records"),
        "data": data_summary(),
    }
    (RESULTS / RESULT_FILES[version]).write_text(json.dumps(result, indent=1, default=float))
    print("summary written:", {k: len(v) for k, v in tables.items()}, "->", RESULTS / RESULT_FILES[version])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summary of an evaluation of Chapter 4.")
    ap.add_argument("--unconstrained", action="store_true")
    main("unconstrained" if ap.parse_args().unconstrained else "final")
