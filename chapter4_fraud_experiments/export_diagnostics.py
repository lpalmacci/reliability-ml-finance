"""Step 6b: the diagnostics of the members and of the procedure cited in Chapter 4.

    python export_diagnostics.py [processes]    ->  results/chapter4_diagnostics.json,
                                                    results/member_validation_auroc.parquet

It trains and calibrates nothing: it reads the saved pools and the results of the evaluations. Needs the
pools of both versions, results/keff_checks.parquet (keff_checks.py), and the summaries of both
evaluations (summarise.py, with and without --unconstrained).

- keff: on the validation part, full pool (K = 36); mean, minimum and maximum across partitions. Second
  construction, unconstrained and final LightGBM members: from keff_checks.parquet. First construction:
  ``confidex.keff_deviations`` on the pools, which must agree with keff_s1 of keff_checks.parquet; with
  one family, rho_within is the mean correlation of the deviations between pairs of members.
- member_auroc: by family, median, minimum and maximum over the members of all the partitions together.
  validation: ``roc_auc_score`` on the calibrated probabilities of the validation part (also written to
  member_validation_auroc.parquet). test: from the performance of the evaluations (calibrated
  probabilities after clipping), the unconstrained members from the comparison evaluation.
- classes_without_errors: every (dataset, construction, partition, threshold, K, draw, class) of the main
  evaluation in which the predicted class has no error on the validation part (share captured undefined
  for every reading); with the reading used for that class and the budget of the class on the test part
  at rho = 1.
- validation_pool: by dataset and construction, the pool of 36 members on the validation part: share of
  probabilities at the clipping constant, also by family, share of degenerate cases and border_share, with
  ``fraud.members.validation_diagnostics`` (recomputed and compared with the saved diagnostics.yaml).
- test_regimes: by dataset, construction and K, the shape of the reconstructed law on the test part
  (unimodal, J, U), from the columns regime_* of the saved draws.
- calibrations: by dataset, construction and partition, the calibration methods recorded by
  ``assemble_pool``.
- saturation: ULB only, at tau_F1, by construction and K, on the test part: share of saturated cases
  (degenerate, or with u_dir >= 1/2 - 1e-9) and mean number of non-saturated cases. The posteriors are
  recomputed as in ``evaluate_strategy`` (same draws, same tau_F1, same fit_posterior with the two
  thresholds), in parallel; for every draw tau_F1, k_alpha and the share of degenerate cases must agree
  with those saved by the evaluation.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

import confidex
from fraud.data import N_PARTITIONS
from fraud.evaluate import K_LEVELS, draws
from fraud.members import clip, load_pool, partition_dir, pool_path, validation_diagnostics
from fraud.registry import RESULTS, ROOT
from fraud.thresholds import cost_tau, select_tau

FAMILIES = ("LightGBM", "XGBoost", "CatBoost", "RandomForest", "ExtraTrees", "LogReg", "MLP", "SVM_RBF", "TabPFN")
S1_FAMILY = "LightGBM_constrained"
DS = ("carclaims", "ulb")
STRATEGIES = ("s1", "s2")
EPS_SATURATED = 1e-9
READ: dict[str, str] = {}
MAIN = RESULTS / "evaluation"
UNCONSTRAINED = RESULTS / "evaluation_unconstrained_lightgbm"


def sha256(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def record(f: Path) -> Path:
    READ[str(f.relative_to(ROOT))] = sha256(f)
    return f


def parquet(f: Path) -> pd.DataFrame:
    return pd.read_parquet(record(f))


def pool(d: str, p: int, s: str, v: str) -> dict[str, np.ndarray]:
    """Load a pool and record the file read (the first construction exists in the final version only)."""
    record(pool_path(d, p, s, "final" if s == "s1" else v))
    return load_pool(d, p, s, v)


def three(x) -> dict[str, float]:
    x = pd.Series(x, dtype=float)
    return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}


def keff() -> list[dict]:
    k = parquet(RESULTS / "keff_checks.parquet")
    out = []
    for d in DS:
        for v in ("unconstrained", "final"):
            z = k[(k.dataset == d) & (k.version == v)]
            assert len(z) == N_PARTITIONS[d], (d, v)
            out.append({"dataset": d, "strategy": "s2", "version": v, "partitions": len(z),
                        "keff": three(z.keff), "keff_without_lightgbm": three(z.keff_without_LightGBM),
                        "rho_within": three(z.rho_within), "rho_between": three(z.rho_between),
                        "share_squared_lightgbm": three(z.share_squared_deviations_LightGBM)})
        r = []
        for p in range(N_PARTITIONS[d]):
            pl = pool(d, p, "s1", "final")
            x = confidex.keff_deviations(clip(pl["P_validation"]), pl["families"])
            ref = k[(k.dataset == d) & (k.partition == p)].keff_s1.unique()
            assert len(ref) == 1 and abs(ref[0] - x["keff"]) < 1e-9, (d, p, ref, x["keff"])
            r.append(x)
        out.append({"dataset": d, "strategy": "s1", "version": "final", "partitions": len(r),
                    "keff": three([x["keff"] for x in r]), "keff_without_lightgbm": None,
                    "rho_within": three([x["rho_within"] for x in r]), "rho_between": None,
                    "share_squared_lightgbm": None})
    return out


def auroc_row(d, s, v, part, fam, x) -> dict:
    x = pd.Series(x, dtype=float)
    return {"dataset": d, "strategy": s, "version": v, "part": part, "family": fam, "n_members": int(len(x)),
            "median": float(x.median()), "min": float(x.min()), "max": float(x.max())}


def member_auroc() -> list[dict]:
    val = []
    for d in DS:
        for p in range(N_PARTITIONS[d]):
            for s, v in (("s2", "unconstrained"), ("s2", "final"), ("s1", "final")):
                pl = pool(d, p, s, v)
                yV = pl["y_validation"].astype(int)
                for j, (name, fam) in enumerate(zip(pl["names"], pl["families"])):
                    val.append({"dataset": d, "partition": p, "strategy": s, "version": v, "member": str(name),
                                "family": str(fam), "AUROC": roc_auc_score(yV, pl["P_validation"][:, j])})
    val = pd.DataFrame(val)
    val.to_parquet(RESULTS / "member_validation_auroc.parquet", index=False)
    t_main = parquet(MAIN / "summary" / "performance_members.parquet")
    t_unc = parquet(UNCONSTRAINED / "summary" / "performance_members.parquet")
    test = {("s2", "unconstrained"): t_unc[t_unc.strategy == "s2"], ("s2", "final"): t_main[t_main.strategy == "s2"],
            ("s1", "final"): t_main[t_main.strategy == "s1"]}
    out = []
    for d in DS:
        for s, v in (("s2", "unconstrained"), ("s2", "final"), ("s1", "final")):
            for fam in (FAMILIES if s == "s2" else ("LightGBM",)):
                f = fam if s == "s2" else S1_FAMILY
                x = val[(val.dataset == d) & (val.strategy == s) & (val.version == v) & (val.family == f)]
                out.append(auroc_row(d, s, v, "validation", fam, x.AUROC))
                y = test[(s, v)]
                y = y[(y.dataset == d) & (y.family == f)]
                assert len(y) == len(x) > 0, (d, s, v, fam)
                out.append(auroc_row(d, s, v, "test", fam, y.AUROC))
    return out


def jobs(strategy: str) -> list[Path]:
    return sorted((MAIN / "jobs").glob(f"*_{strategy}_K*"))


def classes_without_errors() -> list[dict]:
    out = []
    for s in STRATEGIES:
        for job in jobs(s):
            info = json.loads(record(job / "done.json").read_text())
            assert info["strategy"] == s, job
            v = parquet(job / "validation_within_class.parquet")
            c = parquet(job / "within_class.parquet")
            sc = parquet(job / "choice_within_class.parquet").set_index(["strategy", "threshold", "K", "class"])
            keys = ["dataset", "partition", "strategy", "threshold", "K", "draw", "class"]
            for key, g in v.groupby(keys):
                if not g.captured.isna().all():
                    assert g.captured.notna().all(), key
                    continue
                assert (g.k == 0).all(), key
                d, p, s_, t, K, e, cl = key
                kt = c[(c.draw == e) & (c.threshold == t) & (c["class"] == cl) & (c.budget == "rho=1")].k.unique()
                assert len(kt) == 1, key
                out.append({"dataset": d, "strategy": s_, "partition": int(p), "threshold": t, "K": int(K),
                            "draw": int(e), "class": int(cl),
                            "reading": str(sc.loc[(s_, t, K, cl), "validation_choice"]), "k_test": int(kt[0])})
    return sorted(out, key=lambda r: (r["dataset"], r["strategy"], r["partition"], r["threshold"], r["K"], r["draw"], r["class"]))


def mean_over_partitions(df: pd.DataFrame, groups: list[str], cols: list[str]) -> pd.DataFrame:
    """Mean first over the draws of each partition, then over the partitions."""
    return df.groupby(groups + ["partition"])[cols].mean().groupby(groups)[cols].mean().reset_index()


def _pool_diagnostics(d: str, p: int, s: str) -> dict:
    return {"dataset": d, "partition": p, "strategy": s, **validation_diagnostics(load_pool(d, p, s, "final"))}


def validation_pool(n_jobs: int) -> list[dict]:
    from joblib import Parallel, delayed
    work = [(d, p, s) for d in DS for p in range(N_PARTITIONS[d]) for s in STRATEGIES]
    for d, p, s in work:
        record(pool_path(d, p, s, "final"))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_pool_diagnostics)(*x) for x in work)
    for r in rows:
        f = record(partition_dir(r["dataset"], r["partition"]) / "diagnostics.yaml")
        y = yaml.safe_load(f.read_text())["validation_diagnostics"][r["strategy"]]
        for c in ("share_degenerate", "share_probabilities_at_clip", "border_share", "share_at_clip_by_family"):
            assert y[c] == r[c], (f, c, y[c], r[c])
    out = []
    for d in DS:
        for s in STRATEGIES:
            z = [r for r in rows if (r["dataset"], r["strategy"]) == (d, s)]
            assert len(z) == N_PARTITIONS[d]
            fams = z[0]["share_at_clip_by_family"]
            out.append({"dataset": d, "strategy": s,
                        "at_clip": float(np.mean([r["share_probabilities_at_clip"] for r in z])),
                        "at_clip_by_family": {f: float(np.mean([r["share_at_clip_by_family"][f] for r in z])) for f in fams},
                        "degenerate": float(np.mean([r["share_degenerate"] for r in z])),
                        "border_share": float(np.mean([r["border_share"] for r in z]))})
    return out


def saved_draws(s: str) -> pd.DataFrame:
    parts = []
    for job in jobs(s):
        info = json.loads(record(job / "done.json").read_text())
        assert info["strategy"] == s, job
        parts.append(parquet(job / "draws.parquet"))
    return pd.concat(parts, ignore_index=True)


def test_regimes(dr: pd.DataFrame) -> list[dict]:
    reg = ["regime_unimodal", "regime_J", "regime_U"]
    keys = ["dataset", "strategy", "partition", "K", "draw"]
    assert (dr.groupby(keys)[reg].nunique() == 1).all().all(), "the regime depends on the threshold"
    e = dr[dr.threshold == "tau_F1"]
    assert all(len(e[(e.dataset == d) & (e.strategy == s)].partition.unique()) == N_PARTITIONS[d] for d in DS for s in STRATEGIES)
    m = mean_over_partitions(e, ["dataset", "strategy", "K"], reg)
    return [{"dataset": r.dataset, "strategy": r.strategy, "K": int(r.K), "unimodal": float(r.regime_unimodal),
             "J": float(r.regime_J), "U": float(r.regime_U)} for r in m.itertuples()]


def calibrations() -> list[dict]:
    out = []
    for d in DS:
        for s in STRATEGIES:
            for p in range(N_PARTITIONS[d]):
                c = json.loads(record(partition_dir(d, p) / f"{s}_calibration.json").read_text())
                assert len(c) == 36
                out.append({"dataset": d, "strategy": s, "partition": p, "methods": sorted({x["method"] for x in c})})
    return out


def _saturation(p: int, s: str, K: int) -> list[dict]:
    """As evaluate_strategy, ULB at tau_F1 only: saturated cases on the test part for every draw."""
    d = "ulb"
    cfg, crit = confidex.DEFAULT_MODEL, confidex.DEFAULT_CRITERION
    pl = load_pool(d, p, s, "final")
    PVall, PTall = clip(pl["P_validation"], cfg), clip(pl["P_test"], cfg)
    yV = pl["y_validation"].astype(int)
    out = []
    for K_, r, cols in draws(d, p, s, pl["families"], (K,)):
        taus = {"tau_F1": select_tau(PVall[:, cols], yV), "tau_cost": cost_tau(d)}
        post = confidex.fit_posterior(PTall[:, cols], cfg, taus=list(taus.values()))
        o = confidex.confidence_score(post, None, taus["tau_F1"], cfg=crit)
        saturated = post.degenerate | (o["u_dir"] >= 0.5 - EPS_SATURATED)
        out.append({"dataset": d, "strategy": s, "partition": p, "threshold": "tau_F1", "K": K_, "draw": r,
                    "tau": taus["tau_F1"], "share_at_bound": float(saturated.mean()),
                    "k_not_saturated": int((~saturated).sum()),
                    "k_alpha": int((o["status"] == "not_confident").sum()),
                    "share_degenerate": float(post.degenerate.mean())})
    return out


def saturation(n_jobs: int, dr: pd.DataFrame) -> list[dict]:
    from joblib import Parallel, delayed
    work = [(p, s, K) for p in range(N_PARTITIONS["ulb"]) for s in STRATEGIES for K in K_LEVELS]
    rows = pd.DataFrame([r for x in Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_saturation)(*w) for w in work) for r in x])
    keys = ["dataset", "strategy", "partition", "threshold", "K", "draw"]
    m = rows.merge(dr[dr.threshold == "tau_F1"], on=keys, suffixes=("", "_saved"), validate="one_to_one")
    assert len(m) == len(rows), (len(m), len(rows))
    assert (m.tau == m.tau_saved).all() and (m.k_alpha == m.k_alpha_saved).all() \
        and (m.share_degenerate == m.share_degenerate_saved).all(), "posteriors differ from those of the evaluation"
    sm = mean_over_partitions(rows, ["dataset", "strategy", "threshold", "K"], ["share_at_bound", "k_not_saturated"])
    return [{"dataset": r.dataset, "strategy": r.strategy, "threshold": r.threshold, "K": int(r.K),
             "share_at_bound": float(r.share_at_bound), "k_not_saturated": float(r.k_not_saturated)}
            for r in sm.itertuples()]


def main(n_jobs: int = 10) -> None:
    t0 = time.time()
    D = {"keff": keff(), "member_auroc": member_auroc(), "classes_without_errors": classes_without_errors()}
    D["validation_pool"] = validation_pool(n_jobs)
    dr = pd.concat([saved_draws(s) for s in STRATEGIES], ignore_index=True)
    D["test_regimes"] = test_regimes(dr)
    D["calibrations"] = calibrations()
    D["saturation"] = saturation(n_jobs, dr)
    D = {"provenance": {"files": dict(sorted(READ.items())), "script_sha256": sha256(Path(__file__).resolve())}, **D}
    (RESULTS / "chapter4_diagnostics.json").write_text(json.dumps(D, indent=1))
    ce = D["classes_without_errors"]
    print({k: len(v) for k, v in D.items() if isinstance(v, list)}, "files read:", len(READ))
    print("classes without errors on validation:", len(ce), "; with k_test other than 0:",
          [r for r in ce if r["k_test"] != 0] or "none")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
