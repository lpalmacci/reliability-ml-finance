"""Descriptive performance of the members and of the ensembles (Chapter 4, Members and ensembles).

On the test part, for every calibrated member: AUROC, AUPRC (average precision) and Brier score. For the
ensemble of 36 members (pbar = mean after clipping): the same three metrics and, at each threshold, F1,
precision, recall, false positives, missed frauds and share of the cases classified as positive; a
reliability curve of pbar with 10 bins by quantiles. They do not enter the comparison between readings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score, recall_score,
                             roc_auc_score)

from fraud.members import clip, load_pool
from fraud.thresholds import cost_tau, implied_ratio, select_tau


def _three(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}


def performance(dataset: str, p: int, strategies=("s1", "s2"), version: str = "final") -> dict[str, pd.DataFrame]:
    members, ens, rel = [], [], []
    for s in strategies:
        pool = load_pool(dataset, p, s, version)
        PT, PV = clip(pool["P_test"]), clip(pool["P_validation"])
        yT, yV = pool["y_test"].astype(int), pool["y_validation"].astype(int)
        prev = float(yT.mean())
        for j, (name, fam) in enumerate(zip(pool["names"], pool["families"])):
            members.append({"strategy": s, "member": str(name), "family": str(fam), "test_prevalence": prev,
                            **_three(yT, PT[:, j])})
        pbar = PT.mean(axis=1)
        base = {"strategy": s, "K": PT.shape[1], "test_prevalence": prev, **_three(yT, pbar)}
        for tau_name, tau in (("tau_F1", select_tau(PV, yV)), ("tau_cost", cost_tau(dataset))):
            yhat = (pbar >= tau).astype(int)
            ens.append({**base, "threshold": tau_name, "tau": tau, "implied_ratio": implied_ratio(tau),
                        "F1": f1_score(yT, yhat, zero_division=0),
                        "precision": precision_score(yT, yhat, zero_division=0),
                        "recall": recall_score(yT, yhat, zero_division=0),
                        "FP": int(((yhat == 1) & (yT == 0)).sum()),
                        "FN": int(((yhat == 0) & (yT == 1)).sum()),
                        "share_positive": float(yhat.mean())})
        bins = pd.qcut(pbar, 10, labels=False, duplicates="drop")
        for c in np.unique(bins):
            m = bins == c
            rel.append({"strategy": s, "bin": int(c), "n": int(m.sum()), "mean_pbar": float(pbar[m].mean()),
                        "observed_frequency": float(yT[m].mean())})
    members = pd.DataFrame(members)
    by_family = members.groupby(["strategy", "family"])[["AUROC", "AUPRC", "Brier"]].agg(["median", "min", "max"])
    by_family.columns = [f"{a}_{b}" for a, b in by_family.columns]
    by_strategy = members.groupby("strategy")[["AUROC", "AUPRC", "Brier"]].agg(["median", "min", "max"])
    by_strategy.columns = [f"{a}_{b}" for a, b in by_strategy.columns]
    return {"members": members, "summary_by_family": by_family.reset_index(),
            "summary_by_strategy": by_strategy.reset_index(),
            "ensemble": pd.DataFrame(ens), "reliability": pd.DataFrame(rel)}
