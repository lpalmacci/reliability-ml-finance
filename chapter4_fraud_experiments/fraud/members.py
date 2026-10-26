"""Members, calibration and diagnostics on the validation part (Chapter 4, Members and their calibration).

- ``prepare(dataset, p)``: the four parts of the partition in the two paths of the variables; the
  preprocessor is fitted on the training part only.
- ``train_member``: subsample of the member, training, raw score and log-odds on the calibration,
  validation and test parts, saved in results/members/<dataset>_p<k>/<strategy>/<member>.npz. A member
  already saved is not trained again. TabPFN members are not queried here: their raw scores are read
  from results/tabpfn/<dataset>_p<k>/ (``query_tabpfn.py``) and their log-odds is
  logit(clip(r, 1e-6, 1 - 1e-6)).
- ``assemble_pool``: the 36 members of a construction, each calibrated on the calibration part by a
  logistic map of its LOG-ODDS, p = sigma(a z + b), fitted by maximum likelihood without penalty; if the
  fit does not converge (separation), Platt's corrected targets are used instead, and this is recorded.
  Then the clipping of the library. Raw scores, log-odds and calibrated probabilities are saved.
- ``validation_diagnostics`` and ``validation_thresholds``: diagnostics of the full pool (K = 36) on
  the validation part only; they are reported and do not block anything.

Two versions of the pools exist. ``"final"`` is the one of the thesis: in the second construction the
LightGBM members are constrained to leaves with a total Hessian of at least one. ``"unconstrained"``
rebuilds only those four LightGBM members of each partition without the constraint (the members that
degenerated on ULB), and takes the other 32 members of the second construction from the final pools; the
first construction is the same in the two versions. The comparison is reported in the section
"Decisions taken after the pilot and during the evaluation".
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import confidex
from fraud.data import load, load_partition
from fraud.families import logit_of_probability
from fraud.model_registry import family
from fraud.plan import Member, plan, subsample_indices
from fraud.preprocessor import Preprocessor
from fraud.registry import DATASETS, RESULTS
from fraud.thresholds import cost_tau, select_tau

MEMBER_DIRS = {"final": RESULTS / "members", "unconstrained": RESULTS / "members_unconstrained_lightgbm"}
VERSIONS = tuple(MEMBER_DIRS)
#: raw scores of the TabPFN members, shared by the two versions (never queried again)
TABPFN_DIR = RESULTS / "tabpfn"
PRED = ("calibration", "validation", "test")


@dataclass
class Prepared:
    dataset: str
    partition: int
    indices: dict[str, np.ndarray]
    X_raw: dict[str, pd.DataFrame]
    X_proc: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    cat_cols: list[str]
    cat_idx: list[int]
    n_processed_variables: int


def prepare(dataset: str, p: int) -> Prepared:
    ld = load(dataset)
    parts = load_partition(dataset, p)
    X_raw = {k: ld.X_raw.iloc[v].reset_index(drop=True) for k, v in parts.items()}
    pre = Preprocessor(DATASETS[dataset].ordinal_specs)
    X_proc = {"train": pre.fit_transform(X_raw["train"])}
    for k in PRED:
        X_proc[k] = pre.transform(X_raw[k])
    cat_cols = [c for c in ld.X_raw.columns if ld.X_raw[c].dtype == object]
    cat_idx = [ld.X_raw.columns.get_loc(c) for c in cat_cols]
    return Prepared(dataset, p, parts, X_raw, X_proc, {k: ld.y[v] for k, v in parts.items()},
                    cat_cols, cat_idx, int(X_proc["train"].shape[1]))


def partition_dir(dataset: str, p: int, version: str = "final") -> Path:
    return MEMBER_DIRS[version] / f"{dataset}_p{p}"


def is_unconstrained_lightgbm(m: Member, version: str) -> bool:
    """True for the members that the unconstrained version rebuilds: LightGBM of the second construction."""
    return version == "unconstrained" and m.strategy == "s2" and m.family == "LightGBM"


def member_path(m: Member, version: str = "final") -> Path:
    if m.family == "TabPFN":
        return TABPFN_DIR / f"{m.dataset}_p{m.partition}" / f"{m.name}.npz"
    v = "unconstrained" if is_unconstrained_lightgbm(m, version) else "final"
    return MEMBER_DIRS[v] / f"{m.dataset}_p{m.partition}" / m.strategy / f"{m.name}.npz"


def train_member(m: Member, prep: Prepared, version: str = "final") -> dict[str, Any]:
    """Train the member (unless it is already saved) and return its metadata."""
    out = member_path(m, version)
    meta_file = out.with_suffix(".json")
    if out.exists() and meta_file.exists():
        return json.loads(meta_file.read_text())
    if m.family == "TabPFN":
        raise FileNotFoundError(f"{out.name}: TabPFN members are produced by query_tabpfn.py")
    fam = family(m.family, unconstrained_lightgbm=is_unconstrained_lightgbm(m, version))
    yT = prep.y["train"]
    idx = subsample_indices(yT, m.fraction, m.subsample_seed)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fam.build(m.model_seed, m.auxiliary_seed, m.hyperparameters)
        fam.fit(model, X_proc=prep.X_proc["train"][idx],
                X_raw=prep.X_raw["train"].iloc[idx].reset_index(drop=True),
                y=yT[idx], cat_cols=prep.cat_cols, cat_idx=prep.cat_idx)
        t_fit = time.time() - t0
        scores = {k: np.asarray(fam.score(model, prep.X_proc[k], prep.X_raw[k]), float).ravel() for k in PRED}
        logodds = {k: np.asarray(fam.log_odds(model, prep.X_proc[k], prep.X_raw[k]), float).ravel() for k in PRED}
    t_tot = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"raw_{k}": v for k, v in scores.items()}, **{f"logodds_{k}": v for k, v in logodds.items()},
             subsample_indices=prep.indices["train"][idx])
    meta = {**m.to_dict(), "n_train": int(len(idx)), "train_frauds": int(yT[idx].sum()),
            "fit_seconds": round(t_fit, 2), "total_seconds": round(t_tot, 2),
            "warnings": sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught})}
    meta_file.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def member_scores(m: Member, version: str = "final") -> dict[str, dict[str, np.ndarray]]:
    """Raw scores and log-odds of the member on the three parts."""
    with np.load(member_path(m, version)) as z:
        raw = {k: z[f"raw_{k}"] for k in PRED}
        if m.family == "TabPFN":
            logodds = {k: logit_of_probability(v) for k, v in raw.items()}
        else:
            logodds = {k: z[f"logodds_{k}"] for k in PRED}
    return {"raw": raw, "logodds": logodds}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _platt_targets(z: np.ndarray, y: np.ndarray) -> tuple[float, float, bool]:
    """Logistic fit with Platt's corrected targets, (N+ + 1)/(N+ + 2) and 1/(N- + 2): always well posed."""
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    t = np.where(y == 1, (n1 + 1.0) / (n1 + 2.0), 1.0 / (n0 + 2.0))

    def f(ab):
        s = ab[0] * z + ab[1]
        # -[t log sigma(s) + (1 - t) log(1 - sigma(s))], in a numerically stable form
        return float(np.sum(np.logaddexp(0.0, s) - t * s))

    def g(ab):
        r = expit(ab[0] * z + ab[1]) - t
        return np.array([np.sum(r * z), np.sum(r)])

    x0 = np.array([0.0, np.log((n1 + 1.0) / (n0 + 1.0))])
    res = minimize(f, x0, jac=g, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    return float(res.x[0]), float(res.x[1]), bool(res.success)


def platt_logodds(zC: np.ndarray, yC: np.ndarray) -> dict[str, Any]:
    """Fit (a, b) of p = sigma(a z + b) on the calibration part.

    Maximum likelihood without penalty. It does not converge, and Platt's targets are used instead, if
    the classes are separated or the optimiser does not converge (warning, iterations exhausted,
    non-finite coefficients). In one dimension the maximum-likelihood estimate exists if and only if the
    ranges of z in the two classes overlap: with complete or quasi-complete separation ``lbfgs`` stops on
    large coefficients WITHOUT warnings, because the gradient tends to zero, so separation is checked
    directly."""
    zC = np.asarray(zC, dtype=float).ravel()
    yC = np.asarray(yC).astype(int)
    z0, z1 = zC[yC == 0], zC[yC == 1]
    separation = bool(z0.max() <= z1.min() or z1.max() <= z0.min())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lr = LogisticRegression(C=np.inf).fit(zC[:, None], yC)       # C=inf: no penalty (penalty=None is deprecated)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    n_iter = int(lr.n_iter_[0])
    optimiser = (any(issubclass(w.category, ConvergenceWarning) for w in caught)
                 or n_iter >= lr.max_iter or not (np.isfinite(a) and np.isfinite(b)))
    if not (separation or optimiser):
        return {"a": a, "b": b, "method": "maximum_likelihood", "n_iter": n_iter}
    a2, b2, ok = _platt_targets(zC, yC)
    return {"a": a2, "b": b2, "method": "platt_targets", "n_iter": n_iter,
            "reason": "separation" if separation else "optimiser",
            "a_not_converged": a, "b_not_converged": b, "platt_targets_converged": ok,
            "warnings": sorted({str(w.message)[:160] for w in caught})}


def apply_platt(cal: dict[str, Any], z: np.ndarray) -> np.ndarray:
    return expit(cal["a"] * np.asarray(z, dtype=float) + cal["b"])


def clip(P: np.ndarray, cfg: confidex.ModelConfig = confidex.DEFAULT_MODEL) -> np.ndarray:
    """The clipping of the library, the same that ``fit_posterior`` applies before inference."""
    return np.clip(P, cfg.clip_eps, 1.0 - cfg.clip_eps)


def pool_path(dataset: str, p: int, strategy: str, version: str = "final") -> Path:
    return partition_dir(dataset, p, version) / f"{strategy}_pool.npz"


def assemble_pool(dataset: str, p: int, strategy: str, prep: Prepared, version: str = "final") -> dict[str, Any]:
    """Calibrate the 36 members on their log-odds and save the pool."""
    members = plan(dataset, p, strategy)
    G = {k: [] for k in PRED}
    Z = {k: [] for k in PRED}
    for m in members:
        s = member_scores(m, version)
        for k in PRED:
            G[k].append(s["raw"][k])
            Z[k].append(s["logodds"][k])
    G = {k: np.column_stack(v) for k, v in G.items()}
    Z = {k: np.column_stack(v) for k, v in Z.items()}
    yC = prep.y["calibration"]
    P = {k: np.zeros_like(Z[k]) for k in PRED}
    calibrations = []
    for j, m in enumerate(members):
        cal = platt_logodds(Z["calibration"][:, j], yC)
        calibrations.append({"member": m.name, "family": m.family, **cal})
        for k in PRED:
            P[k][:, j] = apply_platt(cal, Z[k][:, j])
    pool = {**{f"raw_{k}": G[k] for k in PRED}, **{f"logodds_{k}": Z[k] for k in PRED},
            **{f"P_{k}": P[k] for k in PRED}, **{f"y_{k}": prep.y[k] for k in PRED},
            "families": np.array([m.family for m in members]),
            "names": np.array([m.name for m in members])}
    out = pool_path(dataset, p, strategy, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **pool)
    out.with_name(f"{strategy}_calibration.json").write_text(json.dumps(calibrations, indent=2))
    return pool


def load_pool(dataset: str, p: int, strategy: str, version: str = "final") -> dict[str, np.ndarray]:
    if version == "unconstrained" and strategy == "s1":
        version = "final"                    # the first construction is the same in the two versions
    with np.load(pool_path(dataset, p, strategy, version), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Diagnostics on the validation part
# ---------------------------------------------------------------------------

def validation_diagnostics(pool: dict[str, np.ndarray], cfg: confidex.ModelConfig = confidex.DEFAULT_MODEL) -> dict[str, Any]:
    """Non-blocking diagnostics of the full pool (K = 36) on the validation part."""
    P = pool["P_validation"]
    eps = cfg.clip_eps
    post = confidex.fit_posterior(clip(P, cfg), cfg)
    fam = pool["families"]
    at_clip = (P <= eps) | (P >= 1 - eps)
    k = confidex.keff_deviations(clip(P, cfg), fam)
    return {
        "n_validation": int(P.shape[0]),
        "share_degenerate": float(post.degenerate.mean()),
        "share_probabilities_at_clip": float(at_clip.mean()),
        "share_at_clip_by_family": {str(f): float(at_clip[:, fam == f].mean()) for f in np.unique(fam)},
        "border_share": confidex.border_share(post),
        "keff": k["keff"], "keff_rho_within": k["rho_within"], "keff_rho_between": k["rho_between"],
        "min_distinct_values_per_member": int(min(len(np.unique(P[:, j])) for j in range(P.shape[1]))),
    }


def validation_thresholds(pool: dict[str, np.ndarray], dataset: str,
                          cfg: confidex.ModelConfig = confidex.DEFAULT_MODEL) -> dict[str, Any]:
    """tau_F1 of the full pool, share of positives and error rate at the two thresholds (validation only)."""
    PV = clip(pool["P_validation"], cfg)
    yV = pool["y_validation"].astype(int)
    pbar = PV.mean(axis=1)
    out = {"prevalence_validation": float(yV.mean()),
           "pbar_quantiles": {f"q{int(q * 100):02d}": float(np.quantile(pbar, q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)}}
    for name, tau in (("tau_F1", select_tau(PV, yV)), ("tau_cost", cost_tau(dataset))):
        yhat = pbar >= tau
        err = yhat != yV
        out[name] = {"tau": float(tau), "share_positive": float(yhat.mean()), "error_rate": float(err.mean()),
                     "FP": int((err & yhat).sum()), "FN": int((err & ~yhat).sum())}
    return out
