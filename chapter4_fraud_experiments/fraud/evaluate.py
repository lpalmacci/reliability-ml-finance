"""Evaluation of one condition (Chapter 4, Ensembles, thresholds and errors; Budgets and comparisons).

For a dataset, a partition and a construction:

- draws: K in {3, 6, 12, 24, 36}; R = 20 draws for each K < 36, the pool itself at K = 36; generator
  ``rng_for("draw", dataset, p, strategy, K, r)``; uniform draws in the first construction, family by
  family in rotation in the second (``draw_members``);
- for every draw: tau_F1 on the validation part and tau_cost; one ``fit_posterior`` on the test part with
  ``taus=[tau_F1, tau_cost]``; the verdict with the predictive law and the directional criterion, and the
  robustness row with the law at the posterior mean of c;
- error = 1[pbar >= tau] != y, with pbar = post.mu (the mean after clipping);
- budgets: k_rho = round(rho * e_val * n_test), with e_val the error rate of the same draw at the same
  threshold on the validation part; k_alpha = the cases the verdict declares not confident; fixed shares
  of 1%, 5% and 10% of the cases;
- metrics for every reading and budget: share of the errors captured, precision, false positives and
  missed frauds captured, share of the budget spent on the cases classified as positive; AUROC on the
  errors;
- the simpler reading chosen on the validation part (share captured at rho = 1, mean over the draws), for
  every threshold, construction and K;
- within each predicted class: among yhat = 1 (errors = false positives) and among yhat = 0 (missed
  frauds), with budget k_c = round(rho * e_val_c * n_test_c) and the simpler reading chosen on the
  validation part within the class; the allocated reading: every simpler reading with the overall budget
  k_rho divided between the classes in proportion to their errors on the validation part, cases ordered
  within each class. The same at k_alpha (within a class: the cases the verdict flags in that class);
- room of the budget: next to every share captured, that of a random ordering, k / n, and of a perfect
  one, min(1, k / E), with n cases and E errors in the set considered;
- ablations (``fraud.ablations``), if requested: variants of the order of Confidex on the same errors and
  budgets;
- paired bootstrap of the test cases: B = 1000 vectors of multinomial weights from the generator
  ``rng_for("bootstrap", dataset, p)``, the same for all readings, draws, thresholds and constructions.
  For every (K, threshold, budget, reading) the mean over the draws of the share captured in each
  replicate is kept: the difference between two readings is the difference of the two vectors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import confidex
from fraud import ablations as abl
from fraud.members import clip, load_pool
from fraud.readings import READINGS, SIMPLE, auroc, baselines, captured, order
from fraud.seeds import key, rng_for
from fraud.thresholds import cost_tau, implied_ratio, select_tau

K_LEVELS: tuple[int, ...] = (3, 6, 12, 24, 36)
R = 20
RHO: tuple[float, ...] = (0.5, 1.0, 2.0)
SHARES: tuple[float, ...] = (0.01, 0.05, 0.10)
B = 1000
THRESHOLDS: tuple[str, ...] = ("tau_F1", "tau_cost")


def draw_members(M: int, K: int, families: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """Draw K members from the pool.

    Uniform when the members are exchangeable; stratified by family with round-robin allocation
    otherwise, so that an ensemble of size K uses ``min(K, number of families)`` distinct families before
    repeating one.
    """
    if K > M:
        raise ValueError(f"K={K} cannot exceed M={M}")
    if families is None or len(np.unique(families)) < 2:
        return rng.choice(M, K, replace=False)
    family_order = rng.permutation(np.unique(families))
    chosen: list[int] = []
    for f in np.resize(family_order, K):
        pool = [c for c in np.where(families == f)[0] if c not in chosen]
        if not pool:
            pool = [c for c in range(M) if c not in chosen]
        chosen.append(int(rng.choice(pool)))
    return np.asarray(chosen)


def draw(strategy: str, M: int, K: int, families: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if K == M:
        return np.arange(M)
    if strategy == "s1":
        return rng.choice(M, K, replace=False)
    return draw_members(M, K, families, rng)


def draws(dataset: str, p: int, strategy: str, families: np.ndarray,
          K_levels: tuple[int, ...] = K_LEVELS) -> list[tuple[int, int, np.ndarray]]:
    M = len(families)
    out = []
    for K in K_levels:
        for r in range(R if K < M else 1):
            out.append((K, r, draw(strategy, M, K, families, rng_for("draw", dataset, p, strategy, K, r))))
    return out


def bootstrap_weights(dataset: str, p: int, n: int, B_: int = B) -> np.ndarray:
    """B vectors of multinomial weights on the test cases (summing to n), the same for all conditions."""
    rng = rng_for("bootstrap", dataset, p)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B_).astype(np.float32)


def split_budget(k: int, E1: int, E0: int, n1: int, n0: int) -> tuple[int, int]:
    """Allocated reading: k divided between yhat = 1 and yhat = 0 in proportion to the errors of each class
    on the validation part (E1 false positives, E0 missed frauds), within the size of each class."""
    share1 = E1 / (E1 + E0) if (E1 + E0) else 0.5
    k1 = min(int(round(k * share1)), n1)
    return k1, min(k - k1, n0)


def order_within(idx: np.ndarray, doubt: np.ndarray) -> np.ndarray:
    """Global indices of the cases of the class ``idx``, from the most doubtful."""
    return idx[order(doubt[idx])]


def room(k: int, n: int, E: int) -> dict[str, float]:
    """Room of the budget: share captured by a random ordering (k / n) and by a perfect one (min(1, k / E))."""
    return {"random": k / n if n else float("nan"), "perfect": min(1.0, k / E) if E else float("nan")}


def budgets(e_val: float, n_test: int, k_alpha: int) -> dict[str, int]:
    out = {f"rho={r:g}": int(round(r * e_val * n_test)) for r in RHO}
    out["k_alpha"] = int(k_alpha)
    out.update({f"share={q:.0%}": int(round(q * n_test)) for q in SHARES})
    return out


@dataclass
class Results:
    rows: list[dict[str, Any]] = field(default_factory=list)              # reading x budget
    within_class: list[dict[str, Any]] = field(default_factory=list)      # within each predicted class
    validation_within_class: list[dict[str, Any]] = field(default_factory=list)
    allocation: list[dict[str, Any]] = field(default_factory=list)        # allocated reading
    ablations: list[dict[str, Any]] = field(default_factory=list)         # variants of Confidex
    draws: list[dict[str, Any]] = field(default_factory=list)             # draw x threshold
    validation: list[dict[str, Any]] = field(default_factory=list)        # simpler readings on validation
    boot: dict[tuple, np.ndarray] = field(default_factory=dict)           # sum over the draws
    n_draws: dict[tuple, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def evaluate_strategy(dataset: str, p: int, strategy: str, W: np.ndarray, res: Results,
                      cfg: confidex.ModelConfig = confidex.DEFAULT_MODEL,
                      crit: confidex.CriterionConfig = confidex.DEFAULT_CRITERION,
                      version: str = "final", K_levels: tuple[int, ...] = K_LEVELS,
                      ablations: bool = False, gate_only: bool = False) -> None:
    pool = load_pool(dataset, p, strategy, version)
    PVall, PTall = clip(pool["P_validation"], cfg), clip(pool["P_test"], cfg)
    yV, yT = pool["y_validation"].astype(int), pool["y_test"].astype(int)
    families = pool["families"]
    nV, nT = len(yV), len(yT)
    t_post = t_rest = t_abl = 0.0
    for K, r, cols in draws(dataset, p, strategy, families, K_levels):
        t0 = time.time()
        PV, PT = PVall[:, cols], PTall[:, cols]
        taus = {"tau_F1": select_tau(PV, yV), "tau_cost": cost_tau(dataset)}
        post = confidex.fit_posterior(PT, cfg, taus=list(taus.values()))
        t1 = time.time(); t_post += t1 - t0
        extra = (abl.extra_posteriors(PV, PT, post, K, list(taus.values()), cfg, with_clip=(dataset == "ulb"),
                                      gate_only=gate_only)
                 if ablations else None)
        t2 = time.time(); t_abl += t2 - t1; t1 = t2
        law = confidex.member_law(post)
        regime = confidex.shape_regime(law)
        pbarV = PV.mean(axis=1)
        base = {"dataset": dataset, "partition": p, "strategy": strategy, "K": K,
                "draw": r, "members": " ".join(map(str, cols)),
                "key": " ".join(map(str, key("draw", dataset, p, strategy, K, r)))}
        for tau_name, tau in taus.items():
            # validation: e_val and choice of the simpler reading (rho = 1)
            errV = ((pbarV >= tau) != yV).astype(int)
            e_val = float(errV.mean())
            kV = int(round(1.0 * e_val * nV))
            doubtV = baselines(PV, pbarV, tau)
            for name in SIMPLE:
                res.validation.append({**base, "threshold": tau_name, "tau": tau, "reading": name,
                                       "k": kV, "captured": captured(order(doubtV[name]), errV, kV)})
            # test
            o = confidex.confidence_score(post, None, tau, cfg=crit)
            op = confidex.confidence_score(post, law, tau, cfg=crit, predictive=False)
            yhat = post.mu >= tau
            err = (yhat != yT).astype(int)
            fp, fn = err & yhat, err & ~yhat
            u = np.where(post.degenerate, 0.5, o["u_dir"])
            u_plug = np.where(post.degenerate, 0.5, op["u_dir"])
            doubt = {"confidex": -u, **baselines(PT, post.mu, tau),
                     "s": PT.std(axis=1, ddof=1), "c mean": -post.c_mean,
                     "confidex plug-in": -u_plug}
            k_alpha = int((o["status"] == "not_confident").sum())
            k_alpha_plug = int((op["status"] == "not_confident").sum())
            bud = budgets(e_val, nT, k_alpha)
            res.draws.append({
                **base, "threshold": tau_name, "tau": tau, "implied_ratio": implied_ratio(tau),
                "e_val": e_val, "n_test": nT, "share_positive_test": float(yhat.mean()),
                "errors": int(err.sum()), "FP": int(fp.sum()), "FN": int(fn.sum()),
                "k_alpha": k_alpha, "k_alpha_plugin": k_alpha_plug,
                "agreement_verdict_plugin": float(np.mean(o["confident"] == op["confident"])),
                "share_degenerate": float(post.degenerate.mean()),
                "errors_degenerate": int(err[post.degenerate].sum()),
                "share_at_border": float(post.at_border.mean()),
                "share_opposite_side": float(o["opposite_side"].mean()),
                **{f"regime_{g}": float(np.mean(regime == g)) for g in ("unimodal", "J", "U")},
                **{f"k_{b}": k for b, k in bud.items()},
            })
            den = W @ err.astype(np.float32)
            columns, keys = [], []
            for name in READINGS:
                ordn = order(doubt[name])
                a = auroc(doubt[name], err)
                bud_r = dict(bud)
                if name == "confidex plug-in":
                    bud_r["k_alpha"] = k_alpha_plug
                for nb, k in bud_r.items():
                    top = ordn[:k]
                    res.rows.append({**base, "threshold": tau_name, "reading": name, "budget": nb,
                                     "k": k, "captured": captured(ordn, err, k),
                                     "precision": float(err[top].mean()) if k else float("nan"),
                                     "FP_captured": int(fp[top].sum()),
                                     "FN_captured": int(fn[top].sum()), "auroc": a,
                                     "share_budget_positive": float(yhat[top].mean()) if k else float("nan"),
                                     **room(k, nT, int(err.sum()))})
                    sel = np.zeros(nT, dtype=np.float32)
                    sel[top] = err[top]
                    columns.append(sel)
                    keys.append((strategy, tau_name, K, nb, name))
            dens = [den] * len(columns)
            # --- within each predicted class, and the allocated reading ---
            yhatV = pbarV >= tau
            classes = {}
            for c in (1, 0):
                mV, mT = yhatV == c, yhat == c
                errVc = errV[mV]
                e_val_c = float(errVc.mean()) if mV.any() else 0.0
                kVc = int(round(1.0 * e_val_c * int(mV.sum())))
                for name in SIMPLE:
                    res.validation_within_class.append({
                        **base, "threshold": tau_name, "class": c, "reading": name, "k": kVc,
                        "captured": captured(order(doubtV[name][mV]), errVc, kVc)})
                idxc = np.flatnonzero(mT)
                classes[c] = idxc
                nTc = len(idxc)
                errc_tot = int(err[idxc].sum())
                den_c = W @ (err * mT).astype(np.float32)
                bud_c = {f"rho={r_:g}": min(int(round(r_ * e_val_c * nTc)), nTc) for r_ in RHO}
                bud_c["k_alpha"] = int(((o["status"] == "not_confident") & mT).sum())
                for name in READINGS[:-1]:
                    ordc = order_within(idxc, doubt[name])
                    for nb, k in bud_c.items():
                        top = ordc[:k]
                        res.within_class.append({
                            **base, "threshold": tau_name, "class": c, "reading": name, "budget": nb,
                            "k": k, "n_class": nTc, "e_val_class": e_val_c, "errors_class": errc_tot,
                            "captured": float(err[top].sum() / errc_tot) if errc_tot else float("nan"),
                            "precision": float(err[top].mean()) if k else float("nan"),
                            **room(k, nTc, errc_tot)})
                        sel = np.zeros(nT, dtype=np.float32)
                        sel[top] = err[top]
                        columns.append(sel)
                        dens.append(den_c)
                        keys.append((strategy, tau_name, K, nb, name, f"class={c}"))
            E1, E0 = int(errV[yhatV].sum()), int(errV[~yhatV].sum())
            share1 = E1 / (E1 + E0) if (E1 + E0) else 0.5
            for nb in [f"rho={r_:g}" for r_ in RHO] + ["k_alpha"]:
                k = bud[nb]
                k1, k0 = split_budget(k, E1, E0, len(classes[1]), len(classes[0]))
                for name in SIMPLE:
                    top = np.concatenate([order_within(classes[1], doubt[name])[:k1],
                                          order_within(classes[0], doubt[name])[:k0]])
                    res.allocation.append({
                        **base, "threshold": tau_name, "reading": name, "budget": nb, "k": k,
                        "k_positive": k1, "k_negative": k0, "share_val_errors_positive": share1,
                        "captured": float(err[top].sum() / err.sum()) if err.sum() else float("nan"),
                        "FP_captured": int(fp[top].sum()), "FN_captured": int(fn[top].sum()),
                        **room(k, nT, int(err.sum()))})
                    sel = np.zeros(nT, dtype=np.float32)
                    sel[top] = err[top]
                    columns.append(sel)
                    dens.append(den)
                    keys.append((strategy, tau_name, K, nb, name, "allocation"))
            # --- ablations: same decisions, errors and budgets; only the order of Confidex changes ---
            if extra is not None:
                var = abl.variants(post, extra, tau, crit)
                ordc = order(doubt["confidex"])
                k_gate = var["gate"]["k_alpha"]
                bud_abl = {f"rho={r_:g}": bud[f"rho={r_:g}"] for r_ in RHO}
                entries = [(name_v, v["doubt"], {**bud_abl, "k_alpha": v["k_alpha"]}) for name_v, v in var.items()]
                entries.append(("confidex (k of the gate)", doubt["confidex"], {"k_alpha": k_gate}))
                for name_v, doubt_v, budgets_v in entries:
                    ordv = ordc if name_v == "confidex (k of the gate)" else order(doubt_v)
                    for nb, k in budgets_v.items():
                        top = ordv[:k]
                        row = {**base, "threshold": tau_name, "variant": name_v, "budget": nb, "k": k,
                               "captured": captured(ordv, err, k),
                               "FP_captured": int(fp[top].sum()), "FN_captured": int(fn[top].sum()),
                               "share_budget_positive": float(yhat[top].mean()) if k else float("nan"),
                               **room(k, nT, int(err.sum()))}
                        if name_v == "gate":
                            ab = var["gate"]["abstained"]
                            row["share_abstention"] = var["gate"]["share_abstention"]
                            row["share_abstained_positive"] = float(yhat[ab].mean()) if ab.any() else float("nan")
                            row["share_positive"] = float(yhat.mean())
                            row["errors_among_abstained"] = float(err[ab].mean()) if ab.any() else float("nan")
                            row["v_star"] = extra.v_star
                        res.ablations.append(row)
                        sel = np.zeros(nT, dtype=np.float32)
                        sel[top] = err[top]
                        columns.append(sel)
                        dens.append(den)
                        keys.append((strategy, tau_name, K, nb, name_v, "ablation"))
                # allocated reading at the k of the gate, for every simpler reading
                k1, k0 = split_budget(k_gate, E1, E0, len(classes[1]), len(classes[0]))
                for name in SIMPLE:
                    top = np.concatenate([order_within(classes[1], doubt[name])[:k1],
                                          order_within(classes[0], doubt[name])[:k0]])
                    res.allocation.append({
                        **base, "threshold": tau_name, "reading": name, "budget": "k_gate", "k": k_gate,
                        "k_positive": k1, "k_negative": k0, "share_val_errors_positive": share1,
                        "captured": float(err[top].sum() / err.sum()) if err.sum() else float("nan"),
                        "FP_captured": int(fp[top].sum()), "FN_captured": int(fn[top].sum()),
                        **room(k_gate, nT, int(err.sum()))})
                    sel = np.zeros(nT, dtype=np.float32)
                    sel[top] = err[top]
                    columns.append(sel)
                    dens.append(den)
                    keys.append((strategy, tau_name, K, "k_gate", name, "allocation"))
            num = W @ np.column_stack(columns)                     # (B, columns)
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = num / np.column_stack(dens)
            for j, ch in enumerate(keys):
                res.boot[ch] = res.boot.get(ch, 0.0) + shares[:, j]
                res.n_draws[ch] = res.n_draws.get(ch, 0) + 1
        t_rest += time.time() - t1
    res.timings[f"{strategy}_posterior_s"] = round(res.timings.get(f"{strategy}_posterior_s", 0) + t_post, 1)
    res.timings[f"{strategy}_ablations_s"] = round(res.timings.get(f"{strategy}_ablations_s", 0) + t_abl, 1)
    res.timings[f"{strategy}_rest_s"] = round(res.timings.get(f"{strategy}_rest_s", 0) + t_rest, 1)


def evaluate(dataset: str, p: int, strategies=("s1", "s2"), version: str = "final",
             K_levels: tuple[int, ...] = K_LEVELS, ablations: bool = False, gate_only: bool = False) -> Results:
    res = Results()
    t0 = time.time()
    nT = len(load_pool(dataset, p, strategies[0], version)["y_test"])
    W = bootstrap_weights(dataset, p, nT)
    res.timings["bootstrap_weights_s"] = round(time.time() - t0, 1)
    for s in strategies:
        evaluate_strategy(dataset, p, s, W, res, version=version, K_levels=K_levels,
                          ablations=ablations, gate_only=gate_only)
    return res


def tables(res: Results) -> dict[str, pd.DataFrame]:
    """All the tables of an evaluation (one condition or more)."""
    rows = pd.DataFrame(res.rows)
    val = pd.DataFrame(res.validation)
    choice = validation_choice(val)
    within = pd.DataFrame(res.within_class)
    valc = pd.DataFrame(res.validation_within_class)
    alloc = pd.DataFrame(res.allocation)
    choice_c = choice_within_class(valc)
    abl_rows = pd.DataFrame(res.ablations)
    return {"rows": rows, "draws": pd.DataFrame(res.draws), "validation": val,
            "choice": choice.reset_index(), "comparison": comparison(res, rows, choice),
            "within_class": within, "validation_within_class": valc, "allocation": alloc,
            "choice_within_class": choice_c.reset_index(),
            "comparison_within_class": comparison_within_class(res, within, choice_c),
            "comparison_allocation": comparison_allocation(res, rows, alloc, choice),
            "ablations": abl_rows, "comparison_ablations": comparison_ablations(res, rows, abl_rows),
            "comparison_gate": comparison_gate(res, abl_rows, alloc, choice)}


def validation_choice(val: pd.DataFrame) -> pd.DataFrame:
    """The simpler reading with the highest mean share captured (rho = 1, validation part), for every
    construction, threshold and K. Ties go to the first reading in the order of SIMPLE."""
    m = (val.groupby(["strategy", "threshold", "K", "reading"], sort=False)["captured"].mean()
         .unstack("reading")[list(SIMPLE)])
    return pd.DataFrame({"validation_choice": m.idxmax(axis=1), "captured_validation": m.max(axis=1)})


def comparison(res: Results, rows: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    """Confidex against the reading chosen on the validation part and against the best on the test part:
    mean difference over the draws and 95% bootstrap interval."""
    mean = rows.groupby(["strategy", "threshold", "K", "budget", "reading"])["captured"].mean()
    marg = rows[rows.reading == "confidex"].groupby(["strategy", "threshold", "K", "budget"])[
        ["k", "random", "perfect", "FP_captured", "FN_captured", "share_budget_positive"]].mean()
    out = []
    for (s, t, K, nb), _ in rows.groupby(["strategy", "threshold", "K", "budget"]):
        sel = choice.loc[(s, t, K), "validation_choice"]
        simple = {name: mean[(s, t, K, nb, name)] for name in SIMPLE}
        best_test = max(simple, key=simple.get)
        bc = res.boot[(s, t, K, nb, "confidex")] / res.n_draws[(s, t, K, nb, "confidex")]
        row = {"strategy": s, "threshold": t, "K": K, "budget": nb,
               "confidex": mean[(s, t, K, nb, "confidex")],
               **{f"{name}": simple[name] for name in SIMPLE},
               "s": mean[(s, t, K, nb, "s")], "c mean": mean[(s, t, K, nb, "c mean")],
               "confidex plug-in": mean[(s, t, K, nb, "confidex plug-in")],
               "validation_choice": sel, "best_test": best_test,
               **marg.loc[(s, t, K, nb)].to_dict()}
        for label, name in (("val", sel), ("test", best_test)):
            bs = res.boot[(s, t, K, nb, name)] / res.n_draws[(s, t, K, nb, name)]
            d = bc - bs
            row[f"diff_{label}"] = mean[(s, t, K, nb, "confidex")] - simple[name]
            row[f"lo_{label}"] = float(np.nanquantile(d, 0.025))
            row[f"hi_{label}"] = float(np.nanquantile(d, 0.975))
        out.append(row)
    return pd.DataFrame(out)


def _interval(res: Results, a: tuple, b: tuple) -> tuple[float, float]:
    d = res.boot[a] / res.n_draws[a] - res.boot[b] / res.n_draws[b]
    return float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))


def choice_within_class(valc: pd.DataFrame) -> pd.DataFrame:
    """The simpler reading chosen on the validation part within the predicted class (rho = 1).

    If the class has no error on the validation part in any draw, the share captured is undefined for
    every reading: none is better, and the tie rule applies (the first reading in the order of the
    protocol). This rule was added during the evaluation (Chapter 4, Decisions taken after the pilot)."""
    m = (valc.groupby(["strategy", "threshold", "K", "class", "reading"], sort=False)["captured"].mean()
         .unstack("reading")[list(SIMPLE)])
    return pd.DataFrame({"validation_choice": m.fillna(-np.inf).idxmax(axis=1), "captured_validation": m.max(axis=1)})


def comparison_within_class(res: Results, within: pd.DataFrame, choice_c: pd.DataFrame) -> pd.DataFrame:
    """Within each predicted class: Confidex against the simpler reading chosen on the validation part
    within the class, with a bootstrap interval (same replicates)."""
    mean = within.groupby(["strategy", "threshold", "K", "class", "budget", "reading"])["captured"].mean()
    agg = within.groupby(["strategy", "threshold", "K", "class", "budget"]).agg(
        k=("k", "mean"), n_class=("n_class", "mean"), errors_class=("errors_class", "mean"),
        random=("random", "mean"), perfect=("perfect", "mean"))
    out = []
    for (s, t, K, c, nb), x in agg.iterrows():
        sel = choice_c.loc[(s, t, K, c), "validation_choice"]
        lo, hi = _interval(res, (s, t, K, nb, "confidex", f"class={c}"), (s, t, K, nb, sel, f"class={c}"))
        out.append({"strategy": s, "threshold": t, "K": K, "class": c, "budget": nb, **x.to_dict(),
                    "confidex": mean[(s, t, K, c, nb, "confidex")],
                    **{name: mean[(s, t, K, c, nb, name)] for name in SIMPLE},
                    "s": mean[(s, t, K, c, nb, "s")], "c mean": mean[(s, t, K, c, nb, "c mean")],
                    "validation_choice": sel,
                    "diff_val": mean[(s, t, K, c, nb, "confidex")] - mean[(s, t, K, c, nb, sel)],
                    "lo_val": lo, "hi_val": hi})
    return pd.DataFrame(out)


def comparison_allocation(res: Results, rows: pd.DataFrame, alloc: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    """Confidex on the overall budget against the allocated reading (the simpler reading chosen on the
    validation part in the main comparison, with the budget divided by class)."""
    mc = rows[rows.reading == "confidex"].groupby(["strategy", "threshold", "K", "budget"])["captured"].mean()
    alloc = alloc[alloc.budget != "k_gate"]
    ma = alloc.groupby(["strategy", "threshold", "K", "budget", "reading"])[
        ["captured", "k_positive", "k_negative", "share_val_errors_positive", "FP_captured", "FN_captured",
         "random", "perfect"]].mean()
    out = []
    for (s, t, K, nb), conf in mc.items():
        if (s, t, K, nb, "distance") not in ma.index:
            continue
        sel = choice.loc[(s, t, K), "validation_choice"]
        x = ma.loc[(s, t, K, nb, sel)]
        lo, hi = _interval(res, (s, t, K, nb, "confidex"), (s, t, K, nb, sel, "allocation"))
        out.append({"strategy": s, "threshold": t, "K": K, "budget": nb, "confidex": conf,
                    "reading": sel, "allocation": x.captured,
                    **{f"alloc_{n}": ma.loc[(s, t, K, nb, n), "captured"] for n in SIMPLE},
                    "k_positive": x.k_positive, "k_negative": x.k_negative,
                    "share_val_errors_positive": x.share_val_errors_positive,
                    "FP_captured": x.FP_captured, "FN_captured": x.FN_captured,
                    "random": x.random, "perfect": x.perfect,
                    "diff": conf - x.captured, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def comparison_ablations(res: Results, rows: pd.DataFrame, abl_rows: pd.DataFrame) -> pd.DataFrame:
    """Every variant against Confidex on the same errors: at the budgets k_rho at the same k, at k_alpha each
    verdict at its own k (the gate against Confidex at the k of the gate); bootstrap interval."""
    if abl_rows.empty:
        return pd.DataFrame()
    mv = abl_rows.groupby(["strategy", "threshold", "K", "budget", "variant"])[
        ["captured", "k", "FP_captured", "FN_captured", "share_budget_positive", "random", "perfect"]].mean()
    mc = rows[rows.reading == "confidex"].groupby(["strategy", "threshold", "K", "budget"])[["captured", "k"]].mean()
    out = []
    for (s, t, K, nb, v), x in mv.iterrows():
        if v == "confidex (k of the gate)":
            continue
        if v == "gate" and nb == "k_alpha":
            ref = mv.loc[(s, t, K, nb, "confidex (k of the gate)")]
            ref_key = (s, t, K, nb, "confidex (k of the gate)", "ablation")
        else:
            ref = mc.loc[(s, t, K, nb)]
            ref_key = (s, t, K, nb, "confidex")
        lo, hi = _interval(res, (s, t, K, nb, v, "ablation"), ref_key)
        out.append({"strategy": s, "threshold": t, "K": K, "budget": nb, "variant": v,
                    **{f"{c}": x[c] for c in x.index}, "confidex": ref.captured, "k_confidex": ref.k,
                    "diff": x.captured - ref.captured, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def comparison_gate(res: Results, abl_rows: pd.DataFrame, alloc: pd.DataFrame, choice: pd.DataFrame) -> pd.DataFrame:
    """The gate against the allocated reading (simpler reading chosen on the validation part) at the same k:
    at rho = 1 and at the k of the gate, with the share of the abstentions among the predicted positives."""
    if abl_rows.empty or "gate" not in set(abl_rows.variant):
        return pd.DataFrame()
    g = abl_rows[abl_rows.variant == "gate"]
    mg = g.groupby(["strategy", "threshold", "K", "budget"])[
        ["captured", "k", "share_abstention", "share_abstained_positive", "share_positive",
         "errors_among_abstained", "random", "perfect"]].mean()
    ma = alloc.groupby(["strategy", "threshold", "K", "budget", "reading"])[["captured", "k"]].mean()
    out = []
    for (s, t, K, nb), x in mg.iterrows():
        if nb not in ("rho=1", "k_alpha"):
            continue
        sel = choice.loc[(s, t, K), "validation_choice"]
        nb_a = "rho=1" if nb == "rho=1" else "k_gate"
        a = ma.loc[(s, t, K, nb_a, sel)]
        lo, hi = _interval(res, (s, t, K, nb, "gate", "ablation"), (s, t, K, nb_a, sel, "allocation"))
        out.append({"strategy": s, "threshold": t, "K": K, "budget": "rho=1" if nb == "rho=1" else "k_gate",
                    **x.to_dict(), "reading": sel, "allocation": a.captured, "k_allocation": a.k,
                    "diff": x.captured - a.captured, "lo": lo, "hi": hi})
    return pd.DataFrame(out)
