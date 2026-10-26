"""All the numerical results of Chapter 3 (Confidex), from fixed seeds.

    python run_experiments.py            ->  results/chapter3_results.json, results/auroc_gap_bootstrap.json

The script takes about eight minutes on a laptop. Each experimental condition draws its random numbers
from its own generator (``synthetic.seed_for``), seeded by a key that depends only on the condition,
so the conditions can be run singly and in any order. The only exception is the bootstrap of the AUROC
differences, which draws from one generator (seed 2026) across the conditions in the order of the loops.

Sections of the output, with the part of Chapter 3 that reports them:

    illustration         the ten constructed ensembles                    (An illustration on ten ensembles)
    five                 five members under independent priors             (Three specifications; Assumptions)
    validation           AUROC of every reading on cases with known truth   (What the signal detects ...)
    budget               errors captured at fixed review budgets            (idem, and Appendix, Table A.2)
    sensitivity          share of verdicts changed by each modelling choice (Sensitivity to the modelling choices)
    shrinkage            sigma_pred / s on unimodal laws                    (Remark: the cancellation is dimensional)
    examples             relative measure, polarised ensembles, MoM         (The concentration is a relative measure)
    anchoring_location   three members near 0.012                           (What the signal says, and its limits)
    gate_corr            correlation behind the admissibility gate          (Remark: an admissibility gate)
    quadrature           grid against adaptive quadrature                   (Inference on the concentration)
    ties                 exact ties and near-ties                           (Proposition: impropriety under exact ties)
    jeffreys             the factor h(c) of the Jeffreys prior              (Inference on the concentration)
    timing               cost of the approximation relative to the predictive law (machine dependent)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, expit, logit, polygamma
from scipy.stats import beta as B, spearmanr

import confidex as cx
from confidex import ModelConfig
from synthetic import DESIGNS, SIMPLE_READINGS, auroc, baselines, recall_at, seed_for, simulate

HERE = Path(__file__).resolve().parent
KS = (3, 6, 12, 24, 36)
BUDGETS = (0.05, 0.10, 0.20)

# the ten constructed ensembles of the chapter (listed in the Appendix)
TEN = np.array([
    [0.95, 0.96, 0.94, 0.95, 0.95, 0.96, 0.94, 0.95, 0.96, 0.95], [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04, 0.05],
    [0.48, 0.50, 0.49, 0.51, 0.50, 0.52, 0.49, 0.48, 0.51, 0.50], [0.12, 0.11, 0.13, 0.12, 0.14, 0.13, 0.12, 0.11, 0.13, 0.12],
    [0.88, 0.87, 0.89, 0.88, 0.86, 0.89, 0.88, 0.87, 0.89, 0.88], [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.9],
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9],
    [0.2, 0.8, 0.5, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1, 0.5], [0.02, 0.99, 0.01, 0.98, 0.03, 0.97, 0.02, 0.96, 0.01, 0.95]])
FIVE = np.array([[0.65, 0.77, 0.40, 0.83, 0.55]])


# ---------------------------------------------------------------------------- illustration
def illustration() -> dict:
    """Posterior mean of c and F_pred(tau) at tau = 0.5 and 0.1 for the ten ensembles."""
    post = cx.fit_posterior(TEN, taus=[0.5, 0.1]); law = cx.member_law(post)
    out = {"cbar": post.c_mean.tolist()}
    for tau in (0.5, 0.1):
        o = cx.confidence_score(post, law, tau)
        op = cx.confidence_score(post, law, tau, predictive=False)
        out[f"F_{tau}"] = o["F_tau"].tolist(); out[f"Fplug_{tau}"] = op["F_tau"].tolist()
        out[f"conf_{tau}"] = o["confident"].tolist(); out[f"confplug_{tau}"] = op["confident"].tolist()
        out[f"opposite_{tau}"] = o["opposite_side"].tolist()
    return out


def five_members() -> dict:
    """Five members under independent priors (inferred location) and under the fixed mean."""
    full = cx.fit_posterior_full(FIVE, points_mu=400, points_c=400)
    fx = cx.fit_posterior(FIVE)
    return {"mu_mean": float(full.mu_mean[0]), "mu_sd": float(full.mu_sd[0]), "ci90": full.mu_ci90[0].tolist(),
            "sigma_pred": float(fx.sigma_pred[0]), "sigma_pred_over_sqrtK": float(fx.sigma_pred[0] / np.sqrt(5))}


# ---------------------------------------------------------------------------- known truth
def validation_condition(d: str, tau: float, K: int) -> tuple[dict, dict]:
    """One condition of the experiment with known truth: AUROC of every reading, the gate at the same
    review budget, and the errors captured at fixed budgets with a bootstrap at 10%."""
    P, q, eta = simulate(d, K, tau, 20000, seed_for(d, tau, K, "val"))
    Pv, _, _ = simulate(d, K, tau, 5000, seed_for(d, tau, K, "gate"))
    post = cx.fit_posterior(P, taus=[tau]); law = cx.member_law(post)
    o = cx.confidence_score(post, law, tau)                          # procedure: predictive law, directional
    ob = cx.confidence_score(post, law, tau, directional=False)      # without the side of the decision
    op = cx.confidence_score(post, law, tau, predictive=False)       # law at the posterior mean of c
    err = ((post.mu >= tau) != (q >= tau)).astype(int)               # against the oracle's decision 1[q >= tau]
    res = {"confidex": auroc(-o["u"], err), "confidex plug-in": auroc(-op["u"], err),
           "band": auroc(-ob["u"], err), "concentration": auroc(-post.c_mean, err)}
    res.update({k: auroc(v, err) for k, v in baselines(P, post.mu, tau).items()})
    res["rho"] = float(spearmanr(np.sqrt(post.rel_disp), eta).correlation)
    res["agree_plugin"] = float(np.mean(o["confident"] == op["confident"]))
    res["opposite"] = float(np.mean(o["opposite_side"]))
    res["error_rate"] = float(err.mean())
    # the admissibility gate, at the same number of cases sent to review
    v_star = cx.calibrate_gate(cx.fit_posterior(Pv))
    og = cx.confidence_score(post, law, tau, v_star=v_star)
    flag = og["status"] != "confident"; k = int(flag.sum())
    top = np.argsort(o["u"], kind="mergesort")[:k]
    excluded, admitted_conf = og["status"] == "abstain", og["status"] == "confident"
    res["gate_recall"] = float(err[flag].sum() / err.sum()); res["rank_recall"] = float(err[top].sum() / err.sum())
    res["gate_err_excluded"] = float(err[excluded].mean()); res["gate_err_admitted_conf"] = float(err[admitted_conf].mean())
    # fixed review budget: share of the errors among the b% most doubtful cases
    sc = {"confidex": -o["u"], "concentration": -post.c_mean, **baselines(P, post.mu, tau)}
    bud = {f"{m}@{b}": recall_at(v, err, b) for m, v in sc.items() for b in BUDGETS}
    best = max(SIMPLE_READINGS, key=lambda m: bud[f"{m}@0.1"])
    rb = seed_for(d, tau, K, "budget"); boots = []
    for _ in range(200):
        ii = rb.integers(0, len(err), len(err))
        boots.append(recall_at(sc["confidex"][ii], err[ii], 0.1) - recall_at(sc[best][ii], err[ii], 0.1))
    bud.update({"best@0.1": best, "gap@0.1": bud["confidex@0.1"] - bud[f"{best}@0.1"],
                "lo@0.1": float(np.quantile(boots, 0.025)), "hi@0.1": float(np.quantile(boots, 0.975)),
                "errors_in_10pct": float(0.1 * len(err) / err.sum())})
    return res, bud


def known_truth(verbose: bool = True) -> tuple[dict, dict]:
    val, budget = {}, {}
    for d in DESIGNS:
        for tau in (0.5, 0.1):
            for K in KS:
                val[f"{d}|{tau}|{K}"], budget[f"{d}|{tau}|{K}"] = validation_condition(d, tau, K)
                if verbose:
                    print(f"  {d:<16} tau={tau}  K={K:<3} confidex AUROC {val[f'{d}|{tau}|{K}']['confidex']:.3f}", flush=True)
    return val, budget


def auroc_gap_bootstrap() -> dict:
    """Difference between the AUROC of the signal and that of the best simpler reading (chosen by AUROC),
    with a 95% bootstrap interval from 200 resamples of the cases (Figure "amplification")."""
    G, rng_b = {}, np.random.default_rng(2026)
    for d in DESIGNS:
        for tau in (0.5, 0.1):
            for K in KS:
                P, q, _ = simulate(d, K, tau, 20000, seed_for(d, tau, K, "val"))
                post = cx.fit_posterior(P, taus=[tau]); o = cx.confidence_score(post, None, tau)
                err = ((post.mu >= tau) != (q >= tau)).astype(int)
                bl = baselines(P, post.mu, tau); best = max(bl, key=lambda k: auroc(bl[k], err))
                sc_c, sc_b = -o["u"], bl[best]
                boots = []
                for _ in range(200):
                    ii = rng_b.integers(0, len(err), len(err))
                    boots.append(auroc(sc_c[ii], err[ii]) - auroc(sc_b[ii], err[ii]))
                G[f"{d}|{tau}|{K}"] = {"gap": auroc(sc_c, err) - auroc(sc_b, err), "lo": float(np.quantile(boots, 0.025)),
                                       "hi": float(np.quantile(boots, 0.975)), "best": best}
    return G


# ---------------------------------------------------------------------------- sensitivity
def sensitivity(verbose: bool = True) -> dict:
    """Share of verdicts that change when each modelling choice is varied (mixed design, 2000 ensembles)."""
    sens = {"prior": {}, "lambda0": {}, "plugin": {}, "fixed_mean": {}}
    for tau in (0.5, 0.25):
        for K in KS:
            P, _, _ = simulate("mixed", K, tau, 2000, seed_for("mixed", tau, K, "sens"))

            def verdict(cfg):
                p = cx.fit_posterior(P, cfg, taus=[tau]); return cx.confidence_score(p, None, tau)["confident"], p

            v0, _ = verdict(ModelConfig(prior_a=1, prior_b=1))                       # uniform prior: reference
            sens["prior"][f"{tau}|{K}"] = {name: float(np.mean(verdict(ModelConfig(prior_a=a, prior_b=b))[0] != v0))
                                           for name, (a, b) in {"2,2": (2, 2), "0.5,0.5": (0.5, 0.5), "2,3": (2, 3)}.items()}
            vK, pK = verdict(ModelConfig())                                          # lambda0 = K: reference
            sens["lambda0"][f"{tau}|{K}"] = {str(sc): float(np.mean(verdict(ModelConfig(lambda0_scale=sc))[0] != vK))
                                             for sc in (0.1, 1 / 3, 3, 10)}
            lawK = cx.member_law(pK)
            vplug = cx.confidence_score(pK, lawK, tau, predictive=False)["confident"]
            sens["plugin"][f"{tau}|{K}"] = float(np.mean(vplug == vK))
            # fixed mean against the inferred mean, on means away from the ends of the interval
            keep = (P.mean(1) > 0.05) & (P.mean(1) < 0.95); Pk = P[keep][:1200]
            pf = cx.fit_posterior(Pk, taus=[tau]); vfix = cx.confidence_score(pf, None, tau)["confident"]
            fb = cx.fit_posterior_full(Pk, points_mu=160, points_c=160, taus=[tau]); Fp = fb.F_pred[tau]
            vfull = np.where(Pk.mean(1) >= tau, Fp < 0.4, Fp > 0.6)
            sens["fixed_mean"][f"{tau}|{K}"] = {"changed": float(np.mean(vfix != vfull)),
                                                "conf_only_fixed": float(np.mean(vfix & ~vfull)),
                                                "conf_only_full": float(np.mean(~vfix & vfull))}
            if verbose:
                print(f"  sensitivity tau={tau} K={K}: fixed mean changes {sens['fixed_mean'][f'{tau}|{K}']['changed']:.3f}", flush=True)
    P, _, _ = simulate("mixed", 6, 0.5, 2000, seed_for("mixed", 0.5, 6, "sens"))
    sens["border4"] = float(np.max(np.abs(cx.fit_posterior(P, ModelConfig(grid_bound=1e-4)).c_mean - cx.fit_posterior(P).c_mean)))
    for K in (3, 12):
        P, _, _ = simulate("mixed", K, 0.5, 2000, seed_for("mixed", 0.5, K, "sens"))
        sens[f"atten_{K}"] = float(np.median(cx.fit_posterior(P, ModelConfig(lambda0_scale=10)).lam
                                             / cx.fit_posterior(P, ModelConfig(lambda0_scale=0.1)).lam))
    return sens


# ---------------------------------------------------------------------------- examples of the remarks
def shrinkage() -> dict:
    """Median of sigma_pred / s on the cases whose reconstructed law is unimodal."""
    out = {}
    for K in KS:
        P, _, _ = simulate("mixed", K, 0.5, 2000, seed_for("mixed", 0.5, K, "shr"))
        p = cx.fit_posterior(P); uni = cx.shape_regime(cx.member_law(p)) == "unimodal"
        out[str(K)] = float(np.median((p.sigma_pred / P.std(axis=1, ddof=1))[uni]))
    return out


def examples() -> dict:
    """The constructed ensembles of 27 members of "The concentration is a relative measure"."""
    ex = {}
    K = 27; qs = (np.arange(K) + 0.5) / K
    for mu in (0.50, 0.15, 0.05):
        lam = mu * (1 - mu) / 0.044 ** 2 - 1; x = B.ppf(qs, mu * lam, (1 - mu) * lam)
        ex[f"beta_{mu}"] = {"s": float(x.std(ddof=1)), "cbar": float(cx.fit_posterior(x[None, :]).c_mean[0])}
    for name, x in (("25-2", np.r_[np.full(25, 0.999), np.full(2, 0.001)]), ("13-14", np.r_[np.full(13, 0.9), np.full(14, 0.1)])):
        p = cx.fit_posterior(x[None, :]); s = x.std(ddof=1)
        ex[name] = {"pbar": float(x.mean()), "cbar": float(p.c_mean[0]), "sigma_pred_over_s": float(p.sigma_pred[0] / s),
                    "lambda_mom": float(x.mean() * (1 - x.mean()) / s ** 2 - 1)}
    return ex


def anchoring_location() -> dict:
    """Three members agreeing within about 20% on probabilities near 0.012."""
    rng = np.random.default_rng(3); X = np.exp(np.log(0.012) + 0.2 * rng.standard_normal((2000, 3)))
    p = cx.fit_posterior(X)
    return {"J_or_U": float(np.mean(cx.shape_regime(cx.member_law(p)) != "unimodal")),
            "sigma_pred_over_s": float(np.median(p.sigma_pred / X.std(axis=1, ddof=1)))}


def gate_correlation() -> dict:
    """Spearman correlation between Var(c | P) and |cbar - 1/2| (mixed design, 5000 ensembles)."""
    out = {}
    for K in KS:
        P, _, _ = simulate("mixed", K, 0.5, 5000, seed_for("mixed", 0.5, K, "gatecorr"))
        p = cx.fit_posterior(P); out[str(K)] = float(spearmanr(p.c_var, np.abs(p.c_mean - 0.5)).correlation)
    return out


# ---------------------------------------------------------------------------- grid, ties, Jeffreys
def synthetic_beta(n: int, K: int, rng) -> np.ndarray:
    """Ensembles drawn from the model itself: mu uniform on [0.02, 0.98], c ~ Beta(2, 3), lambda_0 = K."""
    mu, c = rng.uniform(0.02, 0.98, n), rng.beta(2, 3, n)
    lam = K * c / (1 - c)
    return np.clip(rng.beta((mu * lam)[:, None], ((1 - mu) * lam)[:, None], (n, K)), 1e-6, 1 - 1e-6)


def quad_mean(x: np.ndarray, a0: float = 2.0, b0: float = 3.0) -> float:
    """Posterior mean of c by adaptive quadrature on the logit scale (independent of the grid)."""
    K, mu = len(x), x.mean(); S1, S2 = np.log(x).sum(), np.log1p(-x).sum()

    def lp(t):
        c = expit(t); lam = K * c / (1 - c); a, b = mu * lam, (1 - mu) * lam
        return (a - 1) * S1 + (b - 1) * S2 - K * betaln(a, b) + B.logpdf(c, a0, b0) + np.log(c) + np.log1p(-c)
    lo, hi = logit(1e-6), logit(1 - 1e-6); ts = np.linspace(lo, hi, 4001)
    M = max(lp(t) for t in ts); f = lambda t: np.exp(lp(t) - M)
    Z = quad(f, lo, hi, limit=400, points=ts[::400])[0]
    return quad(lambda t: expit(t) * f(t), lo, hi, limit=400, points=ts[::400])[0] / Z


def quadrature() -> dict:
    qd = {"n": 0, "max_vs_quadrature": 0.0, "grid": {}}
    for K in (3, 6, 12):
        X = synthetic_beta(40, K, seed_for("quadrature", K))
        p = cx.fit_posterior(X); qd["n"] += len(X)
        qd["max_vs_quadrature"] = max(qd["max_vs_quadrature"], float(max(abs(p.c_mean[i] - quad_mean(X[i])) for i in range(len(X)))))
        for J in (500, 2000, 4000):
            dJ = float(np.max(np.abs(cx.fit_posterior(X, ModelConfig(grid_points=J)).c_mean - p.c_mean)))
            qd["grid"][str(J)] = max(qd["grid"].get(str(J), 0.0), dJ)
    return qd


def ties() -> dict:
    """Exact ties (all members at 0.3) and near-ties (a spread of +-0.002) for K = 3, 12, 36."""
    out = {}
    priors = {"2,3": (2, 3), "1,1": (1, 1), "2,2": (2, 2), "0.5,0.5": (0.5, 0.5)}
    for K in (3, 12, 36):
        tie = np.full((1, K), 0.3)
        near = (0.3 + 0.002 * np.linspace(-1, 1, K))[None, :]
        t = {f"tie_cbar_{n}": float(cx.fit_posterior(tie, ModelConfig(prior_a=a, prior_b=b)).c_mean[0]) for n, (a, b) in priors.items()}
        t["tie_cbar_bound1e-9"] = float(cx.fit_posterior(tie, ModelConfig(grid_bound=1e-9, grid_points=1500)).c_mean[0])
        p6 = cx.fit_posterior(near); p9 = cx.fit_posterior(near, ModelConfig(grid_bound=1e-9, grid_points=1500))
        pr = [cx.fit_posterior(near, ModelConfig(prior_a=a, prior_b=b)).c_mean[0] for a, b in priors.values()]
        t.update({"near_border_mass": float(p6.border_mass[0]), "near_change_with_bound": float(abs(p6.c_mean[0] - p9.c_mean[0])),
                  "near_cbar": float(p6.c_mean[0]), "near_prior_spread": float(max(pr) - min(pr))})
        out[str(K)] = t
    return out


def jeffreys() -> dict:
    """The Jeffreys prior on c is pi_J(c) = h(c) / [c (1 - c)]: range and limits of h."""
    c_ = 1 / (1 + np.exp(-np.linspace(-12, 12, 4001))); hs = []
    for mu in (0.01, 0.05, 0.15, 0.5, 0.85, 0.99):
        for lam0 in (3, 12, 36):
            lam = lam0 * c_ / (1 - c_)
            I_lam = mu ** 2 * polygamma(1, mu * lam) + (1 - mu) ** 2 * polygamma(1, (1 - mu) * lam) - polygamma(1, lam)
            hs.append(np.sqrt(I_lam) * (lam0 / (1 - c_) ** 2) * c_ * (1 - c_))
    hs = np.array(hs)
    return {"h_min": float(hs.min()), "h_max": float(hs.max()),
            "h_at_c_small": float(hs[:, 0].mean()), "h_at_c_large": float(hs[:, -1].mean())}


def timing(repeats: int = 5) -> dict:
    """Cost of the approximation (law at the posterior mean of c) relative to the predictive law,
    on 20,000 ensembles of 12 members. The ratio is machine dependent; the chapter reports about 1/6."""
    P, _, _ = simulate("mixed", 12, 0.25, 20000, np.random.default_rng(0))

    def best_of(fn):
        t = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter(); fn(); t = min(t, time.perf_counter() - t0)
        return t
    t_pred = best_of(lambda: cx.fit_posterior(P, taus=[0.25]))
    t_plug = best_of(lambda: cx.member_law(cx.fit_posterior(P)).cdf(0.25))
    return {"predictive_s": t_pred, "approximation_s": t_plug, "ratio": t_plug / t_pred}


# ---------------------------------------------------------------------------- main
def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {"illustration": illustration(), "five": five_members()}
    print("known truth (30 conditions)", flush=True)
    R["validation"], R["budget"] = known_truth()
    print("sensitivity (10 conditions)", flush=True)
    R["sensitivity"] = sensitivity()
    R.update({"shrinkage": shrinkage(), "examples": examples(), "anchoring_location": anchoring_location(),
              "gate_corr": gate_correlation(), "quadrature": quadrature(), "ties": ties(), "jeffreys": jeffreys(),
              "timing": timing()})
    (out_dir / "chapter3_results.json").write_text(json.dumps(R, indent=1))
    print("bootstrap of the AUROC differences", flush=True)
    (out_dir / "auroc_gap_bootstrap.json").write_text(json.dumps(auroc_gap_bootstrap(), indent=1))
    print(f"written to {out_dir} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "results", help="output folder (default: results/)")
    main(ap.parse_args().out)
