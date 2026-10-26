"""Check every number that Chapter 3 (and Table A.2 of the Appendix) reports against the results.

    python check_values.py                 # reads results/chapter3_results.json and results/auroc_gap_bootstrap.json
    python check_values.py --list          # also prints every check, not only the failures

Each check states where the value appears in the thesis, what it measures, the value as the thesis
prints it, and the value recomputed from the results with the same rounding. Statements that are not
numbers ("the concentration is the poorest reading in every condition") are checked as conditions and
printed as "holds". The script exits with status 1 if any check fails.

The ratio of computing times in the section on the law of the members is machine dependent and is
reported for information only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

from synthetic import DESIGNS, SIMPLE_READINGS

HERE = Path(__file__).resolve().parent
KS = (3, 6, 12, 24, 36)
TEN = np.array([
    [0.95, 0.96, 0.94, 0.95, 0.95, 0.96, 0.94, 0.95, 0.96, 0.95], [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04, 0.05],
    [0.48, 0.50, 0.49, 0.51, 0.50, 0.52, 0.49, 0.48, 0.51, 0.50], [0.12, 0.11, 0.13, 0.12, 0.14, 0.13, 0.12, 0.11, 0.13, 0.12],
    [0.88, 0.87, 0.89, 0.88, 0.86, 0.89, 0.88, 0.87, 0.89, 0.88], [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.9],
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9],
    [0.2, 0.8, 0.5, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1, 0.5], [0.02, 0.99, 0.01, 0.98, 0.03, 0.97, 0.02, 0.96, 0.01, 0.95]])

CHECKS: list[tuple[str, str, str, str, bool]] = []      # (where, what, printed, recomputed, ok)


def f(x, nd):
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def value(where, what, printed, recomputed):
    """A number (or a list of numbers) printed in the thesis, against the same rounding of the results."""
    CHECKS.append((where, what, printed, recomputed, printed == recomputed))


def holds(where, what, condition, detail=""):
    """A statement checked as a condition on the results."""
    CHECKS.append((where, what, "holds", "holds" if condition else f"FAILS {detail}", bool(condition)))


def main(results: Path, show_all: bool) -> int:
    R = json.loads((results / "chapter3_results.json").read_text())
    G = json.loads((results / "auroc_gap_bootstrap.json").read_text())
    V, BU, S = R["validation"], R["budget"], R["sensitivity"]
    v = lambda d, t, K, m: V[f"{d}|{t}|{K}"][m]
    best_simple = lambda d, t, K: max(SIMPLE_READINGS, key=lambda m: v(d, t, K, m))
    gap = lambda d, t, K: v(d, t, K, "confidex") - v(d, t, K, best_simple(d, t, K))
    excl = lambda d, t, K: G[f"{d}|{t}|{K}"]["lo"] > 0 or G[f"{d}|{t}|{K}"]["hi"] < 0

    def pts(d, t, K, b="0.1"):        # signal minus best simpler reading at the same budget, points
        g = BU[f"{d}|{t}|{K}"]
        return 100 * (g[f"confidex@{b}"] - max(g[f"{m}@{b}"] for m in SIMPLE_READINGS))

    def star(d, t, K):
        g = BU[f"{d}|{t}|{K}"]
        return not (g["lo@0.1"] <= 0 <= g["hi@0.1"])

    # ---------------------------------------------------------------- coupled and decoupled models
    for c, lam, var in ((0.10, 10, "0.008182"), (0.10, 100, "0.000891"), (0.50, 10, "0.022727"),
                        (0.90, 10, "0.008182"), (0.90, 100, "0.000891"), (0.99, 0.5, "0.006600")):
        value("Table 3.1", f"variance of the coupled model, c={c}, lambda={lam}", var, f"{c * (1 - c) / (lam + 1):.6f}")
    value("Coupled and decoupled models", "c = lambda/(lambda + 10) at lambda = 1, 10, 50", "0.09, 0.50, 0.83",
          ", ".join(f(l / (l + 10), 2) for l in (1, 10, 50)))
    # ---------------------------------------------------------------- three specifications, assumptions
    fv = R["five"]
    value("Three specifications", "mean of the five members", "0.64", f(np.mean([0.65, 0.77, 0.40, 0.83, 0.55]), 2))
    value("Three specifications", "90% credible interval of mu (five members)", "[0.47, 0.75]",
          f"[{f(fv['ci90'][0], 2)}, {f(fv['ci90'][1], 2)}]")
    value("Assumptions", "posterior sd of mu under independent priors", "0.085", f(fv["mu_sd"], 3))
    value("Assumptions", "sigma / sqrt(K) under the fixed mean", "0.084", f(fv["sigma_pred_over_sqrtK"], 3))
    value("Assumptions", "neglected uncertainty at K = 3, 1/sqrt(3), in units of sigma", "0.58", f(1 / np.sqrt(3), 2))
    value("Assumptions", "half-width of the central band, Phi^-1(0.6), in units of sigma", "0.25", f(norm.ppf(0.6), 2))
    # ---------------------------------------------------------------- inference
    value("Inference", "lambda_0 x 100: change of the median precision at K = 3 (about)", "23", str(round(S["atten_3"])))
    value("Inference", "lambda_0 x 100: change of the median precision at K = 12 (about)", "2", str(round(S["atten_12"])))
    holds("Inference", "border 1e-6 -> 1e-4 changes posterior means by at most 3e-6", S["border4"] <= 3e-6, S["border4"])
    q = R["quadrature"]
    value("Inference", "ensembles in the check against quadrature", "120", str(q["n"]))
    holds("Inference", "grid agrees with adaptive quadrature within 1e-5", q["max_vs_quadrature"] < 1e-5, q["max_vs_quadrature"])
    value("Inference", "halving the grid points changes the means by about", "2e-05", f"{q['grid']['500']:.0e}")
    holds("Inference", "doubling or quadrupling the points changes them by less than 1e-5",
          max(q["grid"]["2000"], q["grid"]["4000"]) < 1e-5, q["grid"])
    T = R["ties"]
    value("Inference", "exact tie, K = 3: cbar under Beta(2,3)", "0.73", f(T["3"]["tie_cbar_2,3"], 2))
    value("Inference", "exact tie, K = 3: cbar under the uniform prior", "0.999", f(T["3"]["tie_cbar_1,1"], 3))
    holds("Inference", "near-tie, K = 12 and 36: border 1e-6 -> 1e-9 changes the mean by less than 1e-13",
          max(T["12"]["near_change_with_bound"], T["36"]["near_change_with_bound"]) < 1e-13)
    holds("Inference", "near-tie, K = 12 and 36: the four priors change the mean by less than 2e-4",
          max(T["12"]["near_prior_spread"], T["36"]["near_prior_spread"]) < 2e-4)
    J = R["jeffreys"]
    holds("Inference", "Jeffreys factor h between 1/sqrt(2) and 1",
          abs(J["h_min"] - 1 / np.sqrt(2)) < 1e-6 and abs(J["h_max"] - 1) < 1e-6, J)
    holds("Inference", "h -> 1 as c -> 0 and h -> 1/sqrt(2) as c -> 1",
          abs(J["h_at_c_small"] - 1) < 1e-6 and abs(J["h_at_c_large"] - 1 / np.sqrt(2)) < 1e-4, J)
    # ---------------------------------------------------------------- relative measure
    ex = R["examples"]
    value("Relative measure", "largest variance mu(1-mu) at mu = 0.05 and 0.99", "0.0475, 0.0099", f"{f(0.05 * 0.95, 4)}, {f(0.99 * 0.01, 4)}")
    value("Relative measure", "sd of the laws with lambda = 9 at mu = 0.5 and 0.1", "0.158, 0.095", f"{f(np.sqrt(0.25 / 10), 3)}, {f(np.sqrt(0.09 / 10), 3)}")
    value("Relative measure", "cbar of three ensembles of 27 with sd 0.044 and means 0.50, 0.15, 0.05", "0.80, 0.68, 0.47",
          ", ".join(f(ex[k]["cbar"], 2) for k in ("beta_0.5", "beta_0.15", "beta_0.05")))
    value("Relative measure", "sample sd of the first two", "0.044, 0.044", f"{f(ex['beta_0.5']['s'], 3)}, {f(ex['beta_0.15']['s'], 3)}")
    value("Relative measure", "25 members at 0.999 and 2 at 0.001: mean, cbar", "0.925, 0.046", f"{f(ex['25-2']['pbar'], 3)}, {f(ex['25-2']['cbar'], 3)}")
    value("Relative measure", "13 members at 0.9 and 14 at 0.1: cbar", "0.050", f(ex["13-14"]["cbar"], 3))
    value("Relative measure", "largest variance at a mean of 0.925", "0.069", f(0.925 * 0.075, 3))
    # ---------------------------------------------------------------- law of the members
    sh = R["shrinkage"]
    value("Law of the members", "median sigma_pred / s on unimodal laws, K = 3, 6, 12, 24, 36", "2.30, 1.40, 1.10, 1.00, 0.99",
          ", ".join(f(sh[str(K)], 2) for K in KS))
    value("Law of the members", "sigma_pred / s for the two polarised ensembles of 27", "0.66, 0.80",
          ", ".join(f(ex[k]["sigma_pred_over_s"], 2) for k in ("25-2", "13-14")))
    value("Law of the members", "moment estimate of lambda for 25 against 2", "-0.023", f(ex["25-2"]["lambda_mom"], 3))
    tm = R["timing"]
    CHECKS.append(("Law of the members", "cost of the approximation relative to the predictive law (machine dependent)",
                   "about 1/6", f"{tm['ratio']:.2f}", True))
    # ---------------------------------------------------------------- reliability signal
    g = R["gate_corr"]
    value("Reliability signal", "Spearman(V, |cbar - 1/2|) at K = 3, 6, 12, 24, 36", "+0.41, -0.52, -0.84, -0.96, -0.98",
          ", ".join(f"{g[str(K)]:+.2f}" for K in KS))
    # ---------------------------------------------------------------- sensitivity
    pri = S["prior"]
    allp = [x for d in pri.values() for x in d.values()]
    value("Sensitivity", "prior: largest share of verdicts changed relative to the uniform prior", "3.6%", pct(max(allp)))
    value("Sensitivity", "prior: largest share for K >= 6", "1.3%",
          pct(max(x for k, d in pri.items() if int(k.split("|")[1]) >= 6 for x in d.values())))
    holds("Sensitivity", "prior: for K <= 6 Beta(2,3) is the farthest from the uniform prior",
          all(max(d, key=d.get) == "2,3" for k, d in pri.items() if int(k.split("|")[1]) <= 6))
    holds("Sensitivity", "prior: from K = 12 every difference is below 0.3%",
          max(x for k, d in pri.items() if int(k.split("|")[1]) >= 12 for x in d.values()) < 0.003)
    la = S["lambda0"]
    ext = {k: max(d["0.1"], d["10"]) for k, d in la.items()}
    value("Sensitivity", "anchoring 0.1K and 10K: largest share changed (at K = 3, tau = 0.25)", "11.2% at 0.25|3",
          f"{pct(max(ext.values()))} at {max(ext, key=ext.get)}")
    pl = {k: max(d[str(1 / 3)], d["3"]) for k, d in la.items()}
    value("Sensitivity", "anchoring K/3 and 3K: largest share at K = 3", "5.0%", pct(max(x for k, x in pl.items() if k.endswith("|3"))))
    holds("Sensitivity", "anchoring K/3 and 3K: below 1.8% for K >= 6", max(x for k, x in pl.items() if int(k.split("|")[1]) >= 6) < 0.018)
    pg = S["plugin"]
    holds("Sensitivity", "approximation: same verdict in at least 98.8% of the cases at K = 3",
          min(x for k, x in pg.items() if k.endswith("|3")) >= 0.988)
    holds("Sensitivity", "approximation: same verdict in at least 99.7% from K = 12",
          min(x for k, x in pg.items() if int(k.split("|")[1]) >= 12) >= 0.997)
    fm = S["fixed_mean"]
    value("Sensitivity", "fixed mean, tau = 0.50: verdicts changed at K = 3, 6", "6.8%, 1.7%", f"{pct(fm['0.5|3']['changed'])}, {pct(fm['0.5|6']['changed'])}")
    holds("Sensitivity", "fixed mean, tau = 0.50: below 1% from K = 12", all(fm[f"0.5|{K}"]["changed"] < 0.01 for K in (12, 24, 36)))
    value("Sensitivity", "fixed mean, tau = 0.25: verdicts changed at K = 3, 6, 12, 24, 36", "15.5%, 4.5%, 1.6%, 1.6%, 0.5%",
          ", ".join(pct(fm[f"0.25|{K}"]["changed"]) for K in KS))
    value("Sensitivity", "fixed mean, K = 3, tau = 0.25: confident only under the fixed / only under the inferred mean",
          "12.0%, 3.5%", f"{pct(fm['0.25|3']['conf_only_fixed'])}, {pct(fm['0.25|3']['conf_only_full'])}")
    holds("Sensitivity", "fixed mean, tau = 0.50: at K = 3 and 6 all changes go towards confident",
          all(fm[f"0.5|{K}"]["conf_only_full"] == 0 for K in (3, 6)))
    holds("Sensitivity", "fixed mean, tau = 0.50: from K = 12 changes go both ways",
          all(fm[f"0.5|{K}"]["conf_only_full"] > 0 and fm[f"0.5|{K}"]["conf_only_fixed"] > 0 for K in (12, 24, 36)))
    # ---------------------------------------------------------------- ten ensembles
    il = R["illustration"]; cb = il["cbar"]
    value("Ten ensembles", "locations of ensembles 1-5", "0.95, 0.05, 0.50, 0.12, 0.88", ", ".join(f(TEN[i].mean(), 2) for i in range(5)))
    value("Ten ensembles", "range of the means of ensembles 6-10", "0.49-0.54", f"{f(TEN[5:].mean(1).min(), 2)}-{f(TEN[5:].mean(1).max(), 2)}")
    value("Ten ensembles", "range of cbar for ensembles 1-5", "0.96-0.98", f"{f(min(cb[:5]), 2)}-{f(max(cb[:5]), 2)}")
    holds("Ten ensembles", "ensemble 3 receives the highest cbar", int(np.argmax(cb)) == 2)
    value("Ten ensembles", "F_pred(0.5) of ensemble 3", "0.54", f(il["F_0.5"][2], 2))
    holds("Ten ensembles", "tau = 0.5: ensembles 1, 2, 4, 5 confident, 3 and 6-10 not confident",
          il["conf_0.5"] == [True, True, False, True, True] + [False] * 5)
    holds("Ten ensembles", "tau = 0.1: only ensemble 7 not confident", il["conf_0.1"] == [True] * 6 + [False] + [True] * 3)
    holds("Ten ensembles", "mean and median of the law never on opposite sides of either threshold",
          not any(il["opposite_0.5"] + il["opposite_0.1"]))
    dF = max(max(abs(a - b) for a, b in zip(il[f"F_{t}"], il[f"Fplug_{t}"])) for t in (0.5, 0.1))
    value("Ten ensembles", "largest change of F_pred(tau) under the approximation", "0.012", f(dF, 3))
    holds("Ten ensembles", "the approximation changes no verdict",
          il["conf_0.5"] == il["confplug_0.5"] and il["conf_0.1"] == il["confplug_0.1"])
    # ---------------------------------------------------------------- known truth
    r3 = [v(d, t, 3, "rho") for d in DESIGNS for t in (0.5, 0.1)]
    r36 = [v(d, t, 36, "rho") for d in DESIGNS for t in (0.5, 0.1)]
    value("Known truth", "Spearman(relative dispersion, eta) at K = 3", "0.39-0.59", f"{f(min(r3), 2)}-{f(max(r3), 2)}")
    value("Known truth", "Spearman(relative dispersion, eta) at K = 36", "0.76-0.91", f"{f(min(r36), 2)}-{f(max(r36), 2)}")
    conds = [(d, t, K) for d in DESIGNS for t in (0.5, 0.1) for K in KS]
    others = ("confidex", "confidex plug-in", "band") + SIMPLE_READINGS
    holds("Known truth", "the concentration is the poorest reading in every condition",
          all(v(*c, "concentration") < min(v(*c, m) for m in others) for c in conds))
    conc = [v(*c, "concentration") for c in conds]
    value("Known truth", "range of the AUROC of the concentration", "0.55-0.76", f"{f(min(conc), 2)}-{f(max(conc), 2)}")
    value("Known truth", "informative, tau = 0.5: signal minus best simpler reading at K = 3, 6", "0.007, 0.002",
          f"{f(gap('informative', 0.5, 3), 3)}, {f(gap('informative', 0.5, 6), 3)}")
    holds("Known truth", "informative, tau = 0.5: bootstrap intervals exclude zero at K = 3, 6",
          excl("informative", 0.5, 3) and excl("informative", 0.5, 6))
    holds("Known truth", "informative, tau = 0.5: level from K = 12", all(abs(gap("informative", 0.5, K)) < 0.0015 for K in (12, 24, 36)))
    value("Known truth", "informative, tau = 0.1, K = 3: behind the logit studentised margin by", "0.006 (logit studentised)",
          f"{f(-gap('informative', 0.1, 3), 3)} ({best_simple('informative', 0.1, 3)})")
    value("Known truth", "informative, tau = 0.1: ahead of the logit studentised margin at K = 6, 12, 24, 36",
          "0.008, 0.013, 0.019, 0.022",
          ", ".join(f(v("informative", 0.1, K, "confidex") - v("informative", 0.1, K, "logit studentised"), 3) for K in (6, 12, 24, 36)))
    holds("Known truth", "informative, tau = 0.1: bootstrap intervals exclude zero from K = 6",
          all(excl("informative", 0.1, K) for K in (6, 12, 24, 36)))
    value("Known truth", "informative, tau = 0.1, K = 12: AUROC of the signal and of the distance", "0.972, 0.885",
          f"{f(v('informative', 0.1, 12, 'confidex'), 3)}, {f(v('informative', 0.1, 12, 'distance'), 3)}")
    value("Known truth", "informative, tau = 0.1, K = 12: AUROC of the logit distance", "0.911", f(v("informative", 0.1, 12, "logit distance"), 3))
    d3 = [v(d, 0.5, 3, "distance") - v(d, 0.5, 3, "confidex") for d in ("non-informative", "mixed")]
    d24 = [v(d, 0.5, K, "distance") - v(d, 0.5, K, "confidex") for d in ("non-informative", "mixed") for K in (24, 36)]
    value("Known truth", "non-informative and mixed, tau = 0.5: distance minus signal at K = 3", "0.002-0.005", f"{f(min(d3), 3)}-{f(max(d3), 3)}")
    value("Known truth", "non-informative and mixed, tau = 0.5: largest distance minus signal from K = 24", "0.047", f(max(d24), 3))
    d01 = [v(d, 0.1, K, "logit distance") - v(d, 0.1, K, "confidex") for d in ("non-informative", "mixed") for K in KS]
    value("Known truth", "non-informative and mixed, tau = 0.1: largest logit distance minus signal", "0.028", f(max(d01), 3))
    holds("Known truth", "budget 10%, tau = 0.5, informative: no interval excludes zero", not any(star("informative", 0.5, K) for K in KS))
    c05 = [pts(d, 0.5, K) for d in ("non-informative", "mixed") for K in KS]
    value("Known truth", "budget 10%, tau = 0.5, non-informative and mixed: largest cost, points", "2.5", f(-min(c05), 1))
    value("Known truth", "budget 10%, tau = 0.5, non-informative and mixed: intervals excluding zero", "8 of 10",
          f"{sum(star(d, 0.5, K) for d in ('non-informative', 'mixed') for K in KS)} of 10")
    holds("Known truth", "budget 10%, tau = 0.1: the signal is never behind from K = 6",
          all(pts(d, 0.1, K) >= -0.05 for d in DESIGNS for K in (6, 12, 24, 36)))
    value("Known truth", "budget 10%, tau = 0.1, informative: ahead at K = 12, 24, 36, points", "5.4, 7.2, 8.7",
          ", ".join(f(pts("informative", 0.1, K), 1) for K in (12, 24, 36)))
    value("Known truth", "budget 10%, tau = 0.1, informative: behind at K = 3, points", "3.7", f(-pts("informative", 0.1, 3), 1))
    value("Known truth", "budget 10%, tau = 0.1: largest gain, non-informative and mixed, points", "1.3, 2.5",
          f"{f(max(pts('non-informative', 0.1, K) for K in KS), 1)}, {f(max(pts('mixed', 0.1, K) for K in KS), 1)}")
    value("Known truth", "budget 10%, tau = 0.1, K >= 6: intervals excluding zero in favour of the signal", "7 of 12",
          f"{sum(star(d, 0.1, K) and pts(d, 0.1, K) > 0 for d in DESIGNS for K in (6, 12, 24, 36))} of 12")
    holds("Known truth", "budget 5%, tau = 0.1: the advantage is larger than at 10% in every design and size",
          all(pts(d, 0.1, K, "0.05") > pts(d, 0.1, K) for d in DESIGNS for K in KS))
    p20 = {(d, K): pts(d, 0.1, K, "0.2") for d in DESIGNS for K in KS}
    value("Known truth", "budget 20%, tau = 0.1, informative, K = 3: behind by, points", "6.2", f(-p20[("informative", 3)], 1))
    holds("Known truth", "budget 20%, tau = 0.1: every other difference is about two points or less",
          max(abs(x) for k, x in p20.items() if k != ("informative", 3)) < 2.5)
    op = [v(d, 0.1, 3, "opposite") for d in DESIGNS]
    value("Known truth", "tau = 0.1, K = 3: law on the side opposite to the decision", "2.1%-2.3%", f"{pct(min(op))}-{pct(max(op))}")
    band = [v(d, 0.1, 3, "confidex") - v(d, 0.1, 3, "band") for d in DESIGNS]
    value("Known truth", "tau = 0.1, K = 3: AUROC lost without the side condition", "0.009-0.022", f"{f(min(band), 3)}-{f(max(band), 3)}")
    value("Known truth", "informative, tau = 0.1, K = 3: AUROC without and with the side condition", "0.861, 0.884",
          f"{f(v('informative', 0.1, 3, 'band'), 3)}, {f(v('informative', 0.1, 3, 'confidex'), 3)}")
    holds("Known truth", "tau = 0.5: the criterion with and without the side condition coincide",
          all(abs(v(d, 0.5, K, "confidex") - v(d, 0.5, K, "band")) < 5e-4 for d in DESIGNS for K in KS))
    ag = [v(*c, "agree_plugin") for c in conds]
    au = [abs(v(*c, "confidex") - v(*c, "confidex plug-in")) for c in conds]
    holds("Known truth", "approximation: same verdict in at least 98.7% of the cases", min(ag) >= 0.987, min(ag))
    holds("Known truth", "approximation: AUROC within 0.004", max(au) <= 0.0045, max(au))
    holds("Known truth", "the ordering by u beats the gate in all thirty conditions", all(v(*c, "rank_recall") > v(*c, "gate_recall") for c in conds))
    value("Known truth", "informative, tau = 0.5, K = 12: errors captured by the ordering and by the gate", "94.0%, 76.2%",
          f"{pct(v('informative', 0.5, 12, 'rank_recall'))}, {pct(v('informative', 0.5, 12, 'gate_recall'))}")
    value("Known truth", "mixed, tau = 0.1, K = 12: errors captured by the ordering and by the gate", "62.1%, 41.6%",
          f"{pct(v('mixed', 0.1, 12, 'rank_recall'))}, {pct(v('mixed', 0.1, 12, 'gate_recall'))}")
    holds("Known truth", "tau = 0.1, K <= 12: the cases the gate sets aside are wrong less often than those it admits as confident",
          all(v(d, 0.1, K, "gate_err_excluded") < v(d, 0.1, K, "gate_err_admitted_conf") for d in DESIGNS for K in (3, 6, 12)))
    # ---------------------------------------------------------------- limits
    al = R["anchoring_location"]
    holds("Limits", "three members near 0.012: the approximation is J-shaped", al["J_or_U"] == 1.0)
    value("Limits", "three members near 0.012: predictive law wider than the sample dispersion by about", "15", str(round(al["sigma_pred_over_s"])))
    value("Limits", "fixed mean, K = 3, tau = 0.25: verdicts changed", "15.5%", pct(fm["0.25|3"]["changed"]))
    # ---------------------------------------------------------------- Table A.2
    for d in DESIGNS:
        for t in (0.5, 0.1):
            for b, lab in (("0.05", "5%"), ("0.1", "10%"), ("0.2", "20%")):
                cells = []
                for K in KS:
                    x = pts(d, t, K, b)
                    cells.append(("+0.0" if abs(x) < 0.05 else f"{x:+.1f}") + ("*" if b == "0.1" and star(d, t, K) else ""))
                CHECKS.append(("Table A.2", f"{d}, tau = {t}, budget {lab}", "", " ".join(cells), True))
    expected_a2 = {
        ("informative", 0.5): ("-0.5 -1.7 -0.4 -1.9 -0.7", "+0.2 +0.0 -0.4 -1.0 -0.2", "+0.8 +1.0 +0.1 +0.0 +0.0"),
        ("informative", 0.1): ("+1.9 +8.4 +14.9 +24.9 +30.2", "-3.7* +1.0 +5.4* +7.2* +8.7*", "-6.2 -0.3 -0.3 +0.0 +0.0"),
        ("non-informative", 0.5): ("-0.4 -0.4 -0.3 -1.0 -0.7", "+0.4 -0.9 -1.3* -1.3* -1.7*", "-1.0 -4.1 -2.5 -3.9 -3.9"),
        ("non-informative", 0.1): ("+2.3 +3.6 +4.5 +3.8 +4.7", "+0.6 +1.3* +1.0* +0.7 +0.3", "-2.3 +1.7 +0.5 -0.2 +0.0"),
        ("mixed", 0.5): ("+0.1 +0.0 -0.5 -0.4 -0.6", "-0.9* -1.8* -1.5* -1.9* -2.5*", "-0.7 -1.8 -2.7 -5.2 -3.7"),
        ("mixed", 0.1): ("+1.6 +2.8 +4.6 +3.5 +4.2", "+1.2 +1.2* +2.5* +0.7 +0.3", "-1.7 +2.2 +1.0 +0.0 -0.7"),
    }
    rows = [c for c in CHECKS if c[0] == "Table A.2"]
    CHECKS[:] = [c for c in CHECKS if c[0] != "Table A.2"]
    for (d, t), printed in expected_a2.items():
        for (lab, p) in zip(("5%", "10%", "20%"), printed):
            got = next(c[3] for c in rows if c[1] == f"{d}, tau = {t}, budget {lab}")
            value("Table A.2", f"{d}, tau = {t}, budget {lab} (K = 3 ... 36; * interval excludes zero)", p, got)

    # ---------------------------------------------------------------- report
    failed = [c for c in CHECKS if not c[4]]
    for where, what, printed, got, ok in CHECKS:
        if show_all or not ok:
            print(f"{'ok  ' if ok else 'FAIL'}  [{where}] {what}: thesis {printed} | results {got}")
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check the numbers of Chapter 3 against the results.")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--list", action="store_true", help="print every check, not only the failures")
    a = ap.parse_args()
    sys.exit(main(a.results, a.list))
