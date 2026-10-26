"""Check every number that Chapter 4 (and its Appendix) reports against the results.

    python check_values.py            # reads results/chapter4_results.json and results/chapter4_diagnostics.json
    python check_values.py --list     # also prints every check, not only the failures

Each check states where the value appears in the thesis, what it measures, the value as the thesis
prints it, and the value recomputed from the results with the same rounding. Statements that are not
numbers are checked as conditions and printed as "holds". The settings of the design (sizes, grids,
hyperparameters, library versions) are checked against the code and requirements.txt. The script exits
with status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS, ST, TH, KS = ("carclaims", "ulb"), ("s1", "s2"), ("tau_F1", "tau_cost"), (3, 6, 12, 24, 36)
CHECKS: list[tuple[str, str, str, str, bool]] = []


def value(where, what, printed, recomputed):
    CHECKS.append((where, what, str(printed), str(recomputed), str(printed) == str(recomputed)))


def near(where, what, printed, recomputed, tolerance, fmt="{:.3f}"):
    """A number that is not reproduced bit for bit (see README, Determinism): it must lie within ``tolerance``."""
    CHECKS.append((where, what, f"{printed} (within {tolerance:g})", fmt.format(recomputed),
                   abs(recomputed - printed) <= tolerance))


def holds(where, what, condition, detail=""):
    CHECKS.append((where, what, "holds", "holds" if condition else f"FAILS {detail}", bool(condition)))


def r1(x):
    return round(100 * x, 1)


def r0(x):
    return round(100 * x)


def span(v, f="{:.1f}"):
    return f"{f.format(min(v))} to {f.format(max(v))}"


def main(results: Path, show_all: bool) -> int:
    R = json.loads((results / "chapter4_results.json").read_text())

    def idx(rows, *e):
        return {tuple(r[k] for k in ("dataset", "strategy", "threshold", "K", "budget") + e): r for r in rows}
    P, A, C = idx(R["main"]), idx(R["allocation"]), idx(R["within_class"], "class")
    AB, G = idx(R["ablations"], "variant"), idx(R["gate_allocation"])
    PM = {(r["dataset"], r["strategy"], r["family"]): r for r in R["performance_members"]}
    PE = {(r["dataset"], r["strategy"], r["threshold"]): r for r in R["performance_ensemble"]}
    DATA = {r["dataset"]: r for r in R["data"]}

    # ---------------------------------------------------------------- data and partitions
    c_, u_ = DATA["carclaims"], DATA["ulb"]
    value("Data and partitions", "carclaims: claims, frauds, share of frauds (%), variables", "15,419, 923, 6.0, 30",
          f"{c_['rows']:,}, {c_['frauds']:,}, {100 * c_['frauds'] / c_['rows']:.1f}, {c_['variables']}")
    value("Data and partitions", "ULB: transactions, frauds, share of frauds (%), variables", "284,807, 492, 0.17, 29",
          f"{u_['rows']:,}, {u_['frauds']:,}, {100 * u_['frauds'] / u_['rows']:.2f}, {u_['variables']}")
    parts = ("train", "calibration", "validation", "test")
    holds("Data and partitions", "every partition assigns 55%, 10%, 15% and 20% of the cases to the four parts",
          all([round(100 * x["parts"][p]["share"]) for p in parts] == [55, 10, 15, 20]
              and [round(100 * x["parts"][p]["rows_min"] / x["rows"]) for p in parts] == [55, 10, 15, 20]
              and [round(100 * x["parts"][p]["rows_max"] / x["rows"]) for p in parts] == [55, 10, 15, 20] for x in (c_, u_)))
    value("Data and partitions", "ULB: transactions in groups of identical cases, groups, discordant groups", "14,293, 5,149, 0",
          f"{u_['groups']['rows_in_groups_with_several_rows']:,}, {u_['groups']['groups_with_several_rows']:,}, "
          f"{u_['groups']['discordant_groups']}")
    holds("Data and partitions", "carclaims has no identical cases",
          c_["groups"]["groups_with_several_rows"] == 0 and c_["groups"]["groups"] == c_["rows"])
    ctx = {d: [DATA[d]["parts"]["train"][f"rows_{x}"] * f for x, f in (("min", 0.60), ("max", 0.85))] for d in DS}
    value("Appendix, TabPFN", "size of the context of a member: carclaims, ULB (about)", "5,100-7,200, 94,000-133,000",
          f"{round(ctx['carclaims'][0], -2):,.0f}-{round(ctx['carclaims'][1], -2):,.0f}, "
          f"{round(ctx['ulb'][0], -3):,.0f}-{round(ctx['ulb'][1], -3):,.0f}")

    # ---------------------------------------------------------------- members and ensembles
    f3 = lambda x: f"{x:.3f}"

    def fam_range(d, s="s2"):
        v = {k[2]: r["AUROC_median"] for k, r in PM.items() if k[:2] == (d, s)}
        lo, hi = min(v, key=v.get), max(v, key=v.get)
        return f"{f3(v[lo])} ({lo}) to {f3(v[hi])} ({hi})"
    value("Members and ensembles", "carclaims: range of the median AUROC across families", "0.773 (MLP) to 0.841 (CatBoost)", fam_range("carclaims"))
    value("Members and ensembles", "ULB: range of the median AUROC across families", "0.942 (RandomForest) to 0.978 (TabPFN)", fam_range("ulb"))
    value("Members and ensembles", "median AUROC of the one-family members, carclaims and ULB", "0.835, 0.975",
          ", ".join(f3(PM[(d, "s1", "LightGBM_constrained")]["AUROC_median"]) for d in DS))
    ens_auc = lambda d: [PE[(d, s, "tau_F1")]["AUROC_mean"] for s in ST]
    value("Members and ensembles", "AUROC of the ensembles of 36: carclaims; ULB", "0.848 to 0.852; 0.978 to 0.978",
          f"{span(ens_auc('carclaims'), '{:.3f}')}; {span(ens_auc('ulb'), '{:.3f}')}")
    holds("Members and ensembles", "the ensembles of 36 discriminate about as well as their best family (within 0.015)",
          all(abs(PE[(d, s, "tau_F1")]["AUROC_mean"] - max(r["AUROC_median"] for k, r in PM.items() if k[:2] == (d, s))) <= 0.015
              for d in DS for s in ST))
    ens = lambda d, t, c: span([PE[(d, s, t)][f"{c}_mean"] for s in ST], "{:.0%}")
    value("Members and ensembles", "tau_F1: precision carclaims, ULB; recall carclaims, ULB",
          "21% to 21%, 91% to 92%; 55% to 60%, 77% to 78%",
          f"{ens('carclaims', 'tau_F1', 'precision')}, {ens('ulb', 'tau_F1', 'precision')}; "
          f"{ens('carclaims', 'tau_F1', 'recall')}, {ens('ulb', 'tau_F1', 'recall')}")
    value("Members and ensembles", "cost threshold: precision carclaims, ULB; recall carclaims, ULB",
          "18% to 18%, 23% to 35%; 72% to 73%, 84% to 86%",
          f"{ens('carclaims', 'tau_cost', 'precision')}, {ens('ulb', 'tau_cost', 'precision')}; "
          f"{ens('carclaims', 'tau_cost', 'recall')}, {ens('ulb', 'tau_cost', 'recall')}")
    for d, t, printed in (("carclaims", None, "79 to 83"), ("ulb", "tau_F1", "8 to 11"), ("ulb", "tau_cost", "65 to 80")):
        v = [C[(d, s, tt, K, "rho=1", 1)] for s in ST for tt in ((t,) if t else TH) for K in KS]
        q = [r["errors_class"] / r["n_class"] for r in v]
        value("Members and ensembles", f"{d}{', ' + t if t else ''}: false positives among the cases classified as positive (%)",
              printed, f"{round(100 * min(q))} to {round(100 * max(q))}")

    # ---------------------------------------------------------------- the signal against the simpler readings
    v = [P[(d, s, t, K, "rho=1")] for d in DS for s in ST for t in TH for K in (3, 6) if not (d == "ulb" and t == "tau_F1")]
    value("Signal against simpler readings", "K = 3, 6 (except ULB at tau_F1): gain over the best simpler reading, points",
          "3.7 to 28.0", span([r1(r["mean"]) for r in v]))
    holds("Signal against simpler readings", "... with intervals above zero in every partition", all(r["above_zero"] == r["partitions"] for r in v))
    v = [P[("ulb", s, "tau_F1", K, "rho=1")] for s in ST for K in KS]
    value("Signal against simpler readings", "ULB, tau_F1: differences; most partitions above; most below", "-0.6 to 1.2; 3; 2",
          f"{span([r1(r['mean']) for r in v])}; {max(r['above_zero'] for r in v)}; {max(r['below_zero'] for r in v)}")
    v = [P[("carclaims", s, t, K, "rho=1")] for s in ST for t in TH for K in (24, 36)]
    value("Signal against simpler readings", "carclaims, K >= 24: differences, points", "-2.3 to 0.1", span([r1(r["mean"]) for r in v]))
    v = [P[("carclaims", "s2", t, K, "rho=1")] for t in TH for K in (24, 36)]
    value("Signal against simpler readings", "carclaims, nine families, K >= 24: cost, points; partitions below zero",
          "0.7 to 2.3; 3 to 5", f"{span([-r1(r['mean']) for r in v])}; {min(r['below_zero'] for r in v)} to {max(r['below_zero'] for r in v)}")
    value("Signal against simpler readings", "ULB, cost threshold, K = 24, 36: one family; nine families, points",
          "[-1.3, -1.9]; [10.6, 10.3]",
          f"{[r1(P[('ulb', 's1', 'tau_cost', K, 'rho=1')]['mean']) for K in (24, 36)]}; "
          f"{[r1(P[('ulb', 's2', 'tau_cost', K, 'rho=1')]['mean']) for K in (24, 36)]}")

    # ---------------------------------------------------------------- where the budget goes
    k3 = [P[("carclaims", s, t, 3, "rho=1")] for s in ST for t in TH]
    value("Where the budget goes", "carclaims, K = 3: share of referrals to predicted positives, signal; best simpler reading (%)",
          "73 to 88; 50 to 57", f"{span([r0(r['share_budget_positive']) for r in k3], '{}')}; "
                                f"{span([r0(r['choice_share_budget_positive']) for r in k3], '{}')}")
    v = [P[("ulb", "s2", "tau_cost", K, "rho=1")] for K in KS]
    value("Where the budget goes", "ULB, cost threshold, nine families: the same shares at every size (%)", "85 to 92; 72 to 74",
          f"{span([r0(r['share_budget_positive']) for r in v], '{}')}; {span([r0(r['choice_share_budget_positive']) for r in v], '{}')}")
    v = [A[(d, s, t, K, "rho=1")] for d in DS for s in ST for t in TH for K in (6, 12, 24, 36) if not (d == "ulb" and t == "tau_F1")]
    value("Where the budget goes", "allocated reading ahead from six members, points; partitions not below zero", "4.7 to 26.9; 1",
          f"{span([-r1(r['mean']) for r in v])}; {sum(r['partitions'] - r['below_zero'] for r in v)}")
    v = [A[(d, s, t, 3, "rho=1")] for d in DS for s in ST for t in TH]
    value("Where the budget goes", "K = 3: signal minus allocated reading, points", "-7.8 to 3.8", span([r1(r["mean"]) for r in v]))
    v = [A[("ulb", s, "tau_F1", K, "rho=1")] for s in ST for K in KS]
    value("Where the budget goes", "ULB, tau_F1: signal ahead of the allocated reading, points; most partitions above zero", "1.0 to 3.8; 4",
          f"{span([r1(r['mean']) for r in v])}; {max(r['above_zero'] for r in v)}")
    v = [C[(d, s, t, K, "rho=1", 1)] for d in DS for s in ST for t in TH for K in KS if not (d == "ulb" and t == "tau_F1")]
    value("Where the budget goes", "within predicted positives: differences, points; partitions above zero (max)", "-1.1 to 0.0; 0",
          f"{span([r1(r['mean']) + 0.0 for r in v]).replace('-0.0', '0.0')}; {max(r['above_zero'] for r in v)}")
    v = [C[("ulb", s, "tau_F1", K, "rho=1", 1)] for s in ST for K in KS]
    holds("Where the budget goes", "ULB, tau_F1, predicted positives: about eighty cases and a budget of five to seven",
          80 <= min(r["n_class"] for r in v) and max(r["n_class"] for r in v) < 90 and 4.5 <= min(r["k"] for r in v) and max(r["k"] for r in v) <= 7.5)
    value("Where the budget goes", "ULB, tau_F1, predicted positives: differences; most above; most below", "-1.4 to 0.8; 2; 0",
          f"{span([r1(r['mean']) for r in v])}; {max(r['above_zero'] for r in v)}; {max(r['below_zero'] for r in v)}")
    v = [C[(d, s, t, K, "rho=0.5", 1)] for d in DS for s in ST for t in TH for K in KS if not (d == "ulb" and t == "tau_F1")]
    w = [C[(d, s, t, K, "rho=2", 1)] for d in DS for s in ST for t in TH for K in KS if not (d == "ulb" and t == "tau_F1")]
    value("Where the budget goes", "predicted positives, half the budget: differences, points", "-0.9 to 0.1", span([r1(r["mean"]) for r in v]))
    holds("Where the budget goes", "predicted positives, twice the budget: both readings refer the whole class",
          all(r["k"] == r["n_class"] and r["mean"] == 0 for r in w))
    v = [C[(d, s, t, K, b, 0)] for d in DS for s in ST for t in TH for K in KS for b in ("rho=0.5", "rho=2")]
    value("Where the budget goes", "missed frauds, half and twice the budget: differences, points", "-2.4 to 0.9", span([r1(r["mean"]) for r in v]))
    v = [A[(d, s, t, K, "rho=2")] for d in DS for s in ST for t in TH for K in (6, 12, 24, 36) if not (d == "ulb" and t == "tau_F1")]
    value("Where the budget goes", "twice the budget: allocated reading ahead from six members, points; partitions not below zero",
          "1.5 to 20.0; 3", f"{span([-r1(r['mean']) for r in v])}; {sum(r['partitions'] - r['below_zero'] for r in v)}")
    v = [A[("carclaims", s, t, 3, "rho=0.5")] for s in ST for t in TH]
    w = [A[("ulb", s, "tau_cost", K, "rho=0.5")] for s in ST for K in KS]
    q = [P[("ulb", s, "tau_cost", K, "rho=0.5")]["share_budget_positive"] for s in ST for K in KS]
    x = [A[("carclaims", s, t, K, "rho=0.5")] for s in ST for t in TH for K in (24, 36)]
    value("Where the budget goes", "half the budget: carclaims K = 3; ULB cost threshold; share of referrals to positives (%); carclaims K >= 24",
          "3.6 to 7.4; -0.8 to 4.0; 90 to 100; 1.9 to 12.5",
          f"{span([r1(r['mean']) for r in v])}; {span([r1(r['mean']) for r in w])}; {span([r0(z) for z in q], '{}')}; "
          f"{span([-r1(r['mean']) for r in x])}")
    holds("Where the budget goes", "half the budget, carclaims K = 3: ahead in every partition", all(r["above_zero"] == 5 for r in v))

    # ---------------------------------------------------------------- the verdict
    def kk(d, t=None):
        return [(P[(d, s, tt, K, "k_alpha")]["k"], P[(d, s, tt, K, "rho=1")]["k"]) for s in ST for tt in ((t,) if t else TH) for K in KS]
    a, b = kk("carclaims"), kk("ulb", "tau_cost")
    holds("The verdict", "the verdict declares not confident fewer cases than the expected errors, at every size and threshold",
          all(x_ < y_ for x_, y_ in kk("carclaims") + kk("ulb", "tau_cost") + kk("ulb", "tau_F1")))
    value("The verdict", "cases declared not confident and expected errors: carclaims; ULB at the cost threshold",
          "147 to 506 vs 452 to 684; 69 to 301 vs 169 to 354",
          f"{span([x_ for x_, _ in a], '{:.0f}')} vs {span([y_ for _, y_ in a], '{:.0f}')}; "
          f"{span([x_ for x_, _ in b], '{:.0f}')} vs {span([y_ for _, y_ in b], '{:.0f}')}")
    v = [P[(d, s, t, 3, "k_alpha")]["share_budget_positive"] for d in DS for s in ST for t in TH if not (d == "ulb" and t == "tau_F1")]
    value("The verdict", "K = 3: share of the verdict's referrals to predicted positives (%)", "98 to 100", span([r0(x_) for x_ in v], "{}"))
    v = [P[("carclaims", s, t, K, "k_alpha")] for s in ST for t in TH for K in (3, 6)]
    w = [P[("ulb", "s2", "tau_cost", K, "k_alpha")] for K in KS]
    value("The verdict", "at its own budget, gain: carclaims K = 3, 6; ULB cost threshold nine families; fewest partitions above zero",
          "5.2 to 24.8; 10.2 to 16.9; 9", f"{span([r1(r['mean']) for r in v])}; {span([r1(r['mean']) for r in w])}; {min(r['above_zero'] for r in w)}")
    holds("The verdict", "carclaims K = 3, 6: ahead in every partition", all(r["above_zero"] == r["partitions"] for r in v))
    u = [r1(P[("ulb", "s1", "tau_cost", K, "k_alpha")]["mean"]) for K in KS]
    v = [P[("carclaims", "s2", t, K, "k_alpha")] for t in TH for K in (24, 36)]
    value("The verdict", "ULB, cost threshold, one family: advantage at K = 3 and 36; carclaims nine families K >= 24: cost",
          "24.6, 0.2; 2.0 to 3.1", f"{u[0]}, {u[-1]}; {span([-r1(r['mean']) for r in v])}")
    holds("The verdict", "ULB, cost threshold, one family: the advantage falls with K", u == sorted(u, reverse=True))
    v = [A[("carclaims", s, t, 3, "k_alpha")] for s in ST for t in TH]
    w = [A[("ulb", s, "tau_cost", K, "k_alpha")] for s in ST for K in KS]
    x = [A[("carclaims", s, t, K, "k_alpha")] for s in ST for t in TH for K in (12, 24, 36)]
    value("The verdict", "against the allocated reading: carclaims K = 3; ULB cost threshold; carclaims K >= 12 (behind)",
          "4.4 to 10.4; 2.3 to 5.7; 1.2 to 5.5",
          f"{span([r1(r['mean']) for r in v])}; {span([r1(r['mean']) for r in w])}; {span([-r1(r['mean']) for r in x])}")
    holds("The verdict", "carclaims K = 3: ahead of the allocated reading in every partition", all(r["above_zero"] == 5 for r in v))
    v = [C[(d, s, t, K, "k_alpha", 1)] for d in DS for s in ST for t in TH for K in KS if not (d == "ulb" and t == "tau_F1")]
    value("The verdict", "within predicted positives at the verdict's split: differences; intervals away from zero", "-0.3 to 0.4; 0",
          f"{span([r1(r['mean']) for r in v])}; {max(r['above_zero'] + r['below_zero'] for r in v)}")
    a = [x_ for x_, _ in kk("ulb", "tau_F1")]; b = [y_ for _, y_ in kk("ulb", "tau_F1")]
    value("The verdict", "ULB, tau_F1: cases referred by the verdict; expected errors", "4 to 8; 24 to 27",
          f"{round(min(a))} to {round(max(a))}; {round(min(b))} to {round(max(b))}")

    # ---------------------------------------------------------------- the missed frauds
    v = [C[(d, s, t, K, "rho=1", 0)] for d in DS for s in ST for t in TH for K in KS]
    value("The missed frauds", "signal minus best simpler reading among the cases classified as negative, points", "-1.3 to 1.2",
          span([r1(r["mean"]) for r in v]))

    def both(d, t):
        w_ = [C[(d, s, t, K, "rho=1", 0)] for s in ST for K in KS]
        return [100 * r["confidex"] for r in w_] + [100 * (r["confidex"] - r["mean"]) for r in w_], w_
    b, w = both("carclaims", "tau_F1"); b2, w2 = both("carclaims", "tau_cost")
    value("The missed frauds", "carclaims: missed frauds found by the two readings; random; perfect (%)",
          "9.5 to 14.8; 2.2 to 3.4; 94.0 to 98.0",
          f"{span(b + b2)}; {span([100 * r['random'] for r in w + w2])}; {span([100 * r['perfect'] for r in w + w2])}")
    b, w = both("ulb", "tau_F1"); b2, w2 = both("ulb", "tau_cost")
    value("The missed frauds", "ULB: found at tau_F1; perfect at tau_F1; at most at the cost threshold (%)",
          "13.5 to 17.5; 82.4 to 84.1; 1.2", f"{span(b)}; {span([100 * r['perfect'] for r in w])}; {max(b2):.1f}")
    allc = [x_ for d in DS for t in TH for x_ in both(d, t)[0]]
    allp = [100 * r["perfect"] for r in C.values() if r["budget"] == "rho=1" and r["class"] == 0]
    value("Conclusions", "the best reading recovers at most; a perfect ordering would recover (%)", "17.5; 77.9 to 98.0",
          f"{max(allc):.1f}; {span(allp)}")

    # ---------------------------------------------------------------- fixed shares of the cases
    v = [P[("carclaims", s, t, K, "share=10%")] for s in ST for t in TH for K in (3, 6)]
    w = [P[("carclaims", "s2", t, K, "share=10%")] for t in TH for K in (24, 36)]
    value("Fixed shares", "carclaims, 10% of the cases: gain with K = 3, 6; cost with nine families from K = 24, points",
          "5.3 to 17.9; 2.0 to 2.9", f"{span([r1(r['mean']) for r in v])}; {span([-r1(r['mean']) for r in w])}")
    holds("Fixed shares", "carclaims, 10%, K = 3, 6: ahead in every partition", all(r["above_zero"] == 5 for r in v))
    v = [P[("ulb", s, t, K, "share=1%")] for s in ST for t in TH for K in KS]
    holds("Fixed shares", "ULB: 1% of the cases exceeds the expected errors",
          all(r["k"] > P[(r["dataset"], r["strategy"], r["threshold"], r["K"], "rho=1")]["k"] for r in v))
    value("Fixed shares", "ULB, 1%: lowest share captured by a perfect ordering (%)", "98.9", f"{min(r1(r['perfect']) for r in v):.1f}")
    v = [P[("ulb", s, "tau_cost", K, b_)] for s in ST for K in KS for b_ in ("share=1%", "share=5%", "share=10%")]
    value("Fixed shares", "ULB, cost threshold: gain at every share and size, points", "1.2 to 14.7", span([r1(r["mean"]) for r in v]))
    v = [P[("ulb", "s1", "tau_F1", K, b_)] for K in (12, 24, 36) for b_ in ("share=5%", "share=10%")]
    w = [P[("ulb", "s2", "tau_F1", K, b_)] for K in (24, 36) for b_ in ("share=5%", "share=10%")]
    value("Fixed shares", "ULB, tau_F1, 5% and 10%: cost one family from K = 12; nine families from K = 24, points",
          "4.4 to 22.0; 2.2 to 9.8", f"{span([-r1(r['mean']) for r in v])}; {span([-r1(r['mean']) for r in w])}")
    holds("Fixed shares", "ULB, tau_F1: the errors are a few dozen and these budgets reach thousands of cases",
          max(y_ for _, y_ in kk("ulb", "tau_F1")) < 100 and P[("ulb", "s1", "tau_F1", 3, "share=5%")]["k"] >= 1000)

    # ---------------------------------------------------------------- ablations
    v = [AB[("carclaims", s, t, 3, "rho=1", "lambda0 = 36")] for s in ST for t in TH]
    u = [AB[("ulb", s, t, 3, "rho=1", "lambda0 = 36")] for s in ST for t in TH]
    w = [AB[(d, s, t, K, "rho=1", "lambda0 = 36")] for d in DS for s in ST for t in TH for K in (24, 36)]
    value("Ablations", "lambda0 = 36: cost with K = 3 on carclaims; up to on ULB; at most from K = 24, points", "15.2 to 21.0; 17.8; 0.4",
          f"{span([-r1(r['mean']) for r in v])}; {max(-r1(r['mean']) for r in u):.1f}; {max(abs(r1(r['mean'])) for r in w)}")
    nd = [AB[(d, s, t, K, "rho=1", "non-directional")] for d in DS for s in ST for t in TH for K in KS]
    ex = [r for r in nd if (r["dataset"], r["strategy"], r["threshold"]) == ("ulb", "s2", "tau_cost")]
    value("Ablations", "without the side condition: largest change elsewhere; cost on ULB, cost threshold, nine families, points",
          "0.4; 1.4 to 4.4", f"{max(abs(r1(r['mean'])) for r in nd if r not in ex)}; {span([-r1(r['mean']) for r in ex])}")
    cl = [r for r in R["ablations"] if r["variant"] == "clip 1e-4"]
    value("Ablations", "clipping 1e-4 on ULB: largest change, points", "0.7", max(abs(r1(r["mean"])) for r in cl))
    pl = [AB[(d, s, t, K, "rho=1", "confidex plug-in")] for d in DS for s in ST for t in TH for K in KS]
    p3 = [r for r in pl if r["dataset"] == "carclaims" and r["K"] == 3]
    value("Ablations", "law at cbar: range; carclaims K = 3; largest change from K = 12, points", "-0.3 to 1.7; 1.1 to 1.7; 0.2",
          f"{span([r1(r['mean']) for r in pl])}; {span([r1(r['mean']) for r in p3])}; {max(abs(r1(r['mean'])) for r in pl if r['K'] >= 12)}")
    holds("Ablations", "law at cbar, carclaims K = 3: intervals above zero in every partition", all(r["above_zero"] == r["partitions"] for r in p3))
    g = [AB[("carclaims", s, t, K, "rho=1", "gate")] for s in ST for t in TH for K in KS]
    value("Ablations", "gate on carclaims: combinations ahead of the signal (of 20); largest gain, points", "19; 12.6",
          f"{sum(r['mean'] > 0 for r in g)}; {max(r1(r['mean']) for r in g):.1f}")
    ga = [G[("carclaims", s, t, K, "rho=1")] for s in ST for t in TH for K in KS]
    value("Ablations", "gate on carclaims: set-aside cases classified as positive; all cases classified as positive (%)",
          "69 to 100; 14.6 to 24.7", f"{span([r0(r['share_abstained_positive']) for r in ga], '{}')}; {span([100 * r['share_positive'] for r in ga])}")
    value("Ablations", "gate against the allocated reading on carclaims: K = 3 ahead; K > 3 behind, points", "2.3 to 5.7; 2.6 to 15.0",
          f"{span([r1(r['mean']) for r in ga if r['K'] == 3])}; {span([-r1(r['mean']) for r in ga if r['K'] > 3])}")
    gu = [AB[("ulb", s, t, K, "rho=1", "gate")] for s in ST for t in TH for K in KS]
    value("Ablations", "gate on ULB against the signal, points", "-9.1 to 0.6", span([r1(r["mean"]) for r in gu]))

    # ---------------------------------------------------------------- diagnostics
    D = json.loads((results / "chapter4_diagnostics.json").read_text())
    KF = {(r["dataset"], r["strategy"], r["version"]): r for r in D["keff"]}
    AU = {(r["dataset"], r["strategy"], r["version"], r["part"], r["family"]): r for r in D["member_auroc"]}
    others = [r["median"] for (d, s, ver, pa, fa), r in AU.items()
              if (d, s, ver, pa) == ("ulb", "s2", "unconstrained", "validation") and fa != "LightGBM"]
    # The unconstrained LightGBM members degenerate on ULB, and their fit depends on the number of threads
    # (1 to 8 threads: median AUROC 0.680 to 0.701, share 39.8% to 41.4%): these two numbers have a tolerance.
    near("Decisions after the pilot", "ULB, unconstrained LightGBM: median validation AUROC (depends on the threads)",
         0.687, AU[("ulb", "s2", "unconstrained", "validation", "LightGBM")]["median"], 0.02)
    value("Decisions after the pilot", "ULB: median validation AUROC of the other families, range", "0.947 to 0.983",
          f"{f3(min(others))} to {f3(max(others))}")
    near("Decisions after the pilot", "ULB, unconstrained LightGBM: share of the squared deviations (%) (depends on the threads)",
         41, 100 * KF[("ulb", "s2", "unconstrained")]["share_squared_lightgbm"]["mean"], 2, "{:.1f}")
    value("Decisions after the pilot", "ULB, constrained LightGBM: median validation AUROC; share of the squared deviations (%)",
          "0.972; 7.9", f"{f3(AU[('ulb', 's2', 'final', 'validation', 'LightGBM')]['median'])}; "
                        f"{100 * KF[('ulb', 's2', 'final')]['share_squared_lightgbm']['mean']:.1f}")
    value("Decisions after the pilot", "carclaims, LightGBM: median validation AUROC before and after the constraint", "0.825, 0.830",
          f"{f3(AU[('carclaims', 's2', 'unconstrained', 'validation', 'LightGBM')]['median'])}, "
          f"{f3(AU[('carclaims', 's2', 'final', 'validation', 'LightGBM')]['median'])}")
    ce = D["classes_without_errors"]
    value("Decisions after the pilot", "ensembles on ULB whose predicted positives have no error on validation", "53", len(ce))
    holds("Decisions after the pilot", "... all at the F1 threshold, with a budget of zero for that class",
          all((r["dataset"], r["threshold"], r["class"], r["k_test"]) == ("ulb", "tau_F1", 1, 0) for r in ce))
    groups = Counter((r["dataset"], r["strategy"], r["partition"], r["threshold"], r["K"], r["class"]) for r in ce)
    full = [g_ for g_, n in groups.items() if n == (1 if g_[4] == 36 else 20)]
    holds("Decisions after the pilot", "only the nine-family ensemble of 36 on one partition has the class without errors in every draw, "
          "and the distance from the threshold is used there",
          len(full) == 1 and full[0][1] == "s2" and full[0][4] == 36
          and [r["reading"] for r in ce if (r["dataset"], r["strategy"], r["partition"], r["threshold"], r["K"], r["class"]) == full[0]] == ["distance"])
    f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"
    value("Members and ensembles", "effective number of members, one family: carclaims, ULB", "19.5, 10.3",
          ", ".join(f1(KF[(d, 's1', 'final')]['keff']['mean']) for d in DS))
    value("Members and ensembles", "effective number of members, nine families: carclaims, ULB", "13.6, 7.8",
          ", ".join(f1(KF[(d, 's2', 'final')]['keff']['mean']) for d in DS))
    value("Members and ensembles", "correlation of deviations within families; between families: carclaims, ULB",
          "0.50, 0.67; -0.07, -0.07", f"{', '.join(f2(KF[(d, 's2', 'final')]['rho_within']['mean']) for d in DS)}; "
                                      f"{', '.join(f2(KF[(d, 's2', 'final')]['rho_between']['mean']) for d in DS)}")
    holds("Members and ensembles", "on ULB the effective number of members is close to the number of families (9)",
          abs(KF[("ulb", "s2", "final")]["keff"]["mean"] - 9) <= 1.5)
    VP = {(r["dataset"], r["strategy"]): r for r in D["validation_pool"]}
    RG = {(r["dataset"], r["strategy"], r["K"]): r for r in D["test_regimes"]}
    fam = VP[("ulb", "s2")]["at_clip_by_family"]
    value("Members and ensembles", "validation, ULB: probabilities at the clipping constant, nine families (%); carclaims (%)", "8.9; 0.5",
          f"{100 * VP[('ulb', 's2')]['at_clip']:.1f}; {100 * VP[('carclaims', 's2')]['at_clip']:.1f}")
    holds("Members and ensembles", "more than a fifth of the probabilities of the perceptron, extra trees and random forest at the clipping constant",
          all(fam[f] > 0.2 for f in ("MLP", "ExtraTrees", "RandomForest")))
    value("Members and ensembles", "validation, ULB: probabilities of the one-family members at the clipping constant", "none (0)",
          "none (0)" if VP[("ulb", "s1")]["at_clip"] == 0 else f"{VP[('ulb', 's1')]['at_clip']:.1e} of all")
    holds("Members and ensembles", "no case degenerate and none at the border of the grid on validation, full pools",
          all(r["degenerate"] == 0 and r["border_share"] == 0 for r in VP.values()))
    holds("Members and ensembles", "test, ULB: J-shaped law with nine families in at least 98% of the cases at every size",
          min(RG[("ulb", "s2", K)]["J"] for K in KS) >= 0.98)
    holds("Members and ensembles", "test, ULB: unimodal law with one family in at least 90% of the cases from twelve members",
          min(RG[("ulb", "s1", K)]["unimodal"] for K in (12, 24, 36)) >= 0.90)
    holds("Appendix, randomised trees", "the perceptron, extra trees and random forest are the members most often at the clipping constant on ULB",
          set(sorted(fam, key=fam.get, reverse=True)[:3]) == {"MLP", "ExtraTrees", "RandomForest"})
    holds("Members and their calibration", "every calibration converged (maximum likelihood)",
          all(set(r["methods"]) == {"maximum_likelihood"} for r in D["calibrations"]))
    SA = {(r["dataset"], r["strategy"], r["threshold"], r["K"]): r for r in D["saturation"]}
    cited = [("ulb", "s1", "tau_F1", K) for K in (12, 24, 36)] + [("ulb", "s2", "tau_F1", K) for K in (24, 36)]
    holds("Fixed shares", "where the signal costs at 5% and 10%, fewer cases than the 5% budget are not saturated",
          all(c in SA and SA[c]["k_not_saturated"] < P[c + ("share=5%",)]["k"] for c in cited))

    # ---------------------------------------------------------------- settings of the design, from the code
    from fraud import ablations, evaluate, families, plan, tabpfn_service, thresholds
    from fraud import defaults, preprocessor
    value("Members and their calibration", "members per pool; subsample fraction", "36; (0.6, 0.85)", f"{plan.M}; {plan.FRACTION}")
    value("Ensembles, thresholds and errors", "ensemble sizes; draws for each K < 36", "(3, 6, 12, 24, 36); 20", f"{evaluate.K_LEVELS}; {evaluate.R}")
    holds("Ensembles, thresholds and errors", "tau_F1 over 19 values between 0.05 and 0.95 and 39 quantiles of pbar",
          len(thresholds.TAU_GRID) == 19 and thresholds.TAU_GRID[0] == 0.05 and thresholds.TAU_GRID[-1] == 0.95
          and inspect.signature(thresholds.tau_candidates).parameters["n_quantiles"].default == 39)
    value("Ensembles, thresholds and errors", "cost ratio r: carclaims, ULB", "9, 100", f"{thresholds.COST_RATIO['carclaims']}, {thresholds.COST_RATIO['ulb']}")
    value("Budgets and comparisons", "bootstrap vectors; budget factors rho; fixed shares", "1000; (0.5, 1.0, 2.0); (0.01, 0.05, 0.1)",
          f"{evaluate.B}; {evaluate.RHO}; {evaluate.SHARES}")
    value("Budgets and comparisons", "ablations: lambda0 of the full pool; gate abstention; clipping", "36.0; 0.1; 0.0001",
          f"{ablations.POOL_LAMBDA0}; {ablations.GATE_ABSTENTION}; {ablations.ABLATION_CLIP}")
    value("Members and their calibration", "clipping of the raw probability before the logit", "1e-06", families.EPS_LOGIT)
    src_pre = inspect.getsource(preprocessor)
    holds("Members and their calibration", "categorical levels below 1% of the training cases grouped",
          "min_frequency=max(5, int(0.01 * len(X_train)))" in src_pre)
    p0 = defaults.get_default_params(0)
    holds("Table of the nine families", "random forest and extra trees with 2 cases per leaf",
          p0["random_forest"]["min_samples_leaf"] == 2 and p0["extra_trees"]["min_samples_leaf"] == 2)
    holds("Table of the nine families", "logistic regression with 1000 iterations", p0["logreg"]["max_iter"] == 1000)
    holds("Table of the nine families", "LightGBM (nine families) with leaves of Hessian >= 1; none in the unconstrained version",
          p0["lgbm"]["min_sum_hessian_in_leaf"] == 1.0 and "min_sum_hessian_in_leaf" not in defaults.get_default_params(0, unconstrained=True)["lgbm"])
    value("Table of the nine families", "TabPFN model requested", "v3_default, v3", f"{tabpfn_service.MODEL_PATH}, {tabpfn_service.MODEL_VERSION}")
    src_plan = inspect.getsource(plan._sample_lgbm_constrained)
    for printed, code in (("Number of leaves: integers 8 to 31", "num_leaves=int(rng.integers(8, 32))"),
                          ("Learning rate: log-uniform 0.02 to 0.10", "learning_rate=_log_uniform(rng, 0.02, 0.10)"),
                          ("Maximum depth: 4, 6 or 8", "max_depth=int(rng.choice([4, 6, 8]))"),
                          ("Fraction of the variables per tree: uniform 0.5 to 1", "feature_fraction=float(rng.uniform(0.5, 1.0))"),
                          ("Fraction of the cases per round: uniform 0.6 to 1", "bagging_fraction=float(rng.uniform(0.6, 1.0))"),
                          ("L2 penalty on the leaf values: log-uniform 1 to 10", "lambda_l2=_log_uniform(rng, 1.0, 10.0)"),
                          ("Minimum number of cases per leaf: integers 20 to 80", "min_data_in_leaf=int(rng.integers(20, 81))"),
                          ("Number of rounds: integers 100 to 300", "num_boost_round=int(rng.integers(100, 301))")):
        holds("Appendix, hyperparameters of the one-family construction", printed, code in src_plan and p0["lgbm_s1"]["bagging_freq"] == 1)
    req = dict(line.strip().split("==") for line in open(HERE / "requirements.txt") if "==" in line)
    value("Appendix, models of the ensembles", "LightGBM, XGBoost, CatBoost, scikit-learn, tabpfn-client",
          "4.7.0, 3.4.1, 1.2.10, 1.9.0, 0.6.1",
          ", ".join(req[k] for k in ("lightgbm", "xgboost", "catboost", "scikit-learn", "tabpfn-client")))

    failed = [c for c in CHECKS if not c[4]]
    for where, what, printed, got, ok in CHECKS:
        if show_all or not ok:
            print(f"{'ok  ' if ok else 'FAIL'}  [{where}] {what}: thesis {printed} | results {got}")
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check the numbers of Chapter 4 against the results.")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--list", action="store_true", help="print every check, not only the failures")
    a = ap.parse_args()
    sys.exit(main(a.results, a.list))
