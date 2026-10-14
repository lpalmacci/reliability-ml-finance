"""Validation card: the evidence and verdict of the protocol for one generator on one process.

Results of several training runs (seeds) of the same generator are pooled: pass rates are
computed over contexts and seeds together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNC_ORDER = [("sharpe", "mean"), ("sharpe", "q05"), ("sharpe", "q50"), ("sharpe", "q95"),
              ("maxdd", "mean"), ("maxdd", "q95"), ("ret", "mean"), ("ret", "var95"), ("ret", "es95")]
STRATS = ["BH", "TF", "VT", "TVF"]
FUNC_LABEL = {"mean": "mean", "q05": "$q_{.05}$", "q50": "$q_{.50}$", "q95": "$q_{.95}$", "var95": "VaR",
              "es95": "ES"}
METRIC_LABEL = {"sharpe": "Sharpe", "maxdd": "MaxDD", "ret": "Return"}


def build_card(l0: pd.DataFrame, l1: pd.DataFrame | None, l2: pd.DataFrame, l3: pd.DataFrame | None,
               l4: pd.DataFrame | None, pr_min: float, l1_threshold: float = 0.5) -> dict:
    """Inputs are already filtered to one (process, generator kind, regime)."""
    card = {"dgp": l2.dgp.iloc[0], "kind": l2.kind.iloc[0],
            "regime": l2.regime.iloc[0] if "regime" in l2 and l2.regime.notna().any() else None,
            "runs": int(l2.seed.nunique()) if "seed" in l2 and l2.seed.notna().any() else 1,
            "contexts": int(l2.k.nunique())}
    card["L0"] = {"hash_ok": bool(l0.hash_ok.all()), "valid": bool(l0.valid.all()),
                  "resolvable": float(l2.resolvable.mean())}
    flagged = []
    if l1 is not None and len(l1):
        g = l1.groupby("statistic").agg(rate=("flag_holm", "mean"), ratio=("ratio", "median"))
        flagged = [(s, float(r.rate), float(r.ratio)) for s, r in g.iterrows() if r.rate >= l1_threshold]
        card["L1_power_any"] = float(l1.groupby(["seed", "k"] if "seed" in l1 else ["k"], dropna=False)
                                     .flag_holm.any().mean())
    card["L1_flagged"] = flagged
    pr = l2.groupby(["strategy", "metric", "functional"]).equivalent.mean()
    card["L2"] = {(s, m, f): float(pr.get((s, m, f), np.nan)) for s in STRATS for m, f in FUNC_ORDER}
    card["L2_failed"] = [k for k, v in card["L2"].items() if np.isfinite(v) and v < pr_min]
    card["L3"] = {}
    if l3 is not None and len(l3):
        for s, g in l3.groupby("strategy"):
            card["L3"][s] = {"pr": float(g["pass"].mean()), "regret": float(g.regret.mean()),
                             "tau": float(g.kendall_tau.mean())}
    card["L3_failed"] = [s for s, v in card["L3"].items() if v["pr"] < pr_min]
    card["L4"] = None
    if l4 is not None and len(l4):
        density = "p_moments_hac" if "p_moments_hac" in l4 else "p_berkowitz"
        card["L4"] = {"rejected": [f"{r.strategy}/{r.metric}" for r in l4.itertuples() if r.reject_any],
                      "tests": int(len(l4)), "min_p": float(l4[["p_ks", "p_kupiec", density]].min().min())}
    l4_ok = card["L4"] is None or not card["L4"]["rejected"]
    card["fit"] = bool(card["L0"]["hash_ok"] and card["L0"]["valid"] and not card["L2_failed"]
                       and not card["L3_failed"] and l4_ok)
    return card


def card_markdown(c: dict) -> str:
    lines = [f"### Validation card: {c['kind']} on {c['dgp']}" + (f" ({c['regime']} regime)" if c["regime"] else ""),
             f"Runs: {c['runs']}; contexts: {c['contexts']}.",
             f"- L0: hash consistent {c['L0']['hash_ok']}, scenarios valid {c['L0']['valid']}, "
             f"resolvable cells {c['L0']['resolvable']:.0%}",
             "- L1 flagged: " + (", ".join(f"{s} (rate {r:.2f}, ratio {q:.1f})" for s, r, q in c["L1_flagged"]) or "none"),
             "", "| L2 pass rate | " + " | ".join(f"{METRIC_LABEL[m]} {f}" for m, f in FUNC_ORDER) + " |",
             "|---|" + "---|" * len(FUNC_ORDER)]
    for s in STRATS:
        lines.append(f"| {s} | " + " | ".join(f"{c['L2'][(s, m, f)]:.2f}" for m, f in FUNC_ORDER) + " |")
    lines.append("")
    for s, v in c["L3"].items():
        lines.append(f"- L3 {s}: pass rate {v['pr']:.2f}, mean regret {v['regret']:.3f}, Kendall tau {v['tau']:.2f}")
    if c["L4"] is not None:
        lines.append("- L4 rejected: " + (", ".join(c["L4"]["rejected"]) or "none"))
    lines.append(f"- **Verdict: {'fit' if c['fit'] else 'not fit'} for the declared use**")
    return "\n".join(lines)


STAT_LABEL = {"mean": "mean", "std": "standard deviation", "skew": "skewness", "kurt": "kurtosis",
              "hill": "Hill index", "acf_r": "ACF($r$)", "acf_abs": "ACF($|r|$)", "leverage": "leverage",
              "kurt_agg5": "kurtosis (5 days)", "kurt_agg20": "kurtosis (20 days)", "marginal_w1": "marginal $W_1$",
              "swd_logpath": "sliced $W_2$", "signature_mmd": "signature distance"}


def card_latex(c: dict, pr_min: float, width: str = "13.5cm") -> str:
    """L2 pass-rate matrix with failed cells in bold, framed by the L0 header and the L1/L3/L4 lines."""
    def cell(v):
        return r"\textbf{" + f"{v:.2f}" + "}" if v < pr_min else f"{v:.2f}"

    def line(text):
        return r"\multicolumn{10}{p{" + width + "}}{" + text + r"} \\"

    head = " & ".join(f"{FUNC_LABEL[f]}" for _, f in FUNC_ORDER)
    groups = r" & \multicolumn{4}{c}{Sharpe} & \multicolumn{2}{c}{MaxDD} & \multicolumn{3}{c}{Return} \\"
    rows = [f"{s} & " + " & ".join(cell(c["L2"][(s, m, f)]) for m, f in FUNC_ORDER) + r" \\" for s in STRATS]
    l0 = (f"L0: process hashes {'consistent' if c['L0']['hash_ok'] else 'inconsistent'}; scenarios "
          f"{'valid' if c['L0']['valid'] else 'invalid'}; {100 * c['L0']['resolvable']:.0f}\\% of comparisons resolvable; "
          f"{c['runs']} training runs, {c['contexts']} contexts.")
    l1 = ", ".join(STAT_LABEL.get(s, s.replace("_", r"\_")) for s, _, _ in c["L1_flagged"]) or "none"
    l3 = "; ".join(f"{s} {v['pr']:.2f} (mean regret {v['regret']:.3f}, Kendall {v['tau']:.2f})"
                   for s, v in c["L3"].items())
    l4 = "not applied" if c["L4"] is None else (
        f"{len(c['L4']['rejected'])} of {c['L4']['tests']} cells rejected"
        + (" (" + ", ".join(c["L4"]["rejected"]) + ")" if c["L4"]["rejected"] else ""))
    return "\n".join([r"\begin{tabular}{l" + "r" * len(FUNC_ORDER) + "}", r"\toprule", line(l0), r"\midrule", groups,
                      r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}", "L2 pass rate & " + head + r" \\",
                      r"\midrule", *rows, r"\midrule",
                      line("L1 flagged (power at least 0.5): " + l1 + "."),
                      line("L3 pass rate: " + l3 + "."),
                      line("L4: " + l4 + "."),
                      line(r"\textbf{Verdict: " + ("fit" if c["fit"] else "not fit") + " for the declared use.}"),
                      r"\bottomrule", r"\end{tabular}"])
