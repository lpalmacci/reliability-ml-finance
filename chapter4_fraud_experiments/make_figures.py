"""Step 8: the two figures of Chapter 4, from results/chapter4_results.json.

    python make_figures.py    ->  figures/confidex_ev_differences.pdf|png   (errors captured relative to three references)
                                  figures/confidex_ev_missed_frauds.pdf|png (missed frauds captured among the predicted negatives)

The figures of the thesis are typeset with LaTeX; without it the script falls back to Matplotlib's
mathtext, with the same layout and data (see style.py).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import OI, TEXTWIDTH, save, setup

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
KS = (3, 6, 12, 24, 36)
PANELS = (("carclaims", "tau_F1", r"carclaims, $\tau_{F_1}$"), ("carclaims", "tau_cost", r"carclaims, $\tau=0.1$"),
          ("ulb", "tau_F1", r"ULB, $\tau_{F_1}$"), ("ulb", "tau_cost", r"ULB, $\tau=1/101$"))
STRATEGIES = (("s1", "one family"), ("s2", "nine families"))


def index(rows, *extra):
    return {tuple(r[k] for k in ("dataset", "strategy", "threshold", "K", "budget") + extra): r for r in rows}


def axis_k(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()


def main() -> None:
    usetex = setup()
    OUT.mkdir(exist_ok=True)
    R = json.loads((HERE / "results" / "chapter4_results.json").read_text())
    P, A, C = index(R["main"]), index(R["allocation"]), index(R["within_class"], "class")

    # 1. differences in points at the budget of the expected errors (rho = 1)
    series = (("simple", "against the best simple reading", OI["vermillion"], "o", "-"),
              ("alloc", "against the allocated reading", OI["blue"], "s", "--"),
              ("neg", r"within $\hat y=0$ (missed frauds)", OI["green"], "^", "-."))
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.9), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.18})
    for row, (st, sname) in enumerate(STRATEGIES):
        for col, (ds, thr, title) in enumerate(PANELS):
            ax = axes[row, col]; ax.axhline(0, color="black", lw=0.6)
            for key, lab, colr, mk, ls in series:
                src = {"simple": lambda K: P[(ds, st, thr, K, "rho=1")], "alloc": lambda K: A[(ds, st, thr, K, "rho=1")],
                       "neg": lambda K: C[(ds, st, thr, K, "rho=1", 0)]}[key]
                r = [src(K) for K in KS]
                ax.fill_between(KS, [100 * x["min"] for x in r], [100 * x["max"] for x in r], color=colr, alpha=0.15, lw=0)
                ax.plot(KS, [100 * x["mean"] for x in r], color=colr, marker=mk, ls=ls, label=lab)
            axis_k(ax)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel(r"members $K$")
        axes[row, 0].set_ylabel(f"{sname}\npoints")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    save(fig, OUT / "confidex_ev_differences")

    # 2. missed frauds: share captured within yhat = 0
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (ds, thr, title) in zip(axes, PANELS):
        for st, _ in STRATEGIES:
            ls = "-" if st == "s1" else "--"
            r = [C[(ds, st, thr, K, "rho=1", 0)] for K in KS]
            ax.plot(KS, [100 * x["perfect"] for x in r], color=OI["grey"], ls=ls, lw=1.0)
            ax.plot(KS, [100 * x["confidex"] for x in r], color=OI["vermillion"], marker="o", ls=ls)
            ax.plot(KS, [100 * (x["confidex"] - x["mean"]) for x in r], color=OI["blue"], marker="s", ls=ls, ms=3)
            if st == "s1":
                ax.plot(KS, [100 * x["random"] for x in r], color=OI["grey"], ls=":", lw=1.0)
        axis_k(ax); ax.set_title(title); ax.set_xlabel(r"members $K$"); ax.set_ylim(-3, 103)
    axes[0].set_ylabel(r"missed frauds captured (\%)" if usetex else "missed frauds captured (%)")
    handles = [Line2D([], [], color=OI["grey"], lw=1.0), Line2D([], [], color=OI["vermillion"], marker="o"),
               Line2D([], [], color=OI["blue"], marker="s", ms=3), Line2D([], [], color=OI["grey"], ls=":", lw=1.0),
               Line2D([], [], color="black", ls="-", lw=1.0), Line2D([], [], color="black", ls="--", lw=1.0)]
    labels = ["perfect ordering", "signal", "best simple reading", "random ordering", "one family", "nine families"]
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.19), fontsize=7, handlelength=1.8,
               columnspacing=1.0)
    save(fig, OUT / "confidex_ev_missed_frauds")
    print("two figures written to", OUT, "(LaTeX)" if usetex else "(mathtext: LaTeX not available)")


if __name__ == "__main__":
    main()
