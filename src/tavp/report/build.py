"""Build the tables, figures, validation cards and numbers of the chapter from results/.

Usage: python -m tavp.report.build --config configs/full.yaml [--export <chapter assets dir>]
Only the parts whose results exist are built, so the command can be run at any stage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tavp.experiments import common
from tavp.report import summaries as S
from tavp.report.numbers import Numbers
from tavp.utils import REPO_ROOT

DGP_LABEL = {"gbm": "D1 GBM", "heston": "D2 Heston", "gjr_garch_t": "D3 GJR-GARCH-$t$", "regime": "D4 Regimes"}
DGP_ORDER = ["gbm", "heston", "gjr_garch_t", "regime"]
DEFECT_LABEL = {"P1": "P1 shuffle", "P2": "P2 thinner tails", "P3": "P3 no leverage", "P4": "P4 lower persistence",
                "P5": "P5 volatility scale", "P6": "P6 drift"}
STAT_LABEL = {"mean": "mean", "std": "std. dev.", "skew": "skewness", "kurt": "kurtosis", "hill": "Hill index",
              "acf_r": "ACF($r$)", "acf_abs": "ACF($|r|$)", "leverage": "leverage", "kurt_agg5": "kurt. (5d)",
              "kurt_agg20": "kurt. (20d)", "marginal_w1": "marginal $W_1$", "swd_logpath": "sliced $W_2$",
              "signature_mmd": "signature"}
# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
COLORS = {"L1": "#2a78d6", "L2": "#eb6834", "L3": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}\\%"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ----------------------------------------------------------------------------- E1
def build_e1(root: Path, out: Path, num: Numbers, pr_min: float) -> None:
    l1, l2, l3 = (_read(root, "e1", n) for n in ("l1", "l2", "l3"))
    if l2 is None:
        return
    r1 = S.l1_rates(l1)
    cells = S.l2_cells(l2, pr_min)
    s3 = S.l3_strategies(l3, pr_min)
    rows = []
    for d in DGP_ORDER:
        st = r1[(r1.dgp == d) & (r1.statistic != "any")]
        fam = float(r1[(r1.dgp == d) & (r1.statistic == "any")].flag_holm.iloc[0])
        c = cells[cells.dgp == d]
        t3 = s3[s3.dgp == d]
        rows.append((DGP_LABEL[d], _pct(st.flag_raw.mean(), 1), _pct(st.flag_raw.max()), _pct(fam),
                     f"{int(c['pass'].sum())}/{len(c)}", f"{c.pr.min():.2f}", _pct(c.resolvable.mean()),
                     f"{t3.pr.min():.2f}", f"{t3.regret.mean():.4f}"))
        num.pct(f"e1.{d}.l1.raw.mean", st.flag_raw.mean(), 1)
        num.pct(f"e1.{d}.l1.raw.max", st.flag_raw.max())
        num.pct(f"e1.{d}.l1.family", fam)
        num.add(f"e1.{d}.l2.pass", f"{int(c['pass'].sum())} of {len(c)}")
        num.add(f"e1.{d}.l2.minpr", float(c.pr.min()))
        num.add(f"e1.{d}.l3.minpr", float(t3.pr.min()))
    st = r1[r1.statistic != "any"]
    num.pct("e1.l1.raw.mean", st.flag_raw.mean(), 1)
    num.pct("e1.l1.raw.max", st.flag_raw.max())
    num.add("e1.l1.raw.over9", int((st.flag_raw > 0.09).sum()))
    num.add("e1.l1.cells", int(len(st)))
    num.pct("e1.l1.family.max", r1[r1.statistic == "any"].flag_holm.max())
    num.add("e1.l2.cells.pass", f"{int(cells['pass'].sum())} of {len(cells)}")
    num.add("e1.l2.minpr", float(cells.pr.min()))
    num.pct("e1.l2.resolvable", cells.resolvable.mean())
    num.add("e1.l3.minpr", float(s3.pr.min()))
    num.add("e1.l3.tau.min", float(s3.kendall_tau.min()))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    _write(out / "tables" / "tab_e1_size.tex", rf"""\begin{{tabular}}{{lrrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{L1 false alarms}} & \multicolumn{{3}}{{c}}{{L2 cells}} & \multicolumn{{2}}{{c}}{{L3}} \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}\cmidrule(lr){{8-9}}
Process & mean & max & family & passed & min PR & resolvable & min PR & mean regret \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


# ----------------------------------------------------------------------------- E2
def detection_curves(e1: dict, e2: dict, pr_min: float) -> pd.DataFrame:
    """Detection probability of each level by defect, process and severity (severity 0 = negative control)."""
    l1 = pd.concat([e1["l1"].assign(defect=None), e2["l1"]], ignore_index=True)
    l2 = pd.concat([e1["l2"].assign(defect=None), e2["l2"]], ignore_index=True)
    l3 = pd.concat([e1["l3"].assign(defect=None), e2["l3"]], ignore_index=True)
    out = []
    for defect in sorted(e2["l2"].defect.dropna().unique()):
        for d in DGP_ORDER:
            if not ((e2["l2"].defect == defect) & (e2["l2"].dgp == d)).any():
                continue
            for delta in [0.0] + sorted(e2["l2"].delta.unique()):
                sel = lambda df: df[(df.dgp == d) & ((df.defect == defect) if delta > 0 else df.defect.isna())
                                    & (df.delta == delta)]
                a1, a2, a3 = sel(l1), sel(l2), sel(l3)
                p1 = a1.groupby("k").flag_holm.any().mean()
                c = a2.groupby(["strategy", "metric", "functional"]).equivalent.mean()
                p3 = a3.groupby("strategy")["pass"].mean()
                out.append({"defect": defect, "dgp": d, "delta": delta, "L1": p1, "L2": 1 - c.min(),
                            "L3": 1 - p3.min(), "l2_failed": int((c < pr_min).sum()), "l2_cells": len(c),
                            "l2_failed_strategies": ",".join(sorted({k[0] for k, v in c.items() if v < pr_min})),
                            "l3_failed": ",".join(sorted(p3[p3 < pr_min].index))})
    return pd.DataFrame(out)


def power_figure(curves: pd.DataFrame, path: Path, pr_min: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    defects = [d for d in DEFECT_LABEL if d in set(curves.defect)]
    fig, axes = plt.subplots(len(defects), 4, figsize=(7.2, 1.35 * len(defects) + 0.6), sharex=True, sharey=True)
    first = True
    for i, defect in enumerate(defects):
        for j, d in enumerate(DGP_ORDER):
            ax = axes[i, j]
            ax.set_facecolor("white")
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(colors=INK2, labelsize=6, length=0)
            if i == 0:
                ax.set_title(DGP_LABEL[d].replace("$", "").replace("-t", "-t"), fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(DEFECT_LABEL[defect], fontsize=6.5, color=INK, rotation=0, ha="right", va="center")
            c = curves[(curves.defect == defect) & (curves.dgp == d)].sort_values("delta")
            if c.empty:
                ax.text(0.5, 0.5, "not applicable", ha="center", va="center", fontsize=6, color=INK2,
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.axhline(1 - pr_min, color=INK2, linewidth=0.6, linestyle=":")
            for lev, mk in (("L1", "o"), ("L2", "s"), ("L3", "^")):
                ax.plot(c.delta, c[lev], color=COLORS[lev], linewidth=1.4, marker=mk, markersize=3.2,
                        markeredgecolor="white", markeredgewidth=0.6, label=lev, clip_on=False)
            if first:
                for lev in ("L1", "L2", "L3"):
                    ax.annotate(lev, (1.0, float(c[lev].iloc[-1])), xytext=(4, 0), textcoords="offset points",
                                fontsize=6, color=INK, va="center")
                first = False
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        for ax in axes.ravel():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, ["L1 (any statistic, Holm)", "L2 (worst cell)", "L3 (worst rule)"], loc="upper center",
               ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.supxlabel("severity $\\delta$", fontsize=7, color=INK)
    fig.supylabel("detection probability", fontsize=7, color=INK, x=0.0)
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def build_e2(root: Path, out: Path, num: Numbers, pr_min: float) -> None:
    e1 = {n: _read(root, "e1", n) for n in ("l1", "l2", "l3")}
    e2 = {n: _read(root, "e2", n) for n in ("l1", "l2", "l3")}
    if e2["l2"] is None or e1["l2"] is None:
        return
    curves = detection_curves(e1, e2, pr_min)
    curves.to_csv(out / "tables" / "e2_detection_curves.csv", index=False)
    power_figure(curves, out / "figures" / "fig_power.pdf", pr_min)
    # detection matrix at delta = 1 with the most flagged L1 statistics
    r1 = S.l1_rates(e2["l1"])
    lines = []
    for defect in DEFECT_LABEL:
        for d in DGP_ORDER:
            c = curves[(curves.defect == defect) & (curves.dgp == d) & (curves.delta == 1.0)]
            if c.empty:
                continue
            c = c.iloc[0]
            st = r1[(r1.defect == defect) & (r1.dgp == d) & (r1.delta == 1.0) & (r1.statistic != "any")]
            top = st[st.flag_holm >= 0.5].sort_values("flag_holm", ascending=False).statistic.tolist()
            top_s = ", ".join(STAT_LABEL[s] for s in top[:3]) + (" \\dots" if len(top) > 3 else "") or "none"
            l2s = f"{c.l2_failed}/{c.l2_cells}" + (f" ({c.l2_failed_strategies.replace(',', ', ')})" if c.l2_failed else "")
            lines.append(f"{DEFECT_LABEL[defect]} & {DGP_LABEL[d]} & {c.L1:.2f} & {top_s} & {l2s} & "
                         f"{c.l3_failed.replace(',', ', ') or 'none'} \\\\")
            key = f"e2.{defect}.{d}"
            num.add(key + ".l1", float(c.L1))
            num.add(key + ".l2failed", int(c.l2_failed))
    _write(out / "tables" / "tab_e2_detection.tex", r"""\begin{tabular}{llrp{4.2cm}p{3.2cm}p{1.6cm}}
\toprule
Defect & Process & L1 power & L1 statistics flagged & L2 failed cells & L3 failed \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    # E3: dissociation per (defect, process, severity > 0, strategy)
    dis = S.dissociation(e2["l1"], e2["l2"], pr_min)
    dis.to_csv(out / "tables" / "e3_dissociation.csv", index=False)
    counts = dis.category.value_counts()
    for cat, key in [("both pass", "bothpass"), ("both flag", "bothflag"), ("L1 only (irrelevant defect)", "l1only"),
                     ("L2 only (hidden defect)", "l2only")]:
        num.add(f"e3.{key}", int(counts.get(cat, 0)))
    num.add("e3.total", int(len(dis)))


# ----------------------------------------------------------------------------- E6
def build_e6(root: Path, out: Path, num: Numbers, part: str) -> None:
    t = _read(root, f"e6_{part}", "tests")
    if t is None:
        return
    g = t.groupby(["dgp", "generator"]).agg(rejected=("reject_any", "sum"), tests=("reject_any", "size"),
                                             min_p_ks=("p_ks", "min"), min_p_kupiec=("p_kupiec", "min"),
                                             min_p_berk=("p_berkowitz", "min"),
                                             min_p_berk_marg=("p_berkowitz_marginal", "min"),
                                             coverage=("coverage", "mean")).reset_index()
    g.to_csv(out / "tables" / f"e6_{part}_summary.csv", index=False)
    lines = [f"{DGP_LABEL[r.dgp]} & {r.generator.replace('_', ' ')} & {r.rejected}/{r.tests} & {r.min_p_ks:.3f} & "
             f"{r.min_p_kupiec:.3f} & {r.min_p_berk:.3f} & {r.min_p_berk_marg:.3f} & {r.coverage:.2f} \\\\"
             for r in g.sort_values(["dgp", "generator"], key=lambda s: s.map(
                 {k: i for i, k in enumerate(DGP_ORDER)}) if s.name == "dgp" else s).itertuples()]
    _write(out / "tables" / f"tab_e6_{part}.tex", r"""\begin{tabular}{llrrrrrr}
\toprule
Process & Generator & rejected & min $p$ KS & min $p$ Kupiec & min $p$ Berkowitz & min $p$ marginal & coverage \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    for r in g.itertuples():
        num.add(f"e6{part}.{r.dgp}.{r.generator}.rejected", int(r.rejected))


# ----------------------------------------------------------------------------- main
def main():
    args = common.parse_args(__doc__, lambda p: p.add_argument("--export", default=None))
    cfg = common.load(args.config)
    root = common.results_root(cfg)
    out = REPO_ROOT / "thesis_assets" if not cfg.get("runtime", {}).get("results_suffix") else root / "thesis_assets"
    pr_min = cfg["protocol"]["L2"]["pr_min"]
    num = Numbers()
    sc = root / "train" / "scaling_decision.json"
    if sc.exists():
        s = json.loads(sc.read_text())
        num.pct("scaling.relerr", s["relative_error"], 1)
        num.add("scaling.volg", s["vol_generator"])
        num.add("scaling.volo", s["vol_oracle"])
    num.add("cfg.N.l2", int(cfg["protocol"]["L2"]["N"]))
    num.add("cfg.N.l3", int(cfg["protocol"]["L3"]["N"]))
    num.add("cfg.K", int(cfg["protocol"]["L2"]["K"]))
    build_e1(root, out, num, pr_min)
    b500 = _read(root, "e7_l1null", "l1_b500")
    if b500 is not None:
        rates = b500.groupby(["dgp", "statistic"]).flag_raw_b500.mean()
        num.pct("e7.l1b500.raw.mean", rates.mean(), 1)
        num.pct("e7.l1b500.raw.max", rates.max())
        num.add("e7.l1b500.raw.over9", int((rates > 0.09).sum()))
        num.pct("e7.l1b500.family", b500.groupby(["dgp", "k"]).flag_holm_b500.any().groupby("dgp").mean().max())
    build_e2(root, out, num, pr_min)
    build_e6(root, out, num, "meta")
    build_e6(root, out, num, "case")
    num.write(out / "numbers.tex", out / "numbers.json")
    if args.export:
        dst = Path(args.export)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out, dst)
    print(f"thesis assets written to {out}")


if __name__ == "__main__":
    main()
