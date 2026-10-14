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
    from scipy import stats as sps
    n_cmp = int(l1.groupby(["dgp", "statistic"]).size().iloc[0])
    p_over = float(sps.binom.sf(np.floor(0.09 * n_cmp), n_cmp, 0.05))
    num.add("e1.l1.raw.over9.expected", float(len(st) * p_over), "{:.1f}")
    num.add("e1.l1.comparisons", n_cmp)
    num.add("e1.l1.cells", int(len(st)))
    num.pct("e1.l1.family.max", r1[r1.statistic == "any"].flag_holm.max())
    if "flag_holm_prereg" in l1:
        fam0 = l1.groupby(["dgp", "k"]).flag_holm_prereg.any().groupby("dgp").mean()
        num.pct("e1.l1.family.prereg.max", fam0.max())
        num.add("e1.l1.minp.prereg", f"{l1.p_prereg.min():.4f}")
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
                p1_0 = a1.groupby("k").flag_holm_prereg.any().mean() if "flag_holm_prereg" in a1 else np.nan
                c = a2.groupby(["strategy", "metric", "functional"]).equivalent.mean()
                p3 = a3.groupby("strategy")["pass"].mean()
                out.append({"defect": defect, "dgp": d, "delta": delta, "L1": p1, "L1_prereg": p1_0, "L2": 1 - c.min(),
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
    # detection matrix: smallest severity at which each level detects the defect
    r1 = S.l1_rates(e2["l1"])
    l1_power_min = 0.8

    def first(c, col, thr):
        hit = c[(c.delta > 0) & (c[col] > thr)].delta
        return f"{hit.min():g}" if len(hit) else "--"

    lines = []
    for defect in DEFECT_LABEL:
        for d in DGP_ORDER:
            c = curves[(curves.defect == defect) & (curves.dgp == d)]
            if c.empty:
                continue
            c1 = c[c.delta == 1.0].iloc[0]
            st = r1[(r1.defect == defect) & (r1.dgp == d) & (r1.delta == 1.0) & (r1.statistic != "any")]
            top = st[st.flag_holm >= 0.5].sort_values("flag_holm", ascending=False).statistic.tolist()
            top_s = ", ".join(STAT_LABEL[x] for x in top[:3]) + (" \\dots" if len(top) > 3 else "") or "none"
            l2s = f"{c1.l2_failed}/{c1.l2_cells}" + (f" ({c1.l2_failed_strategies.replace(',', ', ')})" if c1.l2_failed else "")
            d1, d2, d3 = first(c, "L1", l1_power_min - 1e-12), first(c, "L2", 1 - pr_min), first(c, "L3", 1 - pr_min)
            lines.append(f"{DEFECT_LABEL[defect]} & {DGP_LABEL[d]} & {d1} & {top_s} & {d2} & {l2s} & {d3} \\\\")
            key = f"e2.{defect}.{d}"
            num.add(key + ".l1", float(c1.L1))
            num.add(key + ".l2failed", int(c1.l2_failed))
            num.add(key + ".first.l1", d1.replace("--", "none"))
            num.add(key + ".first.l2", d2.replace("--", "none"))
            num.add(key + ".first.l3", d3.replace("--", "none"))
            for r in c[c.delta > 0].itertuples():
                num.add(f"{key}.{r.delta:g}.l1", float(r.L1))
                num.add(f"{key}.{r.delta:g}.l2", float(r.L2))
    _write(out / "tables" / "tab_e2_detection.tex", r"""\begin{tabular}{llrp{3.6cm}rp{3.0cm}r}
\toprule
& & \multicolumn{2}{c}{L1} & \multicolumn{2}{c}{L2} & L3 \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}
Defect & Process & $\delta^{*}$ & statistics flagged at $\delta=1$ & $\delta^{*}$ & failed cells at $\delta=1$ & $\delta^{*}$ \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    num.add("e2.l1.powermin", l1_power_min)
    if "L1_prereg" in curves:
        num.add("e2.l1.prereg.max", float(curves[curves.delta > 0].L1_prereg.max()))
    # power of single L1 statistics and non-equivalence rates of single L2 cells
    for r in r1[r1.statistic != "any"].itertuples():
        num.add(f"e2.stat.{r.defect}.{r.dgp}.{r.delta:g}.{r.statistic}", float(r.flag_holm))
    cells = S.l2_cells(e2["l2"], pr_min)
    for r in cells.itertuples():
        num.add(f"e2.cell.{r.defect}.{r.dgp}.{r.delta:g}.{r.strategy}.{r.metric}.{r.functional}", float(1 - r.pr))
    st1 = r1[r1.statistic != "any"]
    num.add("e2.stat.P1.max.marginal_w1", float(st1[(st1.defect == "P1") & (st1.statistic == "marginal_w1")].flag_holm.max()))
    num.add("e2.stat.P6.0.25.mean.max", float(st1[(st1.defect == "P6") & (st1.delta == 0.25) & (st1.statistic == "mean")].flag_holm.max()))
    c25 = curves[(curves.delta == 0.25)]
    num.add("e2.P5.0.25.l1.min", float(c25[c25.defect == "P5"].L1.min()))
    bh6 = cells[(cells.defect == "P6") & (cells.delta == 0.25) & (cells.strategy == "BH") & (cells.metric == "sharpe")
                & (cells.functional == "mean")]
    num.add("e2.cell.P6.0.25.BH.sharpe.mean.min", float((1 - bh6.pr).min()))
    # P5 at delta = 1: shift of the risk functionals of VT relative to those of BH
    risk = [("maxdd", "mean"), ("maxdd", "q95"), ("ret", "var95"), ("ret", "es95")]
    x = e2["l2"][(e2["l2"].defect == "P5") & (e2["l2"].delta == 1.0) & e2["l2"].strategy.isin(["BH", "VT"])]
    x = x[[(m, f) in risk for m, f in zip(x.metric, x.functional)]]
    x = x.groupby(["dgp", "metric", "functional", "strategy"])["diff"].mean().abs().unstack("strategy")
    ratio = x.VT / x.BH
    num.add("e2.P5.vtbh.median", float(ratio.median()))
    num.add("e2.P5.vtbh.min", float(ratio.min()))
    num.add("e2.P5.vtbh.max", float(ratio.max()))
    s3 = S.l3_strategies(e2["l3"], pr_min)
    num.add("e2.l3.failed", int((~s3["pass"]).sum()))
    num.add("e2.l3.total", int(len(s3)))
    num.add("e2.l3.regret.max", float(s3.regret.max()), "{:.3f}")
    num.add("e2.l3.samechoice.min", float(s3.same_choice.min()))
    num.add("e2.l3.tau.min", float(s3.kendall_tau.min()))
    # E3: dissociation per (defect, process, severity > 0, strategy)
    dis = S.dissociation(e2["l1"], e2["l2"], pr_min, l3=e2["l3"])
    dis.to_csv(out / "tables" / "e3_dissociation.csv", index=False)
    counts = dis.category.value_counts()
    for cat, key in [("both pass", "bothpass"), ("both flag", "bothflag"), ("L1 only (irrelevant defect)", "l1only"),
                     ("L2 only (hidden defect)", "l2only")]:
        num.add(f"e3.{key}", int(counts.get(cat, 0)))
    num.add("e3.total", int(len(dis)))
    for cat, key in [("both pass", "bothpass"), ("L1 only (irrelevant defect)", "l1only"),
                     ("L2 only (hidden defect)", "l2only")]:
        for defect, n in dis[dis.category == cat].defect.value_counts().items():
            num.add(f"e3.{key}.{defect}", int(n))
    l1o = dis[(dis.category == "L1 only (irrelevant defect)") & (dis.defect == "P1")]
    num.add("e3.l1only.P1.bh", int((l1o.strategy == "BH").sum()))
    num.add("e3.bothpass.P1.gbm", int(((dis.category == "both pass") & (dis.defect == "P1") & (dis.dgp == "gbm")).sum()))
    d3 = dis.dropna(subset=["l3_fail"])
    num.add("e3.l3.total", int(len(d3)))
    num.add("e3.l2fail", int(d3.l2_fail.sum()))
    num.add("e3.l2fail.l3pass", int((d3.l2_fail & ~d3.l3_fail.astype(bool)).sum()))
    num.add("e3.l2pass.l3fail", int((~d3.l2_fail & d3.l3_fail.astype(bool)).sum()))
    num.add("e3.bothfail.l23", int((d3.l2_fail & d3.l3_fail.astype(bool)).sum()))
    _write(out / "tables" / "tab_e3_dissociation.tex", dissociation_table(dis))


def dissociation_table(dis: pd.DataFrame) -> str:
    """Counts of (defect, process, severity, rule) by the L1 flag and the L2 verdict, per defect."""
    cats = ["both pass", "L1 only (irrelevant defect)", "L2 only (hidden defect)", "both flag"]
    lines = []
    for defect in DEFECT_LABEL:
        g = dis[dis.defect == defect]
        if g.empty:
            continue
        n = g.category.value_counts()
        lines.append(f"{DEFECT_LABEL[defect]} & " + " & ".join(str(int(n.get(c, 0))) for c in cats) + f" & {len(g)} \\\\")
    n = dis.category.value_counts()
    lines.append(r"\midrule")
    lines.append("All & " + " & ".join(str(int(n.get(c, 0))) for c in cats) + f" & {len(dis)} \\\\")
    return r"""\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{2}{c}{L2 passes} & \multicolumn{2}{c}{L2 fails} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Defect & L1 silent & L1 flags & L1 silent & L1 flags & total \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
"""


# ----------------------------------------------------------------------------- E6
def build_e6(root: Path, out: Path, num: Numbers, part: str) -> None:
    t = _read(root, f"e6_{part}", "tests")
    if t is None:
        return
    joint = "reject_any_berkowitz_joint" if "reject_any_berkowitz_joint" in t else "reject_any"
    g = t.groupby(["dgp", "generator"]).agg(rejected=("reject_any", "sum"), rejected_prereg=(joint, "sum"),
                                             tests=("reject_any", "size"), min_p_ks=("p_ks", "min"),
                                             min_p_kupiec=("p_kupiec", "min"), min_p_hac=("p_moments_hac", "min"),
                                             min_p_berk=("p_berkowitz", "min"),
                                             coverage=("coverage", "mean")).reset_index()
    g.to_csv(out / "tables" / f"e6_{part}_summary.csv", index=False)
    order = {k: i for i, k in enumerate(DGP_ORDER)}
    g = g.sort_values(["dgp", "generator"], key=lambda x: x.map(order) if x.name == "dgp" else x)
    lines = [f"{DGP_LABEL[r.dgp]} & {r.generator.replace('_', ' ')} & {r.rejected}/{r.tests} & "
             f"{r.rejected_prereg}/{r.tests} & {r.min_p_ks:.3f} & {r.min_p_kupiec:.3f} & {r.min_p_hac:.4f} & "
             f"{r.min_p_berk:.1e} & {r.coverage:.2f} \\\\" for r in g.itertuples()]
    _write(out / "tables" / f"tab_e6_{part}.tex", r"""\begin{tabular}{llrrrrrrr}
\toprule
& & \multicolumn{2}{c}{cells rejected} & \multicolumn{4}{c}{smallest $p$-value} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-8}
Process & Generator & applied & pre-reg. & KS & Kupiec & moments & Berkowitz & coverage \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    for r in g.itertuples():
        key = f"e6{part}.{r.dgp}.{r.generator}"
        num.add(key + ".rejected", int(r.rejected))
        num.add(key + ".rejected.prereg", int(r.rejected_prereg))
        num.add(key + ".tests", int(r.tests))
    if part == "meta":
        o = g[g.generator == "oracle"]
        num.add("e6meta.oracle.rejected", int(o.rejected.sum()))
        num.add("e6meta.oracle.rejected.prereg", int(o.rejected_prereg.sum()))
        num.add("e6meta.oracle.tests", int(o.tests.sum()))
        num.add("e6meta.oracle.processes.rejected", int((o.rejected > 0).sum()))
        num.add("e6meta.oracle.processes.rejected.prereg", int((o.rejected_prereg > 0).sum()))
        pits = _read(root, "e6_meta", "pits")
        if pits is not None:
            from scipy import stats as sps
            ar = []
            for (d, st_, m), x in pits[pits.generator == "oracle"].groupby(["dgp", "strategy", "metric"]):
                rho = [np.corrcoef(z[:-1], z[1:])[0, 1] for z in
                       (sps.norm.ppf(np.clip(h.sort_values("origin").pit.to_numpy(), 1e-6, 1 - 1e-6))
                        for _, h in x.groupby("history"))]
                ar.append((d, st_, m, float(np.mean(rho))))
            ar = pd.DataFrame(ar, columns=["dgp", "strategy", "metric", "ar1"])
            ar.to_csv(out / "tables" / "e6_meta_ar1.csv", index=False)
            top = ar.loc[ar.ar1.idxmax()]
            num.add("e6meta.ar1.max", float(top.ar1))
            num.add("e6meta.ar1.max.where", f"{top.strategy} {top.metric}, {DGP_LABEL[top.dgp]}")
        size = _read(root, "e6_size", "size")
        if size is not None:
            for r in size.itertuples():
                num.pct(f"e6size.h{r.histories}.phi{r.phi:g}.{r.test}", r.rejection, 1)
        for gen in g.generator.unique():
            x = g[g.generator == gen]
            num.add(f"e6meta.{gen}.share", float(x.rejected.sum() / x.tests.sum()))


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
    num.add("cfg.l4.histories", int(cfg["meta_validation"]["l4_histories"]))
    build_e1(root, out, num, pr_min)
    ind = _read(root, "e7_l1null", "l1_indep")
    if ind is not None:
        rates = ind.groupby(["dgp", "statistic"]).flag_raw_indep.mean()
        num.pct("e7.l1indep.raw.mean", rates.mean(), 1)
        num.pct("e7.l1indep.raw.max", rates.max())
        num.add("e7.l1indep.raw.over9", int((rates > 0.09).sum()))
        num.pct("e7.l1indep.family", ind.groupby(["dgp", "k"]).flag_holm_indep.any().groupby("dgp").mean().max())
    num.add("cfg.B", int(cfg["protocol"]["L1"]["b_null"]))
    num.add("cfg.B.prereg", int(cfg["protocol"]["L1"].get("b_null_prereg", cfg["protocol"]["L1"]["b_null"])))
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
