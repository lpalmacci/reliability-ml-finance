"""Critical tolerances (additional analysis, not pre-registered; deviation D7).

The verdicts of levels L2 and L3 are binary at the declared tolerances. The stored confidence
intervals give, for every cell, the smallest tolerance at which it would pass: a comparison
declares equivalence at tolerance eta when max(-lo, hi) <= eta, so the cell passes when eta is
at least the PR_min-quantile of max(-lo, hi) over its comparisons (the critical tolerance). The
ratio of the critical to the declared tolerance is below one exactly for the cells that pass.
For level L3 the critical tolerance of a rule is the PR_min-quantile of the upper confidence
bounds of its regret. No new simulation is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tavp.protocol.card import FUNC_ORDER, STRATS
from tavp.report.attain import FUNC_TEXT
from tavp.report.case import BLUES, DGP_LABEL, DGP_ORDER, GRID, INK, INK2
from tavp.report.numbers import Numbers

# (kind, regime, label, short label); regime "none" is the negative control of E1
ROWS = [("oracle", "none", "$O$ oracle (negative control)", "O"),
        ("oracle_drift", "single", "$\\hat{O}$ oracle, estimated drift", "Ô 10y"),
        ("tcvae", "single", "G1 TC-VAE", "G1 10y"),
        ("bootstrap", "single", "B1 stationary bootstrap", "B1 10y"),
        ("garch", "single", "B2 GJR-GARCH-$t$", "B2 10y"),
        ("oracle_drift", "rich", "$\\hat{O}$ oracle, estimated drift", "Ô long"),
        ("tcvae", "rich", "G1 TC-VAE", "G1 rich"),
        ("bootstrap", "rich", "B1 stationary bootstrap", "B1 long"),
        ("garch", "rich", "B2 GJR-GARCH-$t$", "B2 long")]
KEYS = ["kind", "regime", "dgp"]
CELL = ["strategy", "metric", "functional"]


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _quantile(x: pd.Series, q: float) -> float:
    """Smallest value v such that the share of x not above v is at least q."""
    return float(np.quantile(x.to_numpy(), q, method="inverted_cdf"))


def _load(root: Path, level: str) -> pd.DataFrame | None:
    parts = []
    e1 = _read(root, "e1", level)
    if e1 is not None:
        parts.append(e1[e1.kind == "oracle"].assign(regime="none"))
    for exp in ("e4", "e5", "e8"):
        t = _read(root, exp, level)
        if t is not None:
            parts.append(t[t.kind.isin(["oracle_drift", "tcvae", "bootstrap", "garch"])])
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["regime"] = df.regime.fillna("single")
    return df


def critical_cells(l2: pd.DataFrame, pr_min: float) -> pd.DataFrame:
    """Critical tolerance of every cell of level L2, per generator kind, regime and process."""
    x = l2.assign(c=np.maximum(-l2.lo, l2.hi))
    g = x.groupby(KEYS + CELL)
    out = g.c.apply(lambda s: _quantile(s, pr_min)).rename("crit").to_frame()
    out["eta"] = g.eta.first()
    out["pr"] = g.equivalent.mean()
    out = out.reset_index()
    out["ratio"] = out.crit / out.eta
    return out


def critical_rules(l3: pd.DataFrame, pr_min: float) -> pd.DataFrame:
    g = l3.groupby(KEYS + ["strategy"])
    out = g.upper.apply(lambda s: _quantile(s, pr_min)).rename("crit").to_frame()
    out["eta"] = g.eta.first()
    out["pr"] = g["pass"].mean()
    return out.reset_index()


def tolcrit_figure(cells: pd.DataFrame, attain: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.2), sharey=True, sharex=True)
    rows = [r for r in ROWS if ((cells.kind == r[0]) & (cells.regime == r[1])).any()]
    rng = np.random.default_rng(0)  # vertical jitter only, for legibility
    for ax, d in zip(axes, DGP_ORDER):
        ax.axvline(1.0, color=INK2, linewidth=0.8, zorder=1)
        for i, (kind, regime, _, _) in enumerate(rows):
            x = cells[(cells.kind == kind) & (cells.regime == regime) & (cells.dgp == d)]
            if x.empty:
                continue
            A = attain.get((regime, d))
            inA = np.array([A is None or tuple(k) in A for k in x[CELL].itertuples(index=False)])
            y = i + rng.uniform(-0.22, 0.22, len(x))
            ax.scatter(x.ratio[inA], y[inA], s=7, color=BLUES[6], linewidths=0, zorder=3)
            ax.scatter(x.ratio[~inA], y[~inA], s=7, facecolors="none", edgecolors=BLUES[3], linewidths=0.6,
                       zorder=2)
        for yy in (0.5, 4.5):
            ax.axhline(yy, color=GRID, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_title(DGP_LABEL[d].replace("$", ""), fontsize=8, color=INK)
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[3] for r in rows], fontsize=7.5, color=INK)
    axes[0].invert_yaxis()
    fig.supxlabel("critical tolerance / declared tolerance (log scale); a cell passes when the ratio is at most 1",
                  fontsize=7.5, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_tolcrit(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    l2, l3 = _load(root, "l2"), _load(root, "l3")
    if l2 is None or _read(root, "e8", "l2") is None:
        return
    cells = critical_cells(l2, pr_min)
    # the critical tolerance reproduces the verdicts exactly
    mismatch = int(((cells.ratio <= 1) != (cells.pr >= pr_min)).sum())
    num.add("tolcrit.mismatch", mismatch)
    rules = critical_rules(l3, pr_min) if l3 is not None else None
    if rules is not None:
        num.add("tolcrit.l3.mismatch", int(((rules.crit <= rules.eta) != (rules.pr >= pr_min)).sum()))
    cells.to_csv(out / "tables" / "tolcrit_cells.csv", index=False)
    # attainable cells: those that the oracle with estimated drift passes, per amount of data
    o = cells[cells.kind == "oracle_drift"]
    attain = {(r, d): {tuple(k) for k in o[(o.regime == r) & (o.dgp == d) & (o.ratio <= 1)][CELL]
                       .itertuples(index=False)} for r in ("single", "rich") for d in DGP_ORDER}
    lines, prev = [], None
    for kind, regime, lab, _ in ROWS:
        vals = []
        for d in DGP_ORDER:
            x = cells[(cells.kind == kind) & (cells.regime == regime) & (cells.dgp == d)]
            if x.empty:
                vals += ["", "", ""]
                continue
            k_all = float(x.ratio.max())
            A = attain.get((regime, d))
            xa = x if A is None else x[[tuple(k) in A for k in x[CELL].itertuples(index=False)]]
            k_att = float(xa.ratio.max()) if len(xa) else np.nan
            r3 = rules[(rules.kind == kind) & (rules.regime == regime) & (rules.dgp == d)] if rules is not None else None
            l3c = float(r3.crit.max()) if r3 is not None and len(r3) else np.nan
            key = f"tolcrit.{kind}.{regime}.{d}"
            num.add(key + ".all", k_all, "{:.1f}" if k_all >= 10 else "{:.2f}")
            num.add(key + ".attain", k_att, "{:.1f}" if k_att >= 10 else "{:.2f}")
            num.add(key + ".l3", l3c, "{:.3f}")
            num.add(key + ".median", float(x.ratio.median()), "{:.2f}")
            fmt = lambda v: ("--" if np.isnan(v) else (f"{v:.1f}" if v >= 10 else f"{v:.2f}"))
            bold = lambda s, ok: r"\textbf{" + s + "}" if ok else s
            vals += [bold(fmt(k_all), k_all <= 1), "--" if A is None else bold(fmt(k_att), k_att <= 1),
                     bold(f"{l3c:.3f}", l3c <= cfg["use"]["tolerances"]["regret"])]
        group = {"none": "Known law", "single": f"{cfg['data']['train_returns'] // 252} years",
                 "rich": f"{cfg['data']['rich_paths'] * (cfg['data']['scenario_prices'] - 1):,} returns"
                 .replace(",", "{,}")}[regime]
        if prev is not None and regime != prev:
            lines.append(r"\addlinespace")
        lines.append(f"{group if regime != prev else ''} & {lab} & " + " & ".join(vals) + r" \\")
        prev = regime
    head = " & ".join(r"\multicolumn{3}{c}{" + DGP_LABEL[d].split(" ")[0] + "}" for d in DGP_ORDER)
    rules_tex = r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}"
    (out / "tables" / "tab_tolcrit.tex").write_text(r"""\begin{tabular}{ll""" + "r" * 12 + r"""}
\toprule
Training data & Generator & """ + head + r""" \\
""" + rules_tex + r"""
& & """ + " & ".join([r"$\kappa^\ast$ & $\kappa^\ast_{\mathcal{A}}$ & L3"] * 4) + r""" \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    # summaries used in the text
    for kind, regime, _, _ in ROWS:
        x = cells[(cells.kind == kind) & (cells.regime == regime)]
        if x.empty:
            continue
        per = x.groupby("dgp").ratio.max()
        if rules is not None:
            r3 = rules[(rules.kind == kind) & (rules.regime == regime)].groupby("dgp").crit.max()
            if len(r3):
                num.add(f"tolcrit.{kind}.{regime}.l3.min", float(r3.min()), "{:.3f}")
                num.add(f"tolcrit.{kind}.{regime}.l3.max", float(r3.max()), "{:.3f}")
        num.add(f"tolcrit.{kind}.{regime}.all.min", float(per.min()), "{:.2f}")
        num.add(f"tolcrit.{kind}.{regime}.all.max", float(per.max()), "{:.1f}" if per.max() >= 10 else "{:.2f}")
        if regime in ("single", "rich"):
            inA = np.array([tuple(k) in attain[(regime, d)] for d, *k in x[["dgp"] + CELL].itertuples(index=False)])
            att = [float(x[(x.dgp == d).to_numpy() & inA].ratio.max()) for d in DGP_ORDER if (x.dgp == d).any()]
            num.add(f"tolcrit.{kind}.{regime}.attain.min", min(att), "{:.2f}")
            num.add(f"tolcrit.{kind}.{regime}.attain.max", max(att), "{:.1f}" if max(att) >= 10 else "{:.2f}")
    # smallest tolerance that the data allow, per functional: the largest critical tolerance of the
    # oracle with estimated drift over rules and processes (and of the oracle, the Monte Carlo floor)
    flines = []
    for m, f in FUNC_ORDER:
        sel = lambda k, r: cells[(cells.kind == k) & (cells.regime == r) & (cells.metric == m)
                                 & (cells.functional == f)].crit.max()
        fmt = "{:.2f}" if m == "sharpe" else "{:.3f}"
        eta = float(cells[(cells.metric == m) & (cells.functional == f)].eta.iloc[0])
        v = {"o": sel("oracle", "none"), "s": sel("oracle_drift", "single"), "r": sel("oracle_drift", "rich")}
        for k, val in v.items():
            num.add(f"tolfunc.{k}.{m}.{f}", float(val), fmt)
        flines.append(f"{FUNC_TEXT[(m, f)]} & {fmt.format(eta)} & {fmt.format(v['o'])} & {fmt.format(v['s'])} & "
                      f"{fmt.format(v['r'])}" + r" \\")
    reg = {"single": f"{cfg['data']['train_returns'] // 252} years",
           "rich": f"{cfg['data']['rich_paths'] * (cfg['data']['scenario_prices'] - 1):,} returns".replace(",", "{,}")}
    (out / "tables" / "tab_tolfunc.tex").write_text(r"""\begin{tabular}{lrrrr}
\toprule
& & & \multicolumn{2}{c}{$\hat{O}$, estimated drift} \\
\cmidrule(lr){4-5}
Functional & Declared & $O$, known law & """ + reg["single"] + " & " + reg["rich"] + r""" \\
\midrule
""" + "\n".join(flines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    s = cells[(cells.kind == "oracle_drift") & (cells.metric == "sharpe") & (cells.functional == "mean")
              & (cells.strategy == "BH")]
    bh = l2[(l2.kind == "oracle") & (l2.regime == "none") & (l2.strategy == "BH") & (l2.metric == "sharpe")
            & (l2.functional == "mean")]
    for d in DGP_ORDER:
        num.add(f"tolfunc.bh.sharpe.oracle.{d}", float(bh[bh.dgp == d].est_o.mean()), "{:.2f}")
    for r in ("single", "rich"):
        num.add(f"tolfunc.bh.sharpe.mean.{r}.max", float(s[s.regime == r].crit.max()), "{:.2f}")
    tolcrit_figure(cells[cells.kind.isin([k for k, _, _, _ in ROWS])], attain, out / "figures" / "fig_tolcrit.pdf")
