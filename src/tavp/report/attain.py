"""Attainability and positive control (E8, added after the results, not pre-registered; deviation D5).

A cell of level L2 (or a calibrated rule of level L3) is *attainable* with a given amount of
data if the oracle with estimated drift passes it: that generator has the true dynamics and
misses only the drift, which it estimates from the same history. The verdict on the
attainable cells asks whether a generator passes every cell that the data allow it to pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tavp.protocol.card import FUNC_ORDER, STRATS, build_card, card_latex
from tavp.report.case import BLUES, DGP_LABEL, DGP_ORDER, INK, INK2
from tavp.report.numbers import Numbers

KINDS = [("oracle_drift", "$\\hat{O}$ oracle, estimated drift", "$\\hat{O}$"), ("tcvae", "G1 TC-VAE", "G1"),
         ("bootstrap", "B1 stationary bootstrap", "B1"), ("garch", "B2 GJR-GARCH-$t$", "B2")]
REGIMES = ["single", "rich"]
FUNC_TEXT = {("sharpe", "mean"): "Sharpe ratio, mean", ("sharpe", "q05"): "Sharpe ratio, $q_{.05}$",
             ("sharpe", "q50"): "Sharpe ratio, $q_{.50}$", ("sharpe", "q95"): "Sharpe ratio, $q_{.95}$",
             ("maxdd", "mean"): "maximum drawdown, mean", ("maxdd", "q95"): "maximum drawdown, $q_{.95}$",
             ("ret", "mean"): "return, mean", ("ret", "var95"): "return, VaR", ("ret", "es95"): "return, ES"}


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _levels(root: Path) -> dict[str, pd.DataFrame]:
    """Levels of the generators compared in E8, with the regime of the benchmarks of E5 set to single."""
    kinds = [k for k, _, _ in KINDS]
    out = {}
    for level in ("l0", "l1", "l2", "l3"):
        parts = [t for t in (_read(root, e, level) for e in ("e4", "e5", "e8")) if t is not None]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df[df.kind.isin(kinds)].copy()
        df["regime"] = df.regime.fillna("single")
        out[level] = df
    return out


def _sel(df: pd.DataFrame | None, dgp: str, kind: str, regime: str) -> pd.DataFrame | None:
    if df is None:
        return None
    return df[(df.dgp == dgp) & (df.kind == kind) & (df.regime == regime)]


def _cards(lv: dict, pr_min: float) -> dict:
    cards = {}
    for d in DGP_ORDER:
        for r in REGIMES:
            for kind, _, _ in KINDS:
                x2 = _sel(lv["l2"], d, kind, r)
                if x2 is None or x2.empty:
                    continue
                c = build_card(_sel(lv.get("l0"), d, kind, r), _sel(lv.get("l1"), d, kind, r), x2,
                               _sel(lv.get("l3"), d, kind, r), None, pr_min)
                per_seed = (x2.groupby(["seed", "strategy", "metric", "functional"]).equivalent.mean() >= pr_min)
                c["per_seed"] = per_seed
                cards[(d, r, kind)] = c
    return cards


def attain_figure(cards: dict, path: Path, pr_min: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    flab = {"mean": "mean", "q05": "q.05", "q50": "q.50", "q95": "q.95", "var95": "VaR", "es95": "ES"}
    mlab = {"sharpe": "Sharpe", "maxdd": "MaxDD", "ret": "return"}
    keys = [(s, m, f) for s in STRATS for m, f in FUNC_ORDER]
    rows = [f"{s}  {mlab[m]} {flab[f]}" for s, m, f in keys]
    cols = [(r, kind, short) for r in REGIMES for kind, _, short in KINDS]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 8.6), sharey=True)
    im = None
    for ax, d in zip(axes, DGP_ORDER):
        M = np.full((len(keys), len(cols)), np.nan)
        att = np.ones_like(M, dtype=bool)
        for j, (r, kind, _) in enumerate(cols):
            c = cards.get((d, r, kind))
            o = cards.get((d, r, "oracle_drift"))
            if c is not None:
                M[:, j] = [c["L2"][k] for k in keys]
            if o is not None:
                att[:, j] = [o["L2"][k] >= pr_min for k in keys]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        yy, xx = np.where(M >= pr_min)
        ax.scatter(xx, yy, s=8, color="white", linewidths=0, zorder=3)
        yy, xx = np.where(~att)
        ax.scatter(xx, yy, s=12, marker="x", color=INK2, linewidths=0.7, zorder=3)
        ax.axvline(len(KINDS) - 0.5, color="white", linewidth=3)
        for y in (9, 18, 27):
            ax.axhline(y - 0.5, color="white", linewidth=1.5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{short.replace('$', '').replace(chr(92) + 'hat{O}', 'Ô')} {'10y' if r == 'single' else 'long'}"
                            for r, _, short in cols], rotation=90, fontsize=7, color=INK)
        ax.set_title(DGP_LABEL[d].replace("$", ""), fontsize=8, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.025, pad=0.09, aspect=40)
    cb.set_label("L2 pass rate (white dot: cell passes; x: not attainable)", fontsize=7.5, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    cb.outline.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_attain(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    if _read(root, "e8", "l2") is None:
        return
    lv = _levels(root)
    cards = _cards(lv, pr_min)
    keys = [(s, m, f) for s in STRATS for m, f in FUNC_ORDER]
    num.add("e8.long.returns", int(cfg["data"]["rich_paths"] * (cfg["data"]["scenario_prices"] - 1)))
    rows = []
    for d in DGP_ORDER:
        for r in REGIMES:
            o = cards.get((d, r, "oracle_drift"))
            if o is None:
                continue
            att = [k for k in keys if o["L2"][k] >= pr_min]
            att_rules = [s for s, v in o["L3"].items() if v["pr"] >= pr_min]
            num.add(f"e8.A.{r}.{d}", len(att))
            num.add(f"e8.L3A.{r}.{d}", ", ".join(att_rules) or "none")
            num.add(f"e8.L3A.{r}.{d}.n", len(att_rules))
            for kind, _, _ in KINDS:
                c = cards.get((d, r, kind))
                if c is None:
                    continue
                passed = [k for k in keys if c["L2"][k] >= pr_min]
                passed_att = [k for k in att if c["L2"][k] >= pr_min]
                l3_fail_att = [s for s in att_rules if s in c["L3_failed"]]
                fit_att = bool(c["L0"]["hash_ok"] and c["L0"]["valid"] and len(passed_att) == len(att)
                               and not l3_fail_att)
                ps = c["per_seed"]
                att_set = set(att)
                ps_att = ps[[tuple(i[1:]) in att_set for i in ps.index]].groupby("seed").sum()
                key = f"e8.{kind}.{r}.{d}"
                num.add(key + ".pass", len(passed))
                num.add(key + ".passA", len(passed_att))
                num.add(key + ".passA.min", int(ps_att.min()) if len(ps_att) else 0)
                num.add(key + ".passA.max", int(ps_att.max()) if len(ps_att) else 0)
                num.add(key + ".l1flag", len(c["L1_flagged"]))
                num.add(key + ".l3failA", ", ".join(l3_fail_att) or "none")
                num.add(key + ".fitA", "fit" if fit_att else "not fit")
                rows.append({"dgp": d, "regime": r, "kind": kind, "A": len(att), "pass": len(passed),
                             "passA": len(passed_att), "passA_min": int(ps_att.min()) if len(ps_att) else 0,
                             "passA_max": int(ps_att.max()) if len(ps_att) else 0, "l3_rules_att": len(att_rules),
                             "l3failA": ", ".join(l3_fail_att), "fitA": fit_att, "fit": c["fit"],
                             "l1flag": len(c["L1_flagged"]), "resolvable": c["L0"]["resolvable"]})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return
    tab.to_csv(out / "tables" / "e8_attain.csv", index=False)
    for r in REGIMES:
        a = tab[(tab.regime == r) & (tab.kind == "oracle_drift")].A
        num.add(f"e8.A.{r}.min", int(a.min()))
        num.add(f"e8.A.{r}.max", int(a.max()))
        for kind, _, _ in KINDS:
            t = tab[(tab.regime == r) & (tab.kind == kind)]
            fit = [DGP_LABEL[d].split(" ")[0] for d in DGP_ORDER if t[t.dgp == d].fitA.any()]
            num.add(f"e8.{kind}.{r}.fitA.n", len(fit))
            num.add(f"e8.{kind}.{r}.fitA.list", ", ".join(fit) or "none")
            num.add(f"e8.{kind}.{r}.fit.n", int(t.fit.sum()))
    # main table: cells passed (of 36) and attainable cells passed (of A), per process
    labels = {k: lab for k, lab, _ in KINDS}
    reg_label = {"single": f"{cfg['data']['train_returns'] // 252} years",
                 "rich": f"{cfg['data']['rich_paths'] * (cfg['data']['scenario_prices'] - 1):,} returns".replace(",", "{,}")}
    lines = []
    for r in REGIMES:
        for i, (kind, _, _) in enumerate(KINDS):
            cells = []
            for d in DGP_ORDER:
                x = tab[(tab.regime == r) & (tab.kind == kind) & (tab.dgp == d)]
                if x.empty:
                    cells.append(" & ")
                    continue
                x = x.iloc[0]
                pa = f"{x.passA}/{x.A}"
                cells.append(f"{x['pass']} & " + (r"\textbf{" + pa + "}" if x.fitA else pa))
            lab = reg_label[r] if i == 0 else ""
            lines.append(f"{lab} & {labels[kind]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    head = " & ".join(r"\multicolumn{2}{c}{" + DGP_LABEL[d].split(" ")[0] + "}" for d in DGP_ORDER)
    (out / "tables" / "tab_e8_attain.tex").write_text(r"""\begin{tabular}{llrrrrrrrr}
\toprule
Training data & Generator & """ + head + r""" \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
& & all & attain. & all & attain. & all & attain. & all & attain. \\
\midrule
""" + "\n".join(lines[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
""")
    # attainability by functional: rule-process combinations (of 16) in which the cell is attainable
    flines = []
    for m, f in FUNC_ORDER:
        vals = []
        for r in REGIMES:
            n = sum(cards[(d, r, "oracle_drift")]["L2"][(s, m, f)] >= pr_min
                    for d in DGP_ORDER for s in STRATS if (d, r, "oracle_drift") in cards)
            vals.append(str(n))
            num.add(f"e8.Afunc.{r}.{m}.{f}", n)
        flines.append(f"{FUNC_TEXT[(m, f)]} & " + " & ".join(vals) + r" \\")
    (out / "tables" / "tab_e8_functionals.tex").write_text(r"""\begin{tabular}{lrr}
\toprule
Functional & """ + " & ".join(reg_label[r] for r in REGIMES) + r""" \\
\midrule
""" + "\n".join(flines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    attain_figure(cards, out / "figures" / "fig_case_attain.pdf", pr_min)
    # validation cards of the generators trained on the long histories, with the unattainable cells marked
    (out / "cards").mkdir(parents=True, exist_ok=True)
    for (d, r, kind), c in cards.items():
        if r != "rich" or kind not in ("garch", "bootstrap"):
            continue
        o = cards[(d, r, "oracle_drift")]
        A = {k for k in keys if o["L2"][k] >= pr_min}
        rules = {st for st, v in o["L3"].items() if v["pr"] >= pr_min}
        (out / "cards" / f"card_{kind}_long_{d}.tex").write_text(card_latex(c, pr_min, attainable=A,
                                                                           attainable_rules=rules) + "\n")
    # drift errors and fitted parameters
    gens = _read(root, "e8", "generators")
    if gens is not None:
        for r in REGIMES:
            g = gens[(gens.regime == r) & (gens.kind == "oracle_drift")]
            num.add(f"e8.drift.{r}.sd", float(np.sqrt((g.drift_error_annual**2).mean())), "{:.3f}")
            num.add(f"e8.drift.{r}.maxabs", float(g.drift_error_annual.abs().max()), "{:.3f}")
        for (kind, r, d), x in gens.groupby(["kind", "regime", "dgp"]):
            num.add(f"e8.drift.rms.{kind}.{r}.{d}", float(np.sqrt((x.drift_error_annual**2).mean())), "{:.4f}")
        years = cfg["data"]["rich_paths"] * (cfg["data"]["scenario_prices"] - 1) / 252
        num.add("e8.years.rich", int(round(years)))
        num.add("e8.drift.limit.rich", 0.18 / np.sqrt(years), "{:.4f}")
        # a drift error delta (annual) shifts the annualised Sharpe ratio of buy-and-hold by delta / sigma, and the
        # sample mean of Y years has standard error sigma / sqrt(Y): the error of the Sharpe ratio is 1 / sqrt(Y)
        from scipy.stats import norm
        eta = cfg["use"]["tolerances"]["sharpe"]["mean"]
        z = float(norm.ppf((1 + pr_min) / 2))
        num.add("e8.sharpe.se.single", 1 / np.sqrt(cfg["data"]["train_returns"] / 252), "{:.2f}")
        num.add("e8.sharpe.se.rich", 1 / np.sqrt(years), "{:.3f}")
        num.add("e8.years.needed", f"{int(round((z / eta) ** 2, -2)):,}".replace(",", "{,}"))
        g3 = gens[gens.kind == "garch"].copy()
        if len(g3):
            pers = g3.alpha + g3.gamma / 2 + g3.beta
            g3["vol"] = np.sqrt(g3.omega / (1 - pers) / g3.dt)
            for (r, d), x in g3.groupby(["regime", "dgp"]):
                num.add(f"e8.garch.{r}.{d}.vol.min", float(x.vol.min()), "{:.3f}")
                num.add(f"e8.garch.{r}.{d}.vol.max", float(x.vol.max()), "{:.3f}")
                num.add(f"e8.garch.{r}.{d}.drift", float(x.drift_error_annual.mean()), "{:.3f}")
        num.add("e8.vol.true.gjr_garch_t", float(cfg["dgps"]["gjr_garch_t"]["vol_ann"]), "{:.2f}")
        b = gens[(gens.kind == "bootstrap") & (gens.regime == "rich")]
        if len(b):
            dep = b[b.dgp != "gbm"].block_mean
            num.add("e8.bootstrap.rich.block.gbm", float(b[b.dgp == "gbm"].block_mean.mean()), "{:.1f}")
            num.add("e8.bootstrap.rich.block.min", int(round(dep.min())))
            num.add("e8.bootstrap.rich.block.max", int(round(dep.max())))
        gg = gens[(gens.kind == "garch") & (gens.regime == "rich") & (gens.dgp == "gjr_garch_t")]
        for p in ("alpha", "gamma", "beta", "nu"):
            if p in gg:
                num.add(f"e8.garch.rich.gjr_garch_t.{p}", float(gg[p].mean()), "{:.3f}" if p != "nu" else "{:.1f}")
