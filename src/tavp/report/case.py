"""Tables, figures, validation cards and numbers of the case study (E4, E5, E6 case, X1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from tavp.protocol.card import FUNC_ORDER, STRATS, build_card, card_latex, card_markdown
from tavp.protocol.l4 import calibration_tests
from tavp.report.numbers import Numbers

DGP_LABEL = {"gbm": "D1 GBM", "heston": "D2 Heston", "gjr_garch_t": "D3 GJR-GARCH-$t$", "regime": "D4 Regimes"}
DGP_ORDER = ["gbm", "heston", "gjr_garch_t", "regime"]
# (kind, regime, label, exploratory)
GENERATORS = [("tcvae", "single", "G1 TC-VAE, single history", False),
              ("tcvae", "rich", "G1 TC-VAE, rich", False),
              ("vae_mlp", "single", "G2 non-causal VAE", False),
              ("bootstrap", None, "B1 stationary bootstrap", False),
              ("garch", None, "B2 GJR-GARCH-$t$", False),
              ("vae_mlp_wide", "single", "X1 non-causal VAE, 256 units", True)]
# Categorical slots 1-5 of the reference palette (validated, light mode); X1 shares G2's hue.
# One colour per family of generator in every figure of the chapter, the regime in the marker. Slots 1, 2, 3
# and 7 of the reference palette pass the colour-vision checks on all pairs (light mode); the ablation is grey.
FAMILY = {"tcvae": "#2a78d6", "qgan": "#eb6834", "bootstrap": "#1baf7a", "garch": "#4a3aa7", "vae_mlp": "#8a8984",
          "oracle": "#0b0b0b"}
COLORS = {"tcvae_single": FAMILY["tcvae"], "tcvae_rich": FAMILY["tcvae"], "qgan_single": FAMILY["qgan"],
          "qgan_rich": FAMILY["qgan"], "vae_mlp_single": FAMILY["vae_mlp"], "bootstrap": FAMILY["bootstrap"],
          "garch": FAMILY["garch"], "vae_mlp_wide_single": FAMILY["vae_mlp"]}
MARKERS = {"tcvae_single": "o", "tcvae_rich": "s", "qgan_single": "o", "qgan_rich": "s", "vae_mlp_single": "^",
           "bootstrap": "D", "garch": "v", "vae_mlp_wide_single": "x"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STAT_ROWS = [("std", "volatility"), ("acf_r1", "ACF($r$), lag 1"), ("vr", "variance ratio, 59 days"),
             ("acf_abs", "ACF($|r|$), lags 1--20"),
             ("kurt", "kurtosis"), ("hill", "Hill index"), ("leverage", "leverage, lags 1--10")]


def gkey(kind: str, regime: str | None) -> str:
    return f"{kind}_{regime}" if regime else kind


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _levels(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for level in ("l0", "l1", "l2", "l3"):
        parts = [t for t in (_read(root, e, level) for e in ("e4", "e5", "x1_wide")) if t is not None]
        if parts:
            out[level] = pd.concat(parts, ignore_index=True)
    return out


def _sel(df: pd.DataFrame | None, dgp: str, kind: str, regime: str | None) -> pd.DataFrame | None:
    if df is None:
        return None
    m = (df.dgp == dgp) & (df.kind == kind)
    if regime is not None and "regime" in df:
        m &= df.regime == regime
    return df[m]


def l4_pooled(pits: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """L4 tests per (process, generator kind), pooling the held-out histories of all runs."""
    rows = []
    for (d, k), g in pits.groupby(["dgp", "kind"]):
        rows += [{"dgp": d, "kind": k, **r} for r in calibration_tests(g, cfg)]
    return pd.DataFrame(rows)


def cell_matrix(l2: pd.DataFrame, dgp: str) -> pd.DataFrame:
    cols = {}
    for kind, regime, _, expl in GENERATORS:
        if expl:
            continue
        x = _sel(l2, dgp, kind, regime)
        if x is None or x.empty:
            continue
        pr = x.groupby(["strategy", "metric", "functional"]).equivalent.mean()
        cols[gkey(kind, regime)] = [float(pr.get((s, m, f), np.nan)) for s in STRATS for m, f in FUNC_ORDER]
    return pd.DataFrame(cols, index=[f"{s} {m} {f}" for s in STRATS for m, f in FUNC_ORDER])


def cells_figure(l2: pd.DataFrame, path: Path, pr_min: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    short = {"tcvae_single": "G1 single", "tcvae_rich": "G1 rich", "vae_mlp_single": "G2", "bootstrap": "B1",
             "garch": "B2"}
    flab = {"mean": "mean", "q05": "q.05", "q50": "q.50", "q95": "q.95", "var95": "VaR", "es95": "ES"}
    mlab = {"sharpe": "Sharpe", "maxdd": "MaxDD", "ret": "return"}
    rows = [f"{s}  {mlab[m]} {flab[f]}" for s in STRATS for m, f in FUNC_ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 8.6), sharey=True)
    im = None
    for ax, d in zip(axes, DGP_ORDER):
        M = cell_matrix(l2, d)
        im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        yy, xx = np.where(M.to_numpy() >= pr_min)
        ax.scatter(xx, yy, s=8, color="white", linewidths=0)
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([short[c] for c in M.columns], rotation=90, fontsize=7, color=INK)
        ax.set_title(DGP_LABEL[d].replace("$", ""), fontsize=8, color=INK)
        for y in (9, 18, 27):
            ax.axhline(y - 0.5, color="white", linewidth=1.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize=7, color=INK)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.025, pad=0.09, aspect=40)
    cb.set_label("L2 pass rate (white dot: cell passes)", fontsize=7.5, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    cb.outline.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def facts_figure(facts: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    facts = facts.assign(g=[gkey(k, r) if k in ("tcvae", "vae_mlp") else k for k, r in zip(facts.kind, facts.regime)])
    order = ["tcvae_single", "tcvae_rich", "vae_mlp_single", "bootstrap", "garch"]
    names = {"tcvae_single": "G1 single", "tcvae_rich": "G1 rich", "vae_mlp_single": "G2", "bootstrap": "B1",
             "garch": "B2"}
    fig, axes = plt.subplots(len(STAT_ROWS), 4, figsize=(6.3, 9.2), sharey=True)
    for i, (stat, slab) in enumerate(STAT_ROWS):
        for j, d in enumerate(DGP_ORDER):
            ax = axes[i, j]
            f = facts[facts.dgp == d]
            o = f[f.kind == "oracle"][stat].iloc[0]
            ax.axvline(o, color=INK, linewidth=1.0)
            for y, g in enumerate(order):
                x = f[f.g == g][stat].to_numpy()
                if not len(x):
                    continue
                ax.scatter(x, np.full(len(x), y), s=12, color=COLORS[g], marker=MARKERS[g], alpha=0.55,
                           linewidths=0, zorder=3)
                ax.scatter([np.median(x)], [y], s=34, color=COLORS[g], marker=MARKERS[g], edgecolors="white",
                           linewidths=0.8, zorder=4)
            if stat == "kurt":
                ax.set_xscale("log")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([names[g] for g in order], fontsize=6.5, color=INK)
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.tick_params(axis="x", labelsize=6.5, colors=INK2, length=0)
            ax.tick_params(axis="y", length=0)
            ax.grid(True, axis="x", color=GRID, linewidth=0.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(DGP_LABEL[d].replace("$", ""), fontsize=8, color=INK)
            if j == 0:
                ax.annotate(slab, xy=(0, 1.02), xycoords="axes fraction", fontsize=6.5, color=INK, ha="left",
                            va="bottom")
    fig.tight_layout(h_pad=1.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def build_case(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    lv = _levels(root)
    if "l2" not in lv:
        return
    l0, l1, l2, l3 = lv.get("l0"), lv.get("l1"), lv["l2"], lv.get("l3")
    pits = _read(root, "e6_case", "pits")
    l4 = l4_pooled(pits, cfg) if pits is not None else None
    if l4 is not None:
        l4.to_csv(out / "tables" / "e6_case_pooled.csv", index=False)
        # per-run rejections and, for the TC-VAE, pooled rejections without collapsed runs
        for (d, k), g in pits.groupby(["dgp", "kind"]):
            per = [int(pd.DataFrame(calibration_tests(h, cfg)).reject_any.sum()) for _, h in g.groupby("history")]
            key = f"case.{gkey(k, 'single') if k in ('tcvae', 'vae_mlp') else k}.{d}"
            num.add(key + ".l4rej.run.min", min(per))
            num.add(key + ".l4rej.run.max", max(per))
        facts0 = _read(root, "e4", "facts")
        if facts0 is not None:
            coll = facts0[(facts0.kind == "tcvae") & (facts0.regime == "single") & (facts0["std"] < 0.05)]
            for r in coll.itertuples():
                g = pits[(pits.dgp == r.dgp) & (pits.kind == "tcvae")]
                t0 = pd.DataFrame(calibration_tests(g[g.history == r.seed], cfg))
                t1 = pd.DataFrame(calibration_tests(g[g.history != r.seed], cfg))
                num.add(f"case.tcvae_single.{r.dgp}.l4rej.collapsed", int(t0.reject_any.sum()))
                num.add(f"case.tcvae_single.{r.dgp}.l4cov.collapsed", float(t0.coverage.mean()))
                num.add(f"case.tcvae_single.{r.dgp}.l4rej.others", int(t1.reject_any.sum()))
    rows, cards_md = [], []
    for d in DGP_ORDER:
        for kind, regime, label, expl in GENERATORS:
            x2 = _sel(l2, d, kind, regime)
            if x2 is None or x2.empty:
                continue
            x4 = _sel(l4, d, kind, None) if (l4 is not None and regime in (None, "single")) else None
            c = build_card(_sel(l0, d, kind, regime), _sel(l1, d, kind, regime), x2, _sel(l3, d, kind, regime),
                           x4 if x4 is not None and len(x4) else None, pr_min)
            key = f"case.{gkey(kind, regime)}.{d}"
            per_seed = x2.groupby(["seed", "strategy", "metric", "functional"]).equivalent.mean() >= pr_min
            per_seed = per_seed.groupby("seed").sum()
            n_pass = int(sum(v >= pr_min for v in c["L2"].values() if np.isfinite(v)))
            num.add(key + ".l2pass", n_pass)
            num.add(key + ".l2pass.min", int(per_seed.min()))
            num.add(key + ".l2pass.max", int(per_seed.max()))
            num.add(key + ".l1flag", len(c["L1_flagged"]))
            num.add(key + ".l3fail", ", ".join(c["L3_failed"]) or "none")
            num.add(key + ".l3fail.n", len(c["L3_failed"]))
            num.add(key + ".fit", "fit" if c["fit"] else "not fit")
            for s, v in c["L3"].items():
                num.add(f"{key}.l3.{s}.pr", v["pr"])
                num.add(f"{key}.l3.{s}.regret", v["regret"], "{:.3f}")
            l4s = "--" if c["L4"] is None else f"{len(c['L4']['rejected'])}/{c['L4']['tests']}"
            if c["L4"] is not None:
                num.add(key + ".l4rej", len(c["L4"]["rejected"]))
            rows.append({"dgp": d, "label": label, "expl": expl, "runs": c["runs"], "l1": len(c["L1_flagged"]),
                         "l2": n_pass, "l2min": int(per_seed.min()), "l2max": int(per_seed.max()),
                         "l3": ", ".join(c["L3_failed"]) or "none", "l4": l4s, "fit": c["fit"]})
            if not expl:
                stem = f"card_{gkey(kind, regime)}_{d}"
                (out / "cards").mkdir(parents=True, exist_ok=True)
                (out / "cards" / f"{stem}.tex").write_text(card_latex(c, pr_min) + "\n")
                cards_md.append(card_markdown(c))
    (out / "cards" / "cards.md").write_text("\n\n".join(cards_md) + "\n")
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tables" / "case_summary.csv", index=False)
    lines = []
    for d in DGP_ORDER:
        t = tab[tab.dgp == d]
        for i, r in enumerate(t.itertuples()):
            proc = DGP_LABEL[d] if i == 0 else ""
            lab = r.label + ("$^{\\dagger}$" if r.expl else "")
            lines.append(f"{proc} & {lab} & {r.runs} & {r.l1}/13 & {r.l2}/36 & {r.l2min}--{r.l2max} & {r.l3} & {r.l4} & "
                         f"{'fit' if r.fit else 'not fit'} \\\\")
        lines.append(r"\addlinespace")
    (out / "tables" / "tab_case_summary.tex").write_text(r"""\begin{tabular}{llrrrrlrl}
\toprule
& & & L1 & \multicolumn{2}{c}{L2 cells passed} & L3 & L4 & \\
\cmidrule(lr){5-6}
Process & Generator & runs & flagged & pooled & per run & failed rules & rejected & Verdict \\
\midrule
""" + "\n".join(lines[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
""")
    if l3 is not None:
        tf = l3[l3.strategy == "TF"].assign(reg=lambda x: x.regime.fillna("none") if "regime" in x else "none")
        reg = tf.groupby(["kind", "reg", "dgp"]).regret.mean().groupby(["kind", "reg"]).agg(["min", "max"])
        for (kind, rg), r in reg.iterrows():
            k = gkey(kind, rg if rg != "none" else None)
            num.add(f"case.{k}.l3.TF.regret.min", float(r["min"]), "{:.3f}")
            num.add(f"case.{k}.l3.TF.regret.max", float(r["max"]), "{:.3f}")
    res = l2[l2.kind.isin(["tcvae"])].groupby(["dgp", "regime"]).resolvable.mean()
    num.pct("case.tcvae.resolvable.min", float(res.min()), 2)
    num.add("case.fit.count", int(tab[~tab.expl].fit.sum()))
    num.add("case.settings", int((~tab.expl).sum()))
    # rank agreement between L1 (statistics flagged) and L2 (cells passed) across generators and processes
    t = tab[~tab.expl]
    tau, p = sps.kendalltau(-t.l1, t.l2)
    num.add("case.tau.l1l2", float(tau))
    num.add("case.tau.l1l2.p", float(p), "{:.3f}")
    # drift error of the generators: annualised error of the mean horizon return of buy-and-hold
    T = cfg["data"]["scenario_prices"] - 1
    bh = l2[(l2.strategy == "BH") & (l2.metric == "ret") & (l2.functional == "mean")]
    bh = bh.assign(reg=bh.regime.fillna("none"))
    per = bh.groupby(["kind", "reg", "dgp", "seed"])["diff"].mean() * 252 / T
    sd = per.groupby(["kind", "reg", "dgp"]).std()
    sd.reset_index().to_csv(out / "tables" / "case_drift_sd.csv", index=False)
    for (kind, reg, d), v in sd.items():
        num.add(f"case.drift.sd.{gkey(kind, reg if reg != 'none' else None)}.{d}", float(v), "{:.3f}")
    for (kind, reg), v in sd.groupby(["kind", "reg"]):
        num.add(f"case.drift.sd.{gkey(kind, reg if reg != 'none' else None)}", float(v.median()), "{:.3f}")
    num.add("case.drift.sd.theory", float(cfg["dgps"]["gbm"]["sigma"] / np.sqrt(cfg["data"]["train_returns"] / 252)),
            "{:.3f}")
    cells_figure(l2, out / "figures" / "fig_case_cells.pdf", pr_min)
    facts = _read(root, "e4", "facts")
    if facts is not None:
        facts_figure(facts, out / "figures" / "fig_case_facts.pdf")
        f = facts.assign(g=[gkey(k, r) if k in ("tcvae", "vae_mlp") else k for k, r in zip(facts.kind, facts.regime)])
        med = f.groupby(["g", "dgp"])[[s for s, _ in STAT_ROWS]].median()
        for (g, d), r in med.iterrows():
            for s, _ in STAT_ROWS:
                num.add(f"facts.{g}.{d}.{s}", float(r[s]), "{:.3f}" if s in ("acf_r1", "acf_abs", "leverage") else "{:.2f}")
        num.add("facts.tcvae.vr.min", float(f[(f.kind == "tcvae") & (f["std"] >= 0.05)].vr.min()))
        num.add("facts.tcvae.vr.max", float(f[(f.kind == "tcvae") & (f["std"] >= 0.05)].vr.max()))
        tc = f[(f.kind == "tcvae") & (f["std"] >= 0.05)]  # collapsed runs are reported separately
        num.add("facts.tcvae.acf_r1.min", float(tc.acf_r1.min()), "{:.3f}")
        num.add("facts.tcvae.acf_r1.max", float(tc.acf_r1.max()), "{:.3f}")
        num.add("facts.tcvae.acf_abs.max", float(tc.acf_abs.max()), "{:.3f}")
        num.add("facts.oracle.acf_abs.min", float(f[(f.kind == "oracle") & (f.dgp != "gbm")].acf_abs.min()), "{:.3f}")
        num.add("facts.oracle.acf_abs.max", float(f[(f.kind == "oracle") & (f.dgp != "gbm")].acf_abs.max()), "{:.3f}")
        coll = f[(f.kind == "tcvae") & (f["std"] < 0.05)]
        num.add("facts.tcvae.collapsed", int(len(coll)))
        if len(coll):
            r = coll.iloc[0]
            num.add("facts.tcvae.collapsed.vol", float(r["std"]), "{:.3f}")
            hist = "a single history" if r.regime == "single" else "rich data"
            num.add("facts.tcvae.collapsed.which", f"trained on {hist} of process {DGP_LABEL[r.dgp].split()[0]} "
                    f"(run {int(r.seed)})")
        for g, key in (("vae_mlp_single", "vae_mlp"), ("vae_mlp_wide", "vae_mlp_wide")):
            x = f[f.g == g]
            if len(x):
                for st in ("std", "acf_r1", "vr"):
                    num.add(f"facts.{key}.{st}.min", float(x[st].min()))
                    num.add(f"facts.{key}.{st}.max", float(x[st].max()))
        vm = f[f.g == "vae_mlp_single"]["std"]
        num.add("facts.vae_mlp.std.min", float(vm.min()), "{:.3f}")
        num.add("facts.vae_mlp.std.max", float(vm.max()), "{:.3f}")
