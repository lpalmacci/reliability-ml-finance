"""Descriptive figures of the chapter: induced laws, autocorrelation and leverage curves, sample paths,
quantiles, and the PITs of level L4 on the market. Reads results/e4_curves/ (tavp.experiments.e4_curves)
and results/real/e11/. The figures of the pre-registered analysis show the generators of that analysis; the Quant
GAN, added by addendum, has its own figure of dependence curves (fig_qgan_curves)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tavp.report.case import DGP_LABEL, DGP_ORDER, FAMILY, GRID, INK, INK2
from tavp.report.numbers import Numbers

GEN10 = [("tcvae", "single", "G1 TC-VAE"), ("bootstrap", None, "B1 bootstrap"), ("garch", None, "B2 GJR-GARCH-$t$")]
GEN_QGAN = [("tcvae", "single", "G1 TC-VAE, 10 years"), ("qgan", "single", "G3 Quant GAN, 10 years"),
            ("tcvae", "rich", "G1 TC-VAE, rich data"), ("qgan", "rich", "G3 Quant GAN, long history")]
BAND = "#d9d8d3"
STRAT = ["BH", "TF", "VT", "TVF"]


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=6.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.5)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=160)


def _sel(df, kind, regime):
    x = df[df.kind == kind]
    return x[x.regime == regime] if regime else x


def _present(df, gens=GEN10) -> list:
    return [g for g in gens if not _sel(df, g[0], g[1]).empty]


def _dash(regime) -> str:
    return "--" if regime == "rich" else "-"


# ----------------------------------------------------------------------------- figure 1: induced laws
def induced_figure(ind: pd.DataFrame, cfg: dict, path: Path) -> None:
    from scipy.stats import gaussian_kde

    plt = _plt()
    gens = [("oracle", None, "oracle")] + [("tcvae", "rich", "G1 TC-VAE, rich data"),
                                            ("bootstrap", None, "B1 bootstrap, 10 years"),
                                            ("garch", None, "B2 GJR-GARCH-$t$, 10 years")]
    gens = [g for g in gens if not _sel(ind, g[0], g[1]).empty]
    rows = [("sharpe", "q50", "Sharpe ratio (annualised)"), ("maxdd", "mean", "maximum drawdown")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.3, 3.5))
    for i, (metric, func, lab) in enumerate(rows):
        o = _sel(ind, "oracle", None)
        for j, s in enumerate(STRAT):
            ax = axes[i, j]
            _style(ax)
            ov = o[(o.strategy == s) & (o.metric == metric)].value.to_numpy()
            lo, hi = np.quantile(ov, [0.01, 0.99])
            grid = np.linspace(lo, hi, 300)
            centre = np.median(ov) if func == "q50" else np.mean(ov)
            eta = cfg["use"]["tolerances"][metric][func]
            ax.axvspan(centre - eta, centre + eta, color=BAND, zorder=0, linewidth=0)
            for kind, regime, glab in gens:
                v = _sel(ind, kind, regime)
                v = v[(v.strategy == s) & (v.metric == metric)].value.to_numpy()
                if np.ptp(v) == 0:
                    continue
                dens = gaussian_kde(v)(grid)
                ax.plot(grid, dens, color=FAMILY[kind], linewidth=1.6 if kind == "oracle" else 1.1, label=glab,
                        zorder=3 if kind == "oracle" else 2)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(s, fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.5, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 2: dependence curves
def curves_figure(cur: pd.DataFrame, band: pd.DataFrame, path: Path, gens: list | None = None) -> None:
    plt = _plt()
    rows = [("acf_r", "ACF of returns"), ("acf_abs", "ACF of absolute returns"), ("leverage", "leverage")]
    gens = _present(cur, gens or GEN10)
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.3, 5.0), sharex=True)
    for i, (curve, lab) in enumerate(rows):
        for j, d in enumerate(DGP_ORDER):
            ax = axes[i, j]
            _style(ax)
            b = band[(band.dgp == d) & (band.curve == curve)].groupby("x").value
            q = b.quantile([0.05, 0.5, 0.95]).unstack()
            ax.fill_between(q.index, q[0.05], q[0.95], color=BAND, linewidth=0, zorder=1)
            ax.plot(q.index, q[0.5], color=INK, linewidth=1.4, label="oracle", zorder=3)
            for kind, regime, glab in gens:
                c = _sel(cur, kind, regime)
                c = c[(c.dgp == d) & (c.curve == curve)].groupby("x").value.median()
                ax.plot(c.index, c.values, color=FAMILY[kind], linewidth=1.1, linestyle=_dash(regime), label=glab,
                        zorder=2)
            ax.axhline(0, color=INK2, linewidth=0.5)
            if i == 0:
                ax.set_title(DGP_LABEL[d].replace("$", "").replace("-t", "-t"), fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.5, color=INK)
            if i == len(rows) - 1:
                ax.set_xticks([1, 5, 10, 15, 20])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l) if len(l) <= 5 else 3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("lag (days)", fontsize=7, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 3: sample paths
def paths_figure(paths: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    cols = [("oracle", None, "oracle")] + _present(paths)
    fig, axes = plt.subplots(4, len(cols), figsize=(6.3, 5.6), sharex=True, sharey="row")
    for i, d in enumerate(DGP_ORDER):
        for j, (kind, regime, lab) in enumerate(cols):
            ax = axes[i, j]
            _style(ax)
            p = _sel(paths, kind, regime)
            p = p[(p.dgp == d) & (p.seed.fillna(0) == 0)]
            for _, g in p.groupby("path"):
                ax.plot(g.t, np.exp(g.logprice), color=FAMILY[kind], linewidth=0.5, alpha=0.6)
            if i == 0:
                ax.set_title(lab, fontsize=6.5, color=INK)
            if j == 0:
                ax.set_ylabel(DGP_LABEL[d].replace("$", ""), fontsize=6.5, color=INK)
            ax.set_xticks([0, 30, 59])
    fig.supxlabel("trading day of the scenario", fontsize=7, color=INK)
    fig.supylabel("price (starting at 1)", fontsize=7, color=INK, x=0.0)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    _save(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 4: quantiles
def quantile_figure(cur: pd.DataFrame, band: pd.DataFrame, path: Path) -> None:
    from scipy.stats import norm

    plt = _plt()
    rows = [("q_daily", "daily return, standardised"), ("q_horizon", "59-day return")]
    gens = _present(cur)
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.6))
    for i, (curve, lab) in enumerate(rows):
        for j, d in enumerate(DGP_ORDER):
            ax = axes[i, j]
            _style(ax)
            b = band[(band.dgp == d) & (band.curve == curve)]
            med = b.groupby("x").value.median()
            dev = b.assign(dev=b.value - b.x.map(med)).groupby("x").dev.quantile([0.05, 0.95]).unstack()
            z = norm.ppf(med.index)
            ax.fill_between(z, dev[0.05], dev[0.95], color=BAND, linewidth=0, zorder=1)
            ax.axhline(0, color=INK, linewidth=1.2, zorder=2, label="oracle")
            for kind, regime, glab in gens:
                c = _sel(cur, kind, regime)
                c = c[(c.dgp == d) & (c.curve == curve)].groupby("x").value.median()
                ax.plot(norm.ppf(c.index), c.values - med.reindex(c.index).values, color=FAMILY[kind],
                        linewidth=1.1, label=glab, zorder=3)
            if i == 0:
                ax.set_title(DGP_LABEL[d].replace("$", ""), fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.5, color=INK)
            ax.set_xticks([-3, 0, 3])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("quantile level (standard normal scale)", fontsize=7, color=INK)
    fig.supylabel("quantile minus oracle quantile", fontsize=7, color=INK, x=0.0)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    _save(fig, path)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 5: PITs on the market
def pit_figure(pits: pd.DataFrame, path: Path, bins: int = 10) -> None:
    from scipy.stats import binom

    plt = _plt()
    gens = [(g, lab) for g, lab in (("bootstrap", "B1 bootstrap"), ("garch", "B2 GJR-GARCH-$t$"),
                                    ("tcvae", "G1 TC-VAE")) if g in set(pits.generator)]
    metrics = [("ret", "59-day return"), ("sharpe", "Sharpe ratio"), ("maxdd", "maximum drawdown")]
    fig, axes = plt.subplots(len(gens), 3, figsize=(6.3, 1.25 * len(gens) + 0.5), sharex=True, sharey=True)
    for i, (g, glab) in enumerate(gens):
        for j, (m, mlab) in enumerate(metrics):
            ax = axes[i, j]
            _style(ax)
            u = pits[(pits.generator == g) & (pits.metric == m) & (pits.seed == 0)].pit.to_numpy()
            dens, edges = np.histogram(u, bins=bins, range=(0, 1), density=True)
            n = len(u)
            lo, hi = binom.ppf([0.025, 0.975], n, 1 / bins) * bins / n
            ax.axhspan(lo, hi, color=BAND, linewidth=0, zorder=0)
            ax.bar(edges[:-1], dens, width=1 / bins, align="edge", color=FAMILY[g], edgecolor="white", linewidth=0.6,
                   zorder=2)
            ax.axhline(1, color=INK, linewidth=0.8, zorder=3)
            if i == 0:
                ax.set_title(mlab, fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(glab, fontsize=6.5, color=INK)
            ax.set_xticks([0, 0.5, 1])
    fig.supxlabel("PIT of the realised metric", fontsize=7, color=INK)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def build_curves(root: Path, out: Path, num: Numbers, cfg: dict) -> None:
    cur, band = _read(root, "e4_curves", "curves"), _read(root, "e4_curves", "band")
    if cur is None or band is None:
        return
    fig = out / "figures"
    curves_figure(cur, band, fig / "fig_curves.pdf")
    if len(_present(cur, GEN_QGAN)) == len(GEN_QGAN):
        curves_figure(cur, band, fig / "fig_qgan_curves.pdf", GEN_QGAN)
    quantile_figure(cur, band, fig / "fig_quantiles.pdf")
    paths = _read(root, "e4_curves", "paths")
    if paths is not None:
        paths_figure(paths, fig / "fig_paths.pdf")
    ind = _read(root, "e4_curves", "induced")
    if ind is not None:
        induced_figure(ind, cfg, fig / "fig_induced.pdf")
    pits = _read(root / "real", "e11", "pits")
    if pits is not None:
        pit_figure(pits, fig / "fig_pit_market.pdf")
    # numbers quoted with the figures
    from tavp.experiments.e4_curves import BAND_SAMPLES, N_PATHS

    num.add("curves.npaths", N_PATHS)
    num.add("curves.bandsamples", BAND_SAMPLES)
    for d in DGP_ORDER:
        b = band[(band.dgp == d) & (band.curve == "acf_r") & (band.x == 1)].value
        num.add(f"curves.band.acf1.{d}.lo", float(b.quantile(0.05)), "{:.3f}")
        num.add(f"curves.band.acf1.{d}.hi", float(b.quantile(0.95)), "{:.3f}")
