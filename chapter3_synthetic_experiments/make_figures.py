"""The seven figures of Chapter 3, from the saved results.

    python make_figures.py    ->  figures/confidex_*.pdf and .png

The figures of the thesis are typeset with LaTeX (Computer Modern); if LaTeX is not installed the
script falls back to Matplotlib's mathtext, with the same layout and data (see style.py). The
posterior of the five-member example is recomputed on a finer grid for the picture; every other
quantity is read from results/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.stats import beta as B

import confidex as cx
from style import OI, TEXTWIDTH, save, setup
from synthetic import DESIGNS

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
KS = (3, 6, 12, 24, 36)


def main() -> None:
    usetex = setup()
    pct = r"\%" if usetex else "%"
    OUT.mkdir(exist_ok=True)
    R = json.loads((HERE / "results" / "chapter3_results.json").read_text())
    G = json.loads((HERE / "results" / "auroc_gap_bootstrap.json").read_text())
    x = np.linspace(1e-4, 1 - 1e-4, 4000)

    # 1. coupled model: c = 0.9 and three values of the shared precision
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.85, 2.3))
    for lam, col, ls in ((1, OI["sky"], "-"), (10, OI["blue"], "--"), (50, OI["vermillion"], "-")):
        a, b = 0.9 * lam, 0.1 * lam
        ax.plot(x, B.pdf(x, a, b), color=col, ls=ls, label=rf"$\lambda={lam}$ $(\alpha={a:g}$, $\beta={b:g})$")
    ax.axvline(0.9, color=OI["grey"], lw=0.7, ls=":")
    ax.text(0.885, 10.9, r"$\mathbb{E}[p]=c=0.9$", ha="right", va="top", fontsize=8, color=OI["grey"])
    ax.set(xlim=(0, 1), ylim=(0, 11.5), xlabel=r"$p$", ylabel="density")
    ax.legend(loc="upper left")
    save(fig, OUT / "confidex_coupled_density")

    # 2. decoupled model: the same precision (lambda = 9) at three locations
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.85, 2.3))
    for mu, col, ls in ((0.1, OI["blue"], "-"), (0.5, OI["green"], "--"), (0.9, OI["vermillion"], "-")):
        a, b = 9 * mu, 9 * (1 - mu); sd = np.sqrt(mu * (1 - mu) / 10)
        ax.plot(x, B.pdf(x, a, b), color=col, ls=ls, label=rf"$\mu={mu}$ (sd $={sd:.3f}$)")
    ax.set(xlim=(0, 1), ylim=(0, 8), xlabel=r"$p$", ylabel="density")
    ax.legend(loc="upper center")
    save(fig, OUT / "confidex_decoupled_density")

    # 3. joint posterior of location and concentration for five members (independent priors)
    P5 = np.array([[0.65, 0.77, 0.40, 0.83, 0.55]])
    full = cx.fit_posterior_full(P5, points_mu=500, points_c=500, keep_joint=True)
    gm, gc = full.grid_mu.g, full.grid_c.g
    D = full.joint[0] / (full.grid_mu.w[:, None] * full.grid_c.w[None, :])
    mm = full.joint[0].sum(axis=1) / full.grid_mu.w
    fig = plt.figure(figsize=(TEXTWIDTH * 0.72, 3.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=(1, 3.2), width_ratios=(30, 1), hspace=0.06, wspace=0.05)
    axm = fig.add_subplot(gs[0, 0]); axj = fig.add_subplot(gs[1, 0], sharex=axm); cax = fig.add_subplot(gs[1, 1])
    lo, hi = R["five"]["ci90"]; pbar = P5.mean()
    axm.plot(gm, mm, color=OI["blue"]); sel = (gm >= lo) & (gm <= hi)
    axm.fill_between(gm[sel], 0, mm[sel], color=OI["sky"], alpha=0.5, lw=0)
    axm.text(lo - 0.01, 0.55 * mm.max(), f"90{pct}", ha="right", fontsize=7.5, color=OI["blue"])
    axm.axvline(pbar, color=OI["vermillion"], ls="--", lw=1.0)
    axm.set(ylabel=r"$p(\mu\mid P)$", ylim=(0, None)); axm.tick_params(labelbottom=False); axm.grid(False)
    pc = axj.pcolormesh(gm, gc, D.T, shading="auto", cmap="cividis", rasterized=True)
    M = full.joint[0]; order = np.argsort(M.ravel())[::-1]; cum = np.cumsum(M.ravel()[order])
    lev = [D.ravel()[order][np.searchsorted(cum, q)] for q in (0.9, 0.5)]      # highest-posterior regions
    cs = axj.contour(gm, gc, D.T, levels=lev, colors="white", linewidths=0.7)
    axj.clabel(cs, fmt={lev[0]: f"90{pct}", lev[1]: f"50{pct}"}, fontsize=7, inline=True)
    axj.axvline(pbar, color=OI["vermillion"], ls="--", lw=1.0)
    axj.text(pbar + 0.012, 0.95, r"$\mu=\bar p$", color=OI["vermillion"], fontsize=8, va="top")
    axj.set(xlim=(0.25, 1.0), ylim=(0, 1), xlabel=r"location $\mu$", ylabel=r"concentration $c$"); axj.grid(False)
    cb = fig.colorbar(pc, cax=cax); cb.set_label("posterior density", fontsize=8); cax.grid(False); cb.outline.set_linewidth(0.5)
    save(fig, OUT / "confidex_joint_posterior")

    # 4. the ten ensembles: posterior mean of c, and F_pred(tau) at two thresholds
    I = R["illustration"]; idx = np.arange(1, 11)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.35), gridspec_kw={"wspace": 0.32})
    cols = [OI["blue"]] * 5 + [OI["orange"]] * 5
    a1.bar(idx, I["cbar"], color=cols, edgecolor="white", lw=0.5, width=0.72)
    a1.set(xticks=idx, ylim=(0, 1.13), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xlabel="ensemble", ylabel=r"posterior mean $\bar c$")
    a1.set_title(r"(a) concentration", loc="left"); a1.grid(axis="x", visible=False)
    a1.text(3, 1.005, "concentrated", ha="center", va="bottom", fontsize=7.5, color=OI["blue"])
    a1.text(8, 0.36, "dispersed", ha="center", va="bottom", fontsize=7.5, color=OI["orange"])
    a2.axhspan(0.4, 0.6, color=OI["yellow"], alpha=0.35, lw=0, label="central band")
    for tau, mk, col, off in ((0.5, "o", OI["vermillion"], -0.16), (0.1, "s", OI["blue"], 0.16)):
        F = np.array(I[f"F_{tau}"]); conf = np.array(I[f"conf_{tau}"])
        a2.scatter(idx[conf] + off, F[conf], marker=mk, s=22, facecolor="white", edgecolor=col, lw=1.1, zorder=3)
        a2.scatter(idx[~conf] + off, F[~conf], marker=mk, s=22, color=col, zorder=3, label=rf"$\tau={tau}$")
    a2.set(xticks=idx, ylim=(-0.04, 1.04), xlabel="ensemble", ylabel=r"$F_{\mathrm{pred}}(\tau)$")
    a2.set_title(r"(b) position of the threshold", loc="left"); a2.grid(axis="x", visible=False)
    h, l = a2.get_legend_handles_labels()
    a2.legend(h, l, loc="upper center", bbox_to_anchor=(0.6, 1.0), ncol=1, fontsize=7.5)
    save(fig, OUT / "confidex_illustration")

    # 5. known truth: AUROC of every reading by number of members
    V = R["validation"]
    readings = [("confidex", "Confidex", OI["vermillion"], "-", "o", 2.0),
                ("distance", r"$|\bar p-\tau|$", OI["sky"], "--", "s", 1.1),
                ("studentised", r"$|\bar p-\tau|/s$", OI["blue"], "--", "^", 1.1),
                ("logit distance", "logit distance", OI["green"], ":", "s", 1.2),
                ("logit studentised", "logit studentised margin", OI["purple"], ":", "^", 1.2),
                ("concentration", r"concentration $\bar c$", OI["grey"], "-.", "x", 1.1)]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.9), sharex=True, sharey=True, gridspec_kw={"hspace": 0.28, "wspace": 0.1})
    for j, d in enumerate(DESIGNS):
        for i, tau in enumerate((0.5, 0.1)):
            ax = axes[i, j]
            for key, lab, col, ls, mk, lw in readings:
                ax.plot(KS, [V[f"{d}|{tau}|{K}"][key] for K in KS], color=col, ls=ls, marker=mk, lw=lw,
                        ms=3.5 if key != "confidex" else 4, label=lab, zorder=3 if key == "confidex" else 2)
            ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
            ax.minorticks_off(); ax.set_ylim(0.53, 1.0)
            ax.set_title(rf"{d}, $\tau={tau}$")
            if i == 1:
                ax.set_xlabel(r"members $K$")
            if j == 0:
                ax.set_ylabel("AUROC")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, OUT / "confidex_validation")

    # 6. amplification: AUROC difference from the best simpler reading, with bootstrap intervals
    style_of = {"informative": (OI["vermillion"], "o", "-"), "non-informative": (OI["blue"], "s", "--"),
                "mixed": (OI["green"], "^", "-.")}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, tau in zip(axes, (0.5, 0.1)):
        ax.axhline(0, color="black", lw=0.6)
        for d in DESIGNS:
            col, mk, ls = style_of[d]
            g = [G[f"{d}|{tau}|{K}"] for K in KS]
            ax.fill_between(KS, [x_["lo"] for x_ in g], [x_["hi"] for x_ in g], color=col, alpha=0.18, lw=0)
            ax.plot(KS, [x_["gap"] for x_ in g], color=col, marker=mk, ls=ls, label=d)
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
        ax.set_title(rf"$\tau={tau}$"); ax.set_xlabel(r"members $K$")
    axes[0].set_ylabel("AUROC difference")
    axes[1].legend(loc="lower left")
    save(fig, OUT / "confidex_amplification")

    # 7. fixed review budget: errors captured when each reading refers 10% of the cases
    Bd = R["budget"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, tau in zip(axes, (0.5, 0.1)):
        ax.axhline(0, color="black", lw=0.6)
        for d in DESIGNS:
            col, mk, ls = style_of[d]
            g = [Bd[f"{d}|{tau}|{K}"] for K in KS]
            ax.fill_between(KS, [100 * x_["lo@0.1"] for x_ in g], [100 * x_["hi@0.1"] for x_ in g], color=col, alpha=0.18, lw=0)
            ax.plot(KS, [100 * x_["gap@0.1"] for x_ in g], color=col, marker=mk, ls=ls, label=d)
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
        ax.set_title(rf"$\tau={tau}$"); ax.set_xlabel(r"members $K$")
    axes[0].set_ylabel("difference in errors captured (points)")
    axes[1].legend(loc="upper left")
    save(fig, OUT / "confidex_budget")
    print("seven figures written to", OUT, "(LaTeX)" if usetex else "(mathtext: LaTeX not available)")


if __name__ == "__main__":
    main()
