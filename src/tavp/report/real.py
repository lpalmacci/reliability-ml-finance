"""Market-data analysis (deviation D8, not pre-registered): pseudo-real oracles (E10) and level L4 on the
market (E11). Reads results/real/ and writes tables and numbers with the prefix `real.`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tavp.protocol.card import FUNC_ORDER, STRATS, build_card
from tavp.report.numbers import Numbers
from tavp.report.tolcrit import critical_cells

DGPS = [("gjr_real", "GJR-GARCH-$t$"), ("regime_real", "Regimes")]
ROWS = [("oracle", "none", "$O$ oracle (negative control)"),
        ("oracle_drift", "single", "$\\hat{O}$ oracle, estimated drift"),
        ("tcvae", "single", "G1 TC-VAE"), ("bootstrap", "single", "B1 stationary bootstrap"),
        ("garch", "single", "B2 GJR-GARCH-$t$"),
        ("oracle_drift", "rich", "$\\hat{O}$ oracle, estimated drift"), ("tcvae", "rich", "G1 TC-VAE"),
        ("bootstrap", "rich", "B1 stationary bootstrap"), ("garch", "rich", "B2 GJR-GARCH-$t$")]
KEYS = [(s, m, f) for s in STRATS for m, f in FUNC_ORDER]
GEN_LABEL = {"tcvae": "G1 TC-VAE", "bootstrap": "B1 stationary bootstrap", "garch": "B2 GJR-GARCH-$t$"}


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _levels(root: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for level in ("l0", "l1", "l2", "l3"):
        parts = []
        e1 = _read(root, "e1", level)
        if e1 is not None:
            parts.append(e1[e1.kind == "oracle"].assign(regime="none"))
        for exp in ("e4", "e5", "e8"):
            t = _read(root, exp, level)
            if t is not None:
                parts.append(t[t.kind.isin(["oracle_drift", "tcvae", "bootstrap", "garch"])])
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df["regime"] = df.regime.fillna("single")
            out[level] = df
    return out


def _date(x: int) -> str:
    s = str(int(x))
    return f"{s[:4]}-{s[4:6]}"


def build_real(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    rroot = root / "real"
    cal = _read(rroot, "e10", "calibration")
    if cal is None:
        return
    sample = json.loads((rroot / "e10" / "sample.json").read_text())
    num.add("real.days", int(sample["days"]))
    num.add("real.first", _date(sample["first"]))
    num.add("real.last", _date(sample["last"]))
    num.add("real.mean", sample["mean_ann"], "{:.3f}")
    num.add("real.vol", sample["vol_ann"], "{:.3f}")
    g = cal[cal.model == "gjr_garch_t"].iloc[0]
    r = cal[cal.model == "regime"].iloc[0]
    for k in ("mu_ann", "_mu_ann_ml", "vol_ann", "alpha", "gamma", "beta", "nu", "_persistence"):
        num.add(f"real.gjr.{k.strip('_')}", float(g[k]), "{:.1f}" if k == "nu" else "{:.3f}")
    pi = (1 - r.p22) / (2 - r.p11 - r.p22)
    for k in ("p11", "p22", "mu1", "sigma1", "mu2", "sigma2"):
        num.add(f"real.regime.{k}", float(r[k]), "{:.4f}" if k.startswith("p") else "{:.3f}")
    num.add("real.regime.pi", float(pi), "{:.2f}")
    num.add("real.regime.dur1", float(1 / (1 - r.p11)), "{:.0f}")
    num.add("real.regime.dur2", float(1 / (1 - r.p22)), "{:.0f}")
    # parameter table: pseudo-real oracles against the processes D3 and D4 of the main study
    d3, d4 = cfg["dgps"]["gjr_garch_t"], cfg["dgps"]["regime"]
    prow = [("Drift $\\mu$ (annual)", d3["mu_ann"], g.mu_ann, "{:.3f}"),
            ("Volatility (annual, unconditional)", d3["vol_ann"], g.vol_ann, "{:.3f}"),
            ("$\\alpha$", d3["alpha"], g.alpha, "{:.3f}"), ("$\\gamma$", d3["gamma"], g.gamma, "{:.3f}"),
            ("$\\beta$", d3["beta"], g.beta, "{:.3f}"),
            ("Persistence $\\alpha+\\gamma/2+\\beta$", d3["alpha"] + d3["gamma"] / 2 + d3["beta"], g._persistence, "{:.3f}"),
            ("Degrees of freedom $\\nu$", d3["nu"], g.nu, "{:.1f}")]
    rrow = [("$p_{11}$ (stay in the calm regime)", d4["p11"], r.p11, "{:.4f}"),
            ("$p_{22}$ (stay in the turbulent regime)", d4["p22"], r.p22, "{:.4f}"),
            ("$\\mu_1$, $\\sigma_1$ (calm)", f"${d4['mu1']:.2f}$, ${d4['sigma1']:.2f}$", f"${r.mu1:.3f}$, ${r.sigma1:.3f}$", None),
            ("$\\mu_2$, $\\sigma_2$ (turbulent)", f"${d4['mu2']:.2f}$, ${d4['sigma2']:.2f}$", f"${r.mu2:.3f}$, ${r.sigma2:.3f}$", None),
            ("Stationary probability of the calm regime", (1 - d4["p22"]) / (2 - d4["p11"] - d4["p22"]), pi, "{:.2f}")]
    lines = [r"\multicolumn{3}{l}{\emph{GJR-GARCH(1,1)-$t$}} \\"]
    lines += [f"{a} & " + (f"{fmt.format(b)} & {fmt.format(c)}" if fmt else f"{b} & {c}") + r" \\" for a, b, c, fmt in prow]
    lines += [r"\addlinespace", r"\multicolumn{3}{l}{\emph{Two-state regime switching}} \\"]
    lines += [f"{a} & " + (f"{fmt.format(b)} & {fmt.format(c)}" if fmt else f"{b} & {c}") + r" \\" for a, b, c, fmt in rrow]
    (out / "tables" / "tab_real_params.tex").write_text(r"""\begin{tabular}{lrr}
\toprule
Parameter & Main study (D3, D4) & Market, 1926--2026 \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    build_real_protocol(rroot, out, num, cfg, pr_min)
    build_real_l4(rroot, out, num, cfg)


def build_real_protocol(rroot: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    lv = _levels(rroot)
    if "l2" not in lv:
        return
    sel = lambda df, d, k, r: None if df is None else df[(df.dgp == d) & (df.kind == k) & (df.regime == r)]
    cells = critical_cells(lv["l2"], pr_min)
    cards = {}
    for d, _ in DGPS:
        for kind, regime, _ in ROWS:
            x2 = sel(lv["l2"], d, kind, regime)
            if x2 is None or x2.empty:
                continue
            cards[(d, kind, regime)] = build_card(sel(lv.get("l0"), d, kind, regime), sel(lv.get("l1"), d, kind, regime),
                                                  x2, sel(lv.get("l3"), d, kind, regime), None, pr_min)
    lines, prev = [], None
    for kind, regime, lab in ROWS:
        vals = []
        for d, _ in DGPS:
            c = cards.get((d, kind, regime))
            if c is None:
                vals += ["", "", "", ""]
                continue
            o = cards.get((d, "oracle_drift", regime if regime != "none" else "single"))
            A = [k for k in KEYS if o is not None and o["L2"][k] >= pr_min] if regime != "none" else KEYS
            rules_A = {s for s, v in o["L3"].items() if v["pr"] >= pr_min} if regime != "none" and o else set(c["L3"])
            passed = sum(c["L2"][k] >= pr_min for k in KEYS)
            passA = sum(c["L2"][k] >= pr_min for k in A)
            l3A = [s for s in c["L3_failed"] if s in rules_A]
            fitA = c["L0"]["hash_ok"] and c["L0"]["valid"] and passA == len(A) and not l3A
            x = cells[(cells.dgp == d) & (cells.kind == kind) & (cells.regime == regime)]
            kA = float(x[[tuple(k) in set(A) for k in x[["strategy", "metric", "functional"]].itertuples(index=False)]]
                       .ratio.max())
            key = f"real.{kind}.{regime}.{d}"
            num.add(key + ".pass", int(passed))
            num.add(key + ".A", len(A))
            num.add(key + ".passA", int(passA))
            num.add(key + ".kappaA", kA, "{:.1f}" if kA >= 10 else "{:.2f}")
            num.add(key + ".l3failA", ", ".join(l3A) or "none")
            num.add(key + ".l1flag", len(c["L1_flagged"]))
            num.add(key + ".fitA", "fit" if fitA else "not fit")
            num.add(key + ".resolvable", float(c["L0"]["resolvable"]))
            pa = f"{passA}/{len(A)}"
            vals += [str(passed), r"\textbf{" + pa + "}" if fitA else pa, f"{kA:.1f}" if kA >= 10 else f"{kA:.2f}",
                     ", ".join(l3A) or "--"]
        group = {"none": "Known law", "single": f"{cfg['data']['train_returns'] // 252} years",
                 "rich": f"{cfg['data']['rich_paths'] * (cfg['data']['scenario_prices'] - 1):,} returns"
                 .replace(",", "{,}")}[regime]
        if prev is not None and regime != prev:
            lines.append(r"\addlinespace")
        lines.append(f"{group if regime != prev else ''} & {lab} & " + " & ".join(vals) + r" \\")
        prev = regime
    head = " & ".join(r"\multicolumn{4}{c}{" + lab + "}" for _, lab in DGPS)
    (out / "tables" / "tab_real_protocol.tex").write_text(r"""\begin{tabular}{llrrrlrrrl}
\toprule
Training data & Generator & """ + head + r""" \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
& & all & attain. & $\kappa^\ast_{\mathcal{A}}$ & L3 failed & all & attain. & $\kappa^\ast_{\mathcal{A}}$ & L3 failed \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    for d, _ in DGPS:
        o = cards.get((d, "oracle_drift", "single"))
        if o is not None:
            num.add(f"real.A.single.{d}", int(sum(o["L2"][k] >= pr_min for k in KEYS)))
        o = cards.get((d, "oracle_drift", "rich"))
        if o is not None:
            num.add(f"real.A.rich.{d}", int(sum(o["L2"][k] >= pr_min for k in KEYS)))


def build_real_l4(rroot: Path, out: Path, num: Numbers, cfg: dict) -> None:
    tests, wins = _read(rroot, "e11", "tests"), _read(rroot, "e11", "windows")
    if tests is None or wins is None:
        return
    wins = wins.sort_values("start")
    pits = _read(rroot, "e11", "pits")
    num.add("real.l4.windows", int(len(wins)))
    num.add("real.l4.test.first", _date(wins.test_first.min()))
    num.add("real.l4.test.last", _date(wins.test_last.max()))
    num.add("real.l4.train.first", _date(wins.train_first.min()))
    num.add("real.l4.hash_ok", "yes" if bool(pits.hash_ok.all()) else "no")
    lines = []
    for gname in ("tcvae", "bootstrap", "garch"):
        t = tests[tests.generator == gname]
        if t.empty:
            continue
        pooled = t[t.window == -1]
        n = int(pooled.n.iloc[0])
        rej = int(pooled.reject_any.sum())
        cov = float(pooled.coverage.mean())
        per = [int(t[t.window == w].reject_any.sum()) for w in wins.window]
        num.add(f"real.l4.{gname}.rejected", rej)
        num.add(f"real.l4.{gname}.cells", int(len(pooled)))
        num.add(f"real.l4.{gname}.n", n)
        num.pct(f"real.l4.{gname}.coverage", cov)
        num.pct(f"real.l4.{gname}.coverage.min", float(pooled.coverage.min()))
        num.pct(f"real.l4.{gname}.coverage.max", float(pooled.coverage.max()))
        num.add(f"real.l4.{gname}.decades.rejected", int(sum(p > 0 for p in per)))
        num.add(f"real.l4.{gname}.ks", int(pooled.reject_ks.sum()))
        num.add(f"real.l4.{gname}.kupiec", int(pooled.reject_kupiec.sum()))
        num.add(f"real.l4.{gname}.density", int(pooled.reject_density.sum()))
        rejected_cells = ", ".join(f"{a}/{b}" for a, b in pooled[pooled.reject_any][["strategy", "metric"]].itertuples(index=False))
        num.add(f"real.l4.{gname}.which", rejected_cells.replace("_", r"\_") or "none")
        lines.append(f"{GEN_LABEL[gname]} & {rej}/{len(pooled)} & {100 * cov:.0f}\\% & " + " & ".join(map(str, per)) + r" \\")
    head = " & ".join(f"{str(a)[:4]}--{str(b)[2:4]}" for a, b in zip(wins.test_first, wins.test_last))
    (out / "tables" / "tab_real_l4.tex").write_text(r"""\begin{tabular}{lrr""" + "r" * len(wins) + r"""}
\toprule
& \multicolumn{2}{c}{Pooled over decades} & \multicolumn{""" + str(len(wins)) + r"""}{c}{Cells rejected in each held-out period} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-""" + str(3 + len(wins)) + r"""}
Generator & rejected & coverage & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
