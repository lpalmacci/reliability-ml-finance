"""Mean reversion as a planted defect (E9, added after the case study, not pre-registered; deviation D6).

Detection of P7 by level and severity, and comparison with the TC-VAE at the severity matched to
its first-order autocorrelation: which of the attainable cells failed by the TC-VAE are also
failed by an oracle whose only defect is that autocorrelation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tavp.perturb.operators import ACF1_STEP
from tavp.protocol.card import FUNC_ORDER, STRATS
from tavp.report import summaries as S
from tavp.report.attain import _cards, _levels
from tavp.report.case import DGP_LABEL, DGP_ORDER
from tavp.report.numbers import Numbers

CELL = ["strategy", "metric", "functional"]


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _dkey(delta: float) -> str:
    return f"{delta:g}"


def build_mr(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    e9 = {n: _read(root, "e9", n) for n in ("l1", "l2", "l3", "matched")}
    if e9["l2"] is None or e9["matched"] is None:
        return
    matched = e9["matched"]
    grid = sorted(float(d) for d in cfg["meta_validation"]["severities"] if d > 0)
    r1 = S.l1_rates(e9["l1"]) if e9["l1"] is not None else None
    cells = S.l2_cells(e9["l2"], pr_min)
    rules = S.l3_strategies(e9["l3"], pr_min) if e9["l3"] is not None else None
    num.add("e9.acf1.step", -ACF1_STEP, "{:.2f}")
    num.add("e9.matched.delta.min", float(matched.delta.min()), "{:.2f}")
    num.add("e9.matched.delta.max", float(matched.delta.max()), "{:.2f}")
    num.add("e9.matched.acf.min", float(matched.acf_r1.min()), "{:.3f}")
    num.add("e9.matched.acf.max", float(matched.acf_r1.max()), "{:.3f}")
    # detection by process and severity
    lines, rows = [], []
    for d in DGP_ORDER:
        m = matched[matched.dgp == d]
        deltas = sorted(set(grid) | set(m.delta.astype(float)))
        for i, delta in enumerate(deltas):
            c = cells[(cells.dgp == d) & np.isclose(cells.delta, delta)]
            if c.empty:
                continue
            l1any = l1acf = np.nan
            if r1 is not None:
                x = r1[(r1.dgp == d) & np.isclose(r1.delta, delta)]
                l1any = float(x[x.statistic == "any"].flag_holm.iloc[0])
                l1acf = float(x[x.statistic == "acf_r"].flag_holm.iloc[0])
            failed = c[~c["pass"]]
            r3 = rules[(rules.dgp == d) & np.isclose(rules.delta, delta)] if rules is not None else None
            l3f = sorted(r3[~r3["pass"]].strategy) if r3 is not None else []
            tag = ", ".join(sorted(m[np.isclose(m.delta, delta)].regime))
            key = f"e9.{d}.{_dkey(delta)}"
            num.add(key + ".l1", l1any)
            num.add(key + ".acf", l1acf)
            num.add(key + ".l2failed", int(len(failed)))
            num.add(key + ".l3failed", ", ".join(l3f) or "none")
            rows.append({"dgp": d, "delta": delta, "acf1": ACF1_STEP * delta, "matched": tag, "L1": l1any,
                         "L1_acf": l1acf, "l2_failed": len(failed), "l3_failed": ", ".join(l3f)})
            fs = ", ".join(s for s in STRATS if s in set(failed.strategy))
            lab = DGP_LABEL[d] if i == 0 else ""
            delta_txt = f"{delta:g}" + (r"$^{\ast}$" if tag else "")
            lines.append(f"{lab} & {delta_txt} & ${ACF1_STEP * delta:.3f}$ & {l1acf:.2f} & {l1any:.2f} & "
                         f"{len(failed)} & {fs or '--'} & {', '.join(l3f) or '--'}" + r" \\")
        lines.append(r"\addlinespace")
    pd.DataFrame(rows).to_csv(out / "tables" / "e9_detection.csv", index=False)
    (out / "tables" / "tab_e9_detection.tex").write_text(r"""\begin{tabular}{lrrrrrll}
\toprule
Process & $\delta$ & ACF(1) & L1, ACF($r$) & L1, any & L2 failed & L2 rules & L3 failed \\
\midrule
""" + "\n".join(lines[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
""")
    g = pd.DataFrame(rows)
    gg = g[g.delta.isin(grid)]
    num.add("e9.l2failed.min", int(gg.l2_failed.min()))
    num.add("e9.l2failed.max", int(gg.l2_failed.max()))
    g25 = g[np.isclose(g.delta, grid[0])]
    num.add("e9.l2failed.025.min", int(g25.l2_failed.min()))
    num.add("e9.l2failed.025.max", int(g25.l2_failed.max()))
    num.add("e9.l1acf.025.min", float(g[np.isclose(g.delta, grid[0])].L1_acf.min()))
    num.add("e9.l1acf.025.max", float(g[np.isclose(g.delta, grid[0])].L1_acf.max()))
    num.add("e9.l3.grid.failed", int((gg.l3_failed != "").sum()))
    num.add("e9.l3.grid.total", int(len(gg)))
    # comparison with the TC-VAE at the matched severity, on the attainable cells of each regime
    lv = _levels(root)
    cards = _cards(lv, pr_min)
    keys = [(s, m_, f) for s in STRATS for m_, f in FUNC_ORDER]
    l2tc = lv["l2"][lv["l2"].kind == "tcvae"]
    mlines, mrows, tally = [], [], {}
    for r in ("single", "rich"):
        for d in DGP_ORDER:
            o, t = cards.get((d, r, "oracle_drift")), cards.get((d, r, "tcvae"))
            mm = matched[(matched.dgp == d) & (matched.regime == r)]
            if o is None or t is None or mm.empty:
                continue
            delta = float(mm.delta.iloc[0])
            A = [k for k in keys if o["L2"][k] >= pr_min]
            c = cells[(cells.dgp == d) & np.isclose(cells.delta, delta)].set_index(CELL)
            f_tc = {k for k in A if t["L2"][k] < pr_min}
            f_p7 = {k for k in A if not c.loc[k, "pass"]}
            both = f_tc & f_p7
            # fingerprint: mean standardised discrepancy per attainable cell, TC-VAE against P7
            x = l2tc[(l2tc.dgp == d) & (l2tc.regime == r)].groupby(CELL).apply(
                lambda z: (z["diff"] / z.eta).mean(), include_groups=False)
            y = (c["diff"] / c["eta"])
            idx = [k for k in A]
            corr = float(np.corrcoef(x.loc[idx].to_numpy(), y.loc[idx].to_numpy())[0, 1]) if len(idx) > 2 else np.nan
            l3tc = sorted(t["L3_failed"])
            r3 = rules[(rules.dgp == d) & np.isclose(rules.delta, delta)] if rules is not None else None
            l3p7 = sorted(r3[~r3["pass"]].strategy) if r3 is not None else []
            share = len(both) / len(f_tc) if f_tc else np.nan
            for k in f_tc:
                tally.setdefault((r, "both" if k in both else "only", k[1], k[2]), 0)
                tally[(r, "both" if k in both else "only", k[1], k[2])] += 1
            key = f"e9.match.{r}.{d}"
            num.add(key + ".delta", delta, "{:.2f}")
            num.add(key + ".acf", float(mm.acf_r1.iloc[0]), "{:.3f}")
            num.add(key + ".A", len(A))
            num.add(key + ".ftc", len(f_tc))
            num.add(key + ".fp7", len(f_p7))
            num.add(key + ".both", len(both))
            num.pct(key + ".share", share if not np.isnan(share) else 0.0)
            num.add(key + ".corr", corr)
            num.add(key + ".l3tc", ", ".join(l3tc) or "none")
            num.add(key + ".l3p7", ", ".join(l3p7) or "none")
            mrows.append({"regime": r, "dgp": d, "delta": delta, "acf_tcvae": float(mm.acf_r1.iloc[0]), "A": len(A),
                          "f_tcvae": len(f_tc), "f_p7": len(f_p7), "both": len(both), "share": share, "corr": corr,
                          "l3_tcvae": ", ".join(l3tc), "l3_p7": ", ".join(l3p7)})
            reg = {"single": f"{cfg['data']['train_returns'] // 252} years", "rich": "rich data"}[r]
            mlines.append(f"{reg if d == DGP_ORDER[0] else ''} & {DGP_LABEL[d]} & ${mm.acf_r1.iloc[0]:.3f}$ & "
                          f"{delta:.2f} & {len(A)} & {len(f_tc)} & {len(f_p7)} & {len(both)} & {corr:.2f} & "
                          f"{', '.join(l3tc) or '--'} & {', '.join(l3p7) or '--'}" + r" \\")
        mlines.append(r"\addlinespace")
    while mlines and mlines[-1] == r"\addlinespace":
        mlines.pop()
    mlines.append(r"\addlinespace")
    mt = pd.DataFrame(mrows)
    mt.to_csv(out / "tables" / "e9_match.csv", index=False)
    for r in ("single", "rich"):
        x = mt[mt.regime == r]
        if len(x):
            num.pct(f"e9.match.{r}.share.pooled", float(x.both.sum() / max(x.f_tcvae.sum(), 1)))
            num.add(f"e9.match.{r}.corr.min", float(x["corr"].min()))
            num.add(f"e9.match.{r}.corr.max", float(x["corr"].max()))
            num.add(f"e9.match.{r}.ftc.total", int(x.f_tcvae.sum()))
            num.add(f"e9.match.{r}.both.total", int(x.both.sum()))
            num.add(f"e9.match.{r}.p7only.total", int((x.f_p7 - x.both).sum()))
            for kind in ("both", "only"):
                for m_, f in FUNC_ORDER:
                    num.add(f"e9.match.{r}.{kind}.{m_}.{f}", int(tally.get((r, kind, m_, f), 0)))
    (out / "tables" / "tab_e9_match.tex").write_text(r"""\begin{tabular}{llrrrrrrrll}
\toprule
& & & & & \multicolumn{3}{c}{Failed in $\mathcal{A}$} & & \multicolumn{2}{c}{L3 failed} \\
\cmidrule(lr){6-8}\cmidrule(lr){10-11}
Training data & Process & ACF(1) & $\delta$ & $|\mathcal{A}|$ & G1 & P7 & both & $r$ & G1 & P7 \\
\midrule
""" + "\n".join(mlines[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
""")
