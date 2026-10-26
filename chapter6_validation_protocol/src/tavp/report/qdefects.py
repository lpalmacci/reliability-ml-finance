"""The properties of the Quant GAN as planted defects (E14, added after E13, not pre-registered; deviation D11).

Detection of P8 (extreme returns) by level and severity, and comparison with the Quant GAN at the severities matched to
it: which of the attainable cells failed by G3 are also failed by an oracle whose only defect is the frequency of
extreme returns of G3 (P8) or the variance ratio of G3 (P7), the counterpart of E9 for the TC-VAE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tavp.perturb.operators import EXTREME_FACTOR, EXTREME_PROB_STEP
from tavp.protocol.card import FUNC_ORDER, STRATS
from tavp.report import summaries as S
from tavp.report.attain import _cards, _levels
from tavp.report.case import DGP_LABEL, DGP_ORDER
from tavp.report.numbers import Numbers

CELL = ["strategy", "metric", "functional"]


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _fingerprint(l2: pd.DataFrame) -> pd.Series:
    """Mean standardised discrepancy per cell (undefined estimates are left out)."""
    return l2.groupby(CELL).apply(lambda z: (z["diff"] / z.eta).mean(), include_groups=False)


def build_qdefects(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    e14 = {n: _read(root, "e14", n) for n in ("l1", "l2", "l3", "matched", "qgan_facts")}
    if e14["l2"] is None or e14["matched"] is None:
        return
    matched = e14["matched"]
    grid = sorted(float(d) for d in cfg["meta_validation"]["severities"] if d > 0)
    r1 = S.l1_rates(e14["l1"]) if e14["l1"] is not None else None
    cells = S.l2_cells(e14["l2"], pr_min)
    rules = S.l3_strategies(e14["l3"], pr_min) if e14["l3"] is not None else None
    num.add("e14.p8.step", EXTREME_PROB_STEP, "{:.3f}")
    num.pct("e14.p8.step.pct", EXTREME_PROB_STEP, 1)
    num.add("e14.p8.factor", EXTREME_FACTOR, "{:g}")
    num.add("e14.quantile", "99.99\\%")
    f = e14["qgan_facts"]
    if f is not None:
        num.add("e14.qgan.exceed.min", float(f.exceed_ratio.min()), "{:.2f}")
        num.add("e14.qgan.exceed.max", float(f.exceed_ratio.max()), "{:.1f}" if f.exceed_ratio.max() >= 10 else "{:.2f}")
    for r in matched.itertuples():
        key = f"e14.match.{r.regime}.{r.dgp}"
        num.add(key + ".exceed", float(r.exceed_ratio), "{:.1f}" if r.exceed_ratio >= 10 else "{:.2f}")
        num.add(key + ".vr", float(r.vr))
        num.add(key + ".delta8", float(r.delta_p8), "{:.2f}")
        num.add(key + ".delta7", float(r.delta_p7), "{:.2f}")
    p8m = matched[matched.delta_p8 > 0]
    p7m = matched[matched.delta_p7 > 0]
    num.add("e14.p8.matched.n", int(len(p8m)))
    num.add("e14.p7.matched.n", int(len(p7m)))
    num.add("e14.settings", int(len(matched)))
    if len(p8m):
        num.add("e14.p8.matched.min", float(p8m.delta_p8.min()), "{:.2f}")
        num.add("e14.p8.matched.max", float(p8m.delta_p8.max()), "{:.2f}")
    if len(p7m):
        num.add("e14.p7.matched.min", float(p7m.delta_p7.min()), "{:.2f}")
        num.add("e14.p7.matched.max", float(p7m.delta_p7.max()), "{:.2f}")
    num.add("e14.p8.none", ", ".join(f"{DGP_LABEL[r.dgp].split(' ')[0]} ({'10 years' if r.regime == 'single' else 'rich'})"
                                     for r in matched[matched.delta_p8 == 0].itertuples()) or "none")

    def sel(df, defect, d, delta):
        return df[(df.defect == defect) & (df.dgp == d) & np.isclose(df.delta, delta)]

    # detection of P8 by process and severity
    lines, rows = [], []
    for d in DGP_ORDER:
        m = matched[matched.dgp == d]
        deltas = sorted(set(grid) | {float(x) for x in m.delta_p8 if x > 0})
        for i, delta in enumerate(deltas):
            c = sel(cells, "P8", d, delta)
            if c.empty:
                continue
            l1any = l1k = l1h = np.nan
            if r1 is not None:
                x = sel(r1, "P8", d, delta)
                l1any = float(x[x.statistic == "any"].flag_holm.iloc[0])
                l1k = float(x[x.statistic == "kurt"].flag_holm.iloc[0])
                l1h = float(x[x.statistic == "hill"].flag_holm.iloc[0])
            failed = c[~c["pass"]]
            r3 = sel(rules, "P8", d, delta) if rules is not None else None
            l3f = sorted(r3[~r3["pass"]].strategy) if r3 is not None else []
            tag = ", ".join(sorted(m[np.isclose(m.delta_p8, delta)].regime))
            key = f"e14.p8.{d}.{delta:g}"
            num.add(key + ".l1", l1any)
            num.add(key + ".kurt", l1k)
            num.add(key + ".l2failed", int(len(failed)))
            num.add(key + ".l3failed", ", ".join(l3f) or "none")
            rows.append({"dgp": d, "delta": delta, "matched": tag, "L1": l1any, "L1_kurt": l1k, "L1_hill": l1h,
                         "l2_failed": len(failed), "l3_failed": ", ".join(l3f)})
            fs = ", ".join(s for s in STRATS if s in set(failed.strategy))
            lab = DGP_LABEL[d] if i == 0 else ""
            delta_txt = f"{delta:g}" + (r"$^{\ast}$" if tag else "")
            lines.append(f"{lab} & {delta_txt} & {100 * EXTREME_PROB_STEP * delta:.2f}\\% & {l1k:.2f} & {l1h:.2f} & "
                         f"{l1any:.2f} & {len(failed)} & {fs or '--'} & {', '.join(l3f) or '--'}" + r" \\")
        lines.append(r"\addlinespace")
    det = pd.DataFrame(rows)
    det.to_csv(out / "tables" / "e14_detection.csv", index=False)
    (out / "tables" / "tab_e14_detection.tex").write_text(r"""\begin{tabular}{lrrrrrrll}
\toprule
& & extreme & \multicolumn{3}{c}{L1 power} & L2 & & \\
\cmidrule(lr){4-6}
Process & $\delta$ & returns & kurtosis & Hill & any & failed & L2 rules & L3 failed \\
\midrule
""" + "\n".join(lines[:-1]) + "\n" + r"""\bottomrule
\end{tabular}
""")
    gg = det[det.delta.isin(grid)]
    if len(gg):
        num.add("e14.p8.l2failed.min", int(gg.l2_failed.min()))
        num.add("e14.p8.l2failed.max", int(gg.l2_failed.max()))
        num.add("e14.p8.l1.min", float(gg.L1.min()))
        num.add("e14.p8.l1.max", float(gg.L1.max()))
        num.add("e14.p8.l3.grid.failed", int((gg.l3_failed != "").sum()))
        num.add("e14.p8.l3.grid.total", int(len(gg)))
        g1 = det[np.isclose(det.delta, 1.0)]
        num.add("e14.p8.1.l2failed.min", int(g1.l2_failed.min()))
        num.add("e14.p8.1.l2failed.max", int(g1.l2_failed.max()))
    # comparison with the Quant GAN at the matched severities, on the attainable cells of each regime
    lv = _levels(root)
    cards = _cards(lv, pr_min)
    keys = [(s, m_, f_) for s in STRATS for m_, f_ in FUNC_ORDER]
    l2q = lv["l2"][lv["l2"].kind == "qgan"]
    mlines, mrows, tally = [], [], {}
    for r in ("single", "rich"):
        for d in DGP_ORDER:
            o, t = cards.get((d, r, "oracle_drift")), cards.get((d, r, "qgan"))
            mm = matched[(matched.dgp == d) & (matched.regime == r)]
            if o is None or t is None or mm.empty:
                continue
            mm = mm.iloc[0]
            A = [k for k in keys if o["L2"][k] >= pr_min]
            f_g3 = {k for k in A if t["L2"][k] < pr_min}
            x = _fingerprint(l2q[(l2q.dgp == d) & (l2q.regime == r)])
            res = {}
            for defect, delta in (("P8", float(mm.delta_p8)), ("P7", float(mm.delta_p7))):
                if delta <= 0:
                    res[defect] = (set(), np.nan, None)
                    continue
                c = sel(cells, defect, d, delta).set_index(CELL)
                fails = {k for k in A if not c.loc[k, "pass"]}
                y = sel(e14["l2"], defect, d, delta)
                y = _fingerprint(y)
                corr = float(np.corrcoef(x.loc[A].to_numpy(), y.loc[A].to_numpy())[0, 1]) if len(A) > 2 else np.nan
                r3 = sel(rules, defect, d, delta) if rules is not None else None
                res[defect] = (fails, corr, sorted(r3[~r3["pass"]].strategy) if r3 is not None else [])
            expl = {k for k in f_g3 if k in res["P8"][0] or k in res["P7"][0]}
            for k in f_g3:
                lab = "P8" if k in res["P8"][0] else "P7" if k in res["P7"][0] else "none"
                tally[(r, lab, k[1], k[2])] = tally.get((r, lab, k[1], k[2]), 0) + 1
            key = f"e14.match.{r}.{d}"
            num.add(key + ".A", len(A))
            num.add(key + ".fg3", len(f_g3))
            num.add(key + ".fp8", len(res["P8"][0]))
            num.add(key + ".fp7", len(res["P7"][0]))
            num.add(key + ".both8", len(f_g3 & res["P8"][0]))
            num.add(key + ".both7", len(f_g3 & res["P7"][0]))
            num.add(key + ".explained", len(expl))
            for defect in ("P8", "P7"):
                if not np.isnan(res[defect][1]):
                    num.add(f"{key}.corr{defect[1]}", res[defect][1])
            num.add(key + ".l3g3", ", ".join(sorted(t["L3_failed"])) or "none")
            mrows.append({"regime": r, "dgp": d, "exceed": float(mm.exceed_ratio), "delta8": float(mm.delta_p8),
                          "vr": float(mm.vr), "delta7": float(mm.delta_p7), "A": len(A), "f_g3": len(f_g3),
                          "f_p8": len(res["P8"][0]), "f_p7": len(res["P7"][0]), "both8": len(f_g3 & res["P8"][0]),
                          "both7": len(f_g3 & res["P7"][0]), "explained": len(expl), "corr8": res["P8"][1],
                          "corr7": res["P7"][1], "l3_g3": ", ".join(sorted(t["L3_failed"])),
                          "l3_p8": ", ".join(res["P8"][2] or []), "l3_p7": ", ".join(res["P7"][2] or [])})
            dash = lambda v, fmt: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else fmt.format(v)
            reg = {"single": f"{cfg['data']['train_returns'] // 252} years", "rich": "rich data"}[r]
            l3 = lambda z: "--" if z is None else (", ".join(z) or "--")
            mlines.append(f"{reg if d == DGP_ORDER[0] else ''} & {DGP_LABEL[d].split(' ')[0]} & "
                          f"{mm.exceed_ratio:.2f} & {dash(mm.delta_p8 if mm.delta_p8 > 0 else None, '{:.2f}')} & "
                          f"{mm.vr:.2f} & {dash(mm.delta_p7 if mm.delta_p7 > 0 else None, '{:.2f}')} & {len(A)} & "
                          f"{len(f_g3)} & {len(res['P8'][0])} & {len(res['P7'][0])} & {len(expl)} & "
                          f"{dash(res['P8'][1], '{:.2f}')} & {dash(res['P7'][1], '{:.2f}')} & "
                          f"{', '.join(sorted(t['L3_failed'])) or '--'} & {l3(res['P8'][2])} & {l3(res['P7'][2])}" + r" \\")
        mlines.append(r"\addlinespace")
    while mlines and mlines[-1] == r"\addlinespace":
        mlines.pop()
    mt = pd.DataFrame(mrows)
    mt.to_csv(out / "tables" / "e14_match.csv", index=False)
    for r in ("single", "rich"):
        x = mt[mt.regime == r]
        if not len(x):
            continue
        num.add(f"e14.match.{r}.fg3.total", int(x.f_g3.sum()))
        num.add(f"e14.match.{r}.explained.total", int(x.explained.sum()))
        num.pct(f"e14.match.{r}.explained.share", float(x.explained.sum() / max(x.f_g3.sum(), 1)))
        num.add(f"e14.match.{r}.unexplained.total", int(x.f_g3.sum() - x.explained.sum()))
        num.add(f"e14.match.{r}.both8.total", int(x.both8.sum()))
        num.add(f"e14.match.{r}.both7.total", int(x.both7.sum()))
        num.add(f"e14.match.{r}.fp8.total", int(x.f_p8.sum()))
        num.add(f"e14.match.{r}.p7only.total", int((x.f_p7 - x.both7).sum()))
        num.pct(f"e14.match.{r}.share8", float(x.both8.sum() / max(x.f_g3.sum(), 1)))
        num.pct(f"e14.match.{r}.share7", float(x.both7.sum() / max(x.f_g3.sum(), 1)))
        for defect in ("8", "7"):
            c = x[f"corr{defect}"].dropna()
            if len(c):
                num.add(f"e14.match.{r}.corr{defect}.min", float(c.min()))
                num.add(f"e14.match.{r}.corr{defect}.max", float(c.max()))
        for lab in ("P8", "P7", "none"):
            for m_, f_ in FUNC_ORDER:
                num.add(f"e14.match.{r}.{lab}.{m_}.{f_}", int(tally.get((r, lab, m_, f_), 0)))
    (out / "tables" / "tab_e14_match.tex").write_text(r"""\begin{tabular}{llrrrrrrrrrrrlll}
\toprule
& & \multicolumn{4}{c}{G3 and matched severity} & & \multicolumn{4}{c}{Failed in $\mathcal{A}$} & & & \multicolumn{3}{c}{L3 failed} \\
\cmidrule(lr){3-6}\cmidrule(lr){8-11}\cmidrule(lr){14-16}
Data & Process & extremes & $\delta_{8}$ & VR & $\delta_{7}$ & $|\mathcal{A}|$ & G3 & P8 & P7 & explained & $r_{8}$ & $r_{7}$ & G3 & P8 & P7 \\
\midrule
""" + "\n".join(mlines) + "\n" + r"""\bottomrule
\end{tabular}
""")
