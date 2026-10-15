"""Second deep generator, the Quant GAN (G3), pre-registered by addendum (docs/ADDENDUM_QGAN.md; deviation D10).

Fidelity checks Q0 and Q0b against Table 2 of Wiese et al. (2020), and the numbers of experiment E13 that concern
the Quant GAN alone: verdicts against the attainable cells of E8, undefined estimates, comparisons with the other
generators, and the hypotheses HQ1-HQ6 of the addendum, evaluated as stated there on the first three runs and,
alongside, on the five runs of deviation D11. The tables, figures and cards in which G3 appears next to the other
generators are written by tavp.report.case, attain, tolcrit, robust, real and curves. Writes tables and numbers with
the prefixes `q0.`, `e13.`, `facts.qgan.`, `hq.` (three runs) and `hqall.` (five runs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tavp.experiments.e13_qgan import PREREG_SEEDS
from tavp.protocol.card import FUNC_ORDER, STRATS, build_card
from tavp.protocol.l4 import calibration_tests
from tavp.report.attain import _cards as attain_cards
from tavp.report.attain import _levels as attain_levels
from tavp.report.case import DGP_LABEL, DGP_ORDER, gkey, l4_pooled
from tavp.report.numbers import Numbers
from tavp.report.tolcrit import critical_cells, undefined_as_infinite

KEYS = [(s, m, f) for s in STRATS for m, f in FUNC_ORDER]
CELL = ["strategy", "metric", "functional"]
REGIMES = ("single", "rich")
COLLAPSE_VOL = 0.05  # annual volatility below which a run counts as collapsed, as for the TC-VAE
METRIC_LABEL = {"emd_1": "EMD 1", "emd_5": "EMD 5", "emd_20": "EMD 20", "emd_100": "EMD 100", "dy_1": "DY 1",
                "dy_5": "DY 5", "dy_20": "DY 20", "dy_100": "DY 100", "acf_id": r"ACF $r$", "acf_abs": r"ACF $|r|$",
                "acf_sq": r"ACF $r^2$", "leverage": "leverage"}


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _short(d: str) -> str:
    return DGP_LABEL[d].split(" ")[0]


def _lr(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return f"${'' if m == '1' else m + r'\cdot '}10^{{{int(e)}}}$"


# ----------------------------------------------------------------------------- Q0 and Q0b
def build_q0(root: Path, out: Path, num: Numbers) -> None:
    from tavp.experiments.q0_fidelity import METRICS, START
    from tavp.experiments.q0_tune import GRID

    scores, crit = _read(root, "q0", "scores"), _read(root, "q0", "criterion")
    tune, conf, garch = _read(root, "q0", "tune"), _read(root, "q0", "confirm"), _read(root, "q0", "tune_garch")
    if scores is None or crit is None or tune is None or conf is None or garch is None:
        return
    g = garch.iloc[0]
    tcn = scores[scores.model == "TCN"].sort_values("seed")
    paper = {m: scores[scores.model == m].iloc[0] for m in ("paper TCN", "paper GARCH(1,1)")}

    def beats(x) -> tuple[list[str], list[str]]:
        better = [m for m in METRICS if x[m] < g[m]]
        return better, [m for m in METRICS if m not in better]

    rows = []
    for i, x in enumerate(tcn.itertuples(index=False)):
        b, _ = beats(x._asdict())
        rows.append(("Q0", f"run {i + 1}", START["steps"], START["lr_g"], START["r1_gamma"], x.selected_step, b))
    chosen = conf.iloc[0]
    for x in tune.itertuples(index=False):
        b, _ = beats(x._asdict())
        mark = "$^{\\ast}$" if (x.lr_g == chosen.lr_g and x.r1_gamma == chosen.r1_gamma) else ""
        rows.append(("Q0b, tuning", "one run" + mark, x.steps, x.lr_g, x.r1_gamma, x.selected_step, b))
    for i, x in enumerate(conf.itertuples(index=False)):
        b, _ = beats(x._asdict())
        rows.append(("Q0b, confirmation", f"run {i + 1}", x.steps, x.lr_g, x.r1_gamma, x.selected_step, b))
    lines, prev = [], None
    for stage, run, steps, lrg, r1, sel, b in rows:
        if prev is not None and stage != prev:
            lines.append(r"\addlinespace")
        n_emd = sum(m.startswith("emd") for m in b)
        meets = "--" if "tuning" in stage else "yes" if len(b) >= 10 and n_emd == 4 else "no"
        n = lambda v: f"{int(v):,}".replace(",", "{,}")
        lines.append(f"{stage if stage != prev else ''} & {run} & {n(steps)} & {_lr(lrg)} & {r1:g} & {n(sel)} & "
                     f"{len(b)}/12 & {n_emd}/4 & {meets} \\\\")
        prev = stage
    (out / "tables" / "tab_q0.tex").write_text(r"""\begin{tabular}{llrrrrrrl}
\toprule
& & & generator & R1 & selected & \multicolumn{2}{c}{better than GARCH(1,1)} & meets \\
\cmidrule(lr){7-8}
Stage & Training run & steps & learning rate & weight & checkpoint & metrics & EMDs & criterion \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    # the twelve metrics of the paper: our GARCH(1,1), the confirmation runs, and Table 2 of the paper
    fmt = {m: ("{:.4f}" if m.startswith("emd") else "{:.1f}" if m.startswith("dy") else "{:.3f}") for m in METRICS}
    mrows = [("GARCH(1,1), this implementation", g)] + [(f"TCN, confirmation run {i + 1}", x) for i, x in
                                                         enumerate(r for _, r in conf.iterrows())]
    mrows += [("GARCH(1,1), Wiese et al.", paper["paper GARCH(1,1)"]), ("TCN, Wiese et al.", paper["paper TCN"])]
    body = []
    for i, (lab, x) in enumerate(mrows):
        if i in (1, 4):
            body.append(r"\addlinespace")
        cells = []
        for m in METRICS:
            v = fmt[m].format(x[m])
            cells.append(r"\textbf{" + v + "}" if 1 <= i <= 3 and x[m] < g[m] else v)
        body.append(lab + " & " + " & ".join(cells) + r" \\")
    (out / "tables" / "tab_q0_metrics.tex").write_text(r"""\begin{tabular}{l""" + "r" * len(METRICS) + r"""}
\toprule
& """ + " & ".join(METRIC_LABEL[m] for m in METRICS) + r""" \\
\midrule
""" + "\n".join(body) + "\n" + r"""\bottomrule
\end{tabular}
""")
    # numbers
    num.add("q0.metrics", len(METRICS))
    for i, x in enumerate(crit.sort_values("seed").itertuples(index=False)):
        num.add(f"q0.start.better.{i + 1}", int(x.better))
    num.add("q0.start.pass", int(crit["pass"].sum()))
    num.add("q0.start.verdict", "accepted" if int(crit["pass"].sum()) >= 2 else "not accepted")
    num.add("q0.start.lev.min", float(tcn.leverage.min()))
    num.add("q0.start.lev.max", float(tcn.leverage.max()))
    num.add("q0.garch.lev", float(g.leverage))
    num.add("q0.paper.lev.tcn", float(paper["paper TCN"].leverage))
    num.add("q0.paper.lev.garch", float(paper["paper GARCH(1,1)"].leverage))
    acf_lost = all(x[m] >= g[m] for _, x in tcn.iterrows() for m in ("acf_abs", "acf_sq"))
    num.add("q0.start.acf.lost.all", "yes" if acf_lost else "no")
    num.add("q0.tune.configs", len(GRID))
    num.add("q0.tune.better.min", int(tune.better.min()))
    num.add("q0.tune.better.max", int(tune.better.max()))
    num.add("q0.tune.chosen.lrg", _lr(float(chosen.lr_g)))
    num.add("q0.tune.chosen.r1", float(chosen.r1_gamma), "{:g}")
    num.add("q0.tune.chosen.steps", int(chosen.steps))
    num.add("q0.tune.ties", int((tune.better == tune.better.max()).sum()))
    for i, x in enumerate(conf.itertuples(index=False)):
        num.add(f"q0.confirm.better.{i + 1}", int(x.better))
    num.add("q0.confirm.pass", int(conf["pass"].sum()))
    num.add("q0.confirm.verdict", "accepted" if int(conf["pass"].sum()) >= 2 else "not accepted")
    num.add("q0.confirm.step.min", int(conf.selected_step.min()))
    num.add("q0.confirm.step.max", int(conf.selected_step.max()))
    lost = [m for m in METRICS if any(m in beats(x)[1] for _, x in conf.iterrows())]
    num.add("q0.confirm.lost", ", ".join(METRIC_LABEL[m] for m in lost) or "none")
    num.add("q0.confirm.lost.acfonly", "yes" if all(m.startswith("acf") for m in lost) else "no")
    num.add("q0.confirm.lev.min", float(conf.leverage.min()))
    num.add("q0.confirm.lev.max", float(conf.leverage.max()))
    num.add("q0.confirm.emd1.ratio.max", float((conf.emd_1 / g.emd_1).max()))


# ----------------------------------------------------------------------------- E13
def _e13_levels(root: Path) -> dict[str, pd.DataFrame]:
    return {lv: t for lv in ("l0", "l1", "l2", "l3") if (t := _read(root, "e13", lv)) is not None}


def _sel(df: pd.DataFrame | None, d: str, r: str) -> pd.DataFrame | None:
    return None if df is None else df[(df.dgp == d) & (df.regime == r)]


def _drift_sd(l2: pd.DataFrame, T: int, seeds=None) -> pd.Series:
    """Standard deviation across runs of the annual drift error of the scenarios (mean horizon return of buy-and-hold)."""
    bh = l2[(l2.strategy == "BH") & (l2.metric == "ret") & (l2.functional == "mean")]
    if seeds is not None:
        bh = bh[bh.seed.isin(seeds)]
    return (bh.groupby(["dgp", "seed"])["diff"].mean() * 252 / T).groupby("dgp").std()


def _g3(q: dict, cards: dict, pr_min: float, seeds=None) -> dict:
    """Card of G3 and its counts on the attainable cells per (process, regime), pooling the given runs."""
    sub = {k: (v if seeds is None else v[v.seed.isin(seeds)]) for k, v in q.items()}
    out = {}
    for d in DGP_ORDER:
        for r in REGIMES:
            x2, o = _sel(sub["l2"], d, r), cards.get((d, r, "oracle_drift"))
            if x2 is None or x2.empty or o is None:
                continue
            c = build_card(_sel(sub.get("l0"), d, r), _sel(sub.get("l1"), d, r), x2, _sel(sub.get("l3"), d, r), None,
                           pr_min)
            att = [k for k in KEYS if o["L2"][k] >= pr_min]
            att_rules = [s for s, v in o["L3"].items() if v["pr"] >= pr_min]
            ps = x2.groupby(["seed"] + CELL).equivalent.mean() >= pr_min
            ps_att = ps[[tuple(i[1:]) in set(att) for i in ps.index]].groupby("seed").sum()
            passed = [k for k in KEYS if c["L2"][k] >= pr_min]
            pa = [k for k in att if c["L2"][k] >= pr_min]
            l3fa = [s for s in att_rules if s in c["L3_failed"]]
            out[(d, r)] = {"card": c, "att": att, "passed": passed, "passA": len(pa), "l3fa": l3fa,
                           "runs": ps_att.astype(int).tolist(), "outside": len([k for k in passed if k not in att]),
                           "fitA": bool(c["L0"]["hash_ok"] and c["L0"]["valid"] and len(pa) == len(att) and not l3fa)}
    return out


def build_e13(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    q = _e13_levels(root)
    if "l2" not in q or _read(root, "e8", "l2") is None:
        return
    lv = attain_levels(root)
    cards = attain_cards(lv, pr_min)
    pits = _read(root, "e13", "l4_pits")
    l4 = l4_pooled(pits, cfg) if pits is not None else None
    crit = critical_cells(undefined_as_infinite(q["l2"]), pr_min)
    T = cfg["data"]["scenario_prices"] - 1
    g3 = _g3(q, cards, pr_min)
    res = []
    for (d, r), v in g3.items():
        c = v["card"]
        ck = crit[(crit.dgp == d) & (crit.regime == r)]
        kappa_a = float(ck[[tuple(z) in set(v["att"]) for z in ck[CELL].itertuples(index=False)]].ratio.max())
        key = f"e13.{r}.{d}"
        num.add(key + ".pass", len(v["passed"]))
        num.add(key + ".passA", v["passA"])
        num.add(key + ".passA.min", int(min(v["runs"])))
        num.add(key + ".passA.max", int(max(v["runs"])))
        num.add(key + ".l1flag", len(c["L1_flagged"]))
        num.add(key + ".l3failA", ", ".join(v["l3fa"]) or "none")
        num.add(key + ".l3fail", ", ".join(c["L3_failed"]) or "none")
        num.add(key + ".fitA", "fit" if v["fitA"] else "not fit")
        num.add(key + ".fit", "fit" if c["fit"] else "not fit")
        num.add(key + ".kappaA", kappa_a, "{:.1f}" if kappa_a >= 10 else "{:.2f}")
        num.add(key + ".outside", v["outside"])
        if l4 is not None and r == "single":
            num.add(key + ".l4rej", int(l4[l4.dgp == d].reject_any.sum()))
        for s, x in c["L3"].items():
            num.add(f"{key}.l3.{s}.pr", x["pr"])
            num.add(f"{key}.l3.{s}.regret", x["regret"], "{:.3f}")
        res.append({"dgp": d, "regime": r, "A": len(v["att"]), "pass": len(v["passed"]), "passA": v["passA"],
                    "runs": v["runs"], "l1": len(c["L1_flagged"]), "kappaA": kappa_a, "fitA": v["fitA"],
                    "resolvable": c["L0"]["resolvable"], "valid": c["L0"]["valid"]})
    tab = pd.DataFrame(res)
    if tab.empty:
        return
    tab.to_csv(out / "tables" / "e13_summary.csv", index=False)
    num.add("e13.runs", int(q["l2"].seed.nunique()))
    num.add("e13.runs.prereg", PREREG_SEEDS)
    num.add("e13.valid", "yes" if bool(tab.valid.all()) else "no")
    num.pct("e13.resolvable.min", float(tab.resolvable.min()), 2)
    num.add("e13.fitA.n", int(tab.fitA.sum()))
    num.add("e13.fit.n", sum(v["card"]["fit"] for v in g3.values()))
    num.add("e13.outside.total", int(sum(v["outside"] for v in g3.values())))
    if l4 is not None:
        num.add("e13.l4rej.total", int(l4.reject_any.sum()))
        per = [int(pd.DataFrame(calibration_tests(h, cfg)).reject_any.sum()) for _, h in pits.groupby(["dgp", "history"])]
        num.add("e13.l4rej.run.max", max(per))
        num.add("e13.l4.cells", int(len(l4)))
        num.add("e13.l4rej.which", ", ".join(_short(d) for d in DGP_ORDER if l4[l4.dgp == d].reject_any.any()) or "none")
    for r in REGIMES:
        t = tab[tab.regime == r]
        num.add(f"e13.{r}.kappaA.min", float(t.kappaA.min()), "{:.2f}")
        num.add(f"e13.{r}.kappaA.max", float(t.kappaA.max()), "{:.1f}" if t.kappaA.max() >= 10 else "{:.2f}")
    # undefined estimates: scenarios with finite but explosive returns make the drawdown undefined
    bad = q["l2"][~np.isfinite(q["l2"].est_g)]
    ctx = bad.groupby(["dgp", "regime", "seed"]).k.nunique()
    num.add("e13.undefined.contexts", int(ctx.sum()))
    num.add("e13.undefined.runs", int(len(ctx)))
    num.add("e13.undefined.cells", int(len(bad)))
    num.add("e13.contexts", int(q["l2"].k.nunique()))
    if len(ctx):
        (d, r, s), n = ctx.idxmax(), int(ctx.max())
        num.add("e13.undefined.which", f"process {_short(d)}, {'ten years' if r == 'single' else 'rich data'}, run {int(s) + 1}")
        num.add("e13.undefined.max", n)
        num.add("e13.undefined.metrics", ", ".join(sorted(bad.metric.unique())).replace("maxdd", "maximum drawdown"))
    # comparisons with the other generators on the attainable cells, in the eight settings (process, amount of data)
    for kind in ("tcvae", "bootstrap", "garch"):
        other = {(d, r): sum(cards[(d, r, kind)]["L2"][k] >= pr_min for k in g3[(d, r)]["att"])
                 for (d, r) in g3 if (d, r, kind) in cards}
        num.add(f"e13.vs.{kind}.more", sum(g3[s]["passA"] > v for s, v in other.items()))
        num.add(f"e13.vs.{kind}.less", sum(g3[s]["passA"] < v for s, v in other.items()))
        num.add(f"e13.vs.{kind}.settings", len(other))
    facts = _facts(root, num)
    rows3 = _hypotheses(_g3(q, cards, pr_min, seeds=range(PREREG_SEEDS)), cards, lv, q, facts, T, num, "hq",
                        range(PREREG_SEEDS))
    rows5 = _hypotheses(g3, cards, lv, q, facts, T, num, "hqall", None)

    def cell(width: str, text: str) -> str:  # ragged-right paragraph cell, without the array package
        return r"\parbox[t]{" + width + r"}{\raggedright " + text + r"\strut}"

    lines = [f"{h} & {cell('4.0cm', stmt)} & {cell('5.8cm', ev)} & {cell('1.6cm', v3)} & {cell('1.6cm', v5)} \\\\"
             for (h, stmt, ev, v3), (_, _, _, v5) in zip(rows3, rows5)]
    (out / "tables" / "tab_qgan_hyp.tex").write_text(r"""\begin{tabular}{lllll}
\toprule
& & & \multicolumn{2}{c}{Outcome} \\
\cmidrule(lr){4-5}
& Hypothesis (addendum) & Result (first three runs) & three runs & five runs \\
\midrule
""" + "\n\\addlinespace\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
    for prefix, rows in (("hq", rows3), ("hqall", rows5)):
        num.add(f"{prefix}.supported", sum(v == "supported" for *_, v in rows))
        num.add(f"{prefix}.total", len(rows))
    num.add("hq.changed", sum(a[3] != b[3] for a, b in zip(rows3, rows5)))


def _facts(root: Path, num: Numbers) -> pd.DataFrame | None:
    """Stylized facts of the oracles, the TC-VAE and the Quant GAN (the figure is written by tavp.report.case)."""
    f13, f4 = _read(root, "e13", "facts"), _read(root, "e4", "facts")
    if f13 is None or f4 is None:
        return None
    f = pd.concat([f4[f4.kind.isin(["oracle", "tcvae"])], f13], ignore_index=True)
    f = f.assign(g=[gkey(k, r) if k in ("tcvae", "qgan") else k for k, r in zip(f.kind, f.regime)])
    g3 = f[f.kind == "qgan"]
    num.add("facts.qgan.collapsed", int((g3["std"] < COLLAPSE_VOL).sum()))
    num.add("facts.qgan.std.min", float(g3["std"].min()))
    num.add("facts.qgan.std.max", float(g3["std"].max()))
    num.add("facts.qgan.acf_r1.min", float(g3.acf_r1.min()), "{:.3f}")
    num.add("facts.qgan.acf_r1.max", float(g3.acf_r1.max()), "{:.3f}")
    num.add("facts.qgan.acf_r1.absmax", float(g3.acf_r1.abs().max()), "{:.3f}")
    num.add("facts.qgan.acf_r1.over", int((g3.acf_r1.abs() >= 0.02).sum()))
    num.add("facts.qgan.runs", int(len(g3)))
    num.add("facts.qgan.vr.min", float(g3.vr.min()))
    num.add("facts.qgan.vr.max", float(g3.vr.max()))
    num.add("facts.qgan.vr.below90", int((g3.vr < 0.9).sum()))
    num.add("facts.qgan.vr.below1", int((g3.vr < 1.0).sum()))
    tc = f[(f.kind == "tcvae") & (f["std"] >= COLLAPSE_VOL)]
    num.add("facts.tcvae.vr.below1", int((tc.vr < 1.0).sum()))
    num.add("facts.tcvae.runs", int(len(tc)))
    top = g3.sort_values("kurt").iloc[-1]
    num.add("facts.qgan.kurt.max", f"{float(top['kurt']):,.0f}".replace(",", "{,}"))
    num.add("facts.qgan.kurt.max.std", float(top["std"]))
    num.add("facts.qgan.kurt.max.which", f"process {_short(top.dgp)}, "
            f"{'ten years' if top.regime == 'single' else 'rich data'}, run {int(top.seed) + 1}")
    return f


def _v(ok: bool) -> str:
    return "supported" if ok else "not supported"


def _hypotheses(g3: dict, cards: dict, lv: dict, q: dict, f: pd.DataFrame | None, T: int, num: Numbers, prefix: str,
                seeds) -> list[tuple[str, str, str, str]]:
    """HQ1-HQ6 as stated in the addendum, on the runs of G3 in `seeds` (None: all runs)."""
    rows = []
    if f is not None and seeds is not None:
        f = f[(f.kind != "qgan") | f.seed.isin(list(seeds))]

    def fmt(v):
        return f"{v:,.0f}".replace(",", "{,}") if abs(v) >= 100 else f"{v:.2f}" if abs(v) >= 1 else f"{v:.3f}"

    def med(g, d, s):
        return float(f[(f.g == g) & (f.dgp == d)][s].median())

    if f is not None:
        # HQ1: tails on D3, kurtosis and Hill index, medians over runs, both regimes
        d, ok, ev = "gjr_garch_t", [], []
        for s, lab in (("kurt", "kurtosis"), ("hill", "Hill index")):
            o = med("oracle", d, s)
            parts = []
            for r, rl in (("single", "10 years"), ("rich", "rich")):
                a, b = med(f"qgan_{r}", d, s), med(f"tcvae_{r}", d, s)
                ok.append(abs(a - o) < abs(b - o))
                num.add(f"{prefix}1.{s}.{r}", "yes" if ok[-1] else "no")
                parts.append(f"{rl} {fmt(a)} vs {fmt(b)}")
            ev.append(f"{lab}: oracle {fmt(o)}; " + "; ".join(parts))
        num.add(f"{prefix}1.outcome", _v(all(ok)))
        rows.append(("HQ1", "On D3, kurtosis and Hill index of G3 closer to the oracle than those of G1, both regimes",
                     " \\newline ".join(ev), _v(all(ok))))
        # HQ2: |ACF(1)| < 0.02 in every run that does not collapse, and below the median of G1 in absolute value
        g = f[(f.kind == "qgan") & (f["std"] >= COLLAPSE_VOL)]
        a1 = bool((g.acf_r1.abs() < 0.02).all())
        below = []
        for d in DGP_ORDER:
            for r in REGIMES:
                tc = abs(med(f"tcvae_{r}", d, "acf_r1"))
                below.append(bool((g[(g.dgp == d) & (g.regime == r)].acf_r1.abs() < tc).all()))
        a2 = all(below)
        num.add(f"{prefix}2.part1", "yes" if a1 else "no")
        num.add(f"{prefix}2.part2", "yes" if a2 else "no")
        num.add(f"{prefix}2.part2.settings", int(sum(below)))
        num.add(f"{prefix}2.over", int((g.acf_r1.abs() >= 0.02).sum()))
        num.add(f"{prefix}2.runs", int(len(g)))
        num.add(f"{prefix}2.outcome", _v(a1 and a2))
        medbelow = all(abs(med(f"qgan_{r}", d, "acf_r1")) < abs(med(f"tcvae_{r}", d, "acf_r1"))
                       for d in DGP_ORDER for r in REGIMES)
        num.add(f"{prefix}2.median.below", "yes" if medbelow else "no")
        rows.append(("HQ2", "$|$ACF(1)$|$ of G3 below 0.02 in every run that does not collapse, and below the "
                     "$|$median$|$ of G1 (same process and regime)",
                     f"largest $|$ACF(1)$|$ {g.acf_r1.abs().max():.3f}; runs at or above 0.02: "
                     f"{int((g.acf_r1.abs() >= 0.02).sum())} of {len(g)}; below G1 in every run in {sum(below)} of "
                     f"{len(below)} settings; median below G1 in every setting: {'yes' if medbelow else 'no'}", _v(a1 and a2)))
        # HQ3: clustering with rich data, D2-D4
        ok3, ev3 = [], []
        for d in ("heston", "gjr_garch_t", "regime"):
            o, a, b = med("oracle", d, "acf_abs"), med("qgan_rich", d, "acf_abs"), med("tcvae_rich", d, "acf_abs")
            ok3.append(abs(a - o) < abs(b - o))
            ev3.append(f"{_short(d)}: oracle {o:.3f}, G3 {a:.3f}, G1 {b:.3f}")
        num.add(f"{prefix}3.outcome", _v(all(ok3)))
        rows.append(("HQ3", "With rich data (one long history for G3), ACF($|r|$) of G3 closer to the oracle than "
                     "that of G1 with rich data, on D2--D4", "; ".join(ev3), _v(all(ok3))))
    # HQ4: drift with ten years, and no cell passed outside the attainable set
    l2q = q["l2"] if seeds is None else q["l2"][q["l2"].seed.isin(list(seeds))]
    single = lambda df: df[(df.regime == "single")]
    sd_g3 = _drift_sd(single(l2q), T)
    oh = lv["l2"][(lv["l2"].kind == "oracle_drift") & (lv["l2"].regime == "single")]
    sd_oh = _drift_sd(oh, T, seeds=sorted(l2q.seed.unique()))
    ratio = (sd_g3 / sd_oh).reindex(DGP_ORDER)
    ok4a = bool(((ratio - 1).abs() <= 0.25).all())
    outside = sum(g3[(d, "single")]["outside"] for d in DGP_ORDER if (d, "single") in g3)
    for d in DGP_ORDER:
        num.add(f"{prefix}4.ratio.{d}", float(ratio[d]))
        num.add(f"{prefix}4.sd.qgan.{d}", float(sd_g3[d]), "{:.3f}")
        num.add(f"{prefix}4.sd.ohat.{d}", float(sd_oh[d]), "{:.3f}")
    num.add(f"{prefix}4.ratio.min", float(ratio.min()))
    num.add(f"{prefix}4.ratio.max", float(ratio.max()))
    num.add(f"{prefix}4.within", int(((ratio - 1).abs() <= 0.25).sum()))
    num.add(f"{prefix}4.outside", outside)
    num.add(f"{prefix}4.outcome", _v(ok4a and outside == 0))
    rows.append(("HQ4", "With ten years, SD across histories of the annual drift of G3 within 25\\% of that of the "
                 "sample mean; no cell passed outside $\\mathcal{A}$",
                 "ratio of the SDs: " + ", ".join(f"{_short(d)} {ratio[d]:.2f}" for d in DGP_ORDER)
                 + f"; cells passed outside $\\mathcal{{A}}$: {outside}", _v(ok4a and outside == 0)))
    # HQ5: rich data, attainable cells: more than G1 rich on D2-D4, no more than B2 long on D3
    def n_att(d, r, kind):
        c, o = cards.get((d, r, kind)), cards[(d, r, "oracle_drift")]
        return sum(c["L2"][k] >= 0.9 for k in KEYS if o["L2"][k] >= 0.9)

    more = {d: g3[(d, "rich")]["passA"] > n_att(d, "rich", "tcvae") for d in ("heston", "gjr_garch_t", "regime")}
    le_b2 = g3[("gjr_garch_t", "rich")]["passA"] <= n_att("gjr_garch_t", "rich", "garch")
    num.add(f"{prefix}5.outcome", _v(all(more.values()) and le_b2))
    rows.append(("HQ5", "With rich data, G3 passes more attainable cells than G1 with rich data on D2--D4, "
                 "and no more than B2 on the long histories of D3",
                 "; ".join(f"{_short(d)}: G3 {g3[(d, 'rich')]['passA']}, G1 {n_att(d, 'rich', 'tcvae')}"
                           for d in ("heston", "gjr_garch_t", "regime"))
                 + f"; B2 on D3 {n_att('gjr_garch_t', 'rich', 'garch')}", _v(all(more.values()) and le_b2)))
    # HQ6: range across runs of the attainable cells passed, G3 against as many runs of G1, same regime
    ev6, out6 = [], {}
    for r in REGIMES:
        larger = 0
        parts = []
        for d in DGP_ORDER:
            ps = cards[(d, r, "tcvae")]["per_seed"]
            att = set(g3[(d, r)]["att"])
            tseeds = sorted(ps.index.get_level_values("seed").unique())[:len(g3[(d, r)]["runs"])]
            tc = ps[[i[0] in tseeds and tuple(i[1:]) in att for i in ps.index]].groupby("seed").sum()
            tc = tc.reindex(tseeds, fill_value=0)
            rg3, rtc = max(g3[(d, r)]["runs"]) - min(g3[(d, r)]["runs"]), int(tc.max() - tc.min())
            larger += rg3 > rtc
            num.add(f"{prefix}6.{r}.{d}.g3", rg3)
            num.add(f"{prefix}6.{r}.{d}.g1", rtc)
            parts.append(f"{_short(d)} {rg3} vs {rtc}")
        out6[r] = larger >= 3
        num.add(f"{prefix}6.{r}.larger", larger)
        num.add(f"{prefix}6.{r}.outcome", _v(out6[r]))
        ev6.append(f"{'10 years' if r == 'single' else 'rich'}: " + ", ".join(parts))
    rows.append(("HQ6", "Range across runs of the attainable cells passed larger for G3 than for as many runs "
                 "of G1, same regime, on at least three processes", " \\newline ".join(ev6),
                 "; ".join(f"{_v(out6[r])} ({'10 years' if r == 'single' else 'rich'})" for r in REGIMES)))
    num.add(f"{prefix}6.outcome", _v(all(out6.values())))
    return rows


def build_qgan(root: Path, out: Path, num: Numbers, cfg: dict, pr_min: float) -> None:
    build_q0(root, out, num)
    build_e13(root, out, num, cfg, pr_min)
