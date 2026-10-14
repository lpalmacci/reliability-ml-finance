"""Robustness of the verdicts (E7): use variants, sample size and tolerances.

The variants re-run level L2 for the TC-VAE (single history), the bootstrap and the GJR-GARCH
benchmark on the first three histories. The scaling of the tolerances does not need new runs:
the stored 90% confidence intervals of the discrepancies are compared with the scaled
tolerances, and resolvability is re-assessed against half of the scaled tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tavp.report.numbers import Numbers

GENS = [("tcvae", "G1 TC-VAE, single history"), ("bootstrap", "B1 stationary bootstrap"),
        ("garch", "B2 GJR-GARCH-$t$")]
VARIANTS = [("cost", "transaction cost of 10 basis points"), ("obj_low", "objective with halved penalties"),
            ("obj_high", "objective with doubled penalties"), ("n_alt", "$N=10{,}000$ scenarios")]
N_HISTORIES = 3


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _subset(l2: pd.DataFrame) -> pd.DataFrame:
    """Settings of E7: TC-VAE single history, bootstrap and GARCH, first three histories."""
    m = l2.kind.isin(["tcvae", "bootstrap", "garch"]) & (l2.seed < N_HISTORIES)
    if "regime" in l2:
        m &= (l2.kind != "tcvae") | (l2.regime == "single")
    return l2[m]


def passed(l2: pd.DataFrame, pr_min: float, scale: float = 1.0) -> pd.DataFrame:
    """Cells passed per (generator kind, process), with tolerances multiplied by `scale`."""
    eq = (l2.lo >= -scale * l2.eta) & (l2.hi <= scale * l2.eta)
    pr = eq.groupby([l2.kind, l2.dgp, l2.strategy, l2.metric, l2.functional]).mean()
    res = (l2.half_width <= scale * l2.eta / 2).groupby([l2.kind, l2.dgp]).mean()
    out = (pr >= pr_min).groupby(["kind", "dgp"]).agg(["sum", "size"]).rename(columns={"sum": "passed", "size": "cells"})
    out["resolvable"] = res
    return out.reset_index()


def build_robust(root: Path, out: Path, num: Numbers, pr_min: float) -> None:
    parts = [t for t in (_read(root, e, "l2") for e in ("e4", "e5")) if t is not None]
    if not parts:
        return
    main = _subset(pd.concat(parts, ignore_index=True))
    rows = {"main": passed(main, pr_min)}
    for scale, key in ((0.5, "tol_half"), (2.0, "tol_double")):
        rows[key] = passed(main, pr_min, scale)
    for v, _ in VARIANTS:
        t = _read(root, f"e7_{v}", "l2")
        if t is not None:
            rows[v] = passed(_subset(t), pr_min)
    # negative control and power under scaled tolerances
    e1 = _read(root, "e1", "l2")
    if e1 is not None:
        for scale, key in ((0.5, "tol_half"), (2.0, "tol_double")):
            p = passed(e1, pr_min, scale)
            num.add(f"e7.{key}.e1.passed", f"{int(p.passed.sum())} of {int(p.cells.sum())}")
            num.pct(f"e7.{key}.e1.resolvable", float(p.resolvable.mean()))
    labels = {"main": "Main analysis", "tol_half": "Tolerances $\\times0.5$", "tol_double": "Tolerances $\\times2$",
              **{v: lab[0].upper() + lab[1:] for v, lab in VARIANTS}}
    order = ["main", "cost", "obj_low", "obj_high", "n_alt", "tol_half", "tol_double"]
    lines = []
    for key in order:
        if key not in rows:
            continue
        r = rows[key]
        cells = []
        for kind, _ in GENS:
            x = r[r.kind == kind]
            n_pass, n_cells = int(x.passed.sum()), int(x.cells.sum())
            fit = int((x.passed == x.cells).sum())
            cells.append(f"{n_pass}/{n_cells} & {fit}")
            num.add(f"e7.{key}.{kind}.passed", n_pass)
            num.add(f"e7.{key}.{kind}.fit", fit)
        lines.append(f"{labels[key]} & " + " & ".join(cells) + r" \\")
        num.pct(f"e7.{key}.resolvable", float(r.resolvable.mean()))
    pd.concat([r.assign(variant=k) for k, r in rows.items()]).to_csv(out / "tables" / "e7_robust.csv", index=False)
    head = " & ".join(r"\multicolumn{2}{c}{" + lab.split(",")[0] + "}" for _, lab in GENS)
    (out / "tables" / "tab_e7_robust.tex").write_text(r"""\begin{tabular}{lrrrrrr}
\toprule
& """ + head + r""" \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Variant & cells & fit & cells & fit & cells & fit \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
""")
