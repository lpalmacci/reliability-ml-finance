"""Step 6a: how much the LightGBM members weigh in the disagreement of the second construction.

    python keff_checks.py      ->  results/keff_checks.parquet

On the validation part, full pool (K = 36), before ("unconstrained") and after ("final") the constraint on
the Hessian of the LightGBM leaves: the effective number of members and the correlation of the deviations
within and between families (``confidex.keff_deviations``); Keff without the four LightGBM members; mean
correlation within LightGBM, within the other families and between LightGBM and the others; the share of
the sum of squared deviations d_k = p_k - pbar due to the LightGBM members (4 of 36: about 4/36 = 0.11
under exchangeability). For reference, Keff of the first construction (the same in the two versions).
Needs the pools of both versions (train_members.py, with and without --unconstrained).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import confidex
from fraud.data import N_PARTITIONS
from fraud.members import clip, load_pool
from fraud.registry import RESULTS


def mean_correlations(P: np.ndarray, fam: np.ndarray, f: str) -> dict[str, float]:
    D = P - P.mean(axis=1, keepdims=True)
    C = np.corrcoef(D, rowvar=False)
    iu = np.triu_indices(len(fam), 1)
    a, b = fam[iu[0]], fam[iu[1]]
    within_f = (a == f) & (b == f)
    within_others = (a == b) & (a != f)
    between_f_others = (a != b) & ((a == f) | (b == f))
    q = (D[:, fam == f] ** 2).sum() / (D ** 2).sum()
    return {"rho_within_LightGBM": float(C[iu][within_f].mean()),
            "rho_within_others": float(C[iu][within_others].mean()),
            "rho_LightGBM_others": float(C[iu][between_f_others].mean()),
            "share_squared_deviations_LightGBM": float(q)}


def main() -> None:
    rows = []
    for d in ("carclaims", "ulb"):
        for p in range(N_PARTITIONS[d]):
            for v in ("unconstrained", "final"):
                pool = load_pool(d, p, "s2", v)
                P, fam = clip(pool["P_validation"]), pool["families"]
                k = confidex.keff_deviations(P, fam)
                without = fam != "LightGBM"
                rows.append({"dataset": d, "partition": p, "version": v,
                             "keff": k["keff"], "rho_within": k["rho_within"], "rho_between": k["rho_between"],
                             "keff_without_LightGBM": confidex.keff_deviations(P[:, without], fam[without])["keff"],
                             **mean_correlations(P, fam, "LightGBM")})
            P1 = clip(load_pool(d, p, "s1", "final")["P_validation"])
            rows[-1]["keff_s1"] = rows[-2]["keff_s1"] = confidex.keff_deviations(P1)["keff"]
    out = pd.DataFrame(rows)
    out.to_parquet(RESULTS / "keff_checks.parquet", index=False)
    pd.set_option("display.width", 250)
    print(out.groupby(["dataset", "version"]).mean(numeric_only=True).drop(columns="partition").round(3).T)


if __name__ == "__main__":
    main()
