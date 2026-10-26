"""General diagnostics (Chapter 3, Assumptions of the fixed-mean specification).

They do not enter the verdict: they help to read the results and to check the assumptions.
"""
from __future__ import annotations

import numpy as np

from .core import PosteriorC, shape_regime

__all__ = ["keff_deviations", "border_share", "shape_regime"]


def keff_deviations(P: np.ndarray, families: np.ndarray | None = None) -> dict[str, float]:
    """Effective number of members, from the deviations from the mean of each case.

    The deviations d_k = p_k - pbar sum to zero, so their average pairwise correlation is about
    -1/(K-1) whatever the dependence, and it measures nothing. What matters is the *structure* of
    their correlation matrix. With lambda_i its eigenvalues,

        Keff = 1 + (sum lambda_i)^2 / sum lambda_i^2

    equals K under exchangeability (the deviations spread equally over K - 1 directions) and falls
    towards the number of independent groups when subsets of members share their errors. A component
    common to all the members is absorbed into pbar and does not enter the computation (Chapter 3,
    Remark "Effective number of members").

    ``P`` has shape (n, K), normally the validation set. With ``families`` it also returns the average
    correlation of the deviations within and between families.
    """
    P = np.asarray(P, dtype=float)
    n, K = P.shape
    if K < 3:
        raise ValueError("at least 3 members are needed")
    D = P - P.mean(axis=1, keepdims=True)
    C = np.corrcoef(D, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    out = {"keff": float(1.0 + ev.sum() ** 2 / (ev ** 2).sum())}
    if families is not None:
        fam = np.asarray(families)
        iu = np.triu_indices(K, 1)
        same = fam[iu[0]] == fam[iu[1]]
        out["rho_within"] = float(C[iu][same].mean()) if same.any() else float("nan")
        out["rho_between"] = float(C[iu][~same].mean()) if (~same).any() else float("nan")
    return out


def border_share(post: PosteriorC) -> float:
    """Share of cases whose posterior is truncated by the border of the grid."""
    return float(np.mean(post.at_border))
