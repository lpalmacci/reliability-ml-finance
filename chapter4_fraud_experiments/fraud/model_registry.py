"""Registry of the families: name in the plan -> family object.

TabPFN is not here: its scores are produced by ``query_tabpfn.py`` through the remote service and read
from results/tabpfn/.
"""
from __future__ import annotations

from typing import Any

from fraud.families import (CatBoostFamily, ExtraTreesFamily, LightGBMFamily, LogRegFamily, MLPFamily,
                            RandomForestFamily, SVMRBFFamily, XGBFamily)


def family(name: str, unconstrained_lightgbm: bool = False) -> Any:
    """The family of a member. ``unconstrained_lightgbm=True`` returns, for the LightGBM members of the
    second construction, the variant without the constraint on the Hessian of the leaves."""
    table = {
        "LightGBM": lambda: LightGBMFamily("s2_unconstrained" if unconstrained_lightgbm else "s2"),
        "LightGBM_constrained": lambda: LightGBMFamily("s1"),
        "XGBoost": lambda: XGBFamily("XGBoost"),
        "CatBoost": lambda: CatBoostFamily(),
        "RandomForest": lambda: RandomForestFamily("RandomForest"),
        "ExtraTrees": lambda: ExtraTreesFamily("ExtraTrees"),
        "LogReg": lambda: LogRegFamily("LogReg"),
        "MLP": lambda: MLPFamily("MLP"),
        "SVM_RBF": lambda: SVMRBFFamily("SVM_RBF"),
    }
    if name not in table:
        raise KeyError(f"family {name!r} not registered (TabPFN is queried separately)")
    return table[name]()
