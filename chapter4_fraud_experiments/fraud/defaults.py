"""Hyperparameters of the members (Chapter 4, Members and their calibration; Appendix A.2).

The rule of the protocol: the defaults of each library, with these exceptions only.

- class weights: balanced for the logistic regression, the support vector machine, the random forest
  and the extra trees (``balanced_subsample``, which without bootstrap equals ``balanced``); balanced
  weights for CatBoost (set at fit time); ``scale_pos_weight = n0/n1`` for XGBoost (at fit time); 50/50
  oversampling for the perceptron (at fit time, with the auxiliary seed of the member); no weights for
  LightGBM;
- random forest and extra trees with ``min_samples_leaf = 2``;
- support vector machine with a Gaussian kernel: C = 1, gamma "scale", ``probability=False``;
- LightGBM and CatBoost with the default number of iterations of the library (100 and 1000);
- logistic regression with ``max_iter = 1000`` (with the default of 100 the solver does not converge);
- LightGBM in the second construction only: ``min_sum_hessian_in_leaf = 1`` (the constraint of the
  section "Decisions taken after the pilot"). ``unconstrained=True`` gives the members without it, the
  ones that degenerated on ULB, for the comparison kept in the replication material.

``n_jobs`` and ``thread_count`` are not hyperparameters of the models and do not change the results.
``allow_writing_files=False`` stops CatBoost from writing its catboost_info folder.

The first construction uses an explicit binary objective, ``bagging_freq=1`` (so that the sampled
``bagging_fraction`` takes effect) and ``lambda_l1=0``, with the hyperparameters drawn by ``fraud.plan``.
"""
from __future__ import annotations

from typing import Any


def get_default_params(seed: int, unconstrained: bool = False) -> dict[str, dict[str, Any]]:
    """Parameters by family; ``seed`` is the model seed of the member."""
    lgbm = dict(objective="binary", verbosity=-1, seed=seed)       # lgb.train needs the binary objective declared
    if not unconstrained:
        # leaves with a total Hessian of at least one, the default of XGBoost (min_child_weight = 1): with the
        # LightGBM default (1e-3) leaves holding a handful of cases took extreme values on ULB
        lgbm["min_sum_hessian_in_leaf"] = 1.0
    return {
        "xgb": dict(random_state=seed),
        "logreg": dict(class_weight="balanced", max_iter=1000, random_state=seed),
        "svm_rbf": dict(C=1.0, gamma="scale", probability=False, class_weight="balanced", random_state=seed),
        "catboost": dict(random_seed=seed, verbose=False, allow_writing_files=False),
        "lgbm": lgbm,
        "random_forest": dict(min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "extra_trees": dict(min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "mlp": dict(random_state=seed),
        # first construction: the base parameters; the rest is drawn by fraud.plan
        "lgbm_s1": dict(objective="binary", metric="binary_logloss", bagging_freq=1, lambda_l1=0.0,
                        verbosity=-1, seed=seed),
    }


def merge_params(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Shallow merge; ``overrides`` wins on key collision."""
    merged = dict(base)
    merged.update(overrides)
    return merged
