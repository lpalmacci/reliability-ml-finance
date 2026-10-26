"""Plan of the members (Chapter 4, Members and their calibration): who each member is, before any training.

Each member has its own generator ``rng_for("member", dataset, p, strategy, 0, index)``; from it, in this
order: the fraction of the training part in its subsample, U[0.60, 0.85], the seed of the subsample,
the seed of the model, an auxiliary seed (the 50/50 oversampling of the perceptron) and, in the first
construction, the hyperparameters of LightGBM drawn from the distribution given in the Appendix
(The models of the ensembles).

- Strategy 1 (S1): 36 boosted trees of LightGBM with randomised hyperparameters ("LightGBM_constrained").
- Strategy 2 (S2): nine families of different kinds, four members each; member i belongs to
  FAMILIES_S2[i % 9].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from fraud.seeds import integer_seed, key, rng_for

M = 36
FRACTION = (0.60, 0.85)

FAMILIES_S2: tuple[str, ...] = (
    "LightGBM", "XGBoost", "CatBoost", "RandomForest", "ExtraTrees",
    "LogReg", "MLP", "SVM_RBF", "TabPFN",
)
FAMILY_S1 = "LightGBM_constrained"


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _sample_lgbm_constrained(rng: np.random.Generator) -> dict[str, Any]:
    """The distribution of the hyperparameters of the first construction (Appendix, The models of the
    ensembles). Integer ranges include both ends."""
    return dict(
        num_leaves=int(rng.integers(8, 32)),
        learning_rate=_log_uniform(rng, 0.02, 0.10),
        max_depth=int(rng.choice([4, 6, 8])),
        feature_fraction=float(rng.uniform(0.5, 1.0)),
        bagging_fraction=float(rng.uniform(0.6, 1.0)),
        lambda_l2=_log_uniform(rng, 1.0, 10.0),
        min_data_in_leaf=int(rng.integers(20, 81)),
        num_boost_round=int(rng.integers(100, 301)),
    )


@dataclass(frozen=True)
class Member:
    dataset: str
    partition: int
    strategy: str
    index: int
    family: str
    fraction: float
    subsample_seed: int
    model_seed: int
    auxiliary_seed: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> list[int]:
        return key("member", self.dataset, self.partition, self.strategy, 0, self.index)

    @property
    def name(self) -> str:
        return f"{self.strategy}_m{self.index:02d}_{self.family}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key}


def member(dataset: str, p: int, strategy: str, i: int) -> Member:
    rng = rng_for("member", dataset, p, strategy, 0, i)
    fraction = float(rng.uniform(*FRACTION))
    s_sub, s_mod, s_aux = integer_seed(rng), integer_seed(rng), integer_seed(rng)
    if strategy == "s1":
        family, hp = FAMILY_S1, _sample_lgbm_constrained(rng)
    elif strategy == "s2":
        family, hp = FAMILIES_S2[i % len(FAMILIES_S2)], {}
    else:
        raise ValueError(strategy)
    return Member(dataset, p, strategy, i, family, fraction, s_sub, s_mod, s_aux, hp)


def plan(dataset: str, p: int, strategy: str, n_members: int = M) -> list[Member]:
    return [member(dataset, p, strategy, i) for i in range(n_members)]


def subsample_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Stratified subsample without replacement of the training part (sorted indices)."""
    y = np.asarray(y).astype(int)
    if frac >= 1.0:
        return np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return np.sort(idx)
