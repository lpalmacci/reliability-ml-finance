"""Seeds: one random generator for each experimental condition.

    np.random.default_rng([purpose, dataset, partition, strategy, K, index])

with explicit integer codes and 0 in the fields that do not apply (for instance strategy and K for a
partition). The key depends only on the condition, so a condition gives the same numbers whether it
is run alone or with the others, in any order and with any number of parallel processes. The integer
codes below are part of the definition of the experiment and must not be changed.
"""
from __future__ import annotations

import numpy as np

DATASET = {"carclaims": 1, "ulb": 2}
STRATEGY = {"s1": 1, "s2": 2}
PURPOSE = {"partition": 1, "member": 2, "draw": 3, "bootstrap": 4}


def key(purpose: str, dataset: str, partition: int = 0, strategy: str | None = None,
        K: int = 0, index: int = 0) -> list[int]:
    """The integer key of the condition, in the order fixed by the protocol."""
    if partition < 0 or K < 0 or index < 0:
        raise ValueError("partition, K and index must be >= 0")
    return [PURPOSE[purpose], DATASET[dataset], int(partition),
            STRATEGY[strategy] if strategy is not None else 0, int(K), int(index)]


def rng_for(purpose: str, dataset: str, partition: int = 0, strategy: str | None = None,
            K: int = 0, index: int = 0) -> np.random.Generator:
    """Generator of the condition."""
    return np.random.default_rng(key(purpose, dataset, partition, strategy, K, index))


def integer_seed(rng: np.random.Generator) -> int:
    """An integer seed for the libraries that do not accept a Generator (in [0, 2**31 - 1))."""
    return int(rng.integers(0, 2**31 - 1))
