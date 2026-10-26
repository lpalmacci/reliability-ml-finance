"""Partitions: disjoint, covering the dataset, no group split, proportions and strata."""
import numpy as np
import pandas as pd
import pytest

from fraud.data import (N_PARTITIONS, PARTS, PROPORTIONS, group_statistics, groups, load, load_partition, partition,
                        partition_path)
from fraud.seeds import rng_for


def _synthetic(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.integers(0, 40, n), "b": rng.choice(["x", "y", "z"], n),
                      "c": np.where(rng.random(n) < 0.05, np.nan, rng.integers(0, 3, n))})
    y = (rng.random(n) < 0.06).astype(int)
    return X, y


def _check(parts, y, g):
    allp = np.concatenate([parts[k] for k in PARTS])
    assert len(allp) == len(y) and len(np.unique(allp)) == len(y)       # disjoint and covering
    part = np.empty(len(y), int)
    for j, k in enumerate(PARTS):
        part[parts[k]] = j
    G = int(g.max()) + 1                                                  # no group split
    lo, hi = np.full(G, 9), np.full(G, -1)
    np.minimum.at(lo, g, part)
    np.maximum.at(hi, g, part)
    assert np.array_equal(lo, hi)


def test_groups_with_nan_and_discordant_labels():
    X = pd.DataFrame({"a": [1, 1, 2, 2, np.nan, np.nan], "b": ["u", "u", "u", "v", "u", "u"]})
    y = np.array([0, 1, 0, 0, 1, 1])
    g = groups(X)
    assert g[0] == g[1] and g[4] == g[5] and len(np.unique(g)) == 4
    s = group_statistics(y, g)
    assert s["groups_with_several_rows"] == 2 and s["discordant_groups"] == 1


def test_synthetic_partition_by_groups():
    X, y = _synthetic()
    g = groups(X)
    assert group_statistics(y, g)["groups_with_several_rows"] > 0
    parts = partition(y, g, rng_for("partition", "carclaims", 0))
    _check(parts, y, g)
    for k, q in zip(PARTS, PROPORTIONS):
        assert abs(len(parts[k]) / len(y) - q) < 0.03


def test_partition_is_deterministic():
    X, y = _synthetic()
    g = groups(X)
    a = partition(y, g, rng_for("partition", "ulb", 2))
    b = partition(y, g, rng_for("partition", "ulb", 2))
    c = partition(y, g, rng_for("partition", "ulb", 3))
    assert all(np.array_equal(a[k], b[k]) for k in PARTS)
    assert not all(np.array_equal(a[k], c[k]) for k in PARTS)


@pytest.mark.parametrize("dataset", ["carclaims", "ulb"])
def test_saved_partitions(dataset):
    if not partition_path(dataset, 0).exists():
        pytest.skip("partitions not created yet")
    try:
        ld = load(dataset)
    except FileNotFoundError:
        pytest.skip("dataset not available (decompress data/ulb_creditcard.csv.gz)")
    g = groups(ld.X_raw)
    for p in range(N_PARTITIONS[dataset]):
        parts = load_partition(dataset, p)
        _check(parts, ld.y, g)
        ref = partition(ld.y, g, rng_for("partition", dataset, p))        # regenerated on its own: same indices
        assert all(np.array_equal(parts[k], ref[k]) for k in PARTS)
        for k, q in zip(PARTS, PROPORTIONS):
            assert abs(len(parts[k]) / len(ld.y) - q) < 0.001
            assert abs(ld.y[parts[k]].sum() / ld.y.sum() - q) < 0.01     # stratified by class
