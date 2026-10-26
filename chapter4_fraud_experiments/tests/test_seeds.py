"""Seed keys: distinct for distinct conditions, stable, independent of the order."""
import itertools

import numpy as np
import pytest

from fraud.seeds import DATASET, PURPOSE, STRATEGY, key, rng_for


def _all_keys():
    for purpose, d in itertools.product(PURPOSE, DATASET):
        if purpose == "partition":
            for p in range(5):
                yield (purpose, d, p, None, 0, 0)
        elif purpose == "member":
            for p, s, i in itertools.product(range(5), STRATEGY, range(36)):
                yield (purpose, d, p, s, 0, i)
        elif purpose == "draw":
            for p, s, K, r in itertools.product(range(5), STRATEGY, (3, 6, 12, 24, 36), range(20)):
                yield (purpose, d, p, s, K, r)
        else:
            for p in range(5):
                yield (purpose, d, p, None, 0, 0)


def test_protocol_codes():
    assert DATASET == {"carclaims": 1, "ulb": 2}
    assert STRATEGY == {"s1": 1, "s2": 2}
    assert PURPOSE == {"partition": 1, "member": 2, "draw": 3, "bootstrap": 4}
    assert key("partition", "ulb", 3) == [1, 2, 3, 0, 0, 0]
    assert key("draw", "carclaims", 0, "s2", 12, 7) == [3, 1, 0, 2, 12, 7]


def test_different_keys_give_different_generators():
    keys = list(_all_keys())
    assert len({tuple(key(*c)) for c in keys}) == len(keys)
    firsts = {tuple(rng_for(*c).integers(0, 2**63 - 1, size=2)) for c in keys}
    assert len(firsts) == len(keys)


def test_same_key_same_numbers_in_any_order():
    a = rng_for("member", "ulb", 2, "s1", 0, 5).random(10)
    _ = [rng_for("member", "ulb", 2, "s1", 0, i).random(10) for i in range(36)]
    b = rng_for("member", "ulb", 2, "s1", 0, 5).random(10)
    assert np.array_equal(a, b)


def test_negative_fields_rejected():
    with pytest.raises(ValueError):
        key("partition", "ulb", -1)
