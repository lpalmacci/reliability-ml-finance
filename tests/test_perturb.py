import numpy as np

from tavp.dgp.processes import GBM, GJRGarchT
from tavp.generators.base import OracleGenerator
from tavp.perturb.operators import scale_volatility, shift_drift, shuffle_fraction
from tavp.utils import rng


def test_shuffle_preserves_values_per_path():
    r = np.arange(40, dtype=float).reshape(2, 20)
    s = shuffle_fraction(r, 1.0, rng(0, "s"))
    assert not np.array_equal(r, s)
    for i in range(2):
        assert np.array_equal(np.sort(r[i]), np.sort(s[i]))


def test_partial_shuffle_moves_only_selected_positions():
    r = np.arange(100, dtype=float)[None, :]
    s = shuffle_fraction(r, 0.25, rng(1, "s"))
    assert (s != r).sum() <= 25


def test_zero_severity_is_identity():
    d = GJRGarchT.from_annual()
    for k in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        a = OracleGenerator(d, k, 0.0).sample_returns(50, 59, rng(3, "z"))
        b = OracleGenerator(d).sample_returns(50, 59, rng(3, "z"))
        assert np.array_equal(a, b), k


def test_scale_and_drift():
    d = GBM()
    r = d.simulate_returns(5000, 59, rng(4, "p"))
    m = d.drift_per_step()
    r5 = scale_volatility(r, 1.0, m)
    assert np.isclose((r5 - m).std() / (r - m).std(), 1.2)
    r6 = shift_drift(r, 1.0, d.dt)
    assert np.isclose((r6 - r).mean(), 0.05 * d.dt)


def test_target_hash_is_base_process():
    d = GJRGarchT.from_annual()
    assert OracleGenerator(d, "P3", 1.0).target_hash == d.hash()
