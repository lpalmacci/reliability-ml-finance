import numpy as np

from tavp.dgp.processes import GBM, GJRGarchT
from tavp.generators.base import OracleGenerator
from tavp.perturb.operators import (EXTREME_FACTOR, extreme_returns, ma1_coefficient, negative_autocorrelation,
                                   scale_volatility, shift_drift, shuffle_fraction)
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
    for k in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]:
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


def test_negative_autocorrelation_keeps_variance_and_sets_acf1():
    assert np.isclose(ma1_coefficient(-0.1) / (1 + ma1_coefficient(-0.1) ** 2), -0.1)
    d = GBM()
    m = d.drift_per_step()
    r = d.simulate_returns(20000, 59, rng(5, "p7"))
    r7 = negative_autocorrelation(r, 1.0, m, rng(6, "p7"))
    u = r7 - m
    acf1 = (u[:, 1:] * u[:, :-1]).mean() / (u**2).mean()
    assert abs(acf1 + 0.10) < 0.005
    assert abs(u.std() / (r - m).std() - 1) < 0.01
    assert abs(u[:, 0].std() / u[:, 1:].std() - 1) < 0.02
    vr = u.sum(axis=1).var() / (59 * u.var())
    assert abs(vr - (1 + 2 * -0.10 * 58 / 59)) < 0.02


def test_extreme_returns_keep_variance_and_add_extremes():
    d = GBM()
    m = d.drift_per_step()
    r = d.simulate_returns(20000, 59, rng(7, "p8"))
    r8 = extreme_returns(r, 2.0, m, rng(8, "p8"))
    u, u0 = r8 - m, r - m
    assert abs(u.std() / u0.std() - 1) < 0.01
    ratio = np.abs(u / u0)
    assert np.isclose(ratio.max() / ratio.min(), EXTREME_FACTOR)
    assert abs((ratio > ratio.min() * 1.5).mean() - 0.004) < 0.0005
    acf1 = (u[:, 1:] * u[:, :-1]).mean() / (u**2).mean()
    assert abs(acf1) < 0.01


def test_matched_severities_invert_the_defect():
    from tavp.experiments.e14_qgan_defects import p7_severity, p8_rate, p8_severity

    u = np.abs(GBM().simulate_returns(5000, 59, rng(9, "m"))).ravel()
    thr = np.quantile(u, 0.9999)
    target = 5 * p8_rate(u, thr, 0.0)
    assert np.isclose(p8_rate(u, thr, p8_severity(u, thr, target)), target, rtol=1e-6)
    assert p8_severity(u, thr, 0.5 * p8_rate(u, thr, 0.0)) == 0.0
    assert np.isclose(1 + 2 * (-0.1 * p7_severity(0.8, 59)) * 58 / 59, 0.8)
    assert p7_severity(1.05, 59) == 0.0
