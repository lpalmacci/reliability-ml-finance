import math

import numpy as np
import pytest

from tavp.dgp.processes import GBM, GJRGarchT, Heston, RegimeSwitching
from tavp.utils import rng


def test_gbm_exact_moments():
    d = GBM()
    r = d.simulate_returns(20000, 50, rng(1, "gbm"))
    m, s = (d.mu - 0.5 * d.sigma**2) * d.dt, d.sigma * math.sqrt(d.dt)
    assert abs(r.mean() - m) < 4 * s / math.sqrt(r.size)
    assert abs(r.std() / s - 1) < 0.01


def test_heston_stationary_variance_and_drift():
    d = Heston()
    shape, scale = d.stationary_variance_law()
    assert math.isclose(shape * scale, d.theta_v)
    assert 2 * d.kappa * d.theta_v > d.xi**2  # Feller
    r = d.simulate_returns(20000, 20, rng(2, "heston"))
    # E[r^2] ~ theta_v dt under stationarity
    assert abs(np.mean(r**2) / (d.theta_v * d.dt) - 1) < 0.03
    assert abs(r.mean() - d.drift_per_step()) < 4 * r.std() / math.sqrt(r.size)


def test_heston_leverage_sign():
    r = Heston().simulate_returns(20000, 30, rng(3, "lev"))
    lev = np.corrcoef(r[:, :-1].ravel(), (r[:, 1:] ** 2).ravel())[0, 1]
    assert lev < 0


def test_garch_unconditional_variance_and_tails():
    d = GJRGarchT.from_annual()
    assert math.isclose(d.unconditional_variance, 0.18**2 / 252)
    assert math.isclose(d.persistence, 0.98)
    r = d.simulate_returns(4000, 250, rng(4, "garch"))
    assert abs(r.var() / d.unconditional_variance - 1) < 0.08
    k = np.mean((r - r.mean()) ** 4) / r.var() ** 2
    assert k > 3.5


def test_regime_frequencies():
    d = RegimeSwitching()
    assert math.isclose(d.stationary_bull, 0.75)
    states, _ = d.simulate_states_and_returns(20000, 30, rng(5, "ms"))
    assert abs((states == 0).mean() - 0.75) < 0.01


@pytest.mark.parametrize("d", [GBM(), Heston(), GJRGarchT.from_annual(), RegimeSwitching()])
def test_reproducible_and_hash(d):
    a = d.simulate_returns(10, 20, rng(7, "x"))
    b = d.simulate_returns(10, 20, rng(7, "x"))
    assert np.array_equal(a, b)
    assert len(d.hash()) == 12


def test_param_perturbations_zero_severity_is_identity():
    for d, kinds in [(Heston(), ["P3", "P4"]), (GJRGarchT.from_annual(), ["P2", "P3", "P4"]),
                     (RegimeSwitching(), ["P4"])]:
        for k in kinds:
            assert d.perturbed(k, 0.0).hash() == d.hash()


def test_perturbation_invariants():
    g = GJRGarchT.from_annual()
    for k in ["P2", "P3", "P4"]:
        p = g.perturbed(k, 1.0)
        assert math.isclose(p.unconditional_variance, g.unconditional_variance, rel_tol=1e-12)
    assert math.isinf(g.perturbed("P2", 1.0).nu)
    assert math.isclose(g.perturbed("P3", 1.0).persistence, g.persistence)
    assert g.perturbed("P3", 1.0).gamma == 0.0
    h = Heston()
    h4 = h.perturbed("P4", 1.0)
    var_v = lambda x: x.theta_v * x.xi**2 / (2 * x.kappa)
    assert math.isclose(var_v(h4), var_v(h))
    assert math.isclose(2 * h4.kappa * h4.theta_v / h4.xi**2, 2 * h.kappa * h.theta_v / h.xi**2)
    m = RegimeSwitching()
    m4 = m.perturbed("P4", 1.0)
    assert math.isclose(m4.stationary_bull, m.stationary_bull)
    assert math.isclose(1 / (1 - m4.p11), 0.2 / (1 - m.p11))
    with pytest.raises(NotImplementedError):
        GBM().perturbed("P3", 0.5)
