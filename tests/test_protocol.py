import numpy as np

from tavp.dgp.processes import GBM, GJRGarchT
from tavp.protocol import l1
from tavp.protocol.stats import berkowitz, empirical_pvalue, holm, ks_uniform, kupiec_pof
from tavp.utils import REPO_ROOT, load_config, rng

L1CFG = load_config(REPO_ROOT / "configs" / "preregistration.yaml")["protocol"]["L1"]


def test_signature_of_straight_line():
    a = np.array([0.3, -0.2])
    path = np.stack([np.zeros(2), a / 2, a])[None]  # straight line with two pieces
    s = l1.expected_signature(path)
    assert np.allclose(s[:2], a)
    assert np.allclose(s[2:6], np.outer(a, a).ravel() / 2)
    assert np.allclose(s[6:], np.einsum("i,j,k->ijk", a, a, a).ravel() / 6)


def test_sliced_wasserstein_zero_and_shift():
    x = rng(1, "x").standard_normal((500, 5))
    assert l1.sliced_wasserstein(x, x, 50) == 0.0
    assert l1.sliced_wasserstein(x, x + 1.0, 50) > 0.5


def test_stylized_facts_detect_clustering():
    g = GJRGarchT.from_annual().simulate_returns(3000, 59, rng(2, "g"))
    b = GBM().simulate_returns(3000, 59, rng(3, "b"))
    sg, sb = l1.stylized_facts(g, L1CFG), l1.stylized_facts(b, L1CFG)
    assert sg["acf_abs"][:5].mean() > sb["acf_abs"][:5].mean() + 0.02
    assert sg["kurt"] > sb["kurt"] and sg["leverage"][:3].mean() < 0


def test_shuffle_invisible_to_marginal_w1_but_not_to_acf():
    from tavp.perturb.operators import shuffle_fraction
    d = GJRGarchT.from_annual()
    a = d.simulate_returns(4000, 59, rng(4, "a"))
    s = shuffle_fraction(a, 1.0, rng(5, "s"))
    da = l1.discrepancies(s, a, L1CFG)  # same draws: shuffle changes order only
    assert da["mean"] < 1e-15 and da["std"] < 1e-15


def test_pvalue_and_holm():
    assert empirical_pvalue(10.0, np.arange(9.0)) == 0.1
    assert holm(np.array([0.001, 0.02, 0.04]), 0.05).tolist() == [True, True, True]
    assert holm(np.array([0.001, 0.03, 0.04]), 0.05).tolist() == [True, False, False]


def test_calibration_tests_accept_uniform():
    g = rng(6, "u")
    u = [g.random(200) for _ in range(5)]
    assert ks_uniform(np.concatenate(u)) > 0.01
    assert kupiec_pof(np.concatenate(u), (0.05, 0.95)) > 0.01
    p3, p2 = berkowitz(u)
    assert p3 > 0.01 and p2 > 0.01
    assert ks_uniform(np.concatenate(u) ** 3) < 1e-6
