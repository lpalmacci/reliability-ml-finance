"""Tests of the Confidex library. Run them with pytest, or with `python tests/test_confidex.py`."""
import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, expit, logit
from scipy.stats import beta as B

import confidex as cx
from confidex import ModelConfig, CriterionConfig

# the ten constructed ensembles of Chapter 3 (An illustration on ten ensembles)
TEN = np.array([
    [0.95, 0.96, 0.94, 0.95, 0.95, 0.96, 0.94, 0.95, 0.96, 0.95], [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04, 0.05],
    [0.48, 0.50, 0.49, 0.51, 0.50, 0.52, 0.49, 0.48, 0.51, 0.50], [0.12, 0.11, 0.13, 0.12, 0.14, 0.13, 0.12, 0.11, 0.13, 0.12],
    [0.88, 0.87, 0.89, 0.88, 0.86, 0.89, 0.88, 0.87, 0.89, 0.88], [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.9],
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9],
    [0.2, 0.8, 0.5, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1, 0.5], [0.02, 0.99, 0.01, 0.98, 0.03, 0.97, 0.02, 0.96, 0.01, 0.95]])
# posterior means of c for the ten ensembles in the first version of the library (0.1.0)
CBAR_V010 = [0.964904, 0.964904, 0.976265, 0.972710, 0.970534, 0.233140, 0.017244, 0.131480, 0.268060, 0.067054]


def _synthetic(n, K, seed):
    """Ensembles drawn from the model itself: mu uniform, c ~ Beta(2, 3), lambda_0 = K."""
    rng = np.random.default_rng(seed)
    mu, c = rng.uniform(0.02, 0.98, n), rng.beta(2, 3, n)
    lam = K * c / (1 - c)
    return np.clip(rng.beta((mu * lam)[:, None], ((1 - mu) * lam)[:, None], (n, K)), 1e-6, 1 - 1e-6)


def _quad_mean(x, a0=2.0, b0=3.0):
    """Posterior mean of c by adaptive quadrature on the logit scale, an independent check of the grid."""
    K, mu = len(x), x.mean(); S1, S2 = np.log(x).sum(), np.log1p(-x).sum()
    def lp(t):
        c = expit(t); lam = K * c / (1 - c); a, b = mu * lam, (1 - mu) * lam
        return (a - 1) * S1 + (b - 1) * S2 - K * betaln(a, b) + B.logpdf(c, a0, b0) + np.log(c) + np.log1p(-c)
    lo, hi = logit(1e-6), logit(1 - 1e-6); ts = np.linspace(lo, hi, 4001)
    M = max(lp(t) for t in ts); f = lambda t: np.exp(lp(t) - M)
    Z = quad(f, lo, hi, limit=400, points=ts[::400])[0]
    return quad(lambda t: expit(t) * f(t), lo, hi, limit=400, points=ts[::400])[0] / Z


def test_grid_weights_are_trapezoidal():
    gr = cx.LogitGrid.build(1e-6, 1000)
    assert abs(gr.w.sum() - (gr.g[-1] - gr.g[0])) < 1e-12


def test_posterior_mean_matches_quadrature():
    for K in (3, 6, 12):
        X = _synthetic(15, K, seed=K)
        post = cx.fit_posterior(X)
        err = max(abs(post.c_mean[i] - _quad_mean(X[i])) for i in range(len(X)))
        assert err < 2e-5, (K, err)


def test_backward_compatible_with_v010():
    post = cx.fit_posterior(TEN)
    assert np.max(np.abs(post.c_mean - CBAR_V010)) < 1e-5
    assert not post.degenerate.any()


def test_ten_ensembles_verdicts_unchanged_by_directional_criterion():
    post = cx.fit_posterior(TEN); law = cx.member_law(post)
    for tau in (0.1, 0.5):
        a = cx.confidence_score(post, law, tau=tau, directional=False, predictive=False)["confident"]
        b = cx.confidence_score(post, law, tau=tau, predictive=False)["confident"]
        assert np.array_equal(a, b)


def test_ties_are_detected_after_clipping():
    P = np.array([[0.0, 1e-8, 0.0, 2e-8, 0.0, 0.0], [0.3] * 6])
    post = cx.fit_posterior(P)
    assert post.tie.all() and post.degenerate.all()
    out = cx.confidence_score(post, cx.member_law(post), tau=0.5, predictive=False)
    assert (out["status"] == "degenerate").all()


def test_near_ties_are_flagged_at_border_not_degenerate():
    near = 0.3 + 1e-5 * np.linspace(-1, 1, 6)
    fine = 0.3 + 2e-3 * np.linspace(-1, 1, 6)
    post = cx.fit_posterior(np.vstack([near, fine]))
    assert post.at_border[0] and not post.at_border[1]
    assert not post.degenerate.any()


def test_border_verdict_is_conservative():
    # with a wider grid the law becomes narrower: a 'confident' verdict never turns 'not_confident'
    rng = np.random.default_rng(9)
    X = np.exp(np.log(0.38) + 1e-4 * rng.standard_normal((300, 12)))
    p6 = cx.fit_posterior(X); p9 = cx.fit_posterior(X, ModelConfig(grid_bound=1e-9, grid_points=1500))
    assert p6.at_border.mean() > 0.5
    for tau in (0.38002, 0.38005, 0.3801):
        c6 = cx.confidence_score(p6, cx.member_law(p6), tau=tau, directional=False, predictive=False)["confident"]
        c9 = cx.confidence_score(p9, cx.member_law(p9), tau=tau, directional=False, predictive=False)["confident"]
        assert not (c6 & ~c9).any()


def test_band_equivalence():
    X = _synthetic(500, 6, seed=1)
    post = cx.fit_posterior(X); law = cx.member_law(post)
    for tau in (0.1, 0.5):
        out = cx.confidence_score(post, law, tau=tau, directional=False, predictive=False)
        inside = (law.quantile(0.40) <= tau) & (tau <= law.quantile(0.60))
        assert np.array_equal(out["confident"], ~inside)


def test_gate_reaches_the_target_abstention():
    post = cx.fit_posterior(_synthetic(4000, 6, seed=2))
    v = cx.calibrate_gate(post, CriterionConfig(abstention=0.10))
    rate = 1 - cx.admissible(post, v)[~post.degenerate].mean()
    assert abs(rate - 0.10) < 0.01


def test_predictive_close_to_plugin_when_posterior_is_concentrated():
    X = _synthetic(800, 12, seed=3)
    post = cx.fit_posterior(X, taus=[0.5]); law = cx.member_law(post)
    F = law.cdf(0.5); Fp = post.F_pred[0.5]
    tight = post.c_var < np.quantile(post.c_var, 0.25)
    assert np.median(np.abs(F - Fp)[tight]) < 0.02


def test_full_bayes_location_concentrates_with_K():
    X = _synthetic(20, 60, seed=4)
    full = cx.fit_posterior_full(X, points_mu=150, points_c=150)
    assert np.median(np.abs(full.mu_mean - X.mean(axis=1))) < 0.01


def test_directional_criterion_rejects_decisions_contradicted_by_the_law():
    P = np.array([[0.001, 0.001, 0.05]])           # mean 0.017 >= 0.01: positive decision, 2 members of 3 below
    post = cx.fit_posterior(P); law = cx.member_law(post)
    band = cx.confidence_score(post, law, tau=0.01, directional=False, predictive=False)
    dirn = cx.confidence_score(post, law, tau=0.01, predictive=False)
    assert band["confident"][0] and band["opposite_side"][0]
    assert not dirn["confident"][0]


def test_directional_equals_band_when_law_is_on_the_decision_side():
    X = _synthetic(2000, 6, seed=5)
    post = cx.fit_posterior(X); law = cx.member_law(post)
    a = cx.confidence_score(post, law, tau=0.5, directional=False, predictive=False)
    b = cx.confidence_score(post, law, tau=0.5, predictive=False)
    same = ~a["opposite_side"]
    assert np.array_equal(a["confident"][same], b["confident"][same])


def test_keff_on_deviations():
    rng = np.random.default_rng(6); n, K = 5000, 9
    base = rng.uniform(0.1, 0.9, (n, 1))
    indep = np.clip(base + 0.05 * rng.standard_normal((n, K)), 1e-6, 1 - 1e-6)
    shared = rng.standard_normal((n, 3)).repeat(3, axis=1)          # three groups of three replicas
    grouped = np.clip(base + 0.05 * (0.3 * rng.standard_normal((n, K)) + shared), 1e-6, 1 - 1e-6)
    assert abs(cx.keff_deviations(indep)["keff"] - K) < 0.3
    out = cx.keff_deviations(grouped, families=np.repeat([0, 1, 2], 3))
    assert 2.5 < out["keff"] < 4.0 and out["rho_within"] > out["rho_between"]


def test_predictive_variance_is_the_mixture_variance():
    X = _synthetic(50, 6, seed=10)
    post = cx.fit_posterior(X, keep_masses=True)
    lam = post.lambda0 * post.grid.g / (1 - post.grid.g)
    comp_var = post.mu[:, None] * (1 - post.mu[:, None]) / (lam[None, :] + 1)
    assert np.allclose(post.sigma_pred ** 2, (post.masses * comp_var).sum(axis=1))


def test_assess_runs_both_laws_and_they_mostly_agree():
    X = _synthetic(3000, 12, seed=11)
    out = cx.assess(X, tau=0.5)
    assert out["status"].shape == (3000,)
    assert np.mean(out["confident"] == out["confident_plugin"]) > 0.98


def test_predictive_requires_the_threshold_at_fit_time():
    post = cx.fit_posterior(_synthetic(10, 6, seed=12))
    try:
        cx.confidence_score(post, None, tau=0.5)
        raise AssertionError("it should have failed")
    except ValueError:
        pass


def test_shape_regime_labels():
    law = cx.MemberLaw(a=np.array([0.5, 0.5, 2.0, 2.0]), b=np.array([0.5, 2.0, 0.5, 2.0]), sigma=np.zeros(4))
    assert cx.shape_regime(law).tolist() == ["U", "J", "J", "unimodal"]


if __name__ == "__main__":
    import inspect
    import sys
    tests = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction) if n.startswith("test_")]
    for name, f in tests:
        f(); print(f"ok  {name}")
