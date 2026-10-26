"""Log-odds by family and calibration on the log-odds (Chapter 4, Members and their calibration)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from fraud.defaults import get_default_params
from fraud.families import EPS_LOGIT, logit_of_probability
from fraud.members import apply_platt, platt_logodds
from fraud.model_registry import family


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    X_raw = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                          "c": rng.choice(["u", "v", "w"], n).astype(object)})
    X_proc = np.column_stack([X_raw.a, X_raw.b, (X_raw.c == "u").astype(float)]).astype(np.float32)
    y = ((X_raw.a + 0.5 * X_raw.b + rng.normal(scale=0.8, size=n)) > 1.2).astype(int).to_numpy()
    return X_proc, X_raw, y, ["c"], [2]


MARGIN = ("LightGBM", "LightGBM_constrained", "XGBoost", "CatBoost", "LogReg")
PROBABILITY = ("RandomForest", "ExtraTrees")


def test_forests_are_deterministic_in_prediction(data):
    X_proc, X_raw, y, _, _ = data
    for name in ("RandomForest", "ExtraTrees"):
        fam = family(name)
        model = fam.build(1, 2, {})
        fam.fit(model, X_proc=X_proc, X_raw=X_raw, y=y)
        a = [fam.score(model, X_proc, X_raw) for _ in range(5)]
        assert all(np.array_equal(a[0], x) for x in a)


@pytest.mark.parametrize("name", MARGIN + ("SVM_RBF", "MLP") + PROBABILITY)
def test_log_odds_by_family(name, data):
    X_proc, X_raw, y, cat_cols, cat_idx = data
    fam = family(name)
    hp = ({"num_leaves": 8, "learning_rate": 0.05, "max_depth": 4, "feature_fraction": 0.8, "bagging_fraction": 0.8,
           "lambda_l2": 1.0, "min_data_in_leaf": 20, "num_boost_round": 50} if name == "LightGBM_constrained" else {})
    model = fam.build(1, 2, hp)
    if name == "CatBoost":
        model.set_params(iterations=50)
    fam.fit(model, X_proc=X_proc, X_raw=X_raw, y=y, cat_cols=cat_cols, cat_idx=cat_idx)
    z = fam.log_odds(model, X_proc, X_raw)
    r = np.asarray(fam.score(model, X_proc, X_raw), dtype=float)
    assert z.shape == (len(y),) and np.all(np.isfinite(z))
    if name in MARGIN:                       # the margin: sigma(z) is the probability of the model
        assert np.allclose(expit(z), r, atol=1e-6)
        assert not np.allclose(z, logit_of_probability(r)) or np.all((r > EPS_LOGIT) & (r < 1 - EPS_LOGIT))
    elif name == "SVM_RBF":                  # the decision function
        assert np.array_equal(z, model.decision_function(X_proc))
    elif name == "MLP":                      # margin: sigma(z) = predict_proba where it is not saturated
        not_sat = (r > 1e-6) & (r < 1 - 1e-6)
        assert not_sat.sum() > 50
        assert np.allclose(expit(z[not_sat]), r[not_sat], rtol=1e-4, atol=1e-6)
        assert np.all(np.sign(z[~not_sat]) == np.sign(r[~not_sat] - 0.5))
    else:                                    # logit of the raw probability, clipped at 1e-6
        assert np.array_equal(z, logit(np.clip(r, 1e-6, 1 - 1e-6)))


def test_logit_of_probability_clips():
    z = logit_of_probability(np.array([0.0, 0.5, 1.0]))
    assert np.allclose(z, [logit(1e-6), 0.0, logit(1 - 1e-6)])


def test_calibration_uses_the_calibration_part_only():
    rng = np.random.default_rng(3)
    zC = rng.normal(size=500)
    yC = (zC + rng.normal(size=500) > 1).astype(int)
    cal = platt_logodds(zC, yC)
    assert cal["method"] == "maximum_likelihood"
    zV = rng.normal(size=300)                 # validation and test enter only through the application
    assert np.allclose(apply_platt(cal, zV), expit(cal["a"] * zV + cal["b"]))
    assert platt_logodds(zC, yC) == cal
    y2 = yC.copy(); y2[:50] = 1 - y2[:50]
    assert platt_logodds(zC, y2)["a"] != cal["a"]


def test_unpenalised_fit_is_the_mle():
    rng = np.random.default_rng(4)
    zC = rng.normal(size=2000)
    yC = (rng.random(2000) < expit(1.7 * zC - 2.0)).astype(int)
    cal = platt_logodds(zC, yC)
    assert abs(cal["a"] - 1.7) < 0.25 and abs(cal["b"] + 2.0) < 0.25


def test_platt_targets_only_without_convergence():
    zC = np.r_[np.linspace(-3, -0.1, 100), np.linspace(0.1, 3, 20)]   # perfect separation
    yC = np.r_[np.zeros(100, int), np.ones(20, int)]
    cal = platt_logodds(zC, yC)
    assert cal["method"] == "platt_targets" and cal["reason"] == "separation"
    assert cal["platt_targets_converged"]
    assert np.isfinite(cal["a"]) and cal["a"] > 0
    p = apply_platt(cal, zC)
    assert p.max() < 1.0 and p.min() > 0.0            # the targets keep the probabilities away from 0 and 1
    rng = np.random.default_rng(5)
    z2 = rng.normal(size=300)
    assert platt_logodds(z2, (z2 + rng.normal(size=300) > 0).astype(int))["method"] == "maximum_likelihood"


def test_margin_of_the_perceptron_separates_saturated_cases():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(800, 4)).astype(np.float32)
    y = (3 * X[:, 0] + rng.normal(scale=0.3, size=800) > 2.5).astype(int)
    fam = family("MLP")
    model = fam.build(3, 4, {})
    fam.fit(model, X_proc=X, X_raw=None, y=y)
    r = np.asarray(fam.score(model, X, None), dtype=float)
    z = fam.log_odds(model, X, None)
    sat = r < 1e-6
    if sat.sum() > 1:                         # where the clipped probability would tie, the margin keeps the order
        assert len(np.unique(z[sat])) == sat.sum()


def test_lightgbm_constraint_in_the_second_construction_only():
    p = get_default_params(1)
    assert p["lgbm"]["min_sum_hessian_in_leaf"] == 1.0
    assert "min_sum_hessian_in_leaf" not in p["lgbm_s1"]                       # the first construction is unchanged
    assert "min_sum_hessian_in_leaf" not in get_default_params(1, unconstrained=True)["lgbm"]
    assert family("LightGBM", unconstrained_lightgbm=True).variant == "s2_unconstrained"
