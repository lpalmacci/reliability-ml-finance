"""Plan of the members, subsample, and paths of the two versions of the pools."""
from collections import Counter

import numpy as np

from fraud.members import TABPFN_DIR, MEMBER_DIRS, apply_platt, member_path, platt_logodds
from fraud.plan import FAMILIES_S2, plan, subsample_indices


def test_plan_second_construction():
    pl = plan("carclaims", 0, "s2")
    assert len(pl) == 36
    assert Counter(m.family for m in pl) == {f: 4 for f in FAMILIES_S2}
    assert [m.index for m in pl if m.family == "TabPFN"] == [8, 17, 26, 35]
    assert all(0.60 <= m.fraction <= 0.85 for m in pl)
    assert len({tuple(m.key) for m in pl}) == 36


def test_plan_first_construction():
    pl = plan("ulb", 3, "s1")
    assert {m.family for m in pl} == {"LightGBM_constrained"}
    for m in pl:
        hp = m.hyperparameters
        assert 20 <= hp["min_data_in_leaf"] <= 80 and 8 <= hp["num_leaves"] < 32
        assert 0.02 <= hp["learning_rate"] <= 0.10 and 100 <= hp["num_boost_round"] <= 300
    assert plan("ulb", 3, "s1", n_members=6)[5] == pl[5]          # a member alone equals the member in the full plan
    assert plan("ulb", 3, "s1")[0] != plan("ulb", 4, "s1")[0]


def test_stratified_subsample():
    y = np.r_[np.zeros(950, int), np.ones(50, int)]
    idx = subsample_indices(y, 0.7, 123)
    assert len(idx) == 700 and len(np.unique(idx)) == 700
    assert y[idx].sum() == 35
    assert np.array_equal(idx, subsample_indices(y, 0.7, 123))


def test_calibration_is_monotone_for_a_positive_slope():
    rng = np.random.default_rng(0)
    zC = rng.normal(size=400)
    yC = (zC + rng.normal(size=400) > 1).astype(int)
    cal = platt_logodds(zC, yC)
    assert cal["a"] > 0
    zV = np.sort(rng.normal(size=300))
    assert np.all(np.diff(apply_platt(cal, zV)) >= 0)


def test_paths_of_the_two_versions():
    pl = plan("ulb", 0, "s2")
    lgbm = next(m for m in pl if m.family == "LightGBM")
    xgb = next(m for m in pl if m.family == "XGBoost")
    tab = next(m for m in pl if m.family == "TabPFN")
    assert member_path(lgbm, "unconstrained").is_relative_to(MEMBER_DIRS["unconstrained"])   # rebuilt without the constraint
    assert member_path(xgb, "unconstrained") == member_path(xgb, "final")                   # shared with the final pools
    assert member_path(tab, "final").is_relative_to(TABPFN_DIR)                             # saved service outputs
