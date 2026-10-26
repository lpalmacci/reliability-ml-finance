"""Draws (rotation by family), multinomial bootstrap weights, readings and budgets."""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import logit

from fraud import evaluate as ev
from fraud.plan import FAMILIES_S2
from fraud.readings import SIMPLE, auroc, baselines, captured, order
from fraud.seeds import rng_for

FAM = np.array([FAMILIES_S2[i % 9] for i in range(36)])


def test_rotation_by_family():
    for K in (3, 6, 9, 12, 24):
        for r in range(20):
            cols = ev.draw("s2", 36, K, FAM, rng_for("draw", "carclaims", 0, "s2", K, r))
            assert len(set(cols)) == K
            count = Counter(FAM[cols])
            assert len(count) == min(K, 9)                          # min(K, 9) distinct families
            assert max(count.values()) - min(count.values()) <= 1   # at most one apart
    cols = ev.draw("s2", 36, 3, FAM, rng_for("draw", "ulb", 1, "s2", 3, 0))
    assert len(set(FAM[cols])) == 3                                 # K = 3: three different families


def test_uniform_draw_and_full_pool():
    fam1 = np.array(["LightGBM_constrained"] * 36)
    cols = ev.draw("s1", 36, 12, fam1, rng_for("draw", "carclaims", 0, "s1", 12, 0))
    assert len(set(cols)) == 12
    assert np.array_equal(ev.draw("s2", 36, 36, FAM, np.random.default_rng(0)), np.arange(36))
    lst = ev.draws("carclaims", 0, "s1", fam1)
    assert len(lst) == 4 * 20 + 1
    K, r, c = lst[25]                                               # the same draw alone or within the full plan
    assert np.array_equal(c, ev.draw("s1", 36, K, fam1, rng_for("draw", "carclaims", 0, "s1", K, r)))


def test_multinomial_bootstrap_weights():
    W = ev.bootstrap_weights("carclaims", 0, 500, B_=200)
    assert W.shape == (200, 500)
    assert np.all(W >= 0) and np.all(W == np.round(W))
    assert np.all(W.sum(axis=1) == 500)                             # every replicate resamples n cases
    assert abs(W.mean() - 1.0) < 1e-6
    assert np.array_equal(W, ev.bootstrap_weights("carclaims", 0, 500, B_=200))
    assert not np.array_equal(W, ev.bootstrap_weights("carclaims", 1, 500, B_=200))


def test_weighted_share_equals_resampling():
    rng = np.random.default_rng(1)
    n, k = 300, 40
    err = (rng.random(n) < 0.2).astype(int)
    ordn = order(rng.random(n))
    W = ev.bootstrap_weights("ulb", 0, n, B_=50)
    sel = np.zeros(n, np.float32)
    sel[ordn[:k]] = err[ordn[:k]]
    q = (W @ sel) / (W @ err.astype(np.float32))
    for b in range(5):                                              # the same with the cases replicated explicitly
        cnt = W[b].astype(int)
        num = sum(cnt[i] * err[i] for i in ordn[:k])
        den = sum(cnt[i] * err[i] for i in range(n))
        assert abs(q[b] - num / den) < 1e-6
    assert abs(captured(ordn, err, k) - err[ordn[:k]].sum() / err.sum()) < 1e-12


def test_readings_and_studentised_margins_with_zero_dispersion():
    P = np.array([[0.2, 0.2, 0.2], [0.1, 0.3, 0.2], [0.6, 0.7, 0.8]])
    d = baselines(P, P.mean(axis=1), 0.25)
    assert d["studentised"][0] == -np.inf and d["logit studentised"][0] == -np.inf
    assert np.all(np.isfinite(d["distance"]))
    assert order(d["studentised"])[-1] == 0                         # s = 0 -> last


def test_auroc():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5          # ties count one half


def test_budgets():
    b = ev.budgets(0.05, 1000, 37)
    assert b == {"rho=0.5": 25, "rho=1": 50, "rho=2": 100, "k_alpha": 37,
                 "share=1%": 10, "share=5%": 50, "share=10%": 100}


def test_split_of_the_budget_by_class():
    assert ev.split_budget(100, 30, 10, 500, 500) == (75, 25)       # in proportion to the validation errors
    assert ev.split_budget(100, 30, 10, 40, 500) == (40, 60)        # within the size of the class
    assert ev.split_budget(7, 0, 0, 50, 50) == (4, 3)               # no errors: half and half
    k1, k0 = ev.split_budget(53, 17, 29, 1000, 1000)
    assert k1 + k0 == 53


def test_order_within_class():
    doubt = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.95])
    assert list(ev.order_within(np.array([1, 2, 4]), doubt)) == [2, 4, 1]
    pbar = np.array([0.12, 0.30, 0.15, 0.5, 0.11])                  # within a class the two distances give the same order
    tau = 0.1
    d1, d2 = -np.abs(pbar - tau), -np.abs(logit(pbar) - logit(tau))
    every = np.arange(5)
    assert np.array_equal(ev.order_within(every, d1), ev.order_within(every, d2))


def test_choice_within_a_class_without_validation_errors():
    rows = [{"strategy": "s2", "threshold": "tau_F1", "K": 36, "class": c, "reading": r,
             "captured": (float("nan") if c == 1 else
                          {"distance": 0.1, "studentised": 0.3, "logit distance": 0.1, "logit studentised": 0.2}[r])}
            for c in (1, 0) for r in SIMPLE]
    s = ev.choice_within_class(pd.DataFrame(rows))
    assert s.loc[("s2", "tau_F1", 36, 1), "validation_choice"] == SIMPLE[0]     # tie rule: the first
    assert s.loc[("s2", "tau_F1", 36, 0), "validation_choice"] == "studentised"
