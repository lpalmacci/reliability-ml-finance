"""Room of the budget and ablations."""
import numpy as np
from scipy.special import expit

import confidex
from fraud import ablations as abl
from fraud import evaluate as ev
from fraud.readings import order


def test_room():
    assert ev.room(30, 1000, 60) == {"random": 0.03, "perfect": 0.5}
    assert ev.room(80, 1000, 60)["perfect"] == 1.0
    assert np.isnan(ev.room(5, 100, 0)["perfect"])


def _ensemble(n=3000, K=6, seed=0):
    rng = np.random.default_rng(seed)
    lq = np.log(0.1 / 0.9) + 1.5 * rng.standard_normal(n)
    P = expit(lq[:, None] + rng.uniform(0.05, 1.2, n)[:, None] * rng.standard_normal((n, K)))
    return np.clip(P, 1e-6, 1 - 1e-6)


def test_variants_gate_and_lambda0():
    PV, PT = _ensemble(seed=1), _ensemble(seed=2)
    taus = [0.1, 0.25]
    post = confidex.fit_posterior(PT, taus=taus)
    ex = abl.extra_posteriors(PV, PT, post, 6, taus, confidex.DEFAULT_MODEL, with_clip=True)
    for tau in taus:
        v = abl.variants(post, ex, tau)
        g = v["gate"]
        top = order(g["doubt"])[:g["k_alpha"]]
        assert set(top) == set(np.flatnonzero(g["flagged"]))            # the cases flagged by the gate come first
        assert 0.05 < g["share_abstention"] < 0.2
        o = confidex.confidence_score(post, None, tau)
        assert v["non-directional"]["k_alpha"] <= int((o["status"] == "not_confident").sum()) + 1
        assert "clip 1e-4" in v
    P36 = _ensemble(n=500, K=36, seed=3)                                 # at K = 36, lambda0 = 36 is the main analysis
    p36 = confidex.fit_posterior(P36, taus=taus)
    ex36 = abl.extra_posteriors(P36, P36, p36, 36, taus, confidex.DEFAULT_MODEL, with_clip=False)
    assert ex36.lambda0_36 is p36 and ex36.clip_1e4 is None


def test_gate_only_and_abstained_cases():
    PV, PT = _ensemble(seed=4), _ensemble(seed=5)
    taus = [0.1]
    post = confidex.fit_posterior(PT, taus=taus)
    ex = abl.extra_posteriors(PV, PT, post, 6, taus, confidex.DEFAULT_MODEL, with_clip=True, gate_only=True)
    assert ex.lambda0_36 is None and ex.clip_1e4 is None
    v = abl.variants(post, ex, 0.1)
    assert set(v) == {"non-directional", "gate"}
    g = v["gate"]
    assert np.all(g["flagged"][g["abstained"]])                          # abstentions are flagged
    assert abs(g["abstained"].mean() - g["share_abstention"]) < 1e-12
