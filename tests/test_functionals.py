import numpy as np
import pytest

from tavp.tasks.functionals import FUNCTIONALS, tost
from tavp.utils import rng

TRUE = {  # exact values for a standard normal sample
    "mean": 0.0, "q05": -1.6448536, "q50": 0.0, "q95": 1.6448536, "var95": 1.6448536, "es95": 2.0627128,
}


@pytest.mark.parametrize("name", list(TRUE))
def test_standard_errors_give_nominal_coverage(name):
    g = rng(11, "cov", name)
    hits, R, n = 0, 400, 4000
    for _ in range(R):
        est = FUNCTIONALS[name](g.standard_normal(n))
        hits += abs(est.value - TRUE[name]) <= 1.645 * est.se
    cover = hits / R
    assert 0.85 <= cover <= 0.95, cover


def test_tost_declares_equivalence_for_identical_laws():
    g = rng(12, "tost")
    eq = [tost(FUNCTIONALS["mean"](g.standard_normal(40000)), FUNCTIONALS["mean"](g.standard_normal(40000)),
               eta=0.05).equivalent for _ in range(200)]
    assert np.mean(eq) > 0.9


def test_tost_size_at_the_margin():
    # true difference exactly eta: equivalence must be declared in at most ~5% of cases
    g = rng(13, "size")
    eq = [tost(FUNCTIONALS["mean"](g.standard_normal(4000) + 0.05), FUNCTIONALS["mean"](g.standard_normal(4000)),
               eta=0.05).equivalent for _ in range(400)]
    assert np.mean(eq) <= 0.08
