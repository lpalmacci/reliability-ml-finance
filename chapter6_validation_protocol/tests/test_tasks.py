import numpy as np
import pytest

import reference_ipmu as ref
from tavp.dgp.processes import GBM
from tavp.tasks.backtest import Objective, assemble, calibrate, evaluate, scenario_metrics
from tavp.tasks.strategies import (build_strategies, rolling_mean, rolling_vol, trend_following,
                                   trend_vol_filter, vol_target)
from tavp.utils import rng

GRIDS = {"TF": {"fast": [5, 10], "slow": [20, 40], "dz": [0.0, 0.005]},
         "VT": {"lb": [10, 20], "target": [0.1, 0.15], "lmax": [1, 2]},
         "TVF": {"fast": [5], "slow": [20], "dz": [0.0], "lb": [10, 20], "q": [0.8, 0.9]}}


def _logp(n=3, L=300, seed=0):
    r = GBM().simulate_returns(n, L - 1, rng(seed, "t"))
    return np.concatenate([np.zeros((n, 1)), np.cumsum(r, axis=1)], axis=1)


def test_rolling_against_naive():
    x = _logp(2, 80)
    rm = rolling_mean(x, 7)
    assert np.allclose(rm[:, 6:], [[x[i, t - 6:t + 1].mean() for t in range(6, 80)] for i in range(2)])
    rv = rolling_vol(x, 10)
    r = np.diff(x, axis=1)
    naive = [[np.std(r[i, t - 10:t], ddof=1) * np.sqrt(252) for t in range(10, 80)] for i in range(2)]
    assert np.allclose(rv[:, 10:], naive)
    assert np.isnan(rv[:, :10]).all()


def test_port_matches_ipmu_reference():
    x = _logp(1, 200, seed=5)[0]
    for fast, slow, dz in [(5, 20, 0.0), (10, 40, 0.005)]:
        assert np.array_equal(trend_following(x[None], fast, slow, dz)[0], ref.trend_following_ma(x, fast, slow, dz))
    for lb, tgt, lmax in [(10, 0.1, 1.0), (20, 0.15, 2.0)]:
        mine = vol_target(x[None], lb, tgt, lmax)[0]
        theirs = ref.vol_target_long_only(np.exp(x), lb, tgt, periods_per_year=252, max_leverage=lmax)
        assert np.allclose(mine, theirs)
    pos = vol_target(x[None], 20, 0.1, 2.0)[0]
    eq, pnl = ref.backtest_positions(np.exp(x[1:]), pos[1:])
    s_ref, dd_ref = ref.sharpe_and_maxdd(eq, pnl, periods_per_year=252)
    m = evaluate(x[None], pos[None], len(x) - 2, periods=252)
    assert np.isclose(m["sharpe"][0], s_ref, atol=1e-9)
    assert np.isclose(m["maxdd"][0], dd_ref)


@pytest.mark.parametrize("rule,params", [(trend_following, dict(fast=5, slow=20, dz=0.0)),
                                         (vol_target, dict(lb=10, target=0.1, lmax=2.0)),
                                         (trend_vol_filter, dict(fast=5, slow=20, dz=0.0, lb=10, threshold=0.2))])
def test_no_look_ahead(rule, params):
    x = _logp(1, 150)
    y = x.copy()
    y[:, 100:] += np.linspace(0, 0.5, 50)  # change the future after index 99
    assert np.array_equal(rule(x, **params)[:, :100], rule(y, **params)[:, :100])


def test_scale_invariance_of_log_price_rules():
    x = _logp(2, 150)
    assert np.allclose(vol_target(x, 20, 0.1, 2.0), vol_target(x + 3.0, 20, 0.1, 2.0))
    assert np.array_equal(trend_following(x, 5, 20, 0.0), trend_following(x + 3.0, 5, 20, 0.0))


def test_evaluate_indexing_against_loop():
    x = _logp(2, 60)
    pos = np.sign(np.sin(np.arange(60)))[None].repeat(2, 0)
    m = evaluate(x, pos, 20, cost=0.001)
    for i in range(2):
        pnl = [pos[i, t - 1] * (x[i, t] - x[i, t - 1]) - 0.001 * abs(pos[i, t - 1] - pos[i, t - 2]) for t in range(40, 60)]
        assert np.isclose(m["ret"][i], sum(pnl))
        eq = np.exp(np.cumsum(pnl))
        assert np.isclose(m["maxdd"][i], max(0, np.max(1 - eq / np.maximum.accumulate(np.r_[1, eq])[1:])))


def test_metrics_ranges_and_bh():
    x = _logp(500, 260)
    m = evaluate(x, np.ones_like(x), 59)
    assert (m["maxdd"] >= 0).all() and (m["maxdd"] <= 1).all()
    assert np.allclose(m["turnover"], 0)
    r = np.diff(x, axis=1)[:, -59:]
    assert np.allclose(m["sharpe"], r.mean(1) / r.std(1, ddof=1) * np.sqrt(252))


def test_flat_position_has_zero_sharpe():
    x = _logp(2, 100)
    assert np.allclose(evaluate(x, np.zeros_like(x), 30)["sharpe"], 0.0)


def test_assemble_and_calibrate():
    ctx = _logp(1, 501)[0]
    scen = GBM().simulate_returns(100, 59, rng(9, "s"))
    logp = assemble(ctx, scen, 200)
    assert logp.shape == (100, 259)
    assert np.allclose(logp[:, 199], ctx[-1]) and np.allclose(np.diff(logp[:, 199:], axis=1), scen)
    strategies = build_strategies(GRIDS)
    for name, s in strategies.items():
        g, p = calibrate(s, ctx, 200, Objective())
        if name == "BH":
            assert g == {} and p == {}
        else:
            assert g in s.grid
            m = scenario_metrics(s, p, ctx, scen, 200)
            assert m["sharpe"].shape == (100,)
    assert "threshold" in calibrate(strategies["TVF"], ctx, 200, Objective())[1]


def test_cached_positions_match_rules():
    from tavp.tasks.strategies import IndicatorCache, cached_positions
    x = _logp(50, 259, seed=11)
    cache = IndicatorCache(x, 61)
    s = build_strategies(GRIDS)
    for name in ["BH", "TF", "VT", "TVF"]:
        grid = s[name].grid or ({},)
        for g in grid:
            p = dict(g)
            if name == "TVF":
                p.pop("q"); p["threshold"] = 0.18
            full = s[name].positions(x, p)[:, -61:]
            assert np.array_equal(full, cached_positions(name, p, cache)), (name, g)
            m_full = evaluate(x, s[name].positions(x, p), 59)
            m_tail = evaluate(x[:, -61:], cached_positions(name, p, cache), 59)
            assert all(np.allclose(m_full[k], m_tail[k]) for k in m_full)
