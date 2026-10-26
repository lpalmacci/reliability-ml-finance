import numpy as np

from tavp.dgp.processes import GBM, GJRGarchT
from tavp.experiments import common
from tavp.generators.base import EstimatedDriftOracle, GarchBaseline, StationaryBootstrap
from tavp.generators.tcvae import TCVAEGenerator, returns_to_prices, windows_from_history
from tavp.utils import REPO_ROOT, load_config, load_yaml, rng


def test_windows_from_history():
    r = np.array([0.1, -0.2, 0.05, 0.0])
    w = windows_from_history(r, 3)
    assert w.shape == (3, 3)
    assert np.allclose(w[:, 0], 1.0)
    assert np.allclose(np.diff(np.log(w), axis=1), [[0.1, -0.2], [-0.2, 0.05], [0.05, 0.0]])
    assert np.allclose(returns_to_prices(r[None, :2]), w[:1])


def test_bootstrap_resamples_history_with_blocks():
    h = GJRGarchT.from_annual().history(2520, rng(1, "h"))
    b = StationaryBootstrap(h, "x")
    assert b.block_mean >= 1
    s = b.sample_returns(200, 59, rng(2, "b"))
    assert s.shape == (200, 59) and np.isin(s, h).all()


def test_garch_baseline_recovers_parameters_roughly():
    d = GJRGarchT.from_annual()
    h = d.history(5000, rng(3, "g"))
    g = GarchBaseline(h, d.hash())
    assert 0.9 < g.sim.persistence < 1.0
    assert abs(np.sqrt(g.sim.unconditional_variance * 252) - 0.18) < 0.05
    assert g.sample_returns(10, 59, rng(4, "s")).shape == (10, 59)


def test_garch_multistart_never_lowers_the_likelihood():
    from tavp.experiments.runner import gen_id, label
    from tavp.generators.base import fit_gjr_garch_t

    h = 100.0 * GJRGarchT.from_annual().history(3000, rng(7, "g"))
    one, many = fit_gjr_garch_t(h), fit_gjr_garch_t(h, multistart=True)
    assert many.loglikelihood >= one.loglikelihood
    assert np.allclose(GarchBaseline(h / 100.0, "x").fit_params["beta[1]"], one.params["beta[1]"])  # B2 unchanged
    spec = {"kind": "garch_ms", "regime": "rich", "seed": 2}
    assert gen_id(spec) == gen_id({**spec, "kind": "garch"}) and label(spec) != label({**spec, "kind": "garch"})


def test_estimated_drift_oracle_shifts_only_the_drift():
    d = GBM(mu=0.07, sigma=0.18)
    h = d.history(2520, rng(5, "h"))
    g = EstimatedDriftOracle(d, h)
    assert g.target_hash == d.hash()
    assert np.isclose(g.shift, h.mean() - d.drift_per_step())
    a = g.sample_returns(4000, 59, rng(6, "s"))
    b = d.simulate_returns(4000, 59, rng(6, "s"))
    assert np.allclose(a - b, g.shift)  # same random numbers: only the mean moves


def test_long_history_has_as_many_returns_as_the_rich_set():
    cfg = load_config(REPO_ROOT / "configs" / "full.yaml")
    cfg["data"]["rich_paths"] = 20
    d = GBM(mu=0.07, sigma=0.18)
    h = common.long_history(cfg, d, 0)
    assert len(h) == 20 * (cfg["data"]["scenario_prices"] - 1) == common.long_returns(cfg)
    assert np.array_equal(h, common.long_history(cfg, d, 0))
    assert not np.array_equal(h, common.long_history(cfg, d, 1))


def test_tcvae_trains_saves_and_samples(tmp_path):
    d = GBM()
    w = windows_from_history(d.history(1100, rng(5, "w")), 60)
    cfg = load_yaml(REPO_ROOT / "configs" / "generators" / "tcvae.yaml")
    gen = TCVAEGenerator(cfg, d.hash()).fit(w, seed=0, out_dir=tmp_path / "run", epochs=2)
    s1 = gen.sample_returns(50, 59, rng(6, "x"))
    assert s1.shape == (50, 59) and np.isfinite(s1).all()
    loaded = TCVAEGenerator.load(tmp_path / "run")
    assert loaded.target_hash == d.hash()
    assert np.allclose(loaded.sample_returns(50, 59, rng(6, "x")), s1)
