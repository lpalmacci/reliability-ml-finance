"""Common interface of scenario generators, plus the oracle and the classical baselines."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np

from tavp.dgp.base import DGP
from tavp.dgp.processes import GJRGarchT
from tavp.perturb.operators import PATH_DEFECTS, apply_path_defect


class Generator(ABC):
    """Produces scenarios of log-returns. `target_hash` identifies the process it claims to represent."""

    name: str = "generator"
    target_hash: str | None = None

    @abstractmethod
    def sample_returns(self, n: int, T: int, rng: np.random.Generator) -> np.ndarray:
        """Array (n, T) of log-returns."""

    def describe(self) -> dict:
        return {"name": self.name, "target_hash": self.target_hash}


class OracleGenerator(Generator):
    """The ground truth used as a generator, optionally with a planted defect of severity delta.

    With no defect it is the negative control of the protocol. The target hash is
    always the one of the unperturbed process, because a perturbed oracle is a
    defective generator *of that process*.
    """

    def __init__(self, dgp: DGP, defect: str | None = None, delta: float = 0.0):
        self.base = dgp
        self.defect, self.delta = defect, delta
        self.sim = dgp if defect is None or defect in PATH_DEFECTS else dgp.perturbed(defect, delta)
        self.name = "oracle" if defect is None else f"oracle_{defect}_{delta:g}"
        self.target_hash = dgp.hash()

    def sample_returns(self, n, T, rng):
        r = self.sim.simulate_returns(n, T, rng)
        if self.defect in PATH_DEFECTS:
            r = apply_path_defect(self.defect, r, self.delta, self.base, rng)
        return r

    def describe(self):
        return {**super().describe(), "defect": self.defect, "delta": self.delta, "sim_params": self.sim.params()}


class EstimatedDriftOracle(Generator):
    """Positive control of E8: the true dynamics with the drift estimated from a training history.

    Scenarios are oracle scenarios shifted by the difference between the sample mean of the
    training returns and the stationary drift of the process. The generator knows everything
    about the process except what the history cannot tell precisely, its drift, which it
    estimates by the sample mean, as a bootstrap of the same history does. The cells of level
    L2 that it fails are therefore not attainable with histories of that length.
    """

    name = "oracle_drift"

    def __init__(self, dgp: DGP, history: np.ndarray):
        self.base = dgp
        self.target_hash = dgp.hash()
        self.shift = float(np.mean(history) - dgp.drift_per_step())

    def sample_returns(self, n, T, rng):
        return self.base.simulate_returns(n, T, rng) + self.shift

    def describe(self):
        return {**super().describe(), "drift_error_annual": self.shift / self.base.dt}


class StationaryBootstrap(Generator):
    """B1: stationary bootstrap (Politis and Romano, 1994) of a single training history.

    The mean block length is the larger of the Politis-White (2004) automatic
    choices computed on the returns and on the squared returns, so that blocks
    are long enough to carry volatility clustering.
    """

    name = "bootstrap"

    def __init__(self, history: np.ndarray, target_hash: str, block_mean: float | None = None):
        from arch.bootstrap import optimal_block_length

        self.history = np.asarray(history, dtype=float)
        self.target_hash = target_hash
        if block_mean is None:
            b_r = optimal_block_length(self.history)["stationary"].iloc[0]
            b_r2 = optimal_block_length(self.history**2)["stationary"].iloc[0]
            block_mean = float(max(b_r, b_r2, 1.0))
        self.block_mean = block_mean

    def sample_returns(self, n, T, rng):
        m = len(self.history)
        idx = np.empty((n, T), dtype=np.int64)
        idx[:, 0] = rng.integers(0, m, n)
        p_new = 1.0 / self.block_mean
        for t in range(1, T):
            jump = rng.random(n) < p_new
            idx[:, t] = np.where(jump, rng.integers(0, m, n), (idx[:, t - 1] + 1) % m)
        return self.history[idx]

    def describe(self):
        return {**super().describe(), "block_mean": self.block_mean}


# starting values (alpha, gamma, beta) of the multistart fit; every one has persistence below 1
GARCH_STARTS = [(a, g, b) for a in (0.02, 0.05) for g in (0.04, 0.08) for b in (0.84, 0.90)]
GARCH_GAIN_TOL = 1e-3


def fit_gjr_garch_t(x: np.ndarray, multistart: bool = False):
    """ML fit of a GJR-GARCH(1,1)-t with constant mean (arch package) on x (returns in percent).

    The pre-registered benchmark B2 uses the default start of the arch package. With multistart the
    fit is repeated from the starts GARCH_STARTS (mean, degrees of freedom and unconditional variance
    taken from the default fit) and the fit with the largest likelihood is kept: the optimiser can
    stop at a local optimum and report success (deviation D8). A start replaces the default fit only
    if it improves the log-likelihood by more than GARCH_GAIN_TOL, so that a fit already at the
    optimum is left unchanged.
    """
    from arch import arch_model

    model = arch_model(x, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
    res = model.fit(disp="off")
    if multistart:
        names = list(res.params.index)
        for a, g, b in GARCH_STARTS:
            st = res.params.copy()
            st["alpha[1]"], st["gamma[1]"], st["beta[1]"] = a, g, b
            st["omega"] = float(np.var(x)) * (1.0 - a - 0.5 * g - b)
            r = model.fit(disp="off", starting_values=st[names].to_numpy())
            if np.isfinite(r.loglikelihood) and r.loglikelihood > res.loglikelihood + GARCH_GAIN_TOL:
                res = r
    return res


class GarchBaseline(Generator):
    """B2: GJR-GARCH(1,1)-t estimated by maximum likelihood (arch package) on one history."""

    name = "garch"

    def __init__(self, history: np.ndarray, target_hash: str, burn: int = 500, multistart: bool = False):
        self.target_hash = target_hash
        res = fit_gjr_garch_t(100.0 * np.asarray(history), multistart)
        p = res.params
        self.fit_params = {k: float(v) for k, v in p.items()}
        self.loglik = float(res.loglikelihood)
        alpha, gamma, beta = p["alpha[1]"], p["gamma[1]"], p["beta[1]"]
        pers = alpha + 0.5 * gamma + beta
        if pers >= 0.999:  # keep the fitted dynamics but enforce stationarity; recorded in describe()
            warnings.warn(f"fitted GJR-GARCH persistence {pers:.4f} >= 0.999, beta shrunk")
            beta = 0.999 - alpha - 0.5 * gamma
        self.sim = GJRGarchT(mu_d=p["mu"] / 100.0, omega=p["omega"] / 1e4, alpha=alpha, gamma=gamma, beta=beta,
                             nu=p["nu"], burn=burn)

    def sample_returns(self, n, T, rng):
        return self.sim.simulate_returns(n, T, rng)

    def describe(self):
        return {**super().describe(), "fit_params": self.fit_params, "sim_params": self.sim.params()}
