"""The four ground-truth processes D1-D4 of the controlled benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from tavp.dgp.base import DGP
from tavp.utils import TRADING_DAYS


@dataclass(frozen=True)
class GBM(DGP):
    """D1: geometric Brownian motion, r_t = (mu - sigma^2/2) dt + sigma sqrt(dt) Z_t."""

    mu: float = 0.07
    sigma: float = 0.18

    @property
    def name(self) -> str:
        return "gbm"

    def simulate_returns(self, n, T, rng):
        z = rng.standard_normal((n, T))
        return (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * math.sqrt(self.dt) * z

    def drift_per_step(self):
        return (self.mu - 0.5 * self.sigma**2) * self.dt


@dataclass(frozen=True)
class Heston(DGP):
    """D2: Heston stochastic volatility, full-truncation Euler with sub-steps.

    dX = (mu - v/2) dt + sqrt(v) dW1,  dv = kappa (theta_v - v) dt + xi sqrt(v) dW2,
    corr(dW1, dW2) = rho, X = log price. The initial variance is drawn from the
    stationary Gamma(2 kappa theta_v / xi^2, xi^2 / (2 kappa)) law.
    """

    mu: float = 0.07
    kappa: float = 3.0
    theta_v: float = 0.04
    xi: float = 0.4
    rho: float = -0.7
    substeps: int = 10

    @property
    def name(self) -> str:
        return "heston"

    def stationary_variance_law(self) -> tuple[float, float]:
        shape = 2.0 * self.kappa * self.theta_v / self.xi**2
        scale = self.xi**2 / (2.0 * self.kappa)
        return shape, scale

    def simulate_returns(self, n, T, rng):
        shape, scale = self.stationary_variance_law()
        v = rng.gamma(shape, scale, size=n)
        h = self.dt / self.substeps
        sq_h = math.sqrt(h)
        rho_c = math.sqrt(1.0 - self.rho**2)
        out = np.empty((n, T))
        for t in range(T):
            x = np.zeros(n)
            for _ in range(self.substeps):
                z1 = rng.standard_normal(n)
                z2 = self.rho * z1 + rho_c * rng.standard_normal(n)
                vp = np.maximum(v, 0.0)
                sv = np.sqrt(vp)
                x += (self.mu - 0.5 * vp) * h + sv * sq_h * z1
                v = v + self.kappa * (self.theta_v - vp) * h + self.xi * sv * sq_h * z2
            out[:, t] = x
        return out

    def drift_per_step(self):
        return (self.mu - 0.5 * self.theta_v) * self.dt

    def perturbed(self, kind, delta):
        if kind == "P3":  # leverage removed
            return replace(self, rho=(1.0 - delta) * self.rho)
        if kind == "P4":  # faster mean reversion; stationary Var(v) and Feller ratio unchanged
            f = 1.0 + 3.0 * delta
            return replace(self, kappa=self.kappa * f, xi=self.xi * math.sqrt(f))
        return super().perturbed(kind, delta)


@dataclass(frozen=True)
class GJRGarchT(DGP):
    """D3: GJR-GARCH(1,1) with standardised Student-t innovations.

    r_t = mu_d + eps_t, eps_t = sigma_t z_t,
    sigma_{t+1}^2 = omega + (alpha + gamma 1{eps_t < 0}) eps_t^2 + beta sigma_t^2.
    nu = inf gives Gaussian innovations. The state is started at the
    unconditional variance and run for `burn` steps before recording.
    """

    mu_d: float = 0.07 / TRADING_DAYS
    omega: float = 0.18**2 / TRADING_DAYS * 0.02
    alpha: float = 0.02
    gamma: float = 0.12
    beta: float = 0.90
    nu: float = 6.0
    burn: int = 500

    @property
    def name(self) -> str:
        return "gjr_garch_t"

    @classmethod
    def from_annual(cls, mu_ann=0.07, vol_ann=0.18, alpha=0.02, gamma=0.12, beta=0.90, nu=6.0, burn=500):
        pers = alpha + 0.5 * gamma + beta
        var_d = vol_ann**2 / TRADING_DAYS
        return cls(mu_d=mu_ann / TRADING_DAYS, omega=var_d * (1.0 - pers), alpha=alpha, gamma=gamma,
                   beta=beta, nu=nu, burn=burn)

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    def _innovations(self, size, rng):
        if math.isinf(self.nu):
            return rng.standard_normal(size)
        return rng.standard_t(self.nu, size) * math.sqrt((self.nu - 2.0) / self.nu)

    def simulate_returns(self, n, T, rng):
        if self.persistence >= 1.0:
            raise ValueError("non-stationary GJR-GARCH parameters")
        s2 = np.full(n, self.unconditional_variance)
        out = np.empty((n, T))
        for t in range(self.burn + T):
            eps = np.sqrt(s2) * self._innovations(n, rng)
            if t >= self.burn:
                out[:, t - self.burn] = self.mu_d + eps
            s2 = self.omega + (self.alpha + self.gamma * (eps < 0.0)) * eps**2 + self.beta * s2
        return out

    def drift_per_step(self):
        return self.mu_d

    def perturbed(self, kind, delta):
        if kind == "P2":  # thinner tails, 1/nu_delta = (1 - delta)/nu, variance unchanged
            nu = math.inf if delta >= 1.0 else self.nu / (1.0 - delta)
            return replace(self, nu=nu)
        if kind == "P3":  # leverage removed with persistence and omega unchanged
            return replace(self, gamma=(1.0 - delta) * self.gamma, alpha=self.alpha + 0.5 * delta * self.gamma)
        if kind == "P4":  # lower persistence, unconditional variance unchanged
            var_u = self.unconditional_variance
            beta = self.beta - 0.10 * delta
            pers = self.alpha + 0.5 * self.gamma + beta
            return replace(self, beta=beta, omega=var_u * (1.0 - pers))
        return super().perturbed(kind, delta)


@dataclass(frozen=True)
class RegimeSwitching(DGP):
    """D4: two-state Markov-switching log-normal returns (state 1 = bull, state 2 = bear).

    r_t | s_t = i ~ N((mu_i - sigma_i^2/2) dt, sigma_i^2 dt); P(s_{t+1} = i | s_t = i) = p_ii.
    """

    p11: float = 0.99
    p22: float = 0.97
    mu1: float = 0.15
    sigma1: float = 0.12
    mu2: float = -0.20
    sigma2: float = 0.30

    @property
    def name(self) -> str:
        return "regime"

    @property
    def stationary_bull(self) -> float:
        return (1.0 - self.p22) / (2.0 - self.p11 - self.p22)

    def simulate_states_and_returns(self, n, T, rng):
        state = (rng.random(n) >= self.stationary_bull).astype(np.int8)  # 0 = bull, 1 = bear
        mus = np.array([self.mu1, self.mu2])
        sig = np.array([self.sigma1, self.sigma2])
        stay = np.array([self.p11, self.p22])
        out = np.empty((n, T))
        states = np.empty((n, T), dtype=np.int8)
        for t in range(T):
            m, s = mus[state], sig[state]
            out[:, t] = (m - 0.5 * s**2) * self.dt + s * math.sqrt(self.dt) * rng.standard_normal(n)
            states[:, t] = state
            switch = rng.random(n) >= stay[state]
            state = np.where(switch, 1 - state, state).astype(np.int8)
        return states, out

    def simulate_returns(self, n, T, rng):
        return self.simulate_states_and_returns(n, T, rng)[1]

    def drift_per_step(self):
        pi = self.stationary_bull
        m1 = (self.mu1 - 0.5 * self.sigma1**2) * self.dt
        m2 = (self.mu2 - 0.5 * self.sigma2**2) * self.dt
        return pi * m1 + (1.0 - pi) * m2

    def perturbed(self, kind, delta):
        if kind == "P4":  # shorter regimes, stationary probabilities unchanged
            f = 1.0 - 0.8 * delta
            d1, d2 = f / (1.0 - self.p11), f / (1.0 - self.p22)
            return replace(self, p11=1.0 - 1.0 / d1, p22=1.0 - 1.0 / d2)
        return super().perturbed(kind, delta)


def make_dgp(spec: dict) -> DGP:
    """Build a DGP from a configuration entry {'name': ..., **params}."""
    spec = dict(spec)
    name = spec.pop("name")
    if name == "gbm":
        return GBM(**spec)
    if name == "heston":
        return Heston(**spec)
    if name == "gjr_garch_t":
        return GJRGarchT.from_annual(**spec)
    if name == "regime":
        return RegimeSwitching(**spec)
    raise ValueError(f"unknown DGP {name}")
