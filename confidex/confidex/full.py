"""Specification with an inferred location (independent priors), on a two-dimensional grid.

It serves as the yardstick for the fixed-mean assumption (Chapter 3, Three specifications of the
decoupled model, and Sensitivity to the modelling choices). Same conventions as ``core``: logit grids
with trapezoidal weights, sufficient statistics, clipping.

Experimental module. Beware near 0 and 1: with a flat prior on mu and the Beta(2, 3) prior on c
anchored at lambda_0 = K, the posterior of mu can settle far from the members' mean. For five members
between 1e-5 and 4e-5 (mean 2.4e-5) the posterior mean of mu is about 0.06: the prior penalises the
very large precisions that a narrow law near zero would require, and the posterior prefers a J-shaped
law with a higher mean. The comparison with the fixed mean is therefore reliable only for means away
from the ends of the interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import betaln
from scipy.stats import beta as beta_dist

from .config import DEFAULT_MODEL, ModelConfig
from .core import LogitGrid

__all__ = ["PosteriorFull", "fit_posterior_full"]


@dataclass(frozen=True)
class PosteriorFull:
    """Summaries of the specification with an inferred location, of shape ``(n,)``."""

    mu_mean: np.ndarray
    mu_sd: np.ndarray
    mu_ci90: np.ndarray          # (n, 2)
    c_mean: np.ndarray
    c_var: np.ndarray
    F_pred: dict[float, np.ndarray] | None
    joint: np.ndarray | None     # (n, J_mu, J_c), only with keep_joint=True
    grid_mu: LogitGrid
    grid_c: LogitGrid


def fit_posterior_full(P: np.ndarray, cfg: ModelConfig = DEFAULT_MODEL, *,
                       prior_mu: tuple[float, float] = (1.0, 1.0),
                       points_mu: int = 200, points_c: int = 200,
                       bound_mu: float = 1e-4,
                       taus: Sequence[float] | None = None,
                       keep_joint: bool = False) -> PosteriorFull:
    """Joint posterior of (mu, c) with prior Beta(prior_mu) x Beta(prior_a, prior_b).

    With ``taus`` it returns the predictive law over both parameters,
    F_pred(tau) = sum_{j,k} M_jk F_Beta(tau; mu_j lambda_k, (1 - mu_j) lambda_k).
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    n, K = P.shape
    Pc = np.clip(P, cfg.clip_eps, 1.0 - cfg.clip_eps)
    lambda0 = cfg.resolve_lambda0(K)
    gm = LogitGrid.build(bound_mu, points_mu)
    gc = LogitGrid.build(cfg.grid_bound, points_c)
    lam = lambda0 * gc.g / (1.0 - gc.g)
    A = gm.g[:, None] * lam[None, :]
    Bb = (1.0 - gm.g)[:, None] * lam[None, :]
    lprior = (beta_dist.logpdf(gm.g, *prior_mu)[:, None]
              + beta_dist.logpdf(gc.g, cfg.prior_a, cfg.prior_b)[None, :])
    W = gm.w[:, None] * gc.w[None, :]
    lB = K * betaln(A, Bb)
    Fgrid = {float(t): beta_dist.cdf(t, A, Bb) for t in taus} if taus is not None else None

    S1 = np.log(Pc).sum(axis=1)
    S2 = np.log1p(-Pc).sum(axis=1)
    out = dict(mu_mean=np.empty(n), mu_sd=np.empty(n), mu_ci90=np.empty((n, 2)),
               c_mean=np.empty(n), c_var=np.empty(n))
    F_pred = {t: np.empty(n) for t in Fgrid} if Fgrid is not None else None
    joint = np.empty((n, points_mu, points_c)) if keep_joint else None
    for i in range(n):
        ll = (A - 1.0) * S1[i] + (Bb - 1.0) * S2[i] - lB + lprior
        M = np.exp(ll - ll.max()) * W
        M /= M.sum()
        if joint is not None:
            joint[i] = M
        mm, mc = M.sum(axis=1), M.sum(axis=0)
        out["mu_mean"][i] = (gm.g * mm).sum()
        out["mu_sd"][i] = np.sqrt(((gm.g - out["mu_mean"][i]) ** 2 * mm).sum())
        cdf = np.cumsum(mm)
        out["mu_ci90"][i] = gm.g[np.searchsorted(cdf, 0.05)], gm.g[np.searchsorted(cdf, 0.95)]
        out["c_mean"][i] = (gc.g * mc).sum()
        out["c_var"][i] = ((gc.g - out["c_mean"][i]) ** 2 * mc).sum()
        if F_pred is not None:
            for t, Fg in Fgrid.items():
                F_pred[t][i] = (M * Fg).sum()
    return PosteriorFull(F_pred=F_pred, joint=joint, grid_mu=gm, grid_c=gc, **out)
