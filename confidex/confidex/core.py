"""Core of the library: posterior of the concentration c, law of the members, gate and score.

The implementation is vectorised over cases. Its key is the sufficient-statistic form of the
likelihood (Chapter 3, Inference on the concentration): with the location fixed at the ensemble mean,
the model is an exponential family in lambda with sufficient statistics S1 = sum log p and
S2 = sum log(1 - p), so that

    log L(P | c_j) = (a_j - 1) S1 + (b_j - 1) S2 - K log B(a_j, b_j)

costs O(n * J) instead of O(n * K * J), with K entering only as a scalar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import betaln
from scipy.stats import beta as beta_dist

from .config import DEFAULT_CRITERION, DEFAULT_MODEL, CriterionConfig, ModelConfig

__all__ = ["LogitGrid", "PosteriorC", "MemberLaw", "fit_posterior", "member_law",
           "calibrate_gate", "admissible", "confidence_score", "assess", "shape_regime"]


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LogitGrid:
    """Grid on (bound, 1 - bound), equally spaced on the logit scale, with cell weights.

    - *logit spacing*: the informative region is c -> 1, where a concordant ensemble presses the
      posterior against the border.
    - *trapezoidal weights*: half the distance between the neighbouring points, half a cell at the
      ends. They turn point evaluations of the density into masses.
    """

    g: np.ndarray
    w: np.ndarray

    @classmethod
    def build(cls, bound: float, points: int) -> "LogitGrid":
        z = np.linspace(np.log(bound) - np.log1p(-bound),
                        np.log1p(-bound) - np.log(bound), points)
        g = 1.0 / (1.0 + np.exp(-z))
        w = np.empty_like(g)
        w[1:-1] = 0.5 * (g[2:] - g[:-2])
        w[0] = 0.5 * (g[1] - g[0])
        w[-1] = 0.5 * (g[-1] - g[-2])
        return cls(g=g, w=w)


# ---------------------------------------------------------------------------
# Posterior of c
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PosteriorC:
    """Posterior of c for a batch of cases.

    Attributes of shape ``(n,)``:
        mu           mean of the members (after clipping), i.e. the fixed location
        c_mean       posterior mean of c
        c_var        posterior variance of c, the ingredient of the optional gate
        lam          estimated precision, lambda_0 * c_mean / (1 - c_mean)
        tie          True where the members, after clipping, are in an exact tie
        degenerate   equal to ``tie``: improper posterior, summaries undefined
        border_mass  posterior mass at the extreme points of the grid
        at_border    border_mass > border_tol: posterior truncated by the grid
        rel_disp     E[1 / (lambda + 1) | P], expected relative dispersion
        sigma_pred   standard deviation of the predictive law,
                     sqrt(mu (1 - mu) E[1 / (lambda + 1) | P])
    Others:
        masses       masses (n, J), only with ``keep_masses=True``
        F_pred       {tau: F_pred(tau)} of the predictive law, only with ``taus``
    """

    mu: np.ndarray
    c_mean: np.ndarray
    c_var: np.ndarray
    lam: np.ndarray
    tie: np.ndarray
    degenerate: np.ndarray
    border_mass: np.ndarray
    at_border: np.ndarray
    rel_disp: np.ndarray
    sigma_pred: np.ndarray
    masses: np.ndarray | None
    F_pred: dict[float, np.ndarray] | None
    grid: LogitGrid
    lambda0: float


def fit_posterior(P: np.ndarray, cfg: ModelConfig = DEFAULT_MODEL, *,
                  taus: Sequence[float] | None = None,
                  chunk: int = 4096, keep_masses: bool = False) -> PosteriorC:
    """Posterior p(c | P) for every row of ``P`` (shape ``(n, K)``).

    With ``taus`` it also computes, in the same loop and without keeping the masses, the
    distribution function of the predictive law at each threshold:

        F_pred(tau) = sum_j m_j F_Beta(tau; mu lambda_j, (1 - mu) lambda_j)

    The grid masses (``n x J``) take ``8 n J`` bytes and are kept only with ``keep_masses=True``.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    if P.ndim != 2:
        raise ValueError("P must have shape (n, K)")
    n, K = P.shape
    if K < 2:
        raise ValueError(f"at least 2 members are needed, got {K}")
    if not np.all(np.isfinite(P)):
        raise ValueError("P contains non-finite values")
    if P.min() < 0.0 or P.max() > 1.0:
        raise ValueError("the members' probabilities must lie in [0, 1]")

    Pc = np.clip(P, cfg.clip_eps, 1.0 - cfg.clip_eps)
    # ties are checked on the values used for inference, i.e. after clipping
    tie = (Pc.max(axis=1) - Pc.min(axis=1)) < cfg.tie_eps

    lambda0 = cfg.resolve_lambda0(K)
    grid = LogitGrid.build(cfg.grid_bound, cfg.grid_points)
    g, w = grid.g, grid.w
    J = len(g)
    nb = max(1, int(round(cfg.border_fraction * J)))
    lam_grid = lambda0 * g / (1.0 - g)
    log_prior = beta_dist.logpdf(g, cfg.prior_a, cfg.prior_b)

    mu = Pc.mean(axis=1)
    S1 = np.log(Pc).sum(axis=1)
    S2 = np.log1p(-Pc).sum(axis=1)

    c_mean = np.empty(n)
    c_var = np.empty(n)
    border_mass = np.empty(n)
    rel_disp = np.empty(n)
    inv1 = 1.0 / (lam_grid + 1.0)
    masses = np.empty((n, J)) if keep_masses else None
    F_pred = {float(t): np.empty(n) for t in taus} if taus is not None else None
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        a = mu[lo:hi, None] * lam_grid[None, :]
        b = (1.0 - mu[lo:hi, None]) * lam_grid[None, :]
        ll = ((a - 1.0) * S1[lo:hi, None] + (b - 1.0) * S2[lo:hi, None]
              - K * betaln(a, b) + log_prior[None, :])
        ll = np.where(np.isfinite(ll), ll, -np.inf)
        m = np.exp(ll - ll.max(axis=1, keepdims=True)) * w[None, :]
        m /= m.sum(axis=1, keepdims=True)
        if masses is not None:
            masses[lo:hi] = m
        cm = (g[None, :] * m).sum(axis=1)
        c_mean[lo:hi] = cm
        c_var[lo:hi] = (((g[None, :] - cm[:, None]) ** 2) * m).sum(axis=1)
        border_mass[lo:hi] = m[:, :nb].sum(axis=1) + m[:, -nb:].sum(axis=1)
        rel_disp[lo:hi] = (m * inv1[None, :]).sum(axis=1)
        if F_pred is not None:
            for t in F_pred:
                F_pred[t][lo:hi] = (m * beta_dist.cdf(t, a, b)).sum(axis=1)

    c_safe = np.clip(c_mean, 1e-12, 1.0 - 1e-12)
    at_border = border_mass > cfg.border_tol
    return PosteriorC(mu=mu, c_mean=c_mean, c_var=c_var,
                      lam=lambda0 * c_safe / (1.0 - c_safe),
                      tie=tie, degenerate=tie.copy(), border_mass=border_mass,
                      at_border=at_border, rel_disp=rel_disp,
                      sigma_pred=np.sqrt(mu * (1.0 - mu) * rel_disp),
                      masses=masses, F_pred=F_pred, grid=grid, lambda0=lambda0)


# ---------------------------------------------------------------------------
# Optional admissibility gate (ablations only)
# ---------------------------------------------------------------------------

def calibrate_gate(post_val: PosteriorC,
                   cfg: CriterionConfig = DEFAULT_CRITERION) -> float:
    """Threshold v* of the gate: quantile of order (1 - abstention) of Var(c | P) on validation data.

    Degenerate cases are excluded. The threshold is computed once and reused as a constant, so that
    the verdict on a case does not depend on the other cases.
    """
    V = post_val.c_var[~post_val.degenerate]
    if V.size == 0:
        raise ValueError("no non-degenerate case on which to calibrate the gate")
    return float(np.quantile(V, 1.0 - cfg.abstention))


def admissible(post: PosteriorC, v_star: float) -> np.ndarray:
    """True where the case passes the gate: not degenerate and Var(c | P) <= v*."""
    return (~post.degenerate) & (post.c_var <= v_star)


# ---------------------------------------------------------------------------
# Law of the members and criterion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemberLaw:
    """Law of the members at the posterior mean of the concentration, p | c = c_mean, with its
    explicit Beta parameters."""

    a: np.ndarray
    b: np.ndarray
    sigma: np.ndarray

    def cdf(self, t: float | np.ndarray) -> np.ndarray:
        return beta_dist.cdf(t, self.a, self.b)

    def quantile(self, q: float) -> np.ndarray:
        return beta_dist.ppf(q, self.a, self.b)


def member_law(post: PosteriorC) -> MemberLaw:
    """Reconstruct p | c = c_mean (Chapter 3, Proposition "Cancellation of the normalisation")."""
    a = post.mu * post.lam
    b = (1.0 - post.mu) * post.lam
    sigma = np.sqrt(post.mu * (1.0 - post.mu) / (post.lam + 1.0))
    return MemberLaw(a=a, b=b, sigma=sigma)


def confidence_score(post: PosteriorC, law: MemberLaw | None, tau: float,
                     alpha: float | None = None, *,
                     cfg: CriterionConfig = DEFAULT_CRITERION,
                     v_star: float | None = None,
                     predictive: bool = True,
                     directional: bool = True) -> dict[str, np.ndarray]:
    """Central-band criterion (Chapter 3, The reliability signal).

    Law. With ``predictive=True``, the default, F(tau) is the distribution function of the predictive
    law of a further member, which integrates over the posterior of c (it requires ``taus=[tau]`` in
    ``fit_posterior``). With ``predictive=False`` the law at c = c_mean (``law``) is used, which is its
    approximation.

    Verdict. With ``directional=True``, the default, the law must lie beyond the band on the side of
    the decision y = 1[mu >= tau]: F < alpha if y = 1, F > 1 - alpha if y = 0. With the predictive law
    the statement is: a further member would side with the decision with probability greater than
    1 - alpha. With ``directional=False`` it is enough that tau falls outside the band.

    ``status``: 'degenerate' (exact tie), 'abstain' (only with the optional gate, ``v_star``),
    'confident', 'not_confident'.
    """
    alpha = cfg.alpha if alpha is None else alpha
    if predictive:
        if post.F_pred is None or float(tau) not in post.F_pred:
            raise ValueError("predictive law not computed for this threshold: "
                             "pass taus=[tau] to fit_posterior, or use assess()")
        F = post.F_pred[float(tau)]
        sigma = post.sigma_pred
    else:
        law = member_law(post) if law is None else law
        F = law.cdf(tau)
        sigma = law.sigma
    y = post.mu >= tau
    u_band = np.abs(F - 0.5)
    u_dir = np.where(y, 0.5 - F, F - 0.5)          # in [-1/2, 1/2]; < 0 if the law contradicts y
    u = u_dir if directional else u_band
    opposite_side = np.where(y, F > 1.0 - alpha, F < alpha)
    if directional:
        confident = u_dir > 0.5 - alpha
    else:
        confident = (F < alpha) | (F > 1.0 - alpha)
    admitted = admissible(post, v_star) if v_star is not None else ~post.degenerate
    status = np.where(post.degenerate, "degenerate",
                      np.where(~admitted, "abstain",
                               np.where(confident, "confident", "not_confident")))
    z_sigma = np.abs(post.mu - tau) / np.maximum(sigma, 1e-12)
    return {"F_tau": F, "u": u, "u_dir": u_dir, "u_band": u_band, "confident": confident,
            "opposite_side": opposite_side, "admitted": admitted,
            "status": status, "at_border": post.at_border,
            "z_sigma": z_sigma, "z": z_sigma}


def assess(P: np.ndarray, tau: float, cfg: ModelConfig = DEFAULT_MODEL,
           crit: CriterionConfig = DEFAULT_CRITERION) -> dict[str, np.ndarray]:
    """Full procedure on a matrix P (n, K) and a threshold tau.

    Returns the verdict with the predictive law and, for comparison, the one with the law at
    c = c_mean (keys with the suffix ``_plugin``).
    """
    post = fit_posterior(P, cfg, taus=[tau])
    law = member_law(post)
    out = confidence_score(post, law, tau, cfg=crit)
    plug = confidence_score(post, law, tau, cfg=crit, predictive=False)
    out.update({f"{k}_plugin": plug[k] for k in ("F_tau", "u", "confident", "status")})
    out["posterior"] = post
    out["law_plugin"] = law
    return out


def shape_regime(law: MemberLaw) -> np.ndarray:
    """Shape of the reconstructed law: 'U', 'J' or 'unimodal'."""
    lo_a, lo_b = law.a < 1.0, law.b < 1.0
    out = np.full(law.a.shape, "unimodal", dtype=object)
    out[lo_a ^ lo_b] = "J"
    out[lo_a & lo_b] = "U"
    return out
