"""Statistical helpers: empirical p-values, Holm correction, calibration tests for L4."""

from __future__ import annotations

import numpy as np
from scipy import stats


def empirical_pvalue(observed: float, null: np.ndarray) -> float:
    """(1 + #{null >= observed}) / (1 + B): valid p-value for 'observed is too large'."""
    null = np.asarray(null)
    return float((1 + np.sum(null >= observed)) / (1 + null.size))


def holm(p: np.ndarray, level: float) -> np.ndarray:
    """Holm step-down procedure: boolean rejections controlling the family-wise error rate."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, dtype=bool)
    for i, idx in enumerate(order):
        if p[idx] <= level / (m - i):
            reject[idx] = True
        else:
            break
    return reject


def randomized_pit(sample: np.ndarray, realised: float, rng: np.random.Generator) -> float:
    """PIT of a realised value under the empirical law of `sample`, randomised on ties."""
    below = np.sum(sample < realised)
    ties = np.sum(sample == realised)
    return float((below + rng.random() * ties) / len(sample))


def ks_uniform(u: np.ndarray) -> float:
    return float(stats.kstest(u, "uniform").pvalue)


def kupiec_pof(u: np.ndarray, interval: tuple[float, float]) -> float:
    """Kupiec (1995) proportion-of-failures test for the coverage of the central interval."""
    lo, hi = interval
    p = 1.0 - (hi - lo)
    n = len(u)
    x = int(np.sum((u < lo) | (u > hi)))
    phat = x / n

    def ll(q):
        q = min(max(q, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - q) + x * np.log(q)

    lr = -2.0 * (ll(p) - ll(phat))
    return float(stats.chi2.sf(lr, 1))


def berkowitz(u_sequences: list[np.ndarray]) -> tuple[float, float]:
    """Berkowitz (2001) LR tests on z = Phi^{-1}(u), each sequence an independent history.

    Returns the p-values of the joint test (mean 0, variance 1, no AR(1) dependence; 3 df per
    sequence) and of the marginal test (mean 0, variance 1; 2 df per sequence). The LR
    statistics of independent sequences are summed.
    """
    lr3 = lr2 = 0.0
    df3 = df2 = 0
    for u in u_sequences:
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        y, x = z[1:], z[:-1]
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = np.mean(resid**2)
        ll_ar = np.sum(stats.norm.logpdf(resid, scale=np.sqrt(s2)))
        ll_null = np.sum(stats.norm.logpdf(y))
        lr3 += -2.0 * (ll_null - ll_ar)
        df3 += 3
        mu, s2m = z.mean(), z.var()
        lr2 += -2.0 * (np.sum(stats.norm.logpdf(z)) - np.sum(stats.norm.logpdf(z, mu, np.sqrt(s2m))))
        df2 += 2
    return float(stats.chi2.sf(lr3, df3)), float(stats.chi2.sf(lr2, df2))


def moment_test_hac(u_sequences: list[np.ndarray], lags: int | None = None) -> float:
    """Raw-moment calibration test robust to serial dependence (Knuppel, 2015).

    Under calibration z = Phi^{-1}(u) has E[z] = 0 and E[z^2] = 1. The Wald statistic of the two
    moment conditions uses a Bartlett (Newey-West) long-run covariance computed within each
    sequence (lags floor(4 (T/100)^(2/9)) by default) and summed over independent sequences;
    it is chi-squared with 2 degrees of freedom under the null.
    """
    S = np.zeros((2, 2))
    total = np.zeros(2)
    n = 0
    for u in u_sequences:
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        m = np.column_stack([z, z**2 - 1.0])
        T = len(z)
        L = lags if lags is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
        G = m.T @ m
        for lag in range(1, min(L, T - 1) + 1):
            g = m[lag:].T @ m[:-lag]
            G += (1 - lag / (L + 1)) * (g + g.T)
        S += G
        total += m.sum(axis=0)
        n += T
    mbar, S = total / n, S / n
    W = n * mbar @ np.linalg.solve(S, mbar)
    return float(stats.chi2.sf(W, 2))

