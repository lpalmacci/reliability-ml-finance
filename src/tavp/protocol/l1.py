"""Level L1: statistical fidelity of scenarios (diagnostic).

Each statistic s is computed on a generated sample and on an independent oracle sample;
its discrepancy D is |s_G - s_O| (root mean square over lags for vector statistics) or a
distance between the two samples. The null law of D is estimated from pairs of
independent oracle samples of the same size; p-values are empirical and corrected with
the Holm procedure over the family of statistics.
"""

from __future__ import annotations

import numpy as np

from tavp.protocol.stats import empirical_pvalue, holm


def _pooled_acf(x: np.ndarray, lags: int) -> np.ndarray:
    xc = x - x.mean()
    v = np.mean(xc**2)
    return np.array([np.mean(xc[:, :-k] * xc[:, k:]) / v for k in range(1, lags + 1)])


def hill_index(r: np.ndarray, fraction: float) -> float:
    a = np.sort(np.abs(r - r.mean()).ravel())[::-1]
    k = max(int(fraction * a.size), 2)
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def _kurtosis(x: np.ndarray) -> float:
    xc = x - x.mean()
    return float(np.mean(xc**4) / np.mean(xc**2) ** 2)


def stylized_facts(r: np.ndarray, cfg: dict) -> dict[str, np.ndarray | float]:
    """Stylized-fact statistics of a sample of return paths (n, T), pooled across paths."""
    rc = r - r.mean()
    s = r.std()
    r2 = r**2
    lev = np.array([np.mean(rc[:, :-k] * (r2[:, k:] - r2.mean())) / (s * r2.std())
                    for k in range(1, cfg["leverage_lags"] + 1)])
    out = {
        "mean": float(r.mean()),
        "std": float(s),
        "skew": float(np.mean(rc**3) / s**3),
        "kurt": _kurtosis(r),
        "hill": hill_index(r, cfg["hill_fraction"]),
        "acf_r": _pooled_acf(r, cfg["acf_lags"]),
        "acf_abs": _pooled_acf(np.abs(r), cfg["acf_abs_lags"]),
        "leverage": lev,
    }
    T = r.shape[1]
    for m in cfg["aggregation"]:
        blocks = T // m
        agg = r[:, : blocks * m].reshape(len(r), blocks, m).sum(axis=2)
        out[f"kurt_agg{m}"] = _kurtosis(agg)
    return out


def _w1_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """W1 between empirical laws of equal size, column by column."""
    return np.mean(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)), axis=0)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_proj: int, seed: int = 0) -> float:
    """Sliced 2-Wasserstein distance between two samples of equal size (Bonneel et al., 2015)."""
    theta = np.random.default_rng(seed).standard_normal((x.shape[1], n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True)
    px, py = np.sort(x @ theta, axis=0), np.sort(y @ theta, axis=0)
    return float(np.sqrt(np.mean((px - py) ** 2)))


def expected_signature(paths: np.ndarray, level: int = 3) -> np.ndarray:
    """Mean truncated signature (levels 1-3) of 2-d piecewise-linear paths (n, L, 2), via Chen's identity."""
    n, L, d = paths.shape
    inc = np.diff(paths, axis=1)
    s1 = np.zeros((n, d))
    s2 = np.zeros((n, d, d))
    s3 = np.zeros((n, d, d, d))
    for t in range(L - 1):
        a = inc[:, t]
        aa = np.einsum("ni,nj->nij", a, a)
        aaa = np.einsum("nij,nk->nijk", aa, a)
        s3 = s3 + np.einsum("nij,nk->nijk", s2, a) + np.einsum("ni,njk->nijk", s1, aa) / 2.0 + aaa / 6.0
        s2 = s2 + np.einsum("ni,nj->nij", s1, a) + aa / 2.0
        s1 = s1 + a
    feats = [s1, s2.reshape(n, -1), s3.reshape(n, -1)][:level]
    return np.concatenate(feats, axis=1).mean(axis=0)


def signature_distance(rg: np.ndarray, ro: np.ndarray) -> float:
    """Distance between expected signatures of the time-augmented log-price paths.

    The log-price channel is scaled by the reference volatility over the horizon, so that
    the time and price channels are of comparable size. This is the MMD with the linear
    kernel on signatures truncated at level 3 (Chevyrev and Oberhauser).
    """
    T = rg.shape[1]
    scale = ro.std() * np.sqrt(T)

    def paths(r):
        logp = np.concatenate([np.zeros((len(r), 1)), np.cumsum(r, axis=1)], axis=1) / scale
        time = np.broadcast_to(np.linspace(0.0, 1.0, T + 1), logp.shape)
        return np.stack([time, logp], axis=2)

    return float(np.linalg.norm(expected_signature(paths(rg)) - expected_signature(paths(ro))))


def discrepancies(rg: np.ndarray, ro: np.ndarray, cfg: dict) -> dict[str, float]:
    """All L1 discrepancies between a generated sample and an oracle sample of equal size."""
    sg, so = stylized_facts(rg, cfg), stylized_facts(ro, cfg)
    out = {}
    for k in sg:
        diff = np.asarray(sg[k]) - np.asarray(so[k])
        out[k] = float(np.sqrt(np.mean(diff**2))) if diff.ndim else float(abs(diff))
    if "marginal_w1" in cfg["distances"]:
        out["marginal_w1"] = float(_w1_equal(rg, ro).mean())
    if "swd_logpath" in cfg["distances"]:
        out["swd_logpath"] = sliced_wasserstein(np.cumsum(rg, axis=1), np.cumsum(ro, axis=1), cfg["swd_projections"])
    if "signature_mmd" in cfg["distances"]:
        out["signature_mmd"] = signature_distance(rg, ro)
    return out


def test_against_null(obs: dict[str, float], null: dict[str, np.ndarray], level: float) -> dict:
    """Empirical p-values against the null law of each discrepancy, with Holm correction."""
    names = sorted(obs)
    p = np.array([empirical_pvalue(obs[k], null[k]) for k in names])
    reject = holm(p, level)
    return {k: {"D": obs[k], "D_null_median": float(np.median(null[k])), "ratio": obs[k] / float(np.median(null[k])),
                "p": float(pk), "flag_raw": bool(pk < level), "flag_holm": bool(rk)}
            for k, pk, rk in zip(names, p, reject)}
