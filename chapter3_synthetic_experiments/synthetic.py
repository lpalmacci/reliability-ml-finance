"""Synthetic generator with a known truth, and the readings and metrics shared by the experiments of Chapter 3.

The designs are those of Chapter 3, section "What the signal detects when the truth is known". Each case
has a true probability q of the positive class, with logit(q) drawn from a normal law centred on
logit(tau) with standard deviation 1.5, and the K members estimate it with two kinds of noise:

    p_k = expit(logit(q) + b + eta * e_k),    e_k ~ N(0, 1) independent, eta ~ U[0.05, 1.2] per case.

- informative:      b = 0, so the error of the mean comes from the members' own noise and grows with
                    their disagreement;
- non-informative:  the members' noise is centred within each case, so that it creates disagreement
                    without moving the mean on the logit scale, and the error of the mean comes from a
                    common error b, whose scale is itself random and unrelated to eta;
- mixed:            the members' noise is not centred and b ~ N(0, 0.5^2) is added.
"""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

DESIGNS = ("informative", "non-informative", "mixed")
SIMPLE_READINGS = ("distance", "studentised", "logit distance", "logit studentised")


def simulate(design, K, tau, n, rng):
    """Cases with true probability q and the probabilities of K members.

    Returns ``(P, q, eta)``: P of shape (n, K), clipped to [1e-6, 1 - 1e-6], the true probabilities q
    and the scale eta of the members' noise. The order in which random numbers are drawn is part of
    the definition of the experiments and must not be changed.
    """
    lq = logit(tau) + 1.5 * rng.standard_normal(n)
    eta = rng.uniform(0.05, 1.2, n)
    e = eta[:, None] * rng.standard_normal((n, K))
    if design == "informative":
        b = np.zeros(n)
    elif design == "non-informative":
        e = e - e.mean(axis=1, keepdims=True)
        b = rng.uniform(0.05, 1.2, n) * rng.standard_normal(n)
    elif design == "mixed":
        b = 0.5 * rng.standard_normal(n)
    else:
        raise ValueError(design)
    P = np.clip(expit(lq[:, None] + b[:, None] + e), 1e-6, 1 - 1e-6)
    return P, expit(lq), eta


def auroc(score, y, decimals=10):
    """AUROC with which a score of doubt (higher = more doubtful) separates the errors (y = 1) from the
    correct decisions (y = 0).

    It is the Mann-Whitney statistic divided by the product of the group sizes, with ties counted as
    one half (mid-ranks), as in the definition given in the thesis. Scores are rounded to ``decimals``
    decimal places before ranking, so that values that differ only by floating-point noise count as
    ties: the signal saturates at its bound for many cases when the ensemble is large, and without the
    rounding the order of those cases would depend on the last bits of the arithmetic.
    """
    y = np.asarray(y, dtype=int)
    r = rankdata(np.round(np.asarray(score, dtype=float), decimals), method="average")
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def baselines(P, pbar, tau):
    """The four simpler readings of the members' probabilities, as scores of doubt (higher = more doubtful).

    - distance:           -|pbar - tau|
    - studentised:        -|pbar - tau| / s, with s the standard deviation of the members
    - logit distance:     -|logit(pbar) - logit(tau)|
    - logit studentised:  -|mean of the members' logits - logit(tau)| / sd of their logits
    """
    s = P.std(axis=1, ddof=1)
    L = logit(P); sl = L.std(axis=1, ddof=1)
    return {"distance": -np.abs(pbar - tau),
            "studentised": -np.abs(pbar - tau) / s,
            "logit distance": -np.abs(logit(pbar) - logit(tau)),
            "logit studentised": -np.abs(L.mean(axis=1) - logit(tau)) / sl}


def seed_for(*keys):
    """Random generator of one experimental condition.

    Each component of the key is mapped to an integer (``int(1000 * x)`` for a threshold, the integer
    itself for a number of members, the sum of the character codes for a string), and the list seeds
    ``numpy.random.default_rng``. The generator depends only on the condition, so a condition gives the
    same numbers whether it is run alone or with the others, and in any order. The strings used as keys
    ("val", "gate", "budget", "sens", ...) are part of the definition of the experiments.
    """
    return np.random.default_rng([int(1000 * k) if isinstance(k, float)
                                  else (k if isinstance(k, int) else sum(map(ord, k))) for k in keys])


def recall_at(score, err, b):
    """Share of all the errors among the round(b * n) cases with the highest score of doubt
    (ties broken by the order of the cases)."""
    k = int(round(b * len(err)))
    top = np.argsort(-score, kind="mergesort")[:k]
    return float(err[top].sum() / err.sum())
