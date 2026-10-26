"""Confidex: per-sample reliability of the classifications of an ensemble, from the reconstructed law
of the members' probabilities.

Procedure (thesis, Chapter 3):

    1. estimate the posterior p(c | P) of the concentration and flag the degenerate cases
       (exact ties after clipping);
    2. compute the predictive law of the probability that a further member would assign to the case,
       and read where the threshold tau falls relative to it, on the side of the decision
       (central-band criterion).

Minimal use::

    from confidex import assess
    out = assess(P, tau=0.3)            # P of shape (n, K)
    out["status"]      # 'degenerate' | 'confident' | 'not_confident'
    out["u"]           # signed score in [-1/2, 1/2]: negative if the law contradicts the decision
    out["confident_plugin"]             # the same criterion with the law at the posterior mean of c

The admissibility gate (``calibrate_gate``, ``admissible``, argument ``v_star``) is kept for the
ablations but is not part of the procedure.
"""
from .config import (
    DEFAULT_CRITERION,
    DEFAULT_MODEL,
    CriterionConfig,
    ModelConfig,
)
from .core import (
    LogitGrid,
    MemberLaw,
    PosteriorC,
    admissible,
    assess,
    calibrate_gate,
    confidence_score,
    fit_posterior,
    member_law,
    shape_regime,
)
from .full import PosteriorFull, fit_posterior_full
from .diagnostics import border_share, keff_deviations

__all__ = [
    "ModelConfig", "CriterionConfig", "DEFAULT_MODEL", "DEFAULT_CRITERION",
    "LogitGrid", "PosteriorC", "MemberLaw", "PosteriorFull",
    "fit_posterior", "member_law", "calibrate_gate", "admissible", "assess",
    "confidence_score", "shape_regime", "fit_posterior_full",
    "keff_deviations", "border_share",
]
__version__ = "0.5.1"
