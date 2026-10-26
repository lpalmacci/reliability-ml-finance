"""Configuration of the library.

All the default values discussed in Chapter 3 of the thesis live here, each with a reference to the
section or result that justifies it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Parameters of the model of the members' probabilities and of the inference on the concentration c.

    ``prior_a``/``prior_b``
        Prior on c. The default, Beta(2, 3), is the only one of the priors considered with
        E[lambda] = lambda_0 and a finite variance on the scale of the precision (Chapter 3,
        Proposition "Induced prior on the precision"). Beta(1, 1) is the canonical reference of the
        sensitivity analysis, Beta(2, 2) an earlier default, Beta(1/2, 1/2) the option closest to the
        Jeffreys prior.

    ``lambda0``
        Precision at the midpoint of the scale: c = 1/2 <=> lambda = lambda_0. If None, it is set to
        ``lambda0_scale * K``, with K the number of members; the default lambda_0 = K gives the prior
        one pseudo-observation per member (Chapter 3, Inference on the concentration).

    ``lambda0_scale``
        Factor of the adaptive anchoring, used only when ``lambda0`` is None. The sensitivity analysis
        uses {K/3, K, 3K}, i.e. scales in {1/3, 1, 3}, and the extremes {0.1K, 10K}. Setting ``lambda0``
        to a number instead gives a *constant* anchoring across ensemble sizes, the second variant used
        when sizes are compared.

    ``grid_bound`` / ``grid_points``
        Border and resolution of the logit grid. The border matters only for the cases whose posterior
        reaches it: on the synthetic ensembles of Chapter 3, moving it from 1e-6 to 1e-4 changes the
        posterior means by at most 3e-6.

    ``clip_eps``
        Clipping of the members' probabilities before inference. It must stay well below the smallest
        probability the ensemble can return.

    ``tie_eps``
        Threshold below which the members, *after clipping*, are treated as an exact tie. The posterior
        is then improper if b0 <= K/2 (Chapter 3, Proposition "Impropriety under exact ties"), and the
        case must be flagged, not estimated.

    ``border_fraction`` / ``border_tol``
        Diagnostic of the grid border, not an exclusion criterion. A case is flagged ``at_border`` when
        the posterior puts more than ``border_tol`` of its mass in the fraction ``border_fraction`` of
        grid points closest to either end: the precision then exceeds the largest the grid represents
        (near-ties, or members agreeing very closely on small probabilities with many members). The
        verdict remains valid and errs on the side of caution, because the reconstructed law is wider
        than it should be; if the share of such cases is not negligible, reduce ``grid_bound`` (for
        instance 1e-9 with 1500 points).
    """

    prior_a: float = 2.0
    prior_b: float = 3.0
    lambda0: float | None = None
    lambda0_scale: float = 1.0
    grid_bound: float = 1e-6
    grid_points: int = 1000
    clip_eps: float = 1e-6
    tie_eps: float = 1e-9
    border_fraction: float = 0.02
    border_tol: float = 1e-3

    def __post_init__(self) -> None:
        if self.prior_a <= 0 or self.prior_b <= 0:
            raise ValueError("prior_a and prior_b must be > 0")
        if not (0 < self.grid_bound < 0.1):
            raise ValueError("grid_bound must lie in (0, 0.1)")
        if self.grid_points < 100:
            raise ValueError("grid_points must be >= 100")
        if self.lambda0 is not None and self.lambda0 <= 0:
            raise ValueError("lambda0 must be > 0")
        if self.lambda0_scale <= 0:
            raise ValueError("lambda0_scale must be > 0")
        if not (0 < self.border_fraction < 0.5):
            raise ValueError("border_fraction must lie in (0, 0.5)")
        if not (0 < self.border_tol < 1):
            raise ValueError("border_tol must lie in (0, 1)")

    def resolve_lambda0(self, K: int) -> float:
        """Anchoring constant actually used for an ensemble of K members."""
        return float(self.lambda0) if self.lambda0 is not None \
            else float(self.lambda0_scale) * float(K)


@dataclass(frozen=True)
class CriterionConfig:
    """Parameters of the criterion and of the optional admissibility gate.

    ``alpha``
        Level of the central band: a case is not confident if tau falls between the quantile of order
        alpha and the quantile of order 1 - alpha of the members' law. The default, 0.40, is the band
        between the 40th and the 60th percentile. The value is fixed in advance and must not be tuned
        on the results.

    ``abstention``
        Abstention rate of the optional gate (``calibrate_gate``). The gate is NOT part of the
        procedure: on the synthetic experiments of Chapter 3 it spends review capacity on cases that are
        mostly classified correctly (Chapter 3, Remark "An admissibility gate, and why it is not
        used"). It is kept for the ablations only.
    """

    alpha: float = 0.40
    abstention: float = 0.10

    def __post_init__(self) -> None:
        if not (0 < self.alpha < 0.5):
            raise ValueError("alpha must lie in (0, 0.5)")
        if not (0 <= self.abstention < 1):
            raise ValueError("abstention must lie in [0, 1)")


DEFAULT_MODEL = ModelConfig()
DEFAULT_CRITERION = CriterionConfig()
