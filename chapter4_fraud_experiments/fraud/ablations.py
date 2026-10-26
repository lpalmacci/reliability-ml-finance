"""Ablations (Chapter 4, Ablations): variants of the order of doubt of Confidex.

For every draw and threshold, the decision (pbar = post.mu of the main analysis), the errors and the
budgets k_rho stay those of the main analysis: only the order of the cases changes, so that every variant
is compared with Confidex on the same cases and with the same bootstrap replicates. Every variant also has
its own verdict, hence its own k_alpha (cases declared not confident).

- law at the posterior mean of c: the reading "confidex plug-in" of the main analysis (not recomputed here);
- non-directional criterion: order by u_band = |F(tau) - 1/2| increasing, verdict "tau outside the band";
  same posterior;
- gate: v* = ``calibrate_gate`` on the posterior of the VALIDATION part of the same draw (abstention
  0.10); the verdict with the gate flags the abstained cases and the not-confident ones. Order for the
  budgets k_rho: first the abstained cases, then the others, each group by u_dir increasing; at the budget
  of the gate (k = flagged cases) the flagged set is compared with the first k cases of the order of
  Confidex (as in Chapter 3);
- lambda0 = 36 instead of lambda0 = K: one more ``fit_posterior`` on the test part (at K = 36 it is the
  main analysis);
- ULB only: clipping constant 1e-4 instead of 1e-6 in the inference, with one more ``fit_posterior``;
  the decision stays the one with the clipping 1e-6.

Degenerate cases go last in the order of doubt in every variant (u = +1/2), as in the main analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import confidex

GATE_ABSTENTION = 0.10
ABLATION_CLIP = 1e-4
POOL_LAMBDA0 = 36.0


@dataclass
class ExtraPosteriors:
    """The additional posteriors of a draw (independent of the threshold)."""
    lambda0_36: Any
    clip_1e4: Any | None
    v_star: float


def extra_posteriors(PV: np.ndarray, PT: np.ndarray, post: Any, K: int, taus: list[float],
                     cfg: confidex.ModelConfig, with_clip: bool, gate_only: bool = False) -> ExtraPosteriors:
    """``PV`` and ``PT`` are already clipped at cfg.clip_eps (1e-6); the 1e-4 variant clips them again.

    With ``gate_only`` only v* is computed (lambda0 = 36 and the 1e-4 clipping are not recomputed)."""
    v_star = confidex.calibrate_gate(confidex.fit_posterior(PV, cfg), confidex.CriterionConfig(abstention=GATE_ABSTENTION))
    if gate_only:
        return ExtraPosteriors(None, None, float(v_star))
    if K == POOL_LAMBDA0:
        p36 = post
    else:
        p36 = confidex.fit_posterior(PT, confidex.ModelConfig(lambda0=POOL_LAMBDA0), taus=taus)
    pc = confidex.fit_posterior(PT, confidex.ModelConfig(clip_eps=ABLATION_CLIP), taus=taus) if with_clip else None
    return ExtraPosteriors(p36, pc, float(v_star))


def _u(o: dict, post: Any, key: str = "u_dir") -> np.ndarray:
    return np.where(post.degenerate, 0.5, o[key])


def variants(post: Any, extra: ExtraPosteriors, tau: float,
             crit: confidex.CriterionConfig = confidex.DEFAULT_CRITERION) -> dict[str, dict[str, Any]]:
    """For every variant: score of doubt (higher = more doubtful) and k_alpha of its verdict."""
    out: dict[str, dict[str, Any]] = {}
    o_nd = confidex.confidence_score(post, None, tau, cfg=crit, directional=False)
    out["non-directional"] = {"doubt": -_u(o_nd, post, "u_band"),
                              "k_alpha": int((o_nd["status"] == "not_confident").sum())}
    if extra.lambda0_36 is not None:
        o36 = confidex.confidence_score(extra.lambda0_36, None, tau, cfg=crit)
        out["lambda0 = 36"] = {"doubt": -_u(o36, extra.lambda0_36),
                               "k_alpha": int((o36["status"] == "not_confident").sum())}
    if extra.clip_1e4 is not None:
        oc = confidex.confidence_score(extra.clip_1e4, None, tau, cfg=crit)
        out["clip 1e-4"] = {"doubt": -_u(oc, extra.clip_1e4),
                            "k_alpha": int((oc["status"] == "not_confident").sum())}
    og = confidex.confidence_score(post, None, tau, cfg=crit, v_star=extra.v_star)
    abstained = og["status"] == "abstain"
    flagged = abstained | (og["status"] == "not_confident")
    u = _u(og, post)
    out["gate"] = {"doubt": -u + 2.0 * abstained, "k_alpha": int(flagged.sum()),
                   "flagged": flagged, "abstained": abstained,
                   "share_abstention": float(abstained.mean())}
    return out
