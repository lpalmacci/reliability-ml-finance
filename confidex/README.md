# confidex

Per-sample reliability of the classifications of an ensemble of binary classifiers, from the
reconstructed law of the members' probabilities. The library implements the procedure of Chapter 3 of
the thesis (*Confidex*).

For each case, the `K` probabilities that the members assign to the positive class are treated as a
sample from a Beta law whose location is fixed at their mean `pbar` and whose concentration `c` is
inferred. The posterior of `c` is computed on a deterministic grid, and is used to reconstruct the
**predictive law** of the probability that a further member would assign to the case. The ensemble is
declared *confident* on the case when that law lies beyond the central band `[q_alpha, q_{1-alpha}]` on
the side of the decision `1[pbar >= tau]`, that is, when a further member would side with the decision
with probability greater than `1 - alpha` (`alpha = 0.40` by default).

## Installation

```bash
pip install -e .            # from this folder
pip install -e ".[test]"    # with pytest
pytest tests                # 18 tests, about 20 seconds
```

Requirements: Python 3.10 or later, NumPy and SciPy.

## Use

```python
import numpy as np
from confidex import assess

P = np.array([[0.62, 0.71, 0.48, 0.66, 0.55],   # one row per case, one column per member
              [0.08, 0.11, 0.09, 0.10, 0.07]])
out = assess(P, tau=0.3)
out["status"]             # 'degenerate' | 'confident' | 'not_confident'
out["u"]                  # signed score in [-1/2, 1/2]; negative when the law contradicts the decision
out["confident_plugin"]   # the same criterion with the law at the posterior mean of c (approximation)
```

To read the cases in order of doubt, sort them by `u`, from the smallest. The lower-level functions
are `fit_posterior` (posterior of `c`, and the predictive law at the thresholds passed in `taus`),
`member_law` (the Beta law at the posterior mean of `c`) and `confidence_score` (criterion and score).

## Modules

| Module | Content |
|---|---|
| `core` | Posterior of the concentration on a logit grid, predictive law of a further member, law at the posterior mean of `c`, directional central-band criterion, optional admissibility gate (ablations only) |
| `full` | Specification with an inferred location, on a two-dimensional grid (experimental; used only as a yardstick for the fixed-mean assumption) |
| `diagnostics` | Effective number of members from the deviations `p_k - pbar`, shape of the reconstructed law (unimodal, J, U), share of cases at the border of the grid |
| `config` | All the default values, each with the result of Chapter 3 that justifies it |

## Notes

- The summaries of the posterior of `c` (`c_mean`, `c_var`) are available, but they are **not
  comparable across cases with different means**: the concentration measures the dispersion of the
  members relative to the largest dispersion compatible with their mean (Chapter 3, *The concentration
  is a relative measure*). The verdict is read through `confidence_score`.
- The admissibility gate (`calibrate_gate`, `admissible`, argument `v_star`) is kept for the ablations
  of the thesis only: on the synthetic experiments of Chapter 3 it worsens the selection of the cases to
  review.
- Exact ties among the members (after clipping) make the posterior improper for most priors; such
  cases are flagged as `degenerate` rather than estimated.

## Versions

- **0.5.1**: documentation and messages in English; the shape label `'unimodale'` returned by
  `shape_regime` is now `'unimodal'`. The numerical results are identical to those of 0.5.0.
- **0.5.0**: the version with which all the results of Chapters 3 and 4 of the thesis were computed.

## Licence

Apache License 2.0 (see `LICENSE` at the root of the repository).
