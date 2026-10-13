# Deviations log

Every change to `configs/preregistration.yaml` made after the `prereg` tag is recorded here,
with date and reason. Results are always reported under the pre-registered settings first.

## Refinements of the development plan made before pre-registration

These choices were fixed while writing the code, before any experiment, and are part of the
pre-registered configuration.

| Date | Item | Plan v2 | Pre-registered choice | Reason |
|---|---|---|---|---|
| 2026-10-13 | Final TC-VAE model | Best model on a validation loss | Model of the last epoch | The original trainer always keeps the last epoch; its evaluation step only logs an MMD on 1,000 samples. The original procedure is followed. |
| 2026-10-13 | TVF volatility filter | Quantile q of sigma_hat over the trailing 120 days | Threshold = quantile q of sigma_hat over the in-sample part of the context, fixed at calibration | Keeps the filter a calibrated parameter of the rule and makes the L3 grid evaluation tractable at N = 40,000. |
| 2026-10-13 | GJR-GARCH burn-in | 2,000 steps | 500 steps | With persistence 0.98 the initial condition decays by 0.98^500 (about 4e-5). |
| 2026-10-13 | Context warm-up | not specified | 200 log-prices prepended to each scenario; calibration on the last 300 returns of the context | The longest indicator window is 120 days. |
| 2026-10-13 | Turnover | not specified | Sum of absolute position changes over the evaluated period, annualised | Standard definition, used in J. |

## Deviations after pre-registration

None yet.
