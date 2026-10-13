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

| Id | Date | Setting | Pre-registered | Applied | Reason |
|---|---|---|---|---|---|
| D1 | 2026-10-13 | N of L2 and of the meta-validation | 40,000 | 100,000 (configs/full.yaml) | Pre-registered L0.4 contingency. In the negative control (E1, first run) the 90% half-width exceeded eta/2 for the median and mean Sharpe ratio (0.025-0.032 against 0.025) and for the Expected Shortfall of buy-and-hold under GJR-GARCH (0.007 against 0.005), so that even the oracle failed these cells (pass rates 0.46-0.94). N is raised to the pre-registered maximum. The first E1 run is kept in results/e1_n40k for reference. L3 keeps N = 40,000: its negative control passed in every context. |

## Observations from the negative control (not deviations)

- L1, first E1 run: with the pre-registered null of B = 100 pairs, raw per-statistic false-alarm rates average 5.8% and exceed 9% in 7 of 52 (process, statistic) cells; the family-wise rate with the Holm correction, which is the rule used to flag, is 0% on every process. Part of the per-statistic dispersion comes from the null sample being shared by the 50 comparisons: against an independent null of B = 500 pairs, the largest rate (Hill index on D4) falls from 20% to 6%. A sensitivity analysis with B = 500 is part of E7.
