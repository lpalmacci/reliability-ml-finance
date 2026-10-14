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
| D2 | 2026-10-13 | Non-causal ablation | G2 with the hyperparameters of G1 (16 hidden units) | G2 reported as pre-registered; in addition an exploratory variant with 256 hidden units (configs/exploratory.yaml, experiment X1) | After training, every run of G2 produced near-deterministic paths (annual volatility 0.02-0.04 against 0.18-0.20 of the processes): with a 16-unit bottleneck, a map from the whole path to the whole path cannot generate daily noise. G2 therefore does not isolate the effect of causality (VQ4). The wider variant is exploratory and labelled as such wherever it is reported. |
| D3 | 2026-10-14 | Size B of the L1 null | 100 pairs per process | 3,000 pairs per process (configs/full.yaml); the first 100 pairs are the pre-registered null | Found when analysing E2. With B = 100 the smallest empirical p-value is 1/101 = 0.0099, larger than the Holm threshold of the first step, 0.05/13 = 0.0038: the L1 flag could never be raised, whatever the defect (E2: L1 power 0 for every defect, process and severity, although raw per-statistic p-values reached their minimum). B is chosen so that the first Holm threshold corresponds to at least 10 null exceedances, (B+1) * 0.05/13 >= 10. The pair b of the null uses the same random stream for any B, so the pre-registered null is the subset b < 100. Tasks store the discrepancies, so p-values and flags of all experiments are recomputed against the new null (`python -m tavp.experiments.l1_null`) without re-running them; tables report both (`p`, `flag_*` and `p_prereg`, `flag_*_prereg`). |
| D4 | 2026-10-14 | Berkowitz test in L4 | Joint test (mean, variance, AR(1) of the normal scores) | Marginal test (mean and variance) in mode S (configs/full.yaml, `protocol.L4.berkowitz`) | Found in the negative control of L4 (E6, oracle in pseudo-real mode). In mode S the generator targets the stationary law, and the PITs of consecutive non-overlapping origins of one history are dependent even under the true law because volatility is persistent; the joint test also tests their independence and rejected the oracle in 3 of 12 cells on D2 and 1 of 12 on D3 (always through the AR(1) component; KS and Kupiec never rejected). The scope of L4 in mode S, as stated in the protocol, is marginal calibration, which is what the marginal test checks. Tests are recomputed from the stored PITs (`--tests-only`); both decisions are reported. |

## Observations from the negative control (not deviations)

- L1, first E1 run: with the pre-registered null of B = 100 pairs, raw per-statistic false-alarm rates average 5.8% and exceed 9% in 7 of 52 (process, statistic) cells; the family-wise rate with the Holm correction, which is the rule used to flag, is 0% on every process. Part of the per-statistic dispersion comes from the null sample being shared by the 50 comparisons: against an independent null of B = 500 pairs, the largest rate (Hill index on D4) falls from 20% to 6%. A sensitivity analysis against an independent null is part of E7. Note (D3): with B = 100 the Holm rate is 0% by construction, so this observation does not measure the size of the L1 flag; see D3.
