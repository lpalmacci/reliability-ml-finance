# The task-aware validation protocol

This document defines the protocol implemented in `src/tavp/protocol/`. The formal
treatment and the motivation of each component are in the thesis chapter; references to
"the review" are to the systematic review of the previous chapter.

## Inputs

1. **A generator** of scenarios: paths of `T` daily log-returns that continue an observed
   history (the context). Scenarios are anchored at the last observed log-price.
2. **A declared use** `U = (rules, objective, metrics, tolerances)`, fixed before the
   evaluation:
   - a family of decision rules indexed by a finite grid of parameters;
   - the objective `J` on which the parameters are calibrated on the context;
   - the metrics of each rule on a scenario (here: annualised Sharpe ratio, maximum
     drawdown, horizon log-return) and the functionals of their laws (mean, quantiles,
     Value-at-Risk, Expected Shortfall);
   - a tolerance `eta` for each functional: the largest discrepancy considered
     economically irrelevant.
3. **A target law**: in mode S (stationary) the law of `T` consecutive returns of the
   stationary process; in mode C (conditional) the law of the next `T` returns given the
   context. Levels L0-L3 need an oracle that simulates the target law exactly; level L4
   does not.

## Induced laws

For a law `mu` of scenarios and a calibrated rule with parameters `lambda`, the induced law
of the metrics is the push-forward `F_lambda # mu`. The decision levels compare the induced
laws under the generator and under the oracle; the statistical level compares the laws of
the scenarios themselves.

## Levels

**L0: consistency, validity, resolvability (prerequisite).**
- The hash of the parameters of the process used to train the generator must equal the
  hash of the oracle (`Generator.target_hash == DGP.hash()`).
- Scenarios must be finite; invalid scenarios are reported, never discarded.
- Negative control: the oracle, used as a generator of itself, must pass every level
  (experiment E1).
- For every declared functional the half-width of the 90% confidence interval of the
  discrepancy must not exceed `eta / 2`; otherwise `N` is increased.

**L1: statistical fidelity (diagnostic).** `N1` generated and `N1` independent oracle
scenarios are compared through 13 statistics: mean, standard deviation, skewness, kurtosis,
Hill tail index (top 5% of absolute returns), autocorrelation of returns (lags 1-5) and of
absolute returns (lags 1-20), leverage correlation `corr(r_t, r_{t+k}^2)` (lags 1-10),
kurtosis of 5- and 20-day aggregated returns, average 1-Wasserstein distance between
marginals, sliced 2-Wasserstein distance between log-price paths, and the distance between
expected signatures (level 3) of time-augmented log-price paths. The discrepancy of each
statistic is tested against its null law, estimated from `B` pairs of independent oracle
samples, with empirical p-values `(1 + #{null >= observed}) / (1 + B)` and the Holm
correction. L1 does not enter the verdict: it explains failures of L2-L3 or identifies
defects that are irrelevant for the use.

**L2: evaluation fidelity.** For each of `K` contexts, every rule is calibrated on the
context; `N` generated and `N` oracle scenarios are appended to it; for every rule, metric
and functional, the two one-sided tests (TOST) declare equivalence at level 0.05 if the 90%
confidence interval of the discrepancy lies within `[-eta, eta]`. Standard errors are
asymptotic and distribution-free (`src/tavp/tasks/functionals.py`). A cell passes if the
share of contexts (and training runs) in which equivalence is declared is at least
`PR_min = 0.9`.

**L3: optimisation fidelity.** For each of `K3` contexts, every grid point of every
calibrated rule is evaluated on `N` generated scenarios and on two independent oracle
samples, with the same scenarios for all grid points. The regret of choosing the rule on
generated scenarios is measured on the second oracle sample:
`Reg = mean_O2[J(lambda*_O1)] - mean_O2[J(lambda_G)]`. A context passes if the upper bound of
the 90% interval of the regret does not exceed the tolerance; a rule passes if the share of
passing contexts is at least `PR_min`. The Kendall rank correlation of the grid points
under generator and oracle is reported.

**L4: calibration without an oracle.** At non-overlapping origins of a history, the rule
is calibrated on the preceding `T0` returns, the generator produces the induced law of each
metric, and the realised metric of the next `T` returns is mapped to its probability
integral transform (PIT). Uniformity is tested with Kolmogorov-Smirnov, Berkowitz (2001)
and Kupiec (1995) tests with the Holm correction. In mode S the test concerns marginal
calibration. In the chapter, L4 is validated on long oracle paths treated as observed
histories (pseudo-real mode).

## Verdict and validation card

A generator is **fit for the declared use on a process** if it passes L0, every declared
cell of L2, L3 for every calibrated rule and, when applied, L4. The validation card
(`src/tavp/protocol/card.py`) reports the evidence of every level in a fixed format,
including the properties flagged by L1.

## Defaults used in the chapter

See `configs/preregistration.yaml` and `docs/DEVIATIONS.md`. In short: daily data, context
of 500 returns, scenarios of 59 returns, four rules (buy-and-hold, trend-following,
volatility targeting, trend with volatility filter), objective
`J = Sharpe - 0.01 turnover - MaxDD`, `N1 = 5,000`, `B = 100`, `N = 100,000` for L2 (after
deviation D1), `K = 50`, `N = 40,000` and `K3 = 10` for L3.
