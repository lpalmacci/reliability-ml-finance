# Pre-registration addendum: a second deep generator (G3, Quant GAN)

Date: 2026-10-14. Status: committed under the git tag `prereg-qgan` before experiment E13 was run.
The main pre-registration (`configs/preregistration.yaml`, tag `prereg`) is unchanged; this addendum
extends the case study and is recorded as deviation D10 in `docs/DEVIATIONS.md`.

## Why

The case study of the main pre-registration validates one deep generator, the Time-Causal VAE, a
variational model. In the corpus of the systematic review, adversarial networks account for 53% of
the studies, and the review reports that Quant GANs reproduce the distribution and dependence of daily
S&P 500 returns far better than a GARCH(1,1) (Wiese et al., 2020). A second deep generator from the
dominant family shows whether the protocol discriminates between deep generators, and tests the claim
of the original paper against a declared use.

## The generator

Quant GAN, pure TCN model (Wiese, Knobloch, Korn and Kretschmer, 2020, *Quantitative Finance* 20(9),
arXiv:1907.06673). No official code was released and the public replications carry no licence, so the
model is implemented from the paper in `src/tavp/generators/qgan.py` (Apache-2.0):

- specified by the paper and followed exactly: preprocessing (log returns, standardisation, inverse
  Lambert W x Gaussian transform by maximum likelihood, standardisation, rolling windows), generator
  and discriminator (Appendix B: TCNs with skip connections, seven temporal blocks of two dilated
  causal convolutions and two PReLUs, kernel sizes 1 then 2, dilations 1, 1, 2, ..., 32, 80 hidden
  channels, 1x1 output convolution, receptive field 127), three-dimensional Gaussian noise, GAN
  objective with the R1 penalty of Mescheder et al. (2018);
- not reported by the paper and fixed here (`configs/generators/qgan.yaml`): Adam with betas (0, 0.9),
  discriminator learning rate 1e-4 and batch 64 from the public replication of Cascha (2021,
  github.com/ICascha/QuantGANs-replication; no code reused); generator learning rate 3e-5, R1 weight 10 and
  12,000 steps from the tuning round Q0b below (the starting values were 1e-4, 1 and 6,000);
- checkpoint: one every 250 steps; the selected one minimises the mean rank of the twelve metrics of
  Table 2 of the paper computed on the training data only, as the authors chose the best of several
  saved parameter sets by those metrics;
- hardware: Apple GPU (MPS) when available; on the same machine training and sampling are deterministic.

## Fidelity check (Q0, run before this addendum was frozen)

The experiment of the paper (daily log returns May 2009 - December 2018, TCN against a GARCH(1,1) with
constant drift and Gaussian innovations, twelve metrics) is repeated with three training runs on the
daily US market return of the Ken French Data Library over the same period. Acceptance criterion,
fixed before the check: for at least two of the three runs the TCN scores better than the GARCH(1,1)
on at least 10 of the 12 metrics, including the four EMDs.

Result of Q0 (starting values): NOT ACCEPTED. The three runs beat the GARCH(1,1) on 8, 4 and 9 of the 12
metrics. They reproduce the advantage reported by the paper on the return distribution (EMD and DY metric at
short horizons) and on the leverage effect (score 0.35-0.38 against 0.46, close to the 0.33 against 0.46 of
the paper), but not on volatility clustering: the ACF scores of returns, absolute and squared returns are worse
than those of the GARCH(1,1) in every run, mainly at lags 21-127, where the autocorrelation of absolute
returns of the TCN decays too fast.

Q0b, one bounded tuning round fixed and committed before it was run (`tavp.experiments.q0_tune`, commit
7dd24de): 12,000 steps, generator learning rate in {1e-4, 3e-5}, R1 weight in {1, 10}, one new seed per
configuration; choice by the number of metrics beaten; confirmation on three new seeds with the criterion of
Q0 unchanged. Chosen: generator learning rate 3e-5, R1 weight 10. Confirmation: ACCEPTED, the three runs beat
the GARCH(1,1) on 11, 10 and 10 of the 12 metrics, including the four EMDs in each run. The settings are
frozen in `configs/generators/qgan.yaml`.

The absolute values of the DY metric and of the ACF scores are not comparable with Table 2 of the paper,
whose partition and normalisation are not fully specified; the comparison with the GARCH(1,1) is made under
the same implementation, as in the paper. The paper reports one run, selected among several saved parameter
sets by the same metrics; the check above uses three runs and a selection rule fixed in advance.

## Experiment E13

Data, contexts, oracle scenarios, sizes, declared use, tolerances and decision rules are those of the
main study (configs/full.yaml) with no change.

- Training: three runs per process and regime, because each model needs twice the steps planned before the
  fidelity check (24 models). Regime `single`: the first ten years of the first three histories of the
  process, as for every generator. Regime `rich`: the first three long histories of E8, with 151,040 returns
  each; the Quant GAN needs a history longer than its receptive field, so it cannot use
  the independent 60-price paths on which the TC-VAE is trained in its rich regime.
- Levels L0-L3 as in E4, E5 and E8 (`python -m tavp.experiments.e13_qgan --stage protocol`), with the
  attainable cells of E8 for the same amount of data.
- Level L4 on the held-out half of each history (regime `single`) as in E6.
- Level L4 on nine held-out decades of the US market as in E11 (one run per window).
- Not included: the pseudo-real oracles of E10.

## Hypotheses (stated before E13 was run)

- HQ1 (tails). On process D3 the kurtosis and the Hill index of the daily returns of G3 are closer to
  those of the oracle than those of the TC-VAE (medians over runs), in both regimes.
- HQ2 (no spurious mean reversion). The first-order autocorrelation of the returns of G3 is smaller
  than 0.02 in absolute value in every run that does not collapse, on every process, and smaller in
  absolute value than the median of the TC-VAE on each process.
- HQ3 (volatility clustering). With the long histories, the autocorrelation of absolute returns of G3
  is closer to that of the oracle than that of the TC-VAE with rich data on processes D2 to D4.
- HQ4 (data limit). With ten years, the standard deviation across the three histories of the annual drift
  of G3 is within 25% of that of the sample mean of the same histories, and G3 passes no cell outside the
  attainable set.
- HQ5 (attainable cells). Trained on the long histories, G3 passes more attainable cells than the
  TC-VAE with rich data on each of processes D2 to D4, and no more than the GJR-GARCH model trained on
  the same histories on process D3.
- HQ6 (instability). The range across runs of the number of attainable cells passed is larger for G3
  than for the first three runs of the TC-VAE in the same regime, on at least three of the four processes.
- No directional hypothesis for levels L3 and L4; their results are reported.

Every result is reported whatever its outcome; runs that collapse are reported and kept in the pooled
verdicts, as for the TC-VAE.

## Outcome (added on 2026-10-15, after E13 was run)

The hypotheses above are unchanged; this section records their outcomes, computed by `tavp.report.qgan`
exactly as stated (table `thesis_assets/tables/tab_qgan_hyp.tex`, Section 7.9 of the chapter).

- HQ1 not supported: on D3 the Hill index of G3 is closer to the oracle than that of the TC-VAE in both
  regimes, but its kurtosis overshoots by orders of magnitude (medians 4,620 and 109 against 21.9).
- HQ2 not supported: the median first-order autocorrelation of G3 is smaller in absolute value than that of
  the TC-VAE in every setting, but 11 of the 24 runs exceed 0.02 in absolute value.
- HQ3 supported on D2, D3 and D4.
- HQ4 not supported: no cell outside the attainable set is passed, but the standard deviation of the drift of
  G3 across histories is 1.19 to 1.61 times that of the sample mean.
- HQ5 supported.
- HQ6 supported with the long histories (3 of 4 processes), not with ten years (0 of 4).

Observation, not foreseen by the addendum: one run (D3, ten years, run 1) generates finite scenarios with
returns so large that the drawdown of some rules cannot be computed in 4 of the 50 contexts of level L2. Level
L0 declares them valid (every value is finite); level L2 counts the affected comparisons as failures. See
`docs/DEVIATIONS.md`.

## Extension after E13 (deviation D11, 2026-10-15)

After E13 the author asked that the Quant GAN be validated with every analysis of the TC-VAE. Deviation D11 adds
runs 4 and 5 per process and regime, three runs per window on the market, the processes calibrated on the market
(E10), the robustness variants of E7 and the test of its properties as planted defects (E14); its design was committed
before any of these results (`docs/DEVIATIONS.md`). The hypotheses above are not changed: they are evaluated on runs
1-3 as stated, and their outcomes on the five runs are reported alongside in the same table.

## Note on determinism (added 2026-10-26, after the results)

The statement under *The generator* (hardware) that training and sampling are deterministic on the same machine proved
wrong for training. In a full rerun on the thesis machine, with the same code, seeds and library versions, about
half of the Quant GANs coincided bit for bit with those of the thesis, and the others differed from the first
checkpoint on; sampling from a trained model is deterministic. The trained models used in the thesis are therefore
included in `weights/` (`weights/README.md`, `REPRODUCE.md`, *Determinism*). The pre-registered text above is left
as it was committed under the tag `prereg-qgan`.
