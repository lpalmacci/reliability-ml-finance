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
  learning rates 1e-4, batch 64, 6,000 steps, R1 weight 1, starting from the public replication of
  Cascha (2021, github.com/ICascha/QuantGANs-replication; no code reused);
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

Result: FILLED IN BEFORE THE TAG.

## Experiment E13

Data, contexts, oracle scenarios, sizes, declared use, tolerances and decision rules are those of the
main study (configs/full.yaml) with no change.

- Training: five runs per process and regime. Regime `single`: the first ten years of each of the five
  histories of the process, as for every generator. Regime `rich`: the five long histories of E8, with
  151,040 returns each; the Quant GAN needs a history longer than its receptive field, so it cannot use
  the independent 60-price paths on which the TC-VAE is trained in its rich regime.
- Levels L0-L3 as in E4, E5 and E8 (`python -m tavp.experiments.e13_qgan --stage protocol`), with the
  attainable cells of E8 for the same amount of data.
- Level L4 on the held-out half of each history (regime `single`) as in E6.
- Level L4 on nine held-out decades of the US market as in E11 (three runs per window).
- Not included: the pseudo-real oracles of E10.

## Hypotheses (stated before E13 was run)

- HQ1 (tails). On process D3 the kurtosis and the Hill index of the daily returns of G3 are closer to
  those of the oracle than those of the TC-VAE (medians over runs), in both regimes.
- HQ2 (no spurious mean reversion). The first-order autocorrelation of the returns of G3 is smaller
  than 0.02 in absolute value in every run that does not collapse, on every process, and smaller in
  absolute value than the median of the TC-VAE on each process.
- HQ3 (volatility clustering). With the long histories, the autocorrelation of absolute returns of G3
  is closer to that of the oracle than that of the TC-VAE with rich data on processes D2 to D4.
- HQ4 (data limit). With ten years, the standard deviation across histories of the annual drift of
  G3 is within 25% of that of the sample mean, and G3 passes no cell outside the attainable set.
- HQ5 (attainable cells). Trained on the long histories, G3 passes more attainable cells than the
  TC-VAE with rich data on each of processes D2 to D4, and no more than the GJR-GARCH model trained on
  the same histories on process D3.
- HQ6 (instability). The range across runs of the number of attainable cells passed is larger for G3
  than for the TC-VAE in the same regime, on at least three of the four processes.
- No directional hypothesis for levels L3 and L4; their results are reported.

Every result is reported whatever its outcome; runs that collapse are reported and kept in the pooled
verdicts, as for the TC-VAE.
