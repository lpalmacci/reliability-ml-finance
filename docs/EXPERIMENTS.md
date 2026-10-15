# Experiments

Every experiment of the chapter, with its question, its status, the code that runs it, where its
results are written and where the chapter reports them. Status: **P** pre-registered (tag
`prereg`); **A** pre-registered by addendum before it was run (tag `prereg-qgan`); **D** added after
the results, with its design recorded in [`DEVIATIONS.md`](DEVIATIONS.md) before its own results.
Commands take `--config` (default `configs/full.yaml`) and `--jobs`; `make all` runs them in order.

| Exp. | Question | Status | Command (`python -m ...`) | Results | Chapter |
|---|---|---|---|---|---|
| E0 | Does the vectorised backtest reproduce the buy-and-hold figures of the earlier monthly GBM setting (internal, not reported)? | port check | `tavp.experiments.e0_port` | `docs/E0_port_check.json` | -- |
| Training | TC-VAE (G1) and the non-causal ablation (G2), five and three runs per process and regime | P | `tavp.experiments.train` | `results/runs/` | 6.1, 6.3 |
| E1 | Does the protocol reject the true law (false alarms)? Null of level L1 | P (D1, D3) | `tavp.experiments.e1_null` | `results/e1/` | 5.2 |
| E2 | Does each level detect planted defects P1-P6 of graded severity (power)? | P | `tavp.experiments.e2_defects` | `results/e2/` | 5.3 |
| E3 | Do the statistical and the decision levels disagree (dissociation)? | P | from E2, in the report | `results/e2/` | 5.4 |
| E6 meta | Is level L4 valid on simulated histories, and how powerful is it? | P (D4) | `tavp.experiments.e6_calibration --part meta`; `tavp.experiments.l4_size` | `results/e6_meta/`, `results/e6_size/` | 5.5 |
| E12 | Where does detection begin (severities 0.05-0.20)? Effective tolerance of L2 | D9 | `tavp.experiments.e12_fine` | `results/e12/` | 5.3.1 |
| Q0, Q0b | Does the Quant GAN implementation reproduce the paper? One bounded tuning round | A | `tavp.experiments.q0_fidelity`; `tavp.experiments.q0_tune` | `results/q0/` | 6.2 |
| E4 | Is the TC-VAE fit for the declared use? Validation cards, stylized facts | P | `tavp.experiments.e4_case`; `tavp.experiments.e4_facts` | `results/e4/` | 7.1, 7.2 |
| E5 | How do the benchmarks and the ablation compare? | P (D2) | `tavp.experiments.e5_compare` | `results/e5/` | 7.2 |
| X1 | Exploratory wide ablation (256 hidden units) | D2 | `make exploratory` | `results/x1_wide/` | 7.2, App. A |
| E13 | Is the Quant GAN (G3) fit for the declared use? L0-L3, L4 on held-out histories and on the market; hypotheses HQ1-HQ6 | A (D10); runs 4-5 D11 | `tavp.experiments.e13_qgan --stage train/facts/protocol/l4`; `tavp.experiments.e11_market_l4 --config configs/real.yaml --qgan` | `results/e13/`, `results/runs/qgan/`, `results/real/e11/` | 7.1-7.3, 7.6, 7.7, 7.9 |
| E6 case | Level L4 on the held-out half of each history | P (D4) | `tavp.experiments.e6_calibration --part case` | `results/e6_case/` | 7.6 |
| E8 | Which cells can the data support? Does the protocol certify a correctly specified generator? | D5 | `tavp.experiments.e8_attain` | `results/e8/` | 7.3 |
| Critical tolerances | How far is each verdict from its threshold? | D7 | in the report (`tavp.report.tolcrit`) | from E1, E4, E5, E8, E13 | 7.4 |
| E9 | Does the spurious autocorrelation of the TC-VAE explain its failures (planted defect P7)? | D6 | `tavp.experiments.e9_mr` | `results/e9/` | 7.5 |
| E14 | Do the extreme returns (planted defect P8) and the variance ratio (P7) of the Quant GAN explain its failures? | D11 | `tavp.experiments.e14_qgan_defects` | `results/e14/` | 7.5 |
| E10 | The protocol on oracles calibrated on the US market, 1926-2026, for every generator; multistart refit of B2 | D8 (G3: D11) | `tavp.experiments.e10_calibrate`; the experiments above with `configs/real.yaml`, including `tavp.experiments.e13_qgan --config configs/real.yaml --stage train/protocol`; `tavp.experiments.e10_multistart` | `results/real/`, `results/e10ms/` | 7.7 |
| E11 | Level L4 on nine held-out decades of the US market, three runs per deep generator | D8 (G3: run 1 A, runs 2-3 D11) | `tavp.experiments.e11_market_l4 --config configs/real.yaml` (`--qgan` for G3) | `results/real/e11/` | 7.7 |
| E7 | Are the verdicts robust to cost, objective, N, tolerances and the L1 null? | P (G3: D11) | `tavp.experiments.e7_robust` (`--with-qgan` for G3) | `results/e7_*/` | 7.8 |
| Curves | Descriptive figures (induced laws, autocorrelation, quantiles, paths) | descriptive | `tavp.experiments.e4_curves` | `results/e4_curves/` | 4.3, 7.1 |

## Generators

| Code | Generator | Implementation | Training data | Runs |
|---|---|---|---|---|
| O | Oracle (reference law) | `tavp.dgp.processes` | -- | -- |
| Ô | Oracle with estimated drift | `tavp.generators.base.EstimatedDriftOracle` | 10 years; long histories | 5 |
| P1-P8 | Oracles with planted defects | `tavp.perturb.operators` | -- | -- |
| G1 | Time-Causal VAE | `third_party/tcvae` (authors' code and configuration) via `tavp.generators.tcvae` | 10 years; rich set of 2,560 paths | 5 |
| G2, X1 | Non-causal ablations | as G1, MLP encoder and decoder | 10 years | 3 |
| G3 | Quant GAN, pure TCN | `tavp.generators.qgan` (implemented from the paper) | 10 years; long histories | 5 (3 addendum + 2 D11) |
| B1 | Stationary bootstrap | `tavp.generators.base.StationaryBootstrap` | 10 years; long histories | 5 |
| B2 | GJR-GARCH(1,1)-t, maximum likelihood | `tavp.generators.base.GarchBaseline` (`arch`) | 10 years; long histories | 5 |

The reference processes are D1 geometric Brownian motion, D2 Heston, D3 GJR-GARCH(1,1)-t and D4 a
two-state regime-switching model (`configs/preregistration.yaml`), and the two processes estimated
on the market in E10 (`configs/real.yaml`, generated by `tavp.experiments.e10_calibrate`).
