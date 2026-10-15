# tavp: a task-aware validation protocol for generative models of financial time series

This repository implements the validation protocol proposed in the thesis chapter
*A Task-Aware Validation Protocol for Generative Models of Financial Time Series*
(Luca Palmacci, PhD in Models and Methods for Economics and Finance, Sapienza University of
Rome) and reproduces every number, table and figure of that chapter.

**Reviewers: start from [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md).** It gives a reading path from
the chapter to the code and three levels of verification, from five minutes to the full study.

## What the protocol does

A scenario generator is judged by the use for which it is intended. The use is declared in
advance (decision rules, calibration objective, metrics, tolerances), and the generator is
compared with a known reference law on five levels:

| Level | Question | Decides fitness? |
|---|---|---|
| L0 | Do generator and reference refer to the same process; are the scenarios valid; can the tolerances be resolved with the chosen number of scenarios? | prerequisite |
| L1 | Does the generator reproduce the stylized facts and the law of paths? | no (diagnostic) |
| L2 | Are the induced laws of the metrics of calibrated rules equivalent, within the tolerances, to those of the reference? | yes |
| L3 | How much is lost when rules are chosen on generated rather than true scenarios (regret)? | yes |
| L4 | On data with unknown law, are the realised metrics calibrated with respect to the induced laws (PIT)? | yes, when applied |

The protocol is itself validated with known-answer tests: defects of graded severity are
planted in four reference processes (geometric Brownian motion, Heston, GJR-GARCH-t, regime
switching) and the size and power of every level are measured. It is then applied to two deep
generators, the Time-Causal VAE of Acciaio, Eckstein and Hou (2024) and a Quant GAN (Wiese et
al., 2020), against a stationary bootstrap and a GJR-GARCH-t model, on simulated processes, on
processes calibrated on a century of US market returns, and on the market itself.
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) defines the protocol; [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
maps every experiment to its question, code, outputs and section of the chapter.

## Quick start

Requirements: macOS or Linux, Python 3.12 and [uv](https://docs.astral.sh/uv/). Everything runs
on a laptop CPU; the Quant GAN uses the Apple GPU (MPS) when available and the CPU otherwise.

```sh
uv sync                 # environment with the exact versions of uv.lock
make test               # unit tests (about one minute)
make verify-frozen      # check the frozen results and recompute every number of the chapter (minutes)
make quick J=4          # smoke run of the simulated studies with small sizes (15-30 minutes)
```

`make all` runs the full study (about 25 hours on a 12-core laptop). `make thesis` rebuilds the
tables, figures and numbers from `results/`; `make verify` compares them with the frozen
snapshot of the thesis.

## Pre-registration and analyses added later

The scientific settings and the hypotheses were committed under the git tag `prereg` before
any experiment (`git show prereg:configs/preregistration.yaml`). The second deep generator was
pre-registered by an addendum under the tag `prereg-qgan` before its experiment
([`docs/ADDENDUM_QGAN.md`](docs/ADDENDUM_QGAN.md)). Every other change, and every analysis added
after the results, is recorded with its date and reason in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## Repository structure

```
README.md, REVIEWER_GUIDE.md, REPRODUCE.md   entry points
docs/               protocol definition, experiment map, deviations, addendum, data
configs/            pre-registration, run configurations (full, quick, exploratory, real), generators
src/tavp/dgp/       reference processes D1-D4 (oracles)
src/tavp/perturb/   planted defects P1-P8
src/tavp/generators/oracle, oracle with estimated drift, bootstrap, GJR-GARCH-t, TC-VAE wrapper, Quant GAN
src/tavp/tasks/     decision rules, backtest, calibration, functionals and equivalence tests
src/tavp/protocol/  levels L1-L4, statistics, validation cards
src/tavp/experiments/ one module per experiment (E1-E14, Q0) and the training scripts
src/tavp/report/    tables, figures and numbers of the chapter
third_party/tcvae/  Time-Causal VAE code (upstream commit a43effe, unmodified, Apache-2.0)
scripts/            data download, optional Tail-GAN download, freeze and verification
tests/              unit tests
results/frozen/     the snapshot of results used in the thesis (other outputs are regenerated)
thesis_assets/      tables, figures, validation cards and numbers.tex used by the chapter
```

## Data

The simulated studies need no data. The market-data analysis uses the daily US market return
of the Ken French Data Library, which is copyrighted and is therefore downloaded, not
redistributed: `python scripts/get_french.py` fetches it and reports whether the file is the
vintage used in the chapter (see [`docs/DATA.md`](docs/DATA.md)).

## Licence and attribution

Apache License 2.0 (see `LICENSE` and `NOTICE`). The Time-Causal VAE code in
`third_party/tcvae/` is by Songyan Hou and co-authors
(<https://github.com/justinhou95/TimeCausalVAE>) and is redistributed unmodified under the same
licence. The Quant GAN in `src/tavp/generators/qgan.py` is an independent implementation of the
published paper. Tail-GAN (GPL-3.0), mentioned in the chapter as a direction for future work, is
not part of this repository; `scripts/get_tailgan.py` downloads it at a pinned commit for users
who want it.

If you use this code, please cite the thesis chapter (see `CITATION.cff`).
