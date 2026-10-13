# tavp: a task-aware validation protocol for generative models of financial time series

This repository implements the validation protocol proposed in the thesis chapter
*A Task-Aware Validation Protocol for Generative Models of Financial Time Series*
(Luca Palmacci, Sapienza University of Rome) and reproduces every number, table and
figure of that chapter.

The protocol judges a scenario generator by the use for which it is intended. The use is
declared in advance (decision rules, calibration objective, metrics, tolerances), and the
generator is compared with a known reference law on five levels:

| Level | Question | Decides fitness? |
|---|---|---|
| L0 | Do generator and reference refer to the same process; are scenarios valid; can the declared tolerances be resolved with the chosen number of scenarios? | prerequisite |
| L1 | Does the generator reproduce the stylized facts and the law of paths? | no (diagnostic) |
| L2 | Are the induced laws of the metrics of calibrated rules equivalent, within the tolerances, to those of the reference? | yes |
| L3 | How much is lost when rules are chosen on generated rather than true scenarios (regret)? | yes |
| L4 | On data with unknown law, are realised metrics calibrated with respect to the induced laws (PIT)? | yes, when applied |

The protocol itself is validated with known-answer tests: six defects of graded severity
are planted in four reference processes (geometric Brownian motion, Heston, GJR-GARCH-t,
regime switching), and the size and power of every level are measured. It is then applied
to the Time-Causal VAE of Acciaio, Eckstein and Hou (2024), with a stationary bootstrap and
a GJR-GARCH-t model as classical benchmarks and a non-causal ablation.

See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the definition of the protocol and
[`REPRODUCE.md`](REPRODUCE.md) for the step-by-step reproduction.

## Installation

Requirements: macOS or Linux, Python 3.12, [uv](https://docs.astral.sh/uv/). Everything
runs on CPU; no GPU is needed.

```sh
uv sync            # creates .venv with the exact versions of uv.lock
make test          # unit tests (about 10 seconds)
```

## Running

```sh
make quick         # smoke run of the whole pipeline with small sizes (about 20-30 minutes)
make all           # full pre-registered study (several hours on an 8-12 core laptop)
make thesis        # tables, figures and numbers of the chapter from results/
make verify        # recompute and compare with the frozen results of the thesis
```

`make all` runs, in order: training of the neural generators, the negative control (E1),
the planted defects (E2), the case study (E4, E5), the calibration tests (E6) and the
robustness checks (E7). The results of `make quick` are written to `results/quick/` and
never mix with those of the full run.

## Pre-registration

The scientific settings (processes, decision rules, tolerances, sizes, defects and the
hypotheses H1-H7 about which levels should detect them) are in
[`configs/preregistration.yaml`](configs/preregistration.yaml), committed under the git tag
`prereg` before any experiment was run. Every later change is recorded, with its date and
reason, in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md); runtime settings and the applied
deviations are in [`configs/full.yaml`](configs/full.yaml).

## Repository structure

```
configs/            pre-registration, run configurations, generator configurations
src/tavp/dgp/       reference processes D1-D4 (oracles)
src/tavp/perturb/   planted defects P1-P6
src/tavp/generators/oracle, stationary bootstrap, GJR-GARCH-t, wrapper of the TC-VAE code
src/tavp/tasks/     decision rules, backtest, calibration, functionals and equivalence tests
src/tavp/protocol/  levels L1-L4, statistics, validation cards
src/tavp/experiments/ training and experiments E0-E7
src/tavp/report/    tables, figures and numbers of the chapter
third_party/tcvae/  Time-Causal VAE code (upstream commit a43effe, unmodified, Apache-2.0)
tests/              unit tests
results/            outputs (generated; results/frozen/ holds the snapshot used in the thesis)
thesis_assets/      tables, figures, cards and numbers.tex used by the chapter
```

## Reproducibility

All random numbers derive from one master seed through named, independent streams
(`tavp.utils.rng`), so that every result can be regenerated exactly on the same platform.
Each output file is accompanied by a `.provenance.json` file with the configuration hash,
the git commit and the package version. Numbers quoted in the chapter are never typed by
hand: they are read from `thesis_assets/numbers.tex`.

## Licence and attribution

Apache License 2.0 (see `LICENSE` and `NOTICE`). The Time-Causal VAE code in
`third_party/tcvae/` is by Songyan Hou and co-authors
(<https://github.com/justinhou95/TimeCausalVAE>) and is redistributed unmodified under the
same licence.

If you use this code, please cite the thesis chapter (see `CITATION.cff`).
