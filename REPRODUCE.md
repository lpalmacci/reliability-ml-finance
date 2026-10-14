# Reproducing the results of the chapter

This document lists the steps that regenerate every number, table and figure of the chapter
*A Task-Aware Validation Protocol for Generative Models of Financial Time Series*. There
are three levels of verification, from the quickest to the most complete.

| Level | What it checks | Command | Time |
|---|---|---|---|
| 1 | The numbers of the chapter follow from the frozen result tables | `make verify-frozen` | about 1 minute |
| 2 | The whole pipeline runs end to end (small sizes) | `make quick J=3` | about 25 minutes |
| 3 | The full study, re-run from scratch | `make all`, then `make verify` | about 14 hours |

Times are for the machine used for the thesis: Apple M2 Pro (12 cores, 16 GB), macOS,
Python 3.12.11, CPU only.

## 0. Environment

```sh
uv sync          # creates .venv with the exact versions in uv.lock
make test        # unit tests
```

All numerical libraries are restricted to one thread per process (the Makefile exports
`OMP_NUM_THREADS=1` and related variables); parallelism comes from separate worker processes,
whose number can be set with `make <target> J=<workers>`.

At the end of each parallel step Python 3.12 may print `Exception ignored in:
<function ResourceTracker.__del__ ...> ... ChildProcessError: [Errno 10] No child processes`.
It is a harmless message of the multiprocessing shutdown and does not affect the results.

## 1. Verify the frozen results

`results/frozen/` holds the result tables used in the thesis, the numbers of the chapter
(`numbers.json`, `numbers.tex`) and a manifest with the SHA-256 hash of every file and the
git commit that produced them.

```sh
make verify-frozen   # = uv run python scripts/verify.py --frozen
```

The script checks the hashes, rebuilds tables, figures and numbers from the frozen tables
into a temporary directory and compares every number with `results/frozen/numbers.json`.
It prints `VERIFIED` when they coincide.

## 2. Smoke run

```sh
make quick
```

Runs the whole pipeline (training, E1-E7, report) with small sizes defined in
`configs/quick.yaml`. Outputs go to `results/quick/` and never mix with the full results.
The numbers of a quick run are not those of the thesis.

## 3. Full study

```sh
make all       # test, train, meta, case, thesis
make verify    # = uv run python scripts/verify.py
```

`make all` runs the steps below in order. Each step writes its tables to `results/<step>/`
with a `.provenance.json` file (configuration hash, commit, package version). Steps E1-E7
cache each task under `results/<step>/tasks/`, so an interrupted run resumes where it stopped
and produces the same output.

| Step | Command | Output | Time (workers) |
|---|---|---|---|
| Training of the neural generators (G1, G2) | `python -m tavp.experiments.train` | `results/runs/` | 1.5 h (3) |
| E1 negative control and L1 null | `python -m tavp.experiments.e1_null` | `results/e1/` | 1 h (4) |
| E2 planted defects | `python -m tavp.experiments.e2_defects` | `results/e2/` | 3.5 h (4) |
| E6 meta-validation of L4 | `python -m tavp.experiments.e6_calibration --part meta` | `results/e6_meta/` | 15 min (4) |
| Size of the L4 tests under dependence | `python -m tavp.experiments.l4_size` | `results/e6_size/` | 2 min (1) |
| E4 case study (TC-VAE) | `python -m tavp.experiments.e4_case` | `results/e4/` | 2 h (4) |
| Stylized facts of the generators | `python -m tavp.experiments.e4_facts` | `results/e4/facts.parquet` | 5 min (2) |
| E5 benchmarks and non-causal ablation | `python -m tavp.experiments.e5_compare` | `results/e5/` | 2.5 h (4) |
| E6 L4 on held-out histories | `python -m tavp.experiments.e6_calibration --part case` | `results/e6_case/` | 10 min (4) |
| E7 robustness | `python -m tavp.experiments.e7_robust` | `results/e7_*/` | 3 h (6) |
| X1 exploratory wide ablation | `make exploratory` | `results/x1_wide/` | 1.5 h (2) |
| Tables, figures, numbers | `make thesis` | `thesis_assets/` | 1 min |

Every command takes `--config` (default `configs/full.yaml`) and `--jobs`.

## Determinism

All random numbers derive from the master seed of `configs/preregistration.yaml` through
named streams (`tavp.utils.rng`), and PyTorch is seeded from the same streams. On the same
platform and library versions the study is reproduced exactly. On a different platform,
neural training and floating-point reductions can differ in the last digits; such
differences propagate to the trained generators, so that some numbers of the case study may
differ slightly, while the meta-validation (E1, E2, E6 meta), which uses only the oracles,
is unaffected by training.

## Pre-registration and deviations

The scientific settings were committed under the git tag `prereg` before any experiment
(`git show prereg:configs/preregistration.yaml`). Every later change is recorded, with date
and reason, in `docs/DEVIATIONS.md`, and results under the pre-registered settings are kept
alongside: `p_prereg` and `flag_*_prereg` columns for the L1 null (D3), `reject_*_berkowitz_joint`
columns for the L4 tests (D4), `results/e1_n40k/` for the first negative control (D1).

## Correspondence with the chapter

| Chapter | Asset | Source |
|---|---|---|
| Table of false alarms (E1) | `thesis_assets/tables/tab_e1_size.tex` | `results/e1/` |
| Detection figure and table (E2) | `figures/fig_power.pdf`, `tables/tab_e2_detection.tex` | `results/e1/`, `results/e2/` |
| Hypotheses table | numbers `e1.*`, `e2.*` in `numbers.tex` | `results/e1/`, `results/e2/` |
| Dissociation (E3) | `tables/tab_e3_dissociation.tex` | `results/e2/` |
| Validation of L4 (E6) | `tables/tab_e6_meta.tex` | `results/e6_meta/`, `results/e6_size/` |
| Validation cards | `cards/card_*.tex` | `results/e4/`, `results/e6_case/` |
| Case-study figures and summary | `figures/fig_case_*.pdf`, `tables/tab_case_summary.tex` | `results/e4/`, `results/e5/`, `results/e6_case/`, `results/x1_wide/` |
| Robustness (E7) | `tables/tab_e7_robust.tex` | `results/e7_*/` |
| Every number in the text | `numbers.tex` (macro `\tavp{key}`) | all of the above |
