# Reproducing the results of the chapter

This document lists the steps that regenerate every number, table and figure of the chapter
*A Task-Aware Validation Protocol for Generative Models of Financial Time Series*. There
are three levels of verification, from the quickest to the most complete.

| Level | What it checks | Command | Time |
|---|---|---|---|
| 1 | The numbers of the chapter follow from the frozen result tables | `make verify-frozen` | about 1 minute |
| 2 | The pipeline of the simulated studies runs end to end (small sizes; no market data, no Quant GAN) | `make quick J=3` | about 25 minutes |
| 3 | The full study, re-run from scratch | `make all`, then `make verify` | about 25 hours (the Quant GAN uses the Apple GPU if present) |

Times are for the machine used for the thesis: Apple M2 Pro (12 cores, 16 GB), macOS,
Python 3.12.11; CPU only, except the Quant GAN, which uses the GPU (MPS).

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
| E12 fine grid of severities (not pre-registered, D9) | `python -m tavp.experiments.e12_fine` | `results/e12/` | 1.5 h (6) |
| E6 meta-validation of L4 | `python -m tavp.experiments.e6_calibration --part meta` | `results/e6_meta/` | 15 min (4) |
| Size of the L4 tests under dependence | `python -m tavp.experiments.l4_size` | `results/e6_size/` | 2 min (1) |
| E4 case study (TC-VAE) | `python -m tavp.experiments.e4_case` | `results/e4/` | 2 h (4) |
| Stylized facts of the generators | `python -m tavp.experiments.e4_facts` | `results/e4/facts.parquet` | 5 min (2) |
| E5 benchmarks and non-causal ablation | `python -m tavp.experiments.e5_compare` | `results/e5/` | 2.5 h (4) |
| E6 L4 on held-out histories | `python -m tavp.experiments.e6_calibration --part case` | `results/e6_case/` | 10 min (4) |
| E7 robustness | `python -m tavp.experiments.e7_robust` | `results/e7_*/` | 3 h (6) |
| E8 attainability and rich-regime benchmarks (not pre-registered, D5) | `python -m tavp.experiments.e8_attain` | `results/e8/` | 2 h (6) |
| E9 planted mean reversion P7 (not pre-registered, D6; needs the facts of E4) | `python -m tavp.experiments.e9_mr` | `results/e9/` | 1 h (7) |
| X1 exploratory wide ablation | `make exploratory` | `results/x1_wide/` | 1.5 h (2) |
| Market data, download and checksum (D8) | `python scripts/get_french.py` | `data/french/` (not redistributed) | 1 min |
| E10 pseudo-real oracles, calibration (D8) | `python -m tavp.experiments.e10_calibrate` | `configs/real.yaml`, `results/real/e10/` | 1 min |
| E10 protocol on the pseudo-real oracles (D8) | `train`, `e1_null`, `e4_case`, `e5_compare`, `e8_attain` with `--config configs/real.yaml` | `results/real/` | 2.5 h (6) |
| E10 multistart refit of B2, sensitivity (D8) | `python -m tavp.experiments.e10_multistart --config configs/real.yaml --protocol`, then `--config configs/full.yaml` | `results/real/e10ms/`, `results/e10ms/` | 20 min (5) |
| E11 level L4 on the market (D8) | `python -m tavp.experiments.e11_market_l4 --config configs/real.yaml` | `results/real/e11/` | 1.5 h (6) |
| Q0 fidelity check of the Quant GAN on the experiment of its paper (addendum) | `python -m tavp.experiments.q0_fidelity` | `results/q0/` | 35 min (GPU) |
| Q0b single bounded tuning round and confirmation (addendum) | `python -m tavp.experiments.q0_tune` | `results/q0/tune*`, `results/q0/confirm*` | 70 min (GPU) |
| E13 Quant GAN: training, stylized facts, levels L0-L3, L4 on held-out histories (addendum, D10) | `python -m tavp.experiments.e13_qgan --stage train`, then `--stage facts`, `--stage protocol`, `--stage l4` | `results/runs/qgan/`, `results/e13/` | 5 h (GPU, 4 models at a time) |
| E13 Quant GAN on the market (addendum, D10; after E11) | `python -m tavp.experiments.e11_market_l4 --config configs/real.yaml --qgan` | `results/real/runs_market/qgan/`, `results/real/e11/` | 2 h (GPU) |
| Descriptive curves for the figures | `python -m tavp.experiments.e4_curves` | `results/e4_curves/` | 2 min |
| Tables, figures, numbers | `make thesis` | `thesis_assets/` | 1 min |

Every command takes `--config` (default `configs/full.yaml`) and `--jobs`.

## Determinism

All random numbers derive from the master seed of `configs/preregistration.yaml` through
named streams (`tavp.utils.rng`), and PyTorch is seeded from the same streams. On the same
platform and library versions the study is reproduced exactly. On a different platform,
neural training and floating-point reductions can differ in the last digits; such
differences propagate to the trained generators, so that some numbers of the case study may
differ slightly, while the meta-validation (E1, E2, E6 meta), which uses only the oracles,
is unaffected by training. The Quant GAN trains and samples on the Apple GPU (MPS) when it is available:
on the same machine it is deterministic, and its outputs differ from those on the CPU by floating-point rounding
(at most about 1e-7 relative).

This was checked on 2026-10-14 with a clean-room run on the thesis machine: a fresh clone of
the repository, a new environment from `uv.lock`, then `make test`, `make verify-frozen`
(VERIFIED) and `make quick J=3` (11 minutes). The generators trained in the clean room were
bit-identical to those of the original run (the only difference in their files is the
absolute output path recorded in `training_config.json`), and all 2,048 numbers of the smoke
run coincided with those of the original repository.

## Pre-registration and deviations

The scientific settings were committed under the git tag `prereg` before any experiment
(`git show prereg:configs/preregistration.yaml`). Every later change is recorded, with date
and reason, in `docs/DEVIATIONS.md`, and results under the pre-registered settings are kept
alongside: `p_prereg` and `flag_*_prereg` columns for the L1 null (D3), `reject_*_berkowitz_joint`
columns for the L4 tests (D4), `results/e1_n40k/` for the first negative control (D1). Experiments E8 (D5) and E9 (D6), the critical
tolerances (D7), the market-data analysis (D8: E10, its multistart sensitivity, E11) and the fine grid of severities (D9: E12) were added after the results and
are not pre-registered; they do not change any pre-registered verdict. `make real` runs the market-data analysis.

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
| Attainability (E8) | `tables/tab_e8_attain.tex`, `tables/tab_e8_functionals.tex`, `figures/fig_case_attain.pdf` | `results/e4/`, `results/e5/`, `results/e8/` |
| Planted mean reversion (E9) | `tables/tab_e9_detection.tex`, `tables/tab_e9_match.tex` | `results/e4/`, `results/e8/`, `results/e9/` |
| Properties of the Quant GAN as planted defects (E14) | `tables/tab_e14_detection.tex`, `tables/tab_e14_match.tex` | `results/e8/`, `results/e13/`, `results/e14/` |
| Critical tolerances (D7) | `tables/tab_tolcrit.tex`, `tables/tab_tolfunc.tex`, `figures/fig_tolcrit.pdf` | `results/e1/`, `results/e4/`, `results/e5/`, `results/e8/` |
| Fine grid of severities (E12) | `figures/fig_power_fine.pdf`, `tables/tab_e12_thresholds.tex`, `tables/tab_e12_effective.tex` | `results/e1/`, `results/e2/`, `results/e12/` |
| Descriptive figures | `figures/fig_induced.pdf`, `figures/fig_curves.pdf`, `figures/fig_curves_qgan.pdf`, `figures/fig_quantiles.pdf`, `figures/fig_paths.pdf`, `figures/fig_pit_market.pdf`, `figures/fig_ablation_facts.pdf` | `results/e4_curves/`, `results/real/e11/` |
| Quant GAN (Q0, Q0b, E13, D11) | `tables/tab_q0.tex`, `tables/tab_q0_metrics.tex`, `tables/tab_qgan_hyp.tex`, numbers `q0.*`, `e13.*`, `hq.*`; G3 also appears in every table and figure above | `results/q0/`, `results/e13/`, `results/real/e11/`, `results/real/e13/` |
| Market data (D8) | `tables/tab_real_params.tex`, `tables/tab_real_protocol.tex`, `tables/tab_real_l4.tex` | `results/real/`, `results/e10ms/` |
| Every number in the text | `numbers.tex` (macro `\tavp{key}`) | all of the above |
