# Chapter 4 — Confidex on two real fraud datasets

This folder reproduces every number, table and figure of Chapter 4 of the thesis (*Experimental
Framework*) and of its Appendix (*The models of the ensembles*, *Complete results of Chapter 4*): two
public fraud datasets, ensembles of 3 to 36 calibrated members built in two ways, the reliability signal
of Chapter 3 against four simpler readings of the members' probabilities at fixed review budgets, the
comparisons within each predicted class and against the allocated reading, and the ablations.

The replication is complete: `run_all.sh` starts from the two CSV files and ends with the check of every
number of the chapter. The only inputs that are not recomputed are the scores of the TabPFN members,
which come from a remote service (see [TabPFN](#tabpfn)).

## Contents

| Path | Content |
|---|---|
| `run_all.sh` | **Start here.** Runs the eleven steps below in order; every step resumes where it stopped. |
| `fraud/` | The package with the whole method of the experiments (its `__init__.py` lists the modules in the order the pipeline uses them). |
| `make_partitions.py` | Step 1: five partitions of carclaims and ten of ULB, 55/10/15/20%, stratified by class, identical cases in the same part. |
| `query_tabpfn.py` | Optional: queries the TabPFN service again (needs an account; the saved scores are used otherwise). |
| `train_members.py` | Steps 2–3: trains and calibrates the 36 members of each construction on every partition. |
| `run_evaluation.py` | Steps 4–5: the evaluation of every condition, in parallel, with the bootstrap and the ablations. |
| `summarise.py` | Step 6: summary across partitions, `results/chapter4_results.json`. |
| `keff_checks.py`, `export_diagnostics.py` | Steps 7–8: effective number of members and the diagnostics cited in the chapter, `results/chapter4_diagnostics.json`. |
| `make_tables.py`, `make_figures.py` | Steps 9–10: the tables and the two figures of the chapter. |
| `check_values.py` | Step 11: checks every number printed in the chapter and its Appendix against the results, and the settings of the design against the code. |
| `style.py` | Figure style of the thesis (the same as Chapter 3). |
| `tests/` | Unit tests of the partitions, seeds, members, calibration, draws, bootstrap, readings, budgets and ablations. |
| `data/` | The two datasets, with their sources, licences and SHA-256 (`data/README.md`). |
| `results/` | The results of a full run of `run_all.sh` from the data, identical to those of the thesis (see [Results included](#results-included) and [Determinism](#determinism)). |
| `figures/` | The two figures of the chapter, PDF and PNG. |
| `requirements.txt` | Exact package versions with which the results were produced. |

The library is in `../confidex` (version 0.5.1, numerically identical to 0.5.0, with which the thesis was
computed).

## Running

```bash
python3.12 -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ../confidex
gunzip -k data/ulb_creditcard.csv.gz                       # GitHub does not accept files above 100 MB
pytest tests                                               # 43 tests, about 20 s

./run_all.sh                     # everything, with 10 parallel processes
JOBS=4 ./run_all.sh              # with fewer processes
./run_all.sh 9                   # from step 9: tables, figures and checks from the results included
```

| Step | Script | Output | Time (12-core laptop) |
|---|---|---|---|
| 1 | `make_partitions.py` | `results/partitions/` | seconds |
| 2 | `train_members.py` | `results/members/` | about 40 min |
| 3 | `train_members.py --unconstrained` | `results/members_unconstrained_lightgbm/` | a few minutes |
| 4 | `run_evaluation.py freeze`, `run` | `results/evaluation/` | about 2 h 40 min with 10 processes |
| 5 | `run_evaluation.py freeze --unconstrained`, `run --unconstrained` | `results/evaluation_unconstrained_lightgbm/` | about 1.5 h with 10 processes |
| 6 | `summarise.py`, `summarise.py --unconstrained` | `results/chapter4_results*.json`, `results/evaluation*/summary/` | seconds |
| 7 | `keff_checks.py` | `results/keff_checks.parquet` | seconds |
| 8 | `export_diagnostics.py` | `results/chapter4_diagnostics.json`, `results/member_validation_auroc.parquet` | about 75 min with 10 processes |
| 9 | `make_tables.py` | `results/tables/*.tex` | seconds |
| 10 | `make_figures.py` | `figures/*.pdf`, `*.png` | seconds |
| 11 | `check_values.py` | the checks, printed; exit status 1 if any fails | seconds |

The memory needed is modest (about 0.5 GB for each process of the evaluation). Steps 2–5 write large
intermediate files (the scores of every member and the tables of every job, a few GB in all); they are
not part of the repository and are rebuilt by the steps.

## The two versions of the pools

The LightGBM members of the nine-family construction (S2) are constrained to leaves with a total Hessian
of at least one, a decision taken after the first full evaluation (thesis, *Decisions taken after the
pilot and during the evaluation*): with the default of LightGBM these members degenerated on ULB. The
thesis reports the constrained version, which is the main evaluation here (`results/members/`,
`results/evaluation/`). The unconstrained version is kept for the comparison that the thesis cites
(the AUROC and the share of squared deviations of the degenerate members, and the statement that the
conclusions hold for both): `train_members.py --unconstrained` rebuilds only the four LightGBM members
of S2 without the constraint, and `run_evaluation.py --unconstrained` evaluates S2 again on those pools.
The one-family construction (S1) is the same in the two versions.

## Determinism

Every random draw comes from a generator seeded with an integer key that depends only on the condition
(`fraud/seeds.py`): purpose (partition, member, draw, bootstrap), dataset, partition, construction, size
and index. A condition therefore gives the same numbers whether it is run alone or with the others, in any
order and with any number of processes. The evaluation runs one thread per process, so that the
matrix products of the bootstrap do not depend on the number of cores. With the versions in
`requirements.txt` the partitions, the members, the calibrations and every table of the evaluation are
reproduced exactly (bit for bit) from the data.

One column differs from the code with which the thesis was computed: `auroc` in `rows.parquet`, the AUROC
with which each reading separates the errors from the correct decisions. Many scores of doubt are tied
(members in agreement, the signal saturated at the border of the grid); the original code broke the ties
by the order of the cases, which moved this AUROC by up to 0.17 on ULB, whereas `fraud.readings.auroc`
counts them as one half, as in Chapter 3. The column is a by-product of the evaluation: no number, table
or figure of the thesis uses it (the AUROCs that the chapter cites are those of the members and of the
ensembles).

The unconstrained LightGBM members of the second construction on ULB (`--unconstrained`, used only for
the comparison described above) are not reproduced bit for bit. They are the members that degenerated:
with leaves of a handful of cases many candidate splits have the same gain, and which one LightGBM takes
depends on the number of threads. Retrained with 1, 2, 4 and 8 threads, their median AUROC on the
validation part is 0.680, 0.685, 0.692 and 0.701 (0.687 in the thesis) and their share of the squared
deviations 40.8%, 41.4%, 40.7% and 39.8% (41% in the thesis); with the same number of threads the result
is the same. `check_values.py` checks these two numbers with a tolerance (0.02 and 2 points) and states
it. The rest of the unconstrained evaluation varies accordingly; the statement of the thesis that it
concerns, that the conclusions hold for both versions, holds in the run included here as in that of the
thesis. The members of the final
pools, on which every other number of the chapter rests, are identical with any number of threads.

`run_evaluation.py freeze` writes the SHA-256 of the code before the evaluation
(`results/evaluation*/frozen_code.yaml`), and `run` warns if the code changed afterwards. Every step
records the package versions and the git state in a `manifest.yaml`.

## TabPFN

The TabPFN members (four in each pool of S2) are obtained from the remote service of its developers,
whose model may change, so that the same query is not guaranteed to return the same scores. The scores
used in the thesis are saved in `results/tabpfn/<dataset>_p<k>/` (raw scores on the calibration,
validation and test parts and the indices of each member's subsample), with the log of every call
(`tabpfn_calls.jsonl`) and the provenance (`tabpfn_manifest.yaml`); the other steps read them and never
query the service. To query it again, copy `.env.example` to `.env`, set `TABPFN_TOKEN`, delete or move
`results/tabpfn/` and run `python query_tabpfn.py`; the results may then differ from those of the thesis.

## Results included

| Path | Content |
|---|---|
| `results/partitions/` | The indices of the four parts of every partition, with a manifest of counts and SHA-256. |
| `results/tabpfn/` | The saved scores of the TabPFN members (see above). |
| `results/members*/<dataset>_p<k>/` | For each partition: the calibration of every member (`s1_calibration.json`, `s2_calibration.json`), the validation diagnostics (`diagnostics.yaml`) and the manifest. |
| `results/evaluation*/` | The fingerprints of the code, the log of the run and the summaries across partitions (`summary/*.parquet`). |
| `results/chapter4_results.json` | Every quantity of the chapter, from the main evaluation. |
| `results/chapter4_results_unconstrained_lightgbm.json` | The same for S2 with the unconstrained LightGBM members. |
| `results/chapter4_diagnostics.json` | Effective number of members, AUROC of the members by family, classes without validation errors, pools at the clipping constant, regimes of the law, calibration methods and saturation. |
| `results/keff_checks.parquet`, `results/member_validation_auroc.parquet` | The tables behind the diagnostics. |
| `results/tables/` | The tables of the chapter and of its Appendix, as generated. |

With these files, steps 9 to 11 (tables, figures and the check of every number) run at once on a fresh
copy of the repository: `./run_all.sh 9`. Steps 6 to 8 read the intermediate files of steps 2 to 5 (the
pools and the tables of every job), which are not distributed and are rebuilt by those steps.

## Where each result appears in the thesis

| Chapter 4 | Produced by |
|---|---|
| Data and partitions (table *The two datasets and their partitions*) | `results/tables/data_table.tex` |
| Members and their calibration (tables of the nine families and of the hyperparameters of S1) | the settings in `fraud/defaults.py` and `fraud/plan.py`, checked by `check_values.py` |
| Members and ensembles (table *Members and ensembles on the test cases*; effective number of members; probabilities at the clipping constant) | `results/tables/members_table.tex`, `results/chapter4_diagnostics.json` |
| The signal against the simpler readings (figure *Errors captured by the signal relative to three references*; table *Errors captured on the two datasets*) | `figures/confidex_ev_differences.pdf`, `results/tables/main_table.tex` |
| Where the budget goes; the verdict; the missed frauds (figure *Missed frauds captured*) | `results/chapter4_results.json`, `figures/confidex_ev_missed_frauds.pdf` |
| Fixed shares of the cases; ablations | `results/tables/shares_table.tex`, `results/tables/ablations_table.tex`, `results/tables/gate_table.tex` |
| Decisions after the pilot (members with and without the constraint) | `results/chapter4_diagnostics.json`, `results/chapter4_results_unconstrained_lightgbm.json` |
| Appendix: complete results | `results/tables/main_table_all_K.tex`, `budgets_table.tex`, `verdict_table.tex`, `shares_table.tex`, `ablations_table.tex`, `gate_table.tex` |

`check_values.py` lists, for every check, where the value appears in the chapter, what it measures, the
value as the thesis prints it and the value recomputed from the results with the same rounding
(`python check_values.py --list`).

## Figures and LaTeX

As in Chapter 3, the figures are typeset with LaTeX when it is available (`latex`, `dvipng`, `cm-super`),
and otherwise with Matplotlib's mathtext, with the same layout and data.
