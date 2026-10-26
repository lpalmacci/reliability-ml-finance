# Replication package of Chapters 3 to 6

This repository reproduces every number, table and figure of Chapters 3, 4 and 6 of the doctoral thesis
*Reliability of Machine Learning Outputs in Finance* (Luca Palmacci, Sapienza University of Rome), and documents
the systematic review of Chapter 5:

- **Chapter 3, *Confidex*.** A model of the probabilities that the members of an ensemble assign to a
  case, the reconstruction of the law of a further member, and a reliability signal that reads that law
  at the operating threshold; tested on synthetic ensembles whose correct decision is known.
- **Chapter 4, *Experimental Framework*.** The signal on two public fraud datasets, with ensembles of 3
  to 36 calibrated members built from one family of models or from nine, against four simpler readings
  of the members' probabilities at fixed review budgets.
- **Chapter 5, *Generative Models for Financial Time Series: A Systematic Review*.** A PRISMA 2020 review of
  137 studies: the search, the screening and full-text decisions, the coding of every study, and the
  statistics and tables of the chapter.
- **Chapter 6, *A Task-Aware Validation Protocol for Generative Models of Financial Time Series*.** A protocol
  that judges a scenario generator by a declared use, its validation with known-answer tests on four
  reference processes, and its application to the Time-Causal VAE and a Quant GAN, on simulated processes,
  on processes calibrated on the US market and on the market itself, with settings and hypotheses
  pre-registered before the experiments.

## Layout

| Folder | Content | Time to reproduce |
|---|---|---|
| [`confidex/`](confidex/) | The library: posterior of the concentration on a logit grid, predictive law, central-band criterion, signed score, diagnostics. Installable with `pip install -e confidex`; 18 numerical tests. | 20 s (tests) |
| [`chapter3_synthetic_experiments/`](chapter3_synthetic_experiments/) | The experiments of Chapter 3, a notebook that explains and recomputes each of them, the two tables of its Appendix and its seven figures. | about 15 min |
| [`chapter4_fraud_experiments/`](chapter4_fraud_experiments/) | The complete pipeline of Chapter 4, from the two datasets to the tables, the two figures and the check of every number, with the results of a full run from the data, identical to those of the thesis. | about 6 h on 10 cores |
| [`chapter5_systematic_review/`](chapter5_systematic_review/) | The systematic review of Chapter 5: search strings, screening and full-text decisions, coding sheet and codebook, page-referenced extraction notes, and the script that regenerates the corpus statistics, the tables of the chapter and Tables S1–S3. Python standard library only. | seconds |
| [`chapter6_validation_protocol/`](chapter6_validation_protocol/) | The package `tavp` of Chapter 6: reference processes, planted defects, levels L0–L4, every experiment of the chapter, the frozen results used in the thesis and the 97 trained Quant GANs. Its history is that of the original repository, with the pre-registration under the tags `prereg` and `prereg-qgan`. Start from its [`REVIEWER_GUIDE.md`](chapter6_validation_protocol/REVIEWER_GUIDE.md). | 5 min (numbers from the frozen results); about 20 h (full study) |

Each folder has its own README with the commands, the correspondence between its outputs and the
thesis, and the notes on determinism (for Chapter 6, `REPRODUCE.md`). The text of the thesis is not part of the repository.

## Quick start

Python 3.12 is recommended (3.13 also works for Chapter 3).

```bash
# Chapter 3
cd chapter3_synthetic_experiments
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e ../confidex
python run_experiments.py && python make_tables.py && python make_figures.py && python check_values.py

# Chapter 4
cd ../chapter4_fraud_experiments
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e ../confidex
./run_all.sh                    # or ./run_all.sh 9: tables, figures and checks from the results included

# Chapter 5 (no installation needed)
cd ../chapter5_systematic_review
python3 4_analysis/analysis.py

# Chapter 6 (needs uv, https://docs.astral.sh/uv/)
cd ../chapter6_validation_protocol
uv sync && make test && make verify-frozen   # every number of the chapter from the frozen results
```

`check_values.py`, in the folders of Chapters 3 and 4, compares every number printed in the chapter with the value
recomputed from the results, rounded as in the thesis, and states where the number appears.

## Reproducibility

Every random draw comes from a generator seeded by the experimental condition alone, so that each
condition can be rerun on its own and in any order. With the package versions in each
`requirements.txt` the results are reproduced exactly, with one documented exception: the degenerate
LightGBM members of the comparison that Chapter 4 keeps for reference, which depend on the number of
threads (`chapter4_fraud_experiments/README.md`, *Determinism*). The only inputs of Chapter 4 that cannot be
regenerated from the data are the scores of the TabPFN members, which come from a remote service and are
saved in `chapter4_fraud_experiments/results/tabpfn/`. The statistics of Chapter 5 are counts on the coding
sheet and involve no randomness. In Chapter 6 every random number derives from one master seed and the study is
reproduced exactly on the platform of the thesis, except the training of the Quant GAN on the Apple GPU, which is
not reproducible bit for bit; the 97 trained Quant GANs are therefore included
(`chapter6_validation_protocol/weights/`), and with them a full run reproduced every number of the chapter. Its
market data, from the Ken French Data Library, are downloaded by a script, not redistributed.

## Licence and citation

The code is released under the Apache License 2.0 (`LICENSE`). The datasets of Chapter 4 keep their own
licences (ODbL 1.0 / DbCL 1.0 for ULB, CC0 1.0 for carclaims); see `NOTICE` and
`chapter4_fraud_experiments/data/README.md`. The Time-Causal VAE code used in Chapter 6
(`chapter6_validation_protocol/third_party/tcvae/`) is by S. Hou and co-authors and is redistributed unmodified under
the Apache License 2.0; the Quant GAN is an independent implementation of its paper. The records and the coding of
the review of Chapter 5 may be reused for
research purposes with citation of the thesis; full texts and database exports are not redistributed
(`chapter5_systematic_review/README.md`). To cite the software, see `CITATION.cff`; the model of Chapter 3
is joint work with S. Galiani, D. Petturiti and B. Vantaggi (IPMU 2026).
