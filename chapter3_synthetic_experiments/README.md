# Chapter 3 — Confidex on synthetic ensembles with a known truth

This folder reproduces every number, table and figure of Chapter 3 of the thesis (*Confidex*): the
model of the members' probabilities, the reconstructed law, the reliability signal, the sensitivity of
the verdict to each modelling choice, the illustration on ten constructed ensembles and the
experiment on synthetic cases whose correct decision is known.

## Contents

| File | Content |
|---|---|
| `replicate_chapter3.ipynb` | **Start here.** Notebook that explains each experiment with reference to the chapter, recomputes it, draws the figures and checks every recomputed value against the reference files. It is distributed with its outputs, so it can be read without running it. |
| `run_experiments.py` | Computes all the results of the chapter and writes the two reference files below. |
| `synthetic.py` | The synthetic generator with a known truth (informative, non-informative and mixed designs), the four simpler readings, the AUROC and the seeding scheme. |
| `make_tables.py` | Writes the two tables of the Appendix (Tables A.1 and A.2) from the results. |
| `make_figures.py` | Draws the seven figures of the chapter from the results. |
| `check_values.py` | Checks every number printed in the chapter, and every cell of Table A.2, against the results. |
| `style.py` | Figure style of the thesis (Computer Modern, Okabe–Ito palette, 15 cm width). |
| `results/chapter3_results.json` | Reference values: every quantity reported in the chapter. |
| `results/auroc_gap_bootstrap.json` | AUROC difference between the signal and the best simpler reading, with 95% bootstrap intervals (figure *amplification*). |
| `results/tables/` | `settings_table.tex` (Table A.1) and `budget_table.tex` (Table A.2), as generated. |
| `figures/` | The seven figures of the chapter, PDF and PNG. |
| `requirements.txt` | Package versions with which the results are reproduced exactly. |

The library itself is in `../confidex` (version 0.5.1, numerically identical to 0.5.0, with which the
thesis was computed).

## Running

```bash
python3.12 -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ../confidex
pytest ../confidex/tests                                   # 18 tests, about 20 s

python run_experiments.py        # -> results/*.json, about 8 minutes
python make_tables.py            # -> results/tables/*.tex
python make_figures.py           # -> figures/*.pdf, *.png
python check_values.py           # every number of the chapter against the results
```

or, equivalently, run the notebook:

```bash
jupyter lab replicate_chapter3.ipynb       # or: jupyter nbconvert --to notebook --execute --inplace replicate_chapter3.ipynb
```

With `RECOMPUTE = True` (the default) the notebook recomputes everything in about 15–20 minutes; with
`RECOMPUTE = False` it reads the reference files and only draws the figures. Its last section reports,
for every recomputed value, whether it matches the reference to within 1e-12.

## Where each result appears in the thesis

| Chapter 3 | Results | Notebook |
|---|---|---|
| Coupled and decoupled models (Table 3.1, figure *coupled-density*) | analytical; `make_figures.py` | 2 |
| Three specifications; Assumptions (five members, figure *joint-posterior*) | `five` | 6.1 |
| Inference on the concentration (grid, quadrature, ties, Jeffreys prior) | `quadrature`, `ties`, `jeffreys`, `sensitivity.border4`, `sensitivity.atten_*` | 1.1, 6.6 |
| The concentration is a relative measure (figure *decoupled-density*) | `examples` | 6.3 |
| From the concentration to the law of the members (remarks) | `shrinkage`, `examples`, `timing` | 6.2, 6.7 |
| The reliability signal (remark on the gate) | `gate_corr` | 6.5 |
| Sensitivity to the modelling choices | `sensitivity` | 5 |
| An illustration on ten ensembles (figure *illustration*) | `illustration` | 3 |
| What the signal detects when the truth is known (figures *validation*, *amplification*, *budget*; Table A.2) | `validation`, `budget`, `auroc_gap_bootstrap.json` | 4 |
| What the signal says, and its limits | `anchoring_location`, `sensitivity.fixed_mean` | 6.4 |

## Determinism

Every experimental condition draws its random numbers from its own generator, seeded with a key that
depends only on the condition (`synthetic.seed_for`), so results do not depend on the order in which
the conditions are run. The bootstrap of the AUROC differences is the only computation that shares one
generator (seed 2026) across conditions, in the order of the loops. With the versions in
`requirements.txt` the reference values are reproduced exactly, under Python 3.12 and 3.13 alike.

**The AUROC counts ties as one half.** With 24 and 36 members the signal saturates at its bound for a
sizeable share of the cases (up to about a tenth), which creates ties. The AUROC is computed from
mid-ranks, as defined in the thesis, after rounding the scores to ten decimals, so that values that
differ only by floating-point noise are treated as ties. The version of the code used for the thesis
broke ties by the order of the cases instead; on 12 of the 1,441 values of the results this changed the
fifth decimal, and with it the value obtained on different machines, but it changes no number printed
in the thesis, as `check_values.py` verifies.

## Figures and LaTeX

The figures of the thesis are typeset with LaTeX, which needs `latex`, `dvipng` and the `cm-super`
fonts (on Debian or Ubuntu: `apt install texlive-latex-extra cm-super dvipng`; on macOS: MacTeX). Without
them, `make_figures.py` and the notebook fall back to Matplotlib's mathtext, with the same layout and
data.
