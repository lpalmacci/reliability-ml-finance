# Chapter 5 — Generative models for financial time series: a systematic review

This folder documents the systematic review of Chapter 5 of the thesis (*Generative Models for Financial Time
Series: A Systematic Review*), conducted according to the PRISMA 2020 statement (item 27, availability of data,
code and other materials). It allows a reader to follow every record from the database search to the final corpus,
to inspect how each included study was coded, and to reproduce the corpus statistics and the tables of the chapter.

## Contents

| Folder | Content |
|---|---|
| `1_search/` | Search strings as executed, date of the search and number of records per source |
| `2_screening/` | Screening decisions (Scopus, SSRN, arXiv), deduplication and full-text assessment with exclusion codes and reasons; see `2_screening/README.md` |
| `3_extraction/` | `coding_sheet.csv` (137 included studies, 15 reference-set models, 1 methodological source), `codebook.md` and the page-referenced extraction notes (`extraction_notes_IT.md`, in Italian) |
| `4_analysis/` | `analysis.py`, which regenerates the results from the files above |
| `5_tables/` | Outputs of the analysis: `results.md` (the counts and percentages quoted in the chapter), Tables S1–S3 (CSV and LaTeX), the tables of Sections 4 and 6 (`main_tables.tex`), and the completed PRISMA 2020 checklist |

## How to reproduce the results

Requirements: Python 3.8 or later, standard library only.

```
cd chapter5_systematic_review
python 4_analysis/analysis.py
```

The script prints the PRISMA flow counts, the corpus profile, the stylized-fact coverage, the metric families and
the fidelity–utility cross-tabulation, and rewrites the files in `5_tables/` (the PRISMA checklist excepted). The
classification rules for metric families, fidelity and utility are defined at the top of the script and described
in `3_extraction/codebook.md`. The counts that the chapter reports within single subsections of Section 5 are not
printed by the script; they can be counted directly from the coding sheet (columns such as `domain`,
`synthesis_section`, `frequency`, `stylized_facts_tested` and `Q_score`).

## PRISMA flow reproduced by the files

| Stage | Databases | Other methods |
|---|---|---|
| Records identified | 768 (Scopus 271, SSRN 320, arXiv 177) | 27 |
| Excluded at title and abstract screening | 591 | — |
| Duplicates removed | 24 | — |
| Reports sought / not retrieved | 153 / 7 | 27 / 0 |
| Excluded at full text | 19 (EC1 7, EC2 3, EC3 4, EC4 1, EC5 2, EC6 2) | 16 (EC7 15, EC1 1) |
| Reports included | 127 | 11 |
| **Studies included** | **137** (138 reports) | |

## Screening, extraction and quality appraisal

Screening, selection, data extraction and quality appraisal were carried out by a single reviewer, the author of
the thesis. The extracted values were not verified independently by a second reviewer. The extraction notes record,
for the statements taken from each study, the page of the full text on which they appear (Section 2.4 of the
chapter, and its limitations in Section 8.2). The notes are in Italian, the language in which they were taken; all
the other files are in English.

## What is not included, and why

- **Full texts** of the included studies are not redistributed (copyright). Every study is identified by its
  bibliographic reference and, where available, its DOI.
- **Complete database exports** (for instance Scopus exports with abstracts) are not redistributed, in line with the
  licence terms of the databases. Only the identifiers and the basic metadata needed to document the selection are
  provided.
- **Theses** excluded under EC5 are recorded in the screening files but are not cited in the review.

## Section and table references

Section numbers in these files refer to the sections of Chapter 5 without the chapter prefix: Section 2.4 is
Section 5.2.4 of the thesis. Appendix A (search strategies and PRISMA checklist) and Appendix B (Tables S1–S3) of the
chapter are Sections B.1 and B.2 of the Appendix of the thesis.

## Terms of use

The code is released under the Apache License 2.0 (`../LICENSE`). The screening records, the coding sheet and the
extraction notes may be reused for research purposes with citation of the thesis (`../CITATION.cff`); see also
`../NOTICE`.
