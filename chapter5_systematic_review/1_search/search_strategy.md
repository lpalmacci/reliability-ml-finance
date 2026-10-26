# Search strategy

All database searches were run on **10 November 2025** by the author. Strings are reported verbatim as executed.

## Scopus (271 records)

```
( ( TITLE-ABS-KEY ( ( "generative model" OR "generative models" OR "GAN" OR "generative adversarial network" OR "variational autoencoder" OR "VAE" OR "diffusion model" OR "generative modeling" OR "Restricted Boltzmann Machine" ) AND ( "financial time series" OR "financial data" OR "stock returns" OR "synthetic data" OR "time series forecasting" OR "finance" OR "financial modeling" OR "stock prediction" OR "trading" OR "financial risk management" OR "risk forecasting" OR "portfolio optimization" OR "stylized facts" OR "trading strategies" OR "market generator" OR "backtesting" OR "risk measures" OR "scenario simulation" OR "market simulation" OR "limit order book" OR "generation of market scenarios" ) ) AND PUBYEAR > 2015 AND PUBYEAR < 2027 ) ) AND ( LIMIT-TO ( SUBJAREA , "ECON" ) OR LIMIT-TO ( SUBJAREA , "MULT" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) )
```

## SSRN (320 records; title, abstract and keywords, Boolean search)

```
("generative model" OR "generative models" OR "GAN" OR "generative adversarial network" OR "variational autoencoder" OR VAE OR "diffusion model" OR "generative modeling" OR "Restricted Boltzmann Machine") AND ("financial time series" OR "financial data" OR "stock returns" OR "synthetic data" OR "time series forecasting" OR "finance" OR "financial modeling" OR "stock prediction" OR "trading" OR "financial risk management" OR "risk forecasting" OR "portfolio optimization" OR "stylized facts" OR "trading strategies" OR "market generator" OR "backtesting" OR "risk measures" OR "scenario simulation" OR "market simulation" OR "limit order book" OR "generation of market scenarios")
```

## arXiv (177 records; advanced search)

```
date_range: from 2015-12-01 to 2025-11-01; classification: Economics (econ), Quantitative Finance (q-fin); include_cross_list: True; terms: AND all="generative model" OR "GAN" OR "generative adversarial network" OR "variational autoencoder" OR VAE OR "diffusion model" OR "generative modeling" OR "Restricted Boltzmann Machine"; AND all="financial time series" OR "financial data" OR "stock returns" OR "synthetic data" OR "time series forecasting" OR finance OR "financial modeling" OR "stock prediction" OR trading OR "financial risk management" OR "risk forecasting" OR "portfolio optimization" OR "stylized facts" OR "trading strategies" OR "market generator" OR backtesting OR "risk measures" OR "scenario simulation" OR "market simulation" OR "limit order book" OR "generation of market scenarios"
```

## Records identified

| Source | Records | Retained after title/abstract screening |
|---|---|---|
| Scopus | 271 | 73 (65 at first screening + 8 at the second-pass check) |
| SSRN | 320 | 46 |
| arXiv | 177 | 58 |
| **Total databases** | **768** | **177** (24 duplicates removed after screening, leaving 153 reports) |
| Other methods: ML conference proceedings | 18 | all assessed in full text |
| Other methods: author's reference library | 9 | all assessed in full text |

## Notes

- Scopus filters: publication years 2016-2026, English language, subject areas Economics, Econometrics and Finance (ECON) and Multidisciplinary (MULT).
- arXiv: categories econ and q-fin, including cross-listed submissions, 1 December 2015 - 1 November 2025.
- The model block does not name normalizing flows, score-based or energy-based models in general; the Scopus subject filter excludes journals indexed only under computer science, mathematics or physics (see Section 2.1 and the limitations in Section 8.2).
- The 'other methods' channel (prior knowledge of the field) cannot be reproduced by a database query; its 27 records are listed in `2_screening/fulltext_assessment.csv`.
