# Codebook for `coding_sheet.csv`

One row per study. Included studies have identifiers `S001`–`S137` (ordered alphabetically by first author), the fifteen general-purpose models of the methodological reference set `R01`–`R15`, and the benchmark used as a methodological source `M01`. The column `legacy_id` links each row to the working identifiers used in the extraction notes (`extraction_notes_IT.md`).

## Identification

| Column | Content |
|---|---|
| `study_id` | Identifier used in Table S1 and Table S2 |
| `legacy_id` | Working identifier used during extraction |
| `role` | `included` (corpus, 137 studies), `reference_set` (EC7, 15 models), `methodological_source` (TSGBench, EC1) |
| `channel` | `databases` (Scopus, SSRN, arXiv) or `other_methods` (ML proceedings, author's reference library) |
| `citekey` | Key in the bibliography of the review |
| `additional_report_citekey` | Second report of the same study (only S017: Bühler et al. 2020a and 2020b) |
| `authors`, `year`, `title`, `venue` | Bibliographic data of the version cited in the review |
| `peer_reviewed` | `yes` / `no` / `unknown`, status at the time of the search |

## Model, data and task

| Column | Content |
|---|---|
| `family` | Standardised family: `adversarial`, `diffusion`, `vae_autoencoder`, `hybrid`, `energy_based` (including probabilistic circuits), `autoregressive`, `multiple_families`, `distance_minimising` (signature and optimal-transport generators trained without a discriminator), `normalizing_flow`, `evaluation_method` |
| `family_detail` | Family as coded from the full text, e.g. `GAN (decision generator)` for studies that use the adversarial game to produce decisions |
| `architecture` | Backbone and main components |
| `data` | Datasets, as described by the study |
| `asset_class` | Asset classes (several possible, separated by `;`) |
| `frequency` | Sampling frequency of the data |
| `tasks` | Tasks addressed (several possible) |
| `domain` | Application domain to which the study is assigned in Section 5: `synthetic_data_augmentation`, `forecasting`, `microstructure_lob`, `risk_pricing`, `portfolio_strategy`; `evaluation_study` for the three studies discussed in Section 6 |
| `synthesis_section` | Subsection of the review in which the study is primarily discussed |

## Evaluation

| Column | Content |
|---|---|
| `classical_baselines`, `deep_baselines` | Benchmarks used (`N` = none, `?` = not reported) |
| `metrics` | Metrics as coded from the full text. Prefixes: `MB` model-based scores, `FB` feature-based statistics, `DB` distances between distributions, `PATH` path-space metrics, `CAUS` causal or adapted metrics, `RISK` risk measures, `DEC` decision and trading metrics, `PROB` probabilistic scores, `VIS` visualisation, `EFF` efficiency |
| `stylized_facts_tested` | Stylized facts tested explicitly (`none` if none), codes SF1–SF11 below |
| `code_available` | Code released (`yes` / `no` / `unknown`). Coded from links and statements in the full text |

Stylized facts: SF1 heavy tails; SF2 absence of linear autocorrelation; SF3 volatility clustering; SF4 long-range dependence; SF5 leverage effect; SF6 gain/loss asymmetry; SF7 aggregational Gaussianity; SF8 coarse–fine volatility correlation; SF9 cross-sectional dependence; SF10 microstructure regularities; SF11 mean reversion and regimes.

## Quality rubric (Section 2.4)

Each item is coded `Y` (satisfied), `P` (partially satisfied), `N` (not satisfied), `?` (not reported) or `NA` (not applicable). `Q_score` = number of `Y` + 0.5 × number of `P` (maximum 7).

| Item | Criterion |
|---|---|
| `Q1_data` | Data publicly available or fully described |
| `Q2_code` | Code released |
| `Q3_classical_benchmark` | At least one classical benchmark (GARCH-type model, bootstrap or historical simulation, calibrated stochastic model) |
| `Q4_deep_benchmark` | At least one deep generative benchmark |
| `Q5_temporal_oos` | Downstream task evaluated on a temporally separated out-of-sample period |
| `Q6_stylized_facts` | Stylized facts tested with explicit diagnostics |
| `Q7_runs_uncertainty` | Results over several runs or with a measure of statistical uncertainty |

## Derived classifications (computed by `4_analysis/analysis.py`)

These are not stored in the coding sheet; they are derived from `metrics` and `stylized_facts_tested` by the rules in `METRIC_FAMILIES` of the analysis script.

- **Metric families (Table 6.1)**: stylized-fact diagnostics; distances between distributions (KS, Wasserstein, MMD, KL/JS, DTW, Fréchet-type distances, surface distances); model-based scores (discriminative and predictive scores, TSTR); path-space distances (signature MMD, Sig-W1, RS-W1); adapted and causal distances; probabilistic scoring rules (CRPS, likelihood, perplexity); point-forecast errors; risk metrics (VaR, ES, CVaR); decision metrics (P&L, Sharpe ratio, drawdown, hedging, utility); visualisation; efficiency.
- **Fidelity**: the study tests at least one stylized fact or reports a distance between real and generated data (distances between distributions, path-space, or adapted/causal distances).
- **Utility**: the study reports a risk measure or the outcome of a decision computed with the generated data (risk or decision metrics). Point-forecast errors count as neither.
- **Robustness**: counting model-based scores as fidelity changes the number of studies that measure both fidelity and utility from 36 to 37.
- **Robustness to preprints**: restricted to the 90 peer-reviewed studies, 24% measure both fidelity and utility (26% in the full corpus).

## Notes columns

`selection_note` flags studies added at the second-pass check or from other methods, and studies reinstated because outcome-based exclusions are not allowed (Section 2.2). `notes` contains short remarks from the extraction; the detailed, page-referenced notes are in `extraction_notes_IT.md` (in Italian).
