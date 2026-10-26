# Numbers reported in the review

## PRISMA flow
Scopus records: 271 | retained: 73 | of which second pass: 8
Database reports sought: 153 | not retrieved: 7 | excluded: {'EC1': 7, 'EC2': 3, 'EC3': 4, 'EC6': 2, 'EC5': 2, 'EC4': 1} | included: 127
Other-methods reports: 27 | excluded: {'EC7': 15, 'EC1': 1} | included: 11
Included reports: 138 | distinct studies: 137

## Corpus profile (Section 4)
Years: {2017: 1, 2018: 1, 2019: 8, 2020: 8, 2021: 7, 2022: 14, 2023: 19, 2024: 35, 2025: 39, 2026: 5} | <=2019: 10 | 2020-21: 15 | >=2024: 79
Families: {'adversarial': '72 (53%)', 'diffusion': '19 (14%)', 'vae_autoencoder': '15 (11%)', 'hybrid': '8 (6%)', 'energy_based': '6 (4%)', 'multiple_families': '5 (4%)', 'autoregressive': '5 (4%)', 'distance_minimising': '4 (3%)', 'normalizing_flow': '2 (1%)', 'evaluation_method': '1 (1%)'}
Peer reviewed: {'yes': 90, 'no': 44, 'unknown': 3} (66%)
ICAIF studies: 25 | Quantitative Finance: 7
Mean Q: 3.58 | peer reviewed: 3.65 | not peer reviewed: 3.51
Mean Q by period: <=2021 3.12 | 2022-23 3.53 | >=2024 3.75
Asset classes: {'equities': 77, 'equity indices': 26, 'simulated': 14, 'crypto': 10, 'FX': 9, 'options/implied volatility': 8, 'commodities': 7, 'multi-asset': 6, 'macroeconomic': 3, 'credit': 3, 'interest rates': 3, 'non-financial benchmarks with stocks': 1}
Frequencies: {'daily': 95, 'order book / message level': 16, 'intraday': 9, 'monthly': 6, 'simulated': 4, 'hourly': 2, 'high frequency (tick/seconds)': 2, 'annual': 1, 'quarterly': 1, 'daily and intraday': 1}
S&P 500 in data: 36 | LOBSTER: 11 | Google 'Stocks' dataset: corpus 4 reference set 13
Q5 (temporal OOS): {'Y': 82, '?': 22, 'NA': 28, 'N': 2, 'P': 3} | applicable: 109
Q3 classical benchmark Y/P: 75 13
Studies with Q_score = 7: ['S066']

## Stylized facts (Table 3.2, Section 6.2)
SF1 47 (34%)
SF2 22 (16%)
SF3 42 (31%)
SF4 3 (2%)
SF5 12 (9%)
SF6 4 (3%)
SF7 3 (2%)
SF8 2 (1%)
SF9 28 (20%)
SF10 17 (12%)
SF11 8 (6%)
None tested: 57 (42%)
SD SF tested: 24 / 32 (75%)
FC SF tested: 6 / 26 (23%)
MS SF tested: 15 / 16 (94%)
RP SF tested: 21 / 31 (68%)
PS SF tested: 12 / 29 (41%)

## Metric families (Table 6.1)
Stylized-fact diagnostics corpus 80 (58%) | reference set 3 / 15
Distances between distributions corpus 26 (19%) | reference set 7 / 15
Model-based scores corpus 8 (6%) | reference set 15 / 15
Path-space distances corpus 4 (3%) | reference set 0 / 15
Adapted and causal distances corpus 4 (3%) | reference set 0 / 15
Probabilistic scoring rules corpus 7 (5%) | reference set 3 / 15
Point-forecast errors corpus 31 (23%) | reference set 1 / 15
Risk metrics corpus 24 (18%) | reference set 0 / 15
Decision metrics corpus 50 (36%) | reference set 0 / 15
Visualisation corpus 4 (3%) | reference set 10 / 15
Efficiency corpus 2 (1%) | reference set 2 / 15

## Fidelity and utility (Table 6.2)
('SD', 32, 84, 16, 16, 25)
('FC', 26, 23, 58, 15, 46)
('MS', 16, 94, 6, 6, 69)
('RP', 31, 77, 55, 42, 90)
('PS', 29, 48, 86, 41, 93)
('All', 137, 65, 47, 26, 64)
Fidelity: 89 | utility: 64 | both: 36 | fidelity only: 53 (39%) | utility only: 28 (20%) | neither: 20 (15%)
Robustness, counting model-based scores as fidelity: both = 37
Classical benchmark (Y or P): 88 (64%)
Robustness, peer-reviewed studies only (n = 90): fidelity 61% | utility 46% | both 24%
Robustness, preprints only (n = 44): fidelity 75% | utility 50% | both 30%
