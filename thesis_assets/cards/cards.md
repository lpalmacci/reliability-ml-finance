### Validation card: tcvae on gbm (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 4.5), acf_r (rate 1.00, ratio 9.3), hill (rate 0.64, ratio 4.5), kurt (rate 0.97, ratio 25.8), kurt_agg5 (rate 0.95, ratio 7.0), leverage (rate 1.00, ratio 4.8), marginal_w1 (rate 1.00, ratio 2.9), mean (rate 0.92, ratio 11.9), signature_mmd (rate 1.00, ratio 7.5), skew (rate 0.82, ratio 13.1), std (rate 0.81, ratio 23.1), swd_logpath (rate 1.00, ratio 5.3)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.01 | 0.32 | 0.15 | 0.19 | 1.00 | 0.79 | 0.02 | 0.16 | 0.20 |
| TF | 0.11 | 0.22 | 0.14 | 0.22 | 0.87 | 0.92 | 0.36 | 0.40 | 0.34 |
| VT | 0.00 | 0.16 | 0.15 | 0.20 | 1.00 | 0.97 | 0.32 | 0.45 | 0.20 |
| TVF | 0.13 | 0.23 | 0.22 | 0.20 | 0.94 | 0.98 | 0.46 | 0.58 | 0.51 |

- L3 TF: pass rate 0.62, mean regret 0.055, Kendall tau 0.48
- L3 TVF: pass rate 0.54, mean regret 0.047, Kendall tau 0.48
- L3 VT: pass rate 1.00, mean regret 0.003, Kendall tau 0.87
- L4 rejected: TF/maxdd
- **Verdict: not fit for the declared use**

### Validation card: tcvae on gbm (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 4.3), acf_r (rate 1.00, ratio 11.8), hill (rate 0.66, ratio 5.3), kurt (rate 0.94, ratio 19.2), kurt_agg5 (rate 0.90, ratio 6.1), leverage (rate 0.63, ratio 1.8), marginal_w1 (rate 1.00, ratio 2.8), signature_mmd (rate 0.81, ratio 3.3), skew (rate 0.81, ratio 13.2), std (rate 0.76, ratio 8.3), swd_logpath (rate 1.00, ratio 3.4)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.27 | 0.40 | 0.15 | 0.41 | 1.00 | 1.00 | 0.80 | 0.39 | 0.20 |
| TF | 0.26 | 0.07 | 0.23 | 0.41 | 0.90 | 0.92 | 0.70 | 0.10 | 0.05 |
| VT | 0.38 | 0.40 | 0.30 | 0.40 | 1.00 | 1.00 | 0.94 | 0.82 | 0.63 |
| TVF | 0.35 | 0.11 | 0.38 | 0.42 | 0.94 | 1.00 | 0.78 | 0.40 | 0.31 |

- L3 TF: pass rate 0.86, mean regret 0.015, Kendall tau 0.75
- L3 TVF: pass rate 0.82, mean regret 0.015, Kendall tau 0.71
- L3 VT: pass rate 1.00, mean regret 0.001, Kendall tau 0.93
- **Verdict: not fit for the declared use**

### Validation card: qgan on gbm (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 4.8), acf_r (rate 1.00, ratio 9.9), hill (rate 0.87, ratio 17.5), kurt (rate 1.00, ratio 24.3), kurt_agg5 (rate 0.87, ratio 9.4), leverage (rate 1.00, ratio 5.7), marginal_w1 (rate 1.00, ratio 1.4), mean (rate 0.68, ratio 9.5), signature_mmd (rate 0.82, ratio 5.9), std (rate 0.80, ratio 12.5), swd_logpath (rate 0.82, ratio 3.4)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.26 | 0.13 | 0.02 | 1.00 | 1.00 | 0.40 | 0.57 | 0.51 |
| TF | 0.11 | 0.45 | 0.10 | 0.23 | 1.00 | 1.00 | 0.46 | 0.90 | 0.83 |
| VT | 0.15 | 0.22 | 0.26 | 0.05 | 1.00 | 1.00 | 0.40 | 0.70 | 0.70 |
| TVF | 0.14 | 0.42 | 0.22 | 0.22 | 1.00 | 1.00 | 0.54 | 0.88 | 0.81 |

- L3 TF: pass rate 0.64, mean regret 0.036, Kendall tau 0.57
- L3 TVF: pass rate 0.68, mean regret 0.031, Kendall tau 0.62
- L3 VT: pass rate 1.00, mean regret 0.000, Kendall tau 0.85
- L4 rejected: none
- **Verdict: not fit for the declared use**

### Validation card: qgan on gbm (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_r (rate 0.99, ratio 3.9), marginal_w1 (rate 0.52, ratio 1.1), std (rate 0.87, ratio 13.4)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.20 | 0.60 | 0.20 | 0.92 | 1.00 | 1.00 | 0.80 | 0.80 | 0.78 |
| TF | 0.56 | 0.97 | 0.57 | 0.86 | 1.00 | 1.00 | 0.98 | 0.96 | 0.88 |
| VT | 0.20 | 0.60 | 0.21 | 0.93 | 1.00 | 1.00 | 0.94 | 0.96 | 0.95 |
| TVF | 0.60 | 0.96 | 0.60 | 0.86 | 1.00 | 1.00 | 0.99 | 1.00 | 0.98 |

- L3 TF: pass rate 1.00, mean regret 0.002, Kendall tau 0.89
- L3 TVF: pass rate 0.98, mean regret 0.001, Kendall tau 0.89
- L3 VT: pass rate 1.00, mean regret 0.000, Kendall tau 0.97
- **Verdict: not fit for the declared use**

### Validation card: vae_mlp on gbm (single regime)
Runs: 3; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 92%
- L1 flagged: acf_abs (rate 1.00, ratio 59.1), acf_r (rate 1.00, ratio 109.2), hill (rate 1.00, ratio 48.5), kurt (rate 1.00, ratio 268.3), kurt_agg20 (rate 1.00, ratio 14.6), kurt_agg5 (rate 0.99, ratio 173.3), leverage (rate 1.00, ratio 30.9), marginal_w1 (rate 1.00, ratio 26.9), signature_mmd (rate 1.00, ratio 20.6), skew (rate 1.00, ratio 103.2), std (rate 1.00, ratio 652.0), swd_logpath (rate 1.00, ratio 28.1)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 |
| TF | 0.01 | 0.02 | 0.00 | 0.01 | 0.00 | 0.00 | 0.28 | 0.00 | 0.00 |
| VT | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.21 | 0.55 | 0.13 | 0.13 |
| TVF | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.24 | 0.00 | 0.00 |

- L3 TF: pass rate 0.30, mean regret 0.149, Kendall tau 0.20
- L3 TVF: pass rate 0.20, mean regret 0.122, Kendall tau 0.24
- L3 VT: pass rate 0.70, mean regret 0.044, Kendall tau 0.14
- L4 rejected: BH/maxdd, BH/ret, BH/sharpe, TF/maxdd, TF/ret, TF/sharpe, TVF/maxdd, TVF/ret, TVF/sharpe, VT/maxdd, VT/ret, VT/sharpe
- **Verdict: not fit for the declared use**

### Validation card: bootstrap on gbm
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: hill (rate 0.72, ratio 11.8), kurt (rate 1.00, ratio 14.5), marginal_w1 (rate 0.97, ratio 1.3), mean (rate 0.71, ratio 7.0), signature_mmd (rate 0.70, ratio 4.8), skew (rate 0.55, ratio 4.0), std (rate 0.97, ratio 7.6), swd_logpath (rate 0.68, ratio 2.5)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.20 | 0.39 | 0.16 | 0.20 | 1.00 | 1.00 | 0.21 | 0.60 | 0.60 |
| TF | 0.36 | 0.96 | 0.37 | 0.40 | 1.00 | 1.00 | 0.71 | 0.98 | 0.99 |
| VT | 0.20 | 0.40 | 0.18 | 0.20 | 1.00 | 1.00 | 0.41 | 0.80 | 0.79 |
| TVF | 0.35 | 0.93 | 0.40 | 0.41 | 1.00 | 1.00 | 0.69 | 0.99 | 0.99 |

- L3 TF: pass rate 0.78, mean regret 0.027, Kendall tau 0.71
- L3 TVF: pass rate 0.78, mean regret 0.023, Kendall tau 0.69
- L3 VT: pass rate 1.00, mean regret 0.000, Kendall tau 0.97
- L4 rejected: none
- **Verdict: not fit for the declared use**

### Validation card: garch on gbm
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: kurt (rate 0.62, ratio 5.0), mean (rate 0.67, ratio 7.0), signature_mmd (rate 0.67, ratio 4.7), std (rate 0.80, ratio 8.3), swd_logpath (rate 0.64, ratio 2.4)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.15 | 0.24 | 0.10 | 0.25 | 1.00 | 1.00 | 0.20 | 0.58 | 0.59 |
| TF | 0.37 | 0.94 | 0.38 | 0.44 | 1.00 | 1.00 | 0.70 | 0.98 | 0.98 |
| VT | 0.16 | 0.25 | 0.12 | 0.25 | 1.00 | 1.00 | 0.38 | 0.78 | 0.76 |
| TVF | 0.34 | 0.92 | 0.41 | 0.45 | 1.00 | 1.00 | 0.69 | 1.00 | 1.00 |

- L3 TF: pass rate 0.80, mean regret 0.023, Kendall tau 0.71
- L3 TVF: pass rate 0.78, mean regret 0.024, Kendall tau 0.70
- L3 VT: pass rate 1.00, mean regret 0.000, Kendall tau 0.97
- L4 rejected: none
- **Verdict: not fit for the declared use**

### Validation card: tcvae on heston (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 29.3), acf_r (rate 1.00, ratio 8.2), hill (rate 0.99, ratio 11.4), kurt (rate 0.96, ratio 15.4), kurt_agg5 (rate 0.81, ratio 8.6), leverage (rate 1.00, ratio 3.2), marginal_w1 (rate 1.00, ratio 3.8), mean (rate 0.72, ratio 5.1), signature_mmd (rate 0.88, ratio 4.1), skew (rate 0.80, ratio 6.6), std (rate 1.00, ratio 17.1), swd_logpath (rate 1.00, ratio 3.0)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.58 | 0.00 | 0.19 | 0.51 | 0.59 | 0.21 | 0.20 | 0.06 |
| TF | 0.14 | 0.17 | 0.18 | 0.31 | 0.66 | 0.37 | 0.42 | 0.22 | 0.09 |
| VT | 0.04 | 0.66 | 0.00 | 0.14 | 0.88 | 0.91 | 0.39 | 0.55 | 0.56 |
| TVF | 0.10 | 0.24 | 0.18 | 0.31 | 0.72 | 0.77 | 0.53 | 0.44 | 0.40 |

- L3 TF: pass rate 0.84, mean regret 0.042, Kendall tau 0.65
- L3 TVF: pass rate 0.74, mean regret 0.038, Kendall tau 0.56
- L3 VT: pass rate 1.00, mean regret 0.003, Kendall tau 0.87
- L4 rejected: BH/sharpe
- **Verdict: not fit for the declared use**

### Validation card: tcvae on heston (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 28.8), acf_r (rate 1.00, ratio 9.9), hill (rate 0.93, ratio 6.2), kurt (rate 0.87, ratio 10.8), kurt_agg5 (rate 0.91, ratio 8.3), leverage (rate 0.98, ratio 2.8), marginal_w1 (rate 1.00, ratio 3.6), signature_mmd (rate 0.85, ratio 3.6), skew (rate 1.00, ratio 18.0), std (rate 1.00, ratio 10.4), swd_logpath (rate 1.00, ratio 3.0)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.20 | 0.00 | 0.31 | 0.00 | 0.80 | 0.12 | 0.79 | 0.01 | 0.11 |
| TF | 0.32 | 0.11 | 0.24 | 0.02 | 0.83 | 0.00 | 0.66 | 0.05 | 0.00 |
| VT | 0.12 | 0.08 | 0.16 | 0.00 | 0.99 | 0.90 | 0.82 | 0.71 | 0.59 |
| TVF | 0.23 | 0.11 | 0.28 | 0.21 | 0.85 | 0.78 | 0.74 | 0.40 | 0.30 |

- L3 TF: pass rate 1.00, mean regret 0.004, Kendall tau 0.76
- L3 TVF: pass rate 0.92, mean regret 0.008, Kendall tau 0.68
- L3 VT: pass rate 1.00, mean regret 0.001, Kendall tau 0.91
- **Verdict: not fit for the declared use**

### Validation card: qgan on heston (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 11.0), acf_r (rate 1.00, ratio 6.9), hill (rate 0.52, ratio 3.4), kurt (rate 0.91, ratio 11.0), kurt_agg20 (rate 0.65, ratio 4.5), kurt_agg5 (rate 0.63, ratio 6.3), leverage (rate 1.00, ratio 6.5), marginal_w1 (rate 0.90, ratio 1.9), mean (rate 0.80, ratio 9.7), signature_mmd (rate 0.85, ratio 5.8), skew (rate 0.96, ratio 8.2), std (rate 0.80, ratio 14.7), swd_logpath (rate 0.98, ratio 3.2)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.16 | 0.02 | 0.00 | 0.16 | 0.60 | 0.80 | 0.20 | 0.27 | 0.20 |
| TF | 0.11 | 0.33 | 0.09 | 0.05 | 0.64 | 0.98 | 0.30 | 0.30 | 0.30 |
| VT | 0.21 | 0.06 | 0.01 | 0.11 | 0.79 | 0.92 | 0.34 | 0.45 | 0.52 |
| TVF | 0.10 | 0.30 | 0.13 | 0.21 | 0.78 | 0.93 | 0.41 | 0.37 | 0.31 |

- L3 TF: pass rate 0.62, mean regret 0.053, Kendall tau 0.47
- L3 TVF: pass rate 0.68, mean regret 0.044, Kendall tau 0.51
- L3 VT: pass rate 1.00, mean regret 0.003, Kendall tau 0.90
- L4 rejected: TVF/sharpe
- **Verdict: not fit for the declared use**

### Validation card: qgan on heston (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 7.1), acf_r (rate 1.00, ratio 8.1), hill (rate 0.69, ratio 6.4), kurt (rate 0.98, ratio 15.9), leverage (rate 1.00, ratio 4.7), marginal_w1 (rate 0.82, ratio 1.2), mean (rate 0.80, ratio 6.7), signature_mmd (rate 1.00, ratio 5.9), skew (rate 0.84, ratio 8.0), swd_logpath (rate 1.00, ratio 3.8)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.19 | 0.21 | 0.00 | 1.00 | 0.20 | 0.06 | 0.14 | 0.00 |
| TF | 0.09 | 0.24 | 0.08 | 0.05 | 0.82 | 0.73 | 0.15 | 0.23 | 0.22 |
| VT | 0.01 | 0.09 | 0.20 | 0.00 | 1.00 | 0.73 | 0.29 | 0.19 | 0.09 |
| TVF | 0.09 | 0.27 | 0.12 | 0.07 | 0.84 | 0.79 | 0.28 | 0.40 | 0.35 |

- L3 TF: pass rate 0.82, mean regret 0.030, Kendall tau 0.49
- L3 TVF: pass rate 0.72, mean regret 0.042, Kendall tau 0.48
- L3 VT: pass rate 1.00, mean regret 0.001, Kendall tau 0.90
- **Verdict: not fit for the declared use**

### Validation card: vae_mlp on heston (single regime)
Runs: 3; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 92%
- L1 flagged: acf_abs (rate 1.00, ratio 26.2), acf_r (rate 1.00, ratio 72.3), hill (rate 1.00, ratio 25.8), kurt (rate 1.00, ratio 24.9), kurt_agg20 (rate 0.63, ratio 8.2), kurt_agg5 (rate 0.67, ratio 5.0), leverage (rate 1.00, ratio 22.8), marginal_w1 (rate 1.00, ratio 23.6), mean (rate 0.68, ratio 13.8), signature_mmd (rate 1.00, ratio 16.0), skew (rate 1.00, ratio 20.5), std (rate 1.00, ratio 158.7), swd_logpath (rate 1.00, ratio 19.7)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 |
| TF | 0.00 | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.12 | 0.00 | 0.00 |
| VT | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.25 | 0.03 | 0.05 |
| TVF | 0.01 | 0.00 | 0.00 | 0.00 | 0.02 | 0.01 | 0.06 | 0.03 | 0.00 |

- L3 TF: pass rate 0.30, mean regret 0.201, Kendall tau 0.37
- L3 TVF: pass rate 0.33, mean regret 0.101, Kendall tau 0.33
- L3 VT: pass rate 0.43, mean regret 0.084, Kendall tau 0.33
- L4 rejected: BH/maxdd, BH/ret, BH/sharpe, TF/maxdd, TF/ret, TF/sharpe, TVF/maxdd, TVF/ret, TVF/sharpe, VT/maxdd, VT/ret, VT/sharpe
- **Verdict: not fit for the declared use**

### Validation card: bootstrap on heston
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 10.2), acf_r (rate 1.00, ratio 7.9), hill (rate 0.63, ratio 4.1), kurt (rate 1.00, ratio 11.3), kurt_agg20 (rate 0.53, ratio 4.0), kurt_agg5 (rate 0.60, ratio 5.5), leverage (rate 1.00, ratio 7.8), marginal_w1 (rate 0.96, ratio 1.7), mean (rate 0.62, ratio 6.4), signature_mmd (rate 0.80, ratio 3.8), skew (rate 0.96, ratio 8.6), std (rate 0.91, ratio 10.8), swd_logpath (rate 0.93, ratio 2.6)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.22 | 0.29 | 0.20 | 0.60 | 0.46 | 0.40 | 0.12 | 0.16 |
| TF | 0.18 | 0.39 | 0.16 | 0.23 | 0.69 | 0.79 | 0.55 | 0.43 | 0.24 |
| VT | 0.02 | 0.18 | 0.18 | 0.20 | 0.88 | 0.88 | 0.58 | 0.38 | 0.26 |
| TVF | 0.14 | 0.46 | 0.22 | 0.18 | 0.91 | 0.95 | 0.57 | 0.68 | 0.58 |

- L3 TF: pass rate 0.78, mean regret 0.051, Kendall tau 0.60
- L3 TVF: pass rate 0.78, mean regret 0.041, Kendall tau 0.54
- L3 VT: pass rate 1.00, mean regret 0.002, Kendall tau 0.91
- L4 rejected: TVF/ret
- **Verdict: not fit for the declared use**

### Validation card: garch on heston
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 97%
- L1 flagged: acf_abs (rate 1.00, ratio 45.5), acf_r (rate 0.81, ratio 2.9), hill (rate 1.00, ratio 46.3), kurt (rate 1.00, ratio 1145.2), kurt_agg20 (rate 1.00, ratio 152.0), kurt_agg5 (rate 1.00, ratio 529.3), leverage (rate 1.00, ratio 6.7), marginal_w1 (rate 1.00, ratio 4.8), mean (rate 0.81, ratio 10.0), signature_mmd (rate 1.00, ratio 15.6), skew (rate 0.98, ratio 37.0), std (rate 0.98, ratio 39.6), swd_logpath (rate 1.00, ratio 8.7)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.20 | 0.00 | 0.06 | 0.40 | 0.63 | 0.20 | 0.00 | 0.22 |
| TF | 0.12 | 0.73 | 0.10 | 0.25 | 0.61 | 0.44 | 0.46 | 0.46 | 0.00 |
| VT | 0.00 | 0.20 | 0.00 | 0.07 | 0.80 | 0.90 | 0.21 | 0.31 | 0.44 |
| TVF | 0.08 | 0.64 | 0.10 | 0.26 | 0.68 | 0.81 | 0.45 | 0.44 | 0.43 |

- L3 TF: pass rate 1.00, mean regret 0.003, Kendall tau 0.90
- L3 TVF: pass rate 1.00, mean regret 0.001, Kendall tau 0.82
- L3 VT: pass rate 1.00, mean regret 0.001, Kendall tau 0.91
- L4 rejected: BH/maxdd
- **Verdict: not fit for the declared use**

### Validation card: tcvae on gjr_garch_t (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 0.67, ratio 9.8), acf_r (rate 0.61, ratio 4.5), hill (rate 1.00, ratio 31.3), marginal_w1 (rate 1.00, ratio 4.4), std (rate 0.80, ratio 14.8)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.20 | 0.20 | 0.15 | 0.00 | 0.40 | 0.20 | 0.59 | 0.20 | 0.00 |
| TF | 0.10 | 0.18 | 0.11 | 0.08 | 0.39 | 0.02 | 0.45 | 0.03 | 0.00 |
| VT | 0.15 | 0.20 | 0.12 | 0.08 | 0.71 | 0.58 | 0.72 | 0.20 | 0.12 |
| TVF | 0.08 | 0.17 | 0.14 | 0.15 | 0.78 | 0.76 | 0.67 | 0.53 | 0.46 |

- L3 TF: pass rate 0.94, mean regret 0.008, Kendall tau 0.81
- L3 TVF: pass rate 0.94, mean regret 0.012, Kendall tau 0.77
- L3 VT: pass rate 1.00, mean regret 0.000, Kendall tau 0.92
- L4 rejected: BH/maxdd, TF/maxdd, TF/ret, TVF/maxdd
- **Verdict: not fit for the declared use**

### Validation card: tcvae on gjr_garch_t (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 0.96, ratio 9.8), acf_r (rate 0.52, ratio 4.1), hill (rate 1.00, ratio 23.5), marginal_w1 (rate 1.00, ratio 3.6), std (rate 0.80, ratio 8.9)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.39 | 0.00 | 0.52 | 0.25 | 1.00 | 0.13 | 0.60 | 0.00 | 0.00 |
| TF | 0.19 | 0.11 | 0.14 | 0.19 | 0.65 | 0.00 | 0.35 | 0.01 | 0.00 |
| VT | 0.29 | 0.07 | 0.28 | 0.31 | 1.00 | 0.90 | 0.68 | 0.65 | 0.54 |
| TVF | 0.14 | 0.19 | 0.19 | 0.41 | 0.98 | 0.87 | 0.37 | 0.69 | 0.56 |

- L3 TF: pass rate 0.96, mean regret 0.002, Kendall tau 0.87
- L3 TVF: pass rate 0.88, mean regret 0.016, Kendall tau 0.72
- L3 VT: pass rate 1.00, mean regret 0.000, Kendall tau 0.94
- **Verdict: not fit for the declared use**

### Validation card: qgan on gjr_garch_t (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 99%
- L1 flagged: acf_r (rate 0.81, ratio 5.6), hill (rate 0.81, ratio 11.4), marginal_w1 (rate 1.00, ratio 2.3), mean (rate 0.56, ratio 5.5), std (rate 0.94, ratio 12.0)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.17 | 0.00 | 0.20 | 0.26 | 0.47 | 0.38 | 0.40 | 0.20 | 0.00 |
| TF | 0.04 | 0.50 | 0.06 | 0.12 | 0.80 | 0.38 | 0.19 | 0.19 | 0.02 |
| VT | 0.16 | 0.02 | 0.15 | 0.14 | 0.74 | 0.68 | 0.40 | 0.20 | 0.04 |
| TVF | 0.06 | 0.56 | 0.18 | 0.13 | 0.94 | 0.95 | 0.42 | 0.73 | 0.65 |

- L3 TF: pass rate 0.74, mean regret 0.063, Kendall tau 0.63
- L3 TVF: pass rate 0.74, mean regret 0.057, Kendall tau 0.57
- L3 VT: pass rate 0.98, mean regret 0.021, Kendall tau 0.83
- L4 rejected: BH/maxdd
- **Verdict: not fit for the declared use**

### Validation card: qgan on gjr_garch_t (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 99%
- L1 flagged: marginal_w1 (rate 0.74, ratio 1.2), mean (rate 0.72, ratio 5.9)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.30 | 0.00 | 0.32 | 0.00 | 1.00 | 0.60 | 0.20 | 0.00 | 0.00 |
| TF | 0.08 | 0.20 | 0.09 | 0.05 | 0.92 | 0.77 | 0.21 | 0.36 | 0.12 |
| VT | 0.27 | 0.03 | 0.26 | 0.08 | 1.00 | 0.80 | 0.52 | 0.11 | 0.09 |
| TVF | 0.15 | 0.21 | 0.27 | 0.12 | 1.00 | 0.96 | 0.44 | 0.62 | 0.52 |

- L3 TF: pass rate 0.62, mean regret 0.064, Kendall tau 0.56
- L3 TVF: pass rate 0.76, mean regret 0.038, Kendall tau 0.61
- L3 VT: pass rate 1.00, mean regret 0.016, Kendall tau 0.80
- **Verdict: not fit for the declared use**

### Validation card: vae_mlp on gjr_garch_t (single regime)
Runs: 3; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 97%
- L1 flagged: acf_abs (rate 0.60, ratio 7.0), acf_r (rate 1.00, ratio 44.5), hill (rate 1.00, ratio 38.5), leverage (rate 0.66, ratio 5.3), marginal_w1 (rate 1.00, ratio 19.3), mean (rate 0.62, ratio 4.9), std (rate 1.00, ratio 58.3), swd_logpath (rate 1.00, ratio 8.3)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 |
| TF | 0.03 | 0.01 | 0.00 | 0.03 | 0.00 | 0.00 | 0.45 | 0.00 | 0.00 |
| VT | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.00 | 0.43 | 0.00 | 0.00 |
| TVF | 0.03 | 0.01 | 0.01 | 0.09 | 0.00 | 0.00 | 0.44 | 0.00 | 0.00 |

- L3 TF: pass rate 0.20, mean regret 0.234, Kendall tau 0.48
- L3 TVF: pass rate 0.20, mean regret 0.172, Kendall tau 0.43
- L3 VT: pass rate 0.47, mean regret 0.078, Kendall tau 0.38
- L4 rejected: BH/maxdd, BH/ret, BH/sharpe, TF/maxdd, TF/ret, TF/sharpe, TVF/maxdd, TVF/ret, TVF/sharpe, VT/maxdd, VT/ret, VT/sharpe
- **Verdict: not fit for the declared use**

### Validation card: bootstrap on gjr_garch_t
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_r (rate 0.71, ratio 5.5), hill (rate 0.93, ratio 14.8), marginal_w1 (rate 1.00, ratio 2.4), mean (rate 0.53, ratio 4.4), std (rate 0.80, ratio 10.8)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.20 | 0.33 | 0.10 | 0.00 | 0.40 | 0.18 | 0.47 | 0.20 | 0.01 |
| TF | 0.18 | 0.34 | 0.12 | 0.16 | 0.57 | 0.33 | 0.60 | 0.30 | 0.06 |
| VT | 0.18 | 0.24 | 0.11 | 0.00 | 0.93 | 0.79 | 0.62 | 0.41 | 0.51 |
| TVF | 0.14 | 0.46 | 0.28 | 0.12 | 0.98 | 0.97 | 0.68 | 0.71 | 0.61 |

- L3 TF: pass rate 0.90, mean regret 0.008, Kendall tau 0.76
- L3 TVF: pass rate 0.72, mean regret 0.035, Kendall tau 0.67
- L3 VT: pass rate 1.00, mean regret 0.010, Kendall tau 0.89
- L4 rejected: BH/maxdd, TF/maxdd
- **Verdict: not fit for the declared use**

### Validation card: garch on gjr_garch_t
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 99%
- L1 flagged: hill (rate 0.60, ratio 5.2), marginal_w1 (rate 1.00, ratio 2.3), std (rate 0.62, ratio 10.1)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.07 | 0.60 | 0.04 | 0.58 | 0.76 | 0.40 | 0.60 | 0.11 | 0.08 |
| TF | 0.46 | 0.80 | 0.43 | 0.64 | 0.80 | 0.40 | 0.72 | 0.26 | 0.08 |
| VT | 0.15 | 0.59 | 0.11 | 0.56 | 0.89 | 0.87 | 0.60 | 0.42 | 0.36 |
| TVF | 0.46 | 0.75 | 0.44 | 0.51 | 0.98 | 0.98 | 0.72 | 0.85 | 0.80 |

- L3 TF: pass rate 0.96, mean regret 0.012, Kendall tau 0.88
- L3 TVF: pass rate 0.94, mean regret 0.012, Kendall tau 0.83
- L3 VT: pass rate 1.00, mean regret 0.001, Kendall tau 0.94
- L4 rejected: none
- **Verdict: not fit for the declared use**

### Validation card: tcvae on regime (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 36.3), acf_r (rate 0.99, ratio 6.6), hill (rate 1.00, ratio 39.0), kurt (rate 0.97, ratio 40.3), kurt_agg20 (rate 1.00, ratio 7.3), kurt_agg5 (rate 1.00, ratio 23.3), leverage (rate 0.99, ratio 5.3), marginal_w1 (rate 1.00, ratio 4.8), mean (rate 0.74, ratio 7.4), signature_mmd (rate 1.00, ratio 8.3), skew (rate 0.75, ratio 5.0), std (rate 1.00, ratio 38.5), swd_logpath (rate 1.00, ratio 5.8)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.18 | 0.03 | 0.00 | 0.40 | 0.00 | 0.40 | 0.05 | 0.00 |
| TF | 0.16 | 0.27 | 0.18 | 0.20 | 0.40 | 0.01 | 0.46 | 0.06 | 0.00 |
| VT | 0.00 | 0.13 | 0.05 | 0.00 | 0.78 | 0.73 | 0.23 | 0.60 | 0.59 |
| TVF | 0.11 | 0.15 | 0.17 | 0.16 | 0.50 | 0.41 | 0.37 | 0.28 | 0.22 |

- L3 TF: pass rate 0.70, mean regret 0.056, Kendall tau 0.56
- L3 TVF: pass rate 0.74, mean regret 0.055, Kendall tau 0.53
- L3 VT: pass rate 0.98, mean regret 0.024, Kendall tau 0.53
- L4 rejected: BH/maxdd, BH/ret, BH/sharpe, TF/maxdd, TF/ret, TF/sharpe, TVF/maxdd, TVF/ret, TVF/sharpe, VT/maxdd, VT/ret, VT/sharpe
- **Verdict: not fit for the declared use**

### Validation card: tcvae on regime (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 34.5), acf_r (rate 1.00, ratio 7.0), hill (rate 1.00, ratio 34.8), kurt (rate 1.00, ratio 44.7), kurt_agg20 (rate 0.94, ratio 5.8), kurt_agg5 (rate 1.00, ratio 21.4), leverage (rate 1.00, ratio 3.9), marginal_w1 (rate 1.00, ratio 4.2), mean (rate 0.58, ratio 3.7), signature_mmd (rate 1.00, ratio 4.7), skew (rate 0.77, ratio 6.2), std (rate 1.00, ratio 21.7), swd_logpath (rate 1.00, ratio 4.0)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.04 | 0.00 | 0.20 | 0.00 | 0.80 | 0.00 | 0.46 | 0.00 | 0.00 |
| TF | 0.37 | 0.24 | 0.32 | 0.01 | 0.80 | 0.00 | 0.66 | 0.00 | 0.00 |
| VT | 0.21 | 0.34 | 0.16 | 0.03 | 1.00 | 0.99 | 0.81 | 0.86 | 0.83 |
| TVF | 0.29 | 0.17 | 0.22 | 0.12 | 0.89 | 0.54 | 0.57 | 0.35 | 0.24 |

- L3 TF: pass rate 0.90, mean regret 0.010, Kendall tau 0.83
- L3 TVF: pass rate 0.80, mean regret 0.023, Kendall tau 0.66
- L3 VT: pass rate 1.00, mean regret 0.027, Kendall tau 0.53
- **Verdict: not fit for the declared use**

### Validation card: qgan on regime (single regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 5.6), acf_r (rate 1.00, ratio 6.9), hill (rate 0.65, ratio 8.1), kurt (rate 0.87, ratio 14.6), kurt_agg20 (rate 0.75, ratio 5.0), kurt_agg5 (rate 0.84, ratio 8.7), leverage (rate 1.00, ratio 5.8), marginal_w1 (rate 1.00, ratio 2.3), mean (rate 0.68, ratio 4.6), signature_mmd (rate 1.00, ratio 8.5), skew (rate 0.84, ratio 16.6), std (rate 0.82, ratio 13.7), swd_logpath (rate 1.00, ratio 5.4)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.04 | 0.20 | 0.00 | 0.00 | 0.60 | 0.03 | 0.40 | 0.00 | 0.00 |
| TF | 0.02 | 0.20 | 0.02 | 0.03 | 0.83 | 0.83 | 0.06 | 0.48 | 0.36 |
| VT | 0.12 | 0.20 | 0.00 | 0.00 | 0.87 | 0.69 | 0.45 | 0.30 | 0.13 |
| TVF | 0.03 | 0.22 | 0.08 | 0.07 | 0.89 | 0.90 | 0.19 | 0.51 | 0.42 |

- L3 TF: pass rate 0.86, mean regret 0.022, Kendall tau 0.65
- L3 TVF: pass rate 0.82, mean regret 0.021, Kendall tau 0.64
- L3 VT: pass rate 1.00, mean regret 0.019, Kendall tau 0.51
- L4 rejected: BH/sharpe, VT/sharpe
- **Verdict: not fit for the declared use**

### Validation card: qgan on regime (rich regime)
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 3.6), acf_r (rate 1.00, ratio 9.8), kurt (rate 1.00, ratio 38.9), kurt_agg5 (rate 0.91, ratio 14.1), leverage (rate 0.80, ratio 3.6), marginal_w1 (rate 0.74, ratio 1.3), signature_mmd (rate 0.75, ratio 3.6), skew (rate 0.85, ratio 10.2), std (rate 0.66, ratio 5.2), swd_logpath (rate 0.98, ratio 2.5)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.27 | 0.20 | 0.35 | 0.00 | 1.00 | 0.60 | 0.94 | 0.37 | 0.25 |
| TF | 0.11 | 0.32 | 0.08 | 0.06 | 0.74 | 0.54 | 0.38 | 0.25 | 0.17 |
| VT | 0.37 | 0.20 | 0.25 | 0.02 | 1.00 | 0.94 | 1.00 | 0.34 | 0.23 |
| TVF | 0.18 | 0.22 | 0.20 | 0.13 | 0.99 | 0.94 | 0.59 | 0.49 | 0.44 |

- L3 TF: pass rate 0.92, mean regret 0.011, Kendall tau 0.68
- L3 TVF: pass rate 0.86, mean regret 0.015, Kendall tau 0.70
- L3 VT: pass rate 1.00, mean regret 0.016, Kendall tau 0.69
- **Verdict: not fit for the declared use**

### Validation card: vae_mlp on regime (single regime)
Runs: 3; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 80%
- L1 flagged: acf_abs (rate 1.00, ratio 42.6), acf_r (rate 1.00, ratio 87.1), hill (rate 0.97, ratio 23.0), kurt (rate 1.00, ratio 245.8), kurt_agg20 (rate 0.96, ratio 7.7), kurt_agg5 (rate 0.96, ratio 25.0), leverage (rate 1.00, ratio 14.0), marginal_w1 (rate 1.00, ratio 22.2), mean (rate 0.70, ratio 7.1), signature_mmd (rate 1.00, ratio 9.8), skew (rate 0.68, ratio 34.7), std (rate 1.00, ratio 162.5), swd_logpath (rate 1.00, ratio 6.8)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 |
| TF | 0.01 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 | 0.00 |
| VT | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.26 | 0.22 | 0.06 | 0.09 |
| TVF | 0.01 | 0.02 | 0.01 | 0.00 | 0.00 | 0.01 | 0.16 | 0.00 | 0.00 |

- L3 TF: pass rate 0.10, mean regret 0.217, Kendall tau 0.11
- L3 TVF: pass rate 0.03, mean regret 0.161, Kendall tau 0.10
- L3 VT: pass rate 0.80, mean regret 0.039, Kendall tau 0.23
- L4 rejected: BH/maxdd, BH/ret, BH/sharpe, TF/maxdd, TF/ret, TF/sharpe, TVF/maxdd, TVF/ret, TVF/sharpe, VT/maxdd, VT/ret, VT/sharpe
- **Verdict: not fit for the declared use**

### Validation card: bootstrap on regime
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 1.00, ratio 7.1), acf_r (rate 1.00, ratio 6.1), hill (rate 0.82, ratio 9.8), kurt (rate 0.99, ratio 10.5), kurt_agg20 (rate 0.67, ratio 4.1), kurt_agg5 (rate 0.80, ratio 7.9), leverage (rate 1.00, ratio 7.4), marginal_w1 (rate 1.00, ratio 1.9), mean (rate 0.70, ratio 5.1), signature_mmd (rate 1.00, ratio 5.4), skew (rate 0.77, ratio 5.3), std (rate 0.91, ratio 12.6), swd_logpath (rate 1.00, ratio 3.8)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.00 | 0.30 | 0.12 | 0.20 | 1.00 | 0.24 | 0.24 | 0.20 | 0.00 |
| TF | 0.03 | 0.41 | 0.03 | 0.12 | 0.84 | 0.70 | 0.18 | 0.49 | 0.26 |
| VT | 0.00 | 0.12 | 0.14 | 0.18 | 1.00 | 0.93 | 0.60 | 0.75 | 0.16 |
| TVF | 0.08 | 0.45 | 0.12 | 0.18 | 0.99 | 0.94 | 0.36 | 0.69 | 0.52 |

- L3 TF: pass rate 0.84, mean regret 0.020, Kendall tau 0.66
- L3 TVF: pass rate 0.82, mean regret 0.020, Kendall tau 0.64
- L3 VT: pass rate 1.00, mean regret 0.014, Kendall tau 0.61
- L4 rejected: none
- **Verdict: not fit for the declared use**

### Validation card: garch on regime
Runs: 5; contexts: 50.
- L0: hash consistent True, scenarios valid True, resolvable cells 100%
- L1 flagged: acf_abs (rate 0.97, ratio 8.3), hill (rate 0.90, ratio 7.2), kurt (rate 0.92, ratio 106.6), kurt_agg20 (rate 0.84, ratio 13.4), kurt_agg5 (rate 0.95, ratio 42.1), leverage (rate 1.00, ratio 6.9), marginal_w1 (rate 1.00, ratio 2.3), mean (rate 1.00, ratio 9.9), signature_mmd (rate 1.00, ratio 8.2), skew (rate 0.96, ratio 15.0), std (rate 0.88, ratio 15.5), swd_logpath (rate 1.00, ratio 5.3)

| L2 pass rate | Sharpe mean | Sharpe q05 | Sharpe q50 | Sharpe q95 | MaxDD mean | MaxDD q95 | Return mean | Return var95 | Return es95 |
|---|---|---|---|---|---|---|---|---|---|
| BH | 0.16 | 0.00 | 0.16 | 0.52 | 0.33 | 0.20 | 0.00 | 0.00 | 0.00 |
| TF | 0.11 | 0.92 | 0.15 | 0.19 | 0.82 | 0.42 | 0.64 | 0.23 | 0.16 |
| VT | 0.28 | 0.01 | 0.27 | 0.41 | 0.98 | 0.78 | 0.54 | 0.30 | 0.12 |
| TVF | 0.16 | 0.86 | 0.24 | 0.28 | 0.98 | 0.81 | 0.78 | 0.61 | 0.44 |

- L3 TF: pass rate 1.00, mean regret 0.002, Kendall tau 0.89
- L3 TVF: pass rate 0.92, mean regret 0.014, Kendall tau 0.79
- L3 VT: pass rate 0.96, mean regret 0.026, Kendall tau 0.58
- L4 rejected: none
- **Verdict: not fit for the declared use**
