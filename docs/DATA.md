# Data

## Simulated data

All the experiments of Sections 5 and 7.1-7.6 of the chapter (except the fidelity check Q0) use data simulated from the four
reference processes of `configs/preregistration.yaml`. They need no download: every path is generated
from the master seed through named random streams (`tavp.utils.rng`), so the same command produces the
same data on the same platform.

## Market data (Section 7.7, experiments E10, E11 and E13; Section 6.2, check Q0)

- Source: Kenneth R. French Data Library, *Fama/French 3 Factors [Daily]*,
  <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>. The series used is the
  daily return of the value-weighted portfolio of US stocks (CRSP), Mkt-RF + RF, in percent; the
  analysis uses the log-returns log(1 + r / 100) from 1926-07-01 to 2026-08-31.
- Vintage: CRSP 202608 (zip SHA-256 `2f29e22546069914890a712680a6f81f3680ebc3543b52864484d209ca13a7db`).
  The library updates the file every month and may revise past values; `scripts/get_french.py` reports
  whether the downloaded file is the vintage of the chapter, and the analysis truncates the series at
  its last date.
- Licence: the file is copyrighted by Eugene F. Fama and Kenneth R. French and is not redistributed in
  this repository. Download it with

      python scripts/get_french.py

  which writes it to `data/french/` (ignored by git). The frozen results in `results/frozen/` contain
  every quantity derived from the data that the chapter reports, so `make verify-frozen` works without
  the download.
- The Quant GAN fidelity check (Q0) uses the same series from May 2009 to December 2018, the closest
  public series to the daily S&P 500 index used by Wiese et al. (2020).

## Optional external code

Tail-GAN (Cont, Cucuringu, Xu and Zhang, 2025), discussed in the chapter as a direction for future
work, is distributed under the GPL-3.0 and is not part of this repository. `python scripts/get_tailgan.py`
downloads it at a pinned commit into `third_party_ext/tailgan/` (ignored by git) and checks the content.
