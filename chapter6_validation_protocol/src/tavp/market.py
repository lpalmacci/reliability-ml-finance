"""Market data of the additional analysis D8: daily value-weighted US market return (CRSP), from the
Ken French Data Library (file F-F_Research_Data_Factors_daily). The file is copyrighted and is not
redistributed; `scripts/get_french.py` downloads it and checks the vintage used in the chapter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tavp.utils import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "french"
ZIP_NAME = "F-F_Research_Data_Factors_daily_CSV.zip"
CSV_NAME = "F-F_Research_Data_Factors_daily.csv"
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/" + ZIP_NAME
# vintage used in the chapter: CRSP database 202608, 1926-07-01 to 2026-08-31
VINTAGE_SHA256 = "2f29e22546069914890a712680a6f81f3680ebc3543b52864484d209ca13a7db"
VINTAGE_LAST_DATE = 20260831


def parse_factors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Dates (yyyymmdd) and simple daily market returns Mkt-RF + RF (as fractions) of the daily file."""
    dates, rets, started = [], [], False
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if not started:
            started = len(cells) >= 5 and cells[1] == "Mkt-RF" and cells[4] == "RF"
            continue
        if len(cells) < 5 or not cells[0].isdigit():
            break  # end of the daily table
        dates.append(int(cells[0]))
        rets.append((float(cells[1]) + float(cells[4])) / 100.0)
    if not dates:
        raise ValueError(f"no daily table found in {path}")
    return np.array(dates), np.array(rets)


def market_log_returns(last_date: int = VINTAGE_LAST_DATE, path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Daily log-returns of the market up to `last_date` (the vintage of the chapter by default)."""
    d, r = parse_factors(path or DATA_DIR / "extracted" / CSV_NAME)
    keep = d <= last_date
    return d[keep], np.log1p(r[keep])
