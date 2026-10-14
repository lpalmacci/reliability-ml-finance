"""Download the daily Fama-French factors (Ken French Data Library) used by the market-data analysis (D8).

The library updates the file every month, so a new download may contain more recent days and
revised values. The chapter uses the CRSP 202608 vintage; the analysis truncates the series at
its last date, and this script reports whether the downloaded file is that vintage.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile

from tavp.market import CSV_NAME, DATA_DIR, URL, VINTAGE_SHA256, ZIP_NAME, parse_factors


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    z = DATA_DIR / ZIP_NAME
    if not z.exists():
        urllib.request.urlretrieve(URL, z)
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    with zipfile.ZipFile(z) as f:
        f.extract(CSV_NAME, DATA_DIR / "extracted")
    d, _ = parse_factors(DATA_DIR / "extracted" / CSV_NAME)
    same = digest == VINTAGE_SHA256
    print(f"{z.name}: sha256 {digest} ({'vintage of the chapter' if same else 'DIFFERENT vintage'}); "
          f"{len(d)} days, {d[0]} to {d[-1]}")
    if not same:
        print("The series is truncated at the last date of the chapter's vintage; revised values may make "
              "the market-data numbers differ slightly from the frozen ones.")


if __name__ == "__main__":
    main()
