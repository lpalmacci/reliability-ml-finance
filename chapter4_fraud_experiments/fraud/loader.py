"""Loading of the datasets, with a blocking check of fingerprint and counts.

``load_dataset(name)`` returns ``LoadedDataset(X_raw, y, spec, meta)``:

- ``X_raw``: the variables without any encoding; missing categorical values become "MISSING", missing
  numerical values stay NaN and are imputed downstream on the training part only;
- ``y``: the label in {0, 1};
- ``meta``: fingerprint and counts.

The loader stops if the SHA-256 of the CSV, or the rows, frauds and variables after cleaning, differ
from those declared in ``fraud.registry``. The cleaning of carclaims removes one inconsistent row
(Age = 0 with day and month of the claim equal to '0') and treats Age = 0 as missing: the 320 rows with
Age = 0 are exactly those with AgeOfPolicyHolder "16 to 17", so the information is kept.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud.registry import DATASETS
from fraud.specs import DatasetSpec


class DataError(RuntimeError):
    """The dataset is not the one the protocol expects."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


#: the row removed from carclaims: Age = 0, day and month of the claim equal to '0'
CARCLAIMS_EXCLUDED_ROW = 1516


def _hook_carclaims(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning of carclaims."""
    r = df.loc[CARCLAIMS_EXCLUDED_ROW]
    if not (r["Age"] == 0 and str(r["DayOfWeekClaimed"]) == "0" and str(r["MonthClaimed"]) == "0"):
        raise DataError(f"row {CARCLAIMS_EXCLUDED_ROW} is not the expected one: {r.to_dict()}")
    df = df.drop(index=CARCLAIMS_EXCLUDED_ROW).copy()
    df["Age"] = df["Age"].replace(0, np.nan)
    for col in ("Age", "DriverRating", "Year", "WeekOfMonth", "WeekOfMonthClaimed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


_HOOKS = {"carclaims": _hook_carclaims}


@dataclass(frozen=True)
class LoadedDataset:
    X_raw: pd.DataFrame
    y: np.ndarray
    spec: DatasetSpec
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def n_samples(self) -> int:
        return len(self.y)


def _map_target(raw: pd.Series, positive_label: object) -> np.ndarray:
    """The target column as {0, 1}, using the declared positive label."""
    if raw.isna().any():
        raise DataError("target column contains NaN after cleaning.")
    if pd.api.types.is_numeric_dtype(raw):
        y = (raw.astype(int) == int(positive_label)).astype(np.int64).to_numpy()
    else:
        y = (raw.astype(str) == str(positive_label)).astype(np.int64).to_numpy()
    if set(np.unique(y).tolist()) != {0, 1}:
        raise DataError(f"target mapping produced unexpected values: {np.unique(y)}")
    return y


def load_dataset(name: str, path: Path | None = None) -> LoadedDataset:
    """Load a registered dataset and check its fingerprint, rows, frauds and variables."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    spec = DATASETS[name]
    path = Path(path or spec.path)
    if not path.exists():
        hint = " (decompress it first: gunzip -k data/ulb_creditcard.csv.gz)" if name == "ulb" else ""
        raise FileNotFoundError(f"[{name}] CSV not found at {path}{hint}")

    sha = sha256_file(path)
    if sha != spec.sha256:
        raise DataError(f"[{name}] SHA-256 {sha} differs from the expected {spec.sha256}")

    df = pd.read_csv(path, na_values=list(spec.na_tokens) or None)
    n_raw = len(df)
    if spec.loader_hook:
        df = _HOOKS[spec.loader_hook](df)
    df = df.drop(columns=list(spec.drop_cols))
    original_rows = df.index.to_numpy().copy()
    df = df.reset_index(drop=True)

    if df[spec.target].isna().any():
        raise DataError(f"[{name}] rows without a label")
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].fillna("MISSING")
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != object:
            df[col] = df[col].astype(object)

    y = _map_target(df[spec.target], spec.positive_label)
    X_raw = df.drop(columns=[spec.target])

    counts = {"rows": len(y), "frauds": int(y.sum()), "variables": X_raw.shape[1]}
    expected = {"rows": spec.n_rows, "frauds": spec.n_frauds, "variables": spec.n_variables}
    if counts != expected:
        raise DataError(f"[{name}] counts {counts} differ from the expected {expected}")

    meta = {
        "dataset": name, "file": path.name, "sha256": sha, "n_raw": int(n_raw),
        **{f"n_{k}": int(v) for k, v in counts.items()},
        "prevalence": float(y.mean()),
        "numerical_missing": int(X_raw.select_dtypes(exclude="object").isna().sum().sum()),
        "drop_cols": list(spec.drop_cols),
        "original_rows": original_rows,
    }
    return LoadedDataset(X_raw=X_raw, y=y, spec=spec, meta=meta)
