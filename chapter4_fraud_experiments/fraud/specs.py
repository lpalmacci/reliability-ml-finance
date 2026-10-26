"""Dataclasses describing the datasets of the experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OrdinalSpec:
    """An ordered categorical variable and the order of its categories."""
    col: str
    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.col:
            raise ValueError("OrdinalSpec.col must be non-empty.")
        if len(self.categories) < 2:
            raise ValueError(f"OrdinalSpec '{self.col}' needs >= 2 categories; got {self.categories}.")


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a binary-classification CSV and of what the loader must find in it.

    name           : key of the dataset in the registry
    path           : path of the CSV
    target         : name of the target column
    positive_label : value of the target that maps to class 1
    drop_cols      : columns removed before training (identifiers, time stamps)
    na_tokens      : extra string tokens to treat as missing
    ordinal_specs  : ordered categorical variables
    loader_hook    : name of a dataset-specific cleaning step, if any
    sha256         : expected SHA-256 of the CSV; the loader stops if it differs
    n_rows, n_frauds, n_variables : expected counts after cleaning; the loader stops if they differ
    """

    name: str
    path: Path
    target: str
    positive_label: object = 1
    drop_cols: tuple[str, ...] = ()
    na_tokens: tuple[str, ...] = ()
    ordinal_specs: tuple[OrdinalSpec, ...] = ()
    loader_hook: Optional[str] = None
    sha256: str = ""
    n_rows: int = 0
    n_frauds: int = 0
    n_variables: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))
