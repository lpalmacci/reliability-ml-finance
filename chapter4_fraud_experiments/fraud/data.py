"""Data and partitions (Chapter 4, Data and partitions).

- ``load(dataset)``: the cleaned dataset, with fingerprint and counts checked (``fraud.loader``).
- ``groups(X)``: rows with the same vector of variables (after cleaning) form a group.
- ``partition(y, g, rng)``: four parts in the proportions 55/10/15/20, stratified by class and by
  group: a whole group ends up in a single part.
- ``make_partitions(dataset)``: the partitions of the dataset (five for carclaims, ten for ULB), each
  from the generator ``rng_for("partition", dataset, p)``, saved in results/partitions/<dataset>_p<k>.npz
  with their SHA-256 in the manifest.

Stratification with groups. Each group has a stratum according to the labels of its rows: all
legitimate, all frauds, or mixed (groups with discordant labels, which are counted and reported). Within
each stratum the groups are permuted at random and assigned to the parts according to the position of
their midpoint in the cumulative sum of the rows, so that every stratum is split in the proportions
55/10/15/20 up to the size of one group.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fraud.loader import LoadedDataset, load_dataset, sha256_file
from fraud.registry import RESULTS
from fraud.seeds import key, rng_for

PARTS: tuple[str, ...] = ("train", "calibration", "validation", "test")
PROPORTIONS: tuple[float, ...] = (0.55, 0.10, 0.15, 0.20)
#: partitions per dataset: ten for ULB in the main analysis (the ablations use partitions 0-4)
N_PARTITIONS = {"carclaims": 5, "ulb": 10}
PARTITION_DIR = RESULTS / "partitions"


def load(dataset: str) -> LoadedDataset:
    return load_dataset(dataset)


def groups(X: pd.DataFrame) -> np.ndarray:
    """Identifier of the group of identical rows (same vector of variables, missing values included)."""
    return X.groupby(list(X.columns), dropna=False, sort=True).ngroup().to_numpy(np.int64)


def group_statistics(y: np.ndarray, g: np.ndarray) -> dict[str, int]:
    """Groups with more than one row, rows involved, groups with discordant labels."""
    G = int(g.max()) + 1
    n = np.bincount(g, minlength=G)
    pos = np.bincount(g, weights=y, minlength=G)
    multi = n > 1
    discordant = (pos > 0) & (pos < n)
    return {
        "groups": G,
        "groups_with_several_rows": int(multi.sum()),
        "rows_in_groups_with_several_rows": int(n[multi].sum()),
        "excess_rows": int((n[multi] - 1).sum()),
        "discordant_groups": int(discordant.sum()),
        "rows_in_discordant_groups": int(n[discordant].sum()),
        "frauds_in_discordant_groups": int(pos[discordant].sum()),
    }


def partition(y: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Four parts stratified by class and by group (sorted row indices)."""
    y = np.asarray(y).astype(np.int64)
    g = np.asarray(g).astype(np.int64)
    G = int(g.max()) + 1
    n = np.bincount(g, minlength=G)
    pos = np.bincount(g, weights=y, minlength=G)
    stratum = np.where(pos == 0, 0, np.where(pos == n, 1, 2))
    bounds = np.cumsum(PROPORTIONS)
    bounds[-1] = np.inf                       # no group beyond the last part
    part_of_group = np.full(G, -1, dtype=np.int64)
    for s in (0, 1, 2):
        idx = np.flatnonzero(stratum == s)
        if idx.size == 0:
            continue
        perm = rng.permutation(idx)
        size = n[perm]
        midpoint = (np.cumsum(size) - size / 2.0) / size.sum()
        part_of_group[perm] = np.searchsorted(bounds, midpoint, side="right")
    part = part_of_group[g]
    return {name: np.flatnonzero(part == j) for j, name in enumerate(PARTS)}


def _sha_indices(parts: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in PARTS:
        h.update(name.encode())
        h.update(np.ascontiguousarray(parts[name], dtype=np.int64).tobytes())
    return h.hexdigest()


def partition_path(dataset: str, p: int) -> Path:
    return PARTITION_DIR / f"{dataset}_p{p}.npz"


def make_partitions(dataset: str, n_partitions: int | None = None, *, write: bool = True) -> dict[str, Any]:
    """Generate the partitions of the dataset, save them and return the manifest.

    Each partition depends only on its key: adding partitions 5-9 of ULB leaves partitions 0-4
    unchanged (checked through the SHA-256)."""
    n_partitions = N_PARTITIONS[dataset] if n_partitions is None else n_partitions
    ld = load(dataset)
    g = groups(ld.X_raw)
    out: dict[str, Any] = {
        "dataset": dataset, "file": ld.meta["file"], "sha256_dataset": ld.meta["sha256"],
        "rows": ld.meta["n_rows"], "frauds": ld.meta["n_frauds"],
        "variables": ld.meta["n_variables"], "proportions": dict(zip(PARTS, PROPORTIONS)),
        "groups": group_statistics(ld.y, g), "partitions": {},
    }
    for p in range(n_partitions):
        parts = partition(ld.y, g, rng_for("partition", dataset, p))
        entry = {
            "key": key("partition", dataset, p),
            "rows": {k: int(len(v)) for k, v in parts.items()},
            "frauds": {k: int(ld.y[v].sum()) for k, v in parts.items()},
            "sha256_indices": _sha_indices(parts),
        }
        if write:
            PARTITION_DIR.mkdir(parents=True, exist_ok=True)
            f = partition_path(dataset, p)
            buf = io.BytesIO()
            np.savez(buf, **parts)
            f.write_bytes(buf.getvalue())
            entry["file"] = f.name
            entry["sha256_file"] = sha256_file(f)
        out["partitions"][p] = entry
    if write:
        (PARTITION_DIR / f"{dataset}_manifest.yaml").write_text(
            yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    return out


def load_partition(dataset: str, p: int) -> dict[str, np.ndarray]:
    """Read a saved partition and check its fingerprint against the manifest."""
    man = yaml.safe_load((PARTITION_DIR / f"{dataset}_manifest.yaml").read_text())
    entry = man["partitions"][p]
    f = partition_path(dataset, p)
    if sha256_file(f) != entry["sha256_file"]:
        raise RuntimeError(f"{f.name}: SHA-256 differs from the one in the manifest")
    with np.load(f) as z:
        parts = {k: z[k] for k in PARTS}
    if _sha_indices(parts) != entry["sha256_indices"]:
        raise RuntimeError(f"{f.name}: indices differ from those in the manifest")
    return parts
