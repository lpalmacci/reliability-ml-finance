"""Provenance of the results: package versions, git state, configuration (written as YAML)."""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fraud.registry import ROOT

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost",
            "tabpfn-client", "confidex", "joblib", "pyarrow")


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "missing"
    return out


def git_state() -> dict[str, Any]:
    def git(*a: str) -> str:
        try:
            r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return ""
        return r.stdout.strip() if r.returncode == 0 else ""
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def manifest(**extra: Any) -> dict[str, Any]:
    return {"created": pd.Timestamp.now().isoformat(timespec="seconds"), "versions": versions(),
            "git": git_state(), **extra}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
