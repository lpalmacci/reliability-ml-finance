"""Freeze the results used in the thesis: copy the final tables to results/frozen/ with a manifest.

    uv run python scripts/freeze.py

The frozen snapshot contains every table read by `tavp.report.build` (not the task caches or
the trained models), the numbers of the chapter and a manifest with the SHA-256 of each file.
`scripts/verify.py` rebuilds the numbers from a results directory and compares them with it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FROZEN = ROOT / "results", ROOT / "results" / "frozen"
EXPERIMENTS = ["e1", "e1_n40k", "e2", "e4", "e5", "e6_meta", "e6_case", "e6_size", "e7_l1null", "e7_cost", "e7_obj_low",
               "e7_obj_high", "e7_n_alt", "e8", "e9", "e10ms", "e12", "e4_curves", "q0", "x1_wide", "train",
               "real/e1", "real/e4", "real/e5", "real/e8", "real/e10", "real/e10ms", "real/e11", "real/train"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if FROZEN.exists():
        shutil.rmtree(FROZEN)
    files = []
    for exp in EXPERIMENTS:
        src = RESULTS / exp
        if not src.exists():
            print(f"missing: {exp}")
            continue
        for f in sorted(src.glob("*")):
            if f.is_file() and f.suffix in (".parquet", ".json"):
                dst = FROZEN / exp / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                files.append(dst)
    for f in ("numbers.json", "numbers.tex"):
        shutil.copy2(ROOT / "thesis_assets" / f, FROZEN / f)
        files.append(FROZEN / f)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"commit": commit, "files": {str(f.relative_to(FROZEN)): sha256(f) for f in files}}
    (FROZEN / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"frozen {len(files)} files to {FROZEN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
