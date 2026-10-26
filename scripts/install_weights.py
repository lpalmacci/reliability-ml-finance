"""Install the trained Quant GANs of the thesis into results/, so that `make qgan` uses them instead of retraining.

    uv run python scripts/install_weights.py              # check the SHA-256 of every file, then copy
    uv run python scripts/install_weights.py --manifest   # rewrite weights/MANIFEST.json (maintainer only)

Training the Quant GAN on the Apple GPU (MPS) is not reproducible bit for bit, even on the same machine (see
REPRODUCE.md, Determinism), so its numbers can be regenerated exactly only from the trained models. weights/ holds,
for each of the 97 models of the chapter (Q0, Q0b, E13, E13 on the processes calibrated on the market, E11 on the
market), the selected generator, its metadata and its training log, at the paths they have under results/. The
training scripts skip a model whose meta.json exists, and sampling from a trained model is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS, RESULTS = ROOT / "weights", ROOT / "results"
MANIFEST = WEIGHTS / "MANIFEST.json"
FILES = ("generator.pt", "meta.json", "training_log.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    files = sorted(p for p in WEIGHTS.rglob("*") if p.is_file() and p.name in FILES)
    manifest = {"files": {str(p.relative_to(WEIGHTS)): sha256(p) for p in files}}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {len(files)} files, {len(files) // len(FILES)} models")


def install() -> int:
    manifest = json.loads(MANIFEST.read_text())["files"]
    bad = [f for f, h in manifest.items() if not (WEIGHTS / f).exists() or sha256(WEIGHTS / f) != h]
    if bad:
        print(f"weights: {len(bad)} files missing or altered: {bad[:5]}")
        return 1
    models = sorted({str(Path(f).parent) for f in manifest})
    copied = same = 0
    conflicts = []
    for m in models:
        files = [f for f in manifest if str(Path(f).parent) == m]
        present = [f for f in files if (RESULTS / f).exists()]
        if any(sha256(RESULTS / f) != manifest[f] for f in present):
            conflicts.append(m)  # a model trained here: leave it whole rather than mix files of two runs
            continue
        for f in files:
            if f in present:
                same += 1
                continue
            (RESULTS / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WEIGHTS / f, RESULTS / f)
            copied += 1
    print(f"weights: {len(models)} models, {len(manifest)} files checked, {copied} copied, {same} already in place")
    if conflicts:
        print(f"{len(conflicts)} models in results/ differ from those of the thesis (retrained here?) and were left "
              f"unchanged, for example results/{conflicts[0]}; remove them to install the weights of the thesis")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="rewrite weights/MANIFEST.json")
    args = ap.parse_args()
    if args.manifest:
        write_manifest()
        return 0
    return install()


if __name__ == "__main__":
    sys.exit(main())
