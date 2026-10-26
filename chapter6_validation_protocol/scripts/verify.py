"""Verify the numbers of the thesis.

    uv run python scripts/verify.py              # rebuild from results/ and compare with the frozen numbers
    uv run python scripts/verify.py --frozen     # rebuild from results/frozen/ (checks the snapshot itself)

1. Checks the SHA-256 of every file of results/frozen/ against its manifest.
2. Rebuilds the tables, figures and numbers of the chapter from the chosen results directory
   into a temporary directory.
3. Compares every number with results/frozen/numbers.json and reports the differences.
On the platform used for the thesis the comparison is exact; on other platforms, neural
training and floating-point reductions can differ slightly, so differences in the last digits
of some numbers are possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "frozen"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", action="store_true", help="rebuild from the frozen snapshot instead of results/")
    ap.add_argument("--config", default="configs/full.yaml")
    args = ap.parse_args()
    manifest = json.loads((FROZEN / "MANIFEST.json").read_text())
    bad = [f for f, h in manifest["files"].items() if hashlib.sha256((FROZEN / f).read_bytes()).hexdigest() != h]
    print(f"manifest: {len(manifest['files'])} files, {len(bad)} altered" + (f": {bad}" if bad else ""))
    root = FROZEN if args.frozen else ROOT / "results"
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "tavp.report.build", "--config", args.config, "--root", str(root), "--out", tmp]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        new = json.loads((Path(tmp) / "numbers.json").read_text())
    ref = json.loads((FROZEN / "numbers.json").read_text())
    missing = sorted(set(ref) - set(new))
    extra = sorted(set(new) - set(ref))
    diff = sorted(k for k in set(ref) & set(new) if ref[k] != new[k])
    print(f"numbers: {len(ref)} in the thesis, {len(diff)} different, {len(missing)} missing, {len(extra)} new")
    for k in diff[:50]:
        print(f"  {k}: thesis {ref[k]}  rebuilt {new[k]}")
    for k in missing[:20]:
        print(f"  missing: {k}")
    ok = not bad and not diff and not missing
    print("VERIFIED" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
