"""Download the official Tail-GAN code (Cont, Cucuringu, Xu and Zhang, 2025) at a pinned commit.

Tail-GAN is distributed under the GNU GPL-3.0, which is not compatible with including its code in this
Apache-2.0 repository. It is therefore an optional external dependency: this script downloads it from
GitHub into third_party_ext/tailgan/ (not tracked by git) and checks that the extracted files are exactly
those of the pinned commit. Nothing else in the repository needs it.

    python scripts/get_tailgan.py
"""

from __future__ import annotations

import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = "chaozhang-ox/Tail-GAN"
COMMIT = "bfbe9501a593d7797b2cfb80de1f20c78d99cda3"  # 2025-10-15
TREE_SHA256 = "d3b13f6fc6e08dedcceddaf8243ab9fa12224751bd5e68a30ee80d0f89c3bc14"  # 18 files
DEST = Path(__file__).resolve().parents[1] / "third_party_ext" / "tailgan"


def tree_sha256(root: Path) -> str:
    items = sorted((str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in root.rglob("*") if p.is_file())
    return hashlib.sha256("\n".join(f"{a} {b}" for a, b in items).encode()).hexdigest()


def main() -> None:
    if DEST.exists() and tree_sha256(DEST) == TREE_SHA256:
        print(f"{DEST} already holds commit {COMMIT[:12]}")
        return
    url = f"https://github.com/{REPO}/archive/{COMMIT}.zip"
    data = urllib.request.urlopen(url).read()
    tmp = DEST.parent / f"_tmp_{COMMIT[:12]}"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp)
    src = next(tmp.iterdir())
    digest = tree_sha256(src)
    if digest != TREE_SHA256:
        shutil.rmtree(tmp)
        raise SystemExit(f"content of {url} differs from the pinned commit (tree sha256 {digest})")
    shutil.rmtree(DEST, ignore_errors=True)
    src.rename(DEST)
    shutil.rmtree(tmp)
    print(f"Tail-GAN {COMMIT[:12]} (GPL-3.0) extracted to {DEST}")


if __name__ == "__main__":
    main()
