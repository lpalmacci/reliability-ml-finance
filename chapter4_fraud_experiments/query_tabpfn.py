"""Step 2 (optional): the TabPFN members of the second construction, through the remote service.

The scores used in the thesis are included in results/tabpfn/ and are never recomputed by the other
scripts; run this only to query the service again. It needs an account, with the token in the
environment variable TABPFN_TOKEN or in a file .env in this folder (``TABPFN_TOKEN=...``), and the
service may return different outputs (see fraud/tabpfn_service.py). The script resumes where it
stopped: a member already saved is not queried again.

For every partition, the four TabPFN members (indices 8, 17, 26 and 35 of the rotation): the context is
the member's subsample of the training part, with the variables in their original form; one fit call and
one predict call on the calibration, validation and test parts concatenated (in-context inference is
independent row by row). The raw scores are saved; the calibration is done later with the other members.
Every call is logged in results/tabpfn/tabpfn_calls.jsonl: time, client version, requested model,
metadata declared by the server, tokens estimated and used, duration, errors.

    python query_tabpfn.py                 # both datasets
    python query_tabpfn.py carclaims       # one dataset
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from fraud import tabpfn_service as tw
from fraud.data import N_PARTITIONS, PARTS, load, load_partition
from fraud.manifest import manifest, write_yaml
from fraud.members import TABPFN_DIR
from fraud.plan import plan, subsample_indices

LOG = TABPFN_DIR / "tabpfn_calls.jsonl"
MAX_QUOTA = 0.95          # do not start a call that would take the use beyond 95% of the limit


def log(event: dict) -> None:
    TABPFN_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"time": pd.Timestamp.now().isoformat(timespec="seconds"), **event}, default=str) + "\n")


def path(dataset: str, p: int, name: str) -> Path:
    return TABPFN_DIR / f"{dataset}_p{p}" / f"{name}.npz"


def query_member(dataset: str, p: int, m, ld, parts) -> None:
    out = path(dataset, p, m.name)
    if out.exists():
        return
    iT = parts["train"]
    idx = subsample_indices(ld.y[iT], m.fraction, m.subsample_seed)
    X_fit = ld.X_raw.iloc[iT[idx]].reset_index(drop=True)
    y_fit = ld.y[iT[idx]]
    pred = ["calibration", "validation", "test"]
    X_pred = pd.concat([ld.X_raw.iloc[parts[k]] for k in pred]).reset_index(drop=True)
    n_part = [len(parts[k]) for k in pred]

    estimate = tw.estimate_cost(len(y_fit), len(X_pred), X_fit.shape[1])
    before = tw.usage()
    for lim, used in (("daily_token_limit", "daily_tokens_used"), ("monthly_token_limit", "monthly_tokens_used")):
        if before[used] + estimate["estimated"] > MAX_QUOTA * before[lim]:
            log({"event": "pause", "dataset": dataset, "partition": p, "member": m.name,
                 "reason": f"{used} {before[used]} + {estimate['estimated']} > {MAX_QUOTA} x {before[lim]}"})
            raise SystemExit(3)

    base = {"dataset": dataset, "partition": p, "member": m.name, "index": m.index, "key": m.key,
            "client": tw.client_version(), "model_path": tw.MODEL_PATH, "random_state": m.model_seed,
            "n_train": len(y_fit), "train_frauds": int(y_fit.sum()), "n_pred": len(X_pred),
            "n_col": X_fit.shape[1], "estimate": estimate}
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            clf = tw.build_tabpfn_classifier(seed=m.model_seed)
            clf.fit(X_fit, y_fit)
            t_fit = time.time() - t0
            after_fit = tw.usage()
            log({**base, "event": "fit", "attempt": attempt, "seconds": round(t_fit, 1),
                 "tokens": after_fit["daily_tokens_used"] - before["daily_tokens_used"]})
            t1 = time.time()
            s = tw.score(clf, X_pred)
            t_pred = time.time() - t1
            after = tw.usage()
            server_meta = getattr(clf, "_last_meta", {})
            log({**base, "event": "predict", "attempt": attempt, "seconds": round(t_pred, 1),
                 "tokens": after["daily_tokens_used"] - after_fit["daily_tokens_used"],
                 "member_tokens": after["daily_tokens_used"] - before["daily_tokens_used"],
                 "daily_use": after["daily_tokens_used"], "monthly_use": after["monthly_tokens_used"],
                 "n_estimators": (server_meta or {}).get("n_estimators"), "server": server_meta})
            break
        except SystemExit:
            raise
        except Exception as exc:                                    # noqa: BLE001
            log({**base, "event": "error", "attempt": attempt, "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()[-2000:]})
            if attempt == 3:
                raise
            time.sleep(60 * attempt ** 2)

    a, b = n_part[0], n_part[0] + n_part[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, raw_calibration=s[:a], raw_validation=s[a:b], raw_test=s[b:], subsample_indices=iT[idx])
    out.with_suffix(".json").write_text(json.dumps(
        {**m.to_dict(), "n_train": len(y_fit), "fit_seconds": round(t_fit, 1), "predict_seconds": round(t_pred, 1),
         "server": server_meta, "member_tokens": after["daily_tokens_used"] - before["daily_tokens_used"]},
        indent=2, default=str))


def main(argv: list[str]) -> int:
    datasets = argv[1:] or ["carclaims", "ulb"]
    write_yaml(TABPFN_DIR / "tabpfn_manifest.yaml",
               manifest(script="query_tabpfn.py", model_path=tw.MODEL_PATH, datasets=datasets,
                        initial_usage=tw.usage()))
    for d in datasets:
        ld = load(d)
        for p in range(N_PARTITIONS[d]):
            parts = load_partition(d, p)
            assert set(parts) == set(PARTS)
            for m in plan(d, p, "s2"):
                if m.family == "TabPFN":
                    query_member(d, p, m, ld, parts)
                    print(f"{d} p{p} {m.name} ok", flush=True)
    log({"event": "end", "datasets": datasets, "usage": tw.usage()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
