"""Step 4: the evaluation of every condition, in parallel.

    python run_evaluation.py freeze                      # SHA-256 of the code before the run
    python run_evaluation.py run --jobs 10               # main evaluation: both constructions, final pools
    python run_evaluation.py freeze --unconstrained
    python run_evaluation.py run --unconstrained --jobs 10
                                                         # second construction on the pools with the
                                                         # unconstrained LightGBM members (comparison)

Conditions: carclaims partitions 0-4 and ULB partitions 0-9, two constructions, K in {3, 6, 12, 24, 36}, two
thresholds (in the same job). A job is (dataset, partition, construction, K): the choice of the simpler
reading on the validation part and the bootstrap are done for each K, so the jobs are independent; the
seeds depend only on the condition (``fraud.seeds``), so neither the order nor the number of processes
changes the numbers. The ablations are run on partitions 0-4 of each dataset.

Every job saves in results/<evaluation>/jobs/<job>/ its tables (parquet), the bootstrap means (npz) and a
marker done.json; a job already done is not run again. The descriptive performance of members and
ensembles is computed for every (dataset, partition) in results/<evaluation>/performance/. Errors and
timings go to results/<evaluation>/log.jsonl. <evaluation> is "evaluation" for the main run and
"evaluation_unconstrained_lightgbm" for the comparison.

On a 12-core laptop the main evaluation takes about 3 hours with 10 processes, the comparison about 1.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "run":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")            # parallelism comes from the processes, one thread each

import numpy as np
import pandas as pd

from fraud.data import N_PARTITIONS
from fraud.registry import RESULTS, ROOT

STRATEGIES = ("s1", "s2")
N_ABLATION_PARTITIONS = 5
EVALUATIONS = {"final": "evaluation", "unconstrained": "evaluation_unconstrained_lightgbm"}
CODE_FILES = sorted([*Path(ROOT / "fraud").glob("*.py"), ROOT / "run_evaluation.py", ROOT / "train_members.py",
                     ROOT / "query_tabpfn.py"])


def out_dir(version: str) -> Path:
    return RESULTS / EVALUATIONS[version]


def log(event: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"time": pd.Timestamp.now().isoformat(timespec="seconds"), **event}, default=str) + "\n")


def fingerprints() -> dict[str, str]:
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in CODE_FILES}


def freeze(version: str) -> None:
    from fraud.manifest import manifest, write_yaml
    d = out_dir(version)
    write_yaml(d / "frozen_code.yaml", manifest(script="run_evaluation.py", members=version, code_sha256=fingerprints()))
    log({"event": "code frozen"}, d / "log.jsonl")
    print("code fingerprints written to", d / "frozen_code.yaml")


def changed_since_freeze(version: str) -> list[str]:
    import yaml
    f = out_dir(version) / "frozen_code.yaml"
    if not f.exists():
        raise SystemExit(f"run first: python run_evaluation.py freeze{' --unconstrained' if version != 'final' else ''}")
    ref = yaml.safe_load(f.read_text())["code_sha256"]
    return [k for k, v in fingerprints().items() if ref.get(k) != v]


def conditions() -> list[tuple[str, int]]:
    return [(d, p) for d in ("carclaims", "ulb") for p in range(N_PARTITIONS[d])]


def job_list(version: str) -> list[tuple]:
    from fraud.evaluate import K_LEVELS
    strategies = ("s2",) if version == "unconstrained" else STRATEGIES
    out = [(d, p, s, K, p < N_ABLATION_PARTITIONS) for d, p in conditions() for s in strategies for K in K_LEVELS]
    # longest first: ULB, ablations, K < 36
    return sorted(out, key=lambda x: (x[0] != "ulb", not x[4], x[3] == 36, x[1], x[2], x[3]))


def job_name(d: str, p: int, s: str, K: int) -> str:
    return f"{d}_p{p}_{s}_K{K}"


def one_job(d: str, p: int, s: str, K: int, ablations: bool, version: str) -> str:
    from fraud import evaluate as ev
    out = out_dir(version) / "jobs" / job_name(d, p, s, K)
    if (out / "done.json").exists():
        return "already done"
    t0 = time.time()
    try:
        res = ev.evaluate(d, p, strategies=(s,), version=version, K_levels=(K,), ablations=ablations)
        tab = ev.tables(res)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tab.items():
            if not df.empty:
                df.to_parquet(out / f"{name}.parquet", index=False)
        keys = sorted(res.boot)
        np.savez_compressed(out / "bootstrap.npz", keys=np.array(["|".join(map(str, c)) for c in keys]),
                            means=np.vstack([res.boot[c] / res.n_draws[c] for c in keys]).astype(np.float32))
        info = {"dataset": d, "partition": p, "strategy": s, "K": K, "ablations": ablations, "members": version,
                "seconds": round(time.time() - t0, 1), "timings": res.timings}
        (out / "done.json").write_text(json.dumps(info, indent=2))
        log({"event": "job", **info}, out_dir(version) / "log.jsonl")
        return f"ok {time.time() - t0:.0f}s"
    except BaseException as exc:                                         # noqa: BLE001
        log({"event": "job error", "dataset": d, "partition": p, "strategy": s, "K": K,
             "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]},
            out_dir(version) / "log.jsonl")
        return f"ERROR {type(exc).__name__}: {exc}"


def one_performance(d: str, p: int, version: str) -> str:
    from fraud.performance import performance
    out = out_dir(version) / "performance" / f"{d}_p{p}"
    if (out / "done.json").exists():
        return "already done"
    try:
        strategies = ("s2",) if version == "unconstrained" else STRATEGIES
        pr = performance(d, p, strategies=strategies, version=version)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in pr.items():
            df.assign(dataset=d, partition=p).to_parquet(out / f"{name}.parquet", index=False)
        (out / "done.json").write_text("{}")
        return "ok"
    except BaseException as exc:                                         # noqa: BLE001
        log({"event": "performance error", "dataset": d, "partition": p, "error": f"{type(exc).__name__}: {exc}",
             "traceback": traceback.format_exc()[-3000:]}, out_dir(version) / "log.jsonl")
        return f"ERROR {exc}"


def run(version: str, n_jobs: int) -> None:
    from joblib import Parallel, delayed
    changed = changed_since_freeze(version)
    lg = out_dir(version) / "log.jsonl"
    if changed:
        log({"event": "code changed after freezing", "files": changed}, lg)
        print("WARNING: code changed after freezing:", changed, flush=True)
    jobs = job_list(version)
    log({"event": "evaluation started", "jobs": len(jobs), "processes": n_jobs}, lg)
    t0 = time.time()
    outcomes = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(delayed(one_job)(*x, version) for x in jobs)
    perf = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one_performance)(d, p, version) for d, p in conditions())
    errors = [(x, e) for x, e in zip(jobs, outcomes) if e.startswith("ERROR")]
    errors += [((d, p), e) for (d, p), e in zip(conditions(), perf) if e.startswith("ERROR")]
    log({"event": "evaluation finished", "seconds": round(time.time() - t0, 1), "errors": len(errors)}, lg)
    print(f"done: {len(jobs)} jobs, {len(errors)} errors, {time.time() - t0:.0f}s", flush=True)
    for x, e in errors:
        print(" ", x, e, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluation of Chapter 4.")
    ap.add_argument("command", choices=("freeze", "run"))
    ap.add_argument("--unconstrained", action="store_true", help="use the pools with the unconstrained LightGBM members")
    ap.add_argument("--jobs", type=int, default=10, help="parallel processes (default 10)")
    a = ap.parse_args()
    version = "unconstrained" if a.unconstrained else "final"
    if a.command == "freeze":
        freeze(version)
    else:
        run(version, a.jobs)
