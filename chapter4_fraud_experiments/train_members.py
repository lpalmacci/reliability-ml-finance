"""Step 3: the members of the two constructions, their calibrated pools and the validation diagnostics.

    python train_members.py                      # all partitions, final pools        (about 45 minutes)
    python train_members.py carclaims 0          # one partition
    python train_members.py --unconstrained      # the LightGBM members of S2 without the constraint
                                                 # on the Hessian of the leaves (needs the final pools)

The TabPFN members must already be in results/tabpfn/ (they are included). The script resumes where it
stopped: a member already saved is not trained again. For every partition it writes, in
results/members/<dataset>_p<k>/ (results/members_unconstrained_lightgbm/ with --unconstrained):

    <strategy>/<member>.npz|.json   raw scores and log-odds of each member        (not distributed, ~4 GB)
    <strategy>_pool.npz             the calibrated pool of 36 members             (not distributed)
    <strategy>_calibration.json     the calibration of each member (a, b, method)
    diagnostics.yaml                diagnostics of the full pool on the validation part only
    manifest.yaml                   package versions and git state

The unconstrained version rebuilds only the four LightGBM members of the second construction in each
partition and takes the other members from the final pools (Chapter 4, Decisions taken after the pilot).
"""
from __future__ import annotations

import argparse
import time

import confidex
from fraud.data import N_PARTITIONS
from fraud.manifest import manifest, write_yaml
from fraud.members import (assemble_pool, member_path, partition_dir, prepare, train_member,
                           validation_diagnostics, validation_thresholds)
from fraud.plan import plan

BORDER_RULE = 0.01      # if more than 1% of the validation cases lie at the border of the grid, use a wider grid


def main(dataset: str, p: int, version: str = "final") -> None:
    t0 = time.time()
    prep = prepare(dataset, p)
    t_prep = time.time() - t0
    print(f"{dataset} p{p} ({version}): preparation {t_prep:.1f}s, {prep.n_processed_variables} encoded variables",
          flush=True)
    timings = {"preparation_s": round(t_prep, 1)}
    diag = {}
    strategies = ("s2",) if version == "unconstrained" else ("s1", "s2")
    for strategy in strategies:
        t1 = time.time()
        members = plan(dataset, p, strategy)
        missing = [m.name for m in members if m.family == "TabPFN" and not member_path(m).exists()]
        if missing:
            raise SystemExit(f"TabPFN members missing: {missing}")
        seconds = {}
        for m in members:
            meta = train_member(m, prep, version)
            sec = meta.get("total_seconds", (meta.get("fit_seconds") or 0) + (meta.get("predict_seconds") or 0))
            seconds[m.family] = round(seconds.get(m.family, 0.0) + float(sec), 2)
            print(f"  {m.name:28s} n={meta['n_train']:>6} {sec:>7}s"
                  + (f"  warnings: {meta['warnings']}" if meta.get("warnings") else ""), flush=True)
        t_members = time.time() - t1
        pool = assemble_pool(dataset, p, strategy, prep, version)
        t2 = time.time()
        diag[strategy] = {**validation_diagnostics(pool), "validation_thresholds": validation_thresholds(pool, dataset)}
        timings[strategy] = {"members_s_this_run": round(t_members, 1), "members_s_by_family": seconds,
                             "members_s_total": round(sum(seconds.values()), 1),
                             "diagnostics_s": round(time.time() - t2, 1)}
        print(f"  {strategy}: {diag[strategy]}", flush=True)
    beyond = {s: d["border_share"] > BORDER_RULE for s, d in diag.items()}
    grid = ({"grid_bound": 1e-9, "grid_points": 1500} if any(beyond.values())
            else {"grid_bound": confidex.DEFAULT_MODEL.grid_bound, "grid_points": confidex.DEFAULT_MODEL.grid_points})
    out = partition_dir(dataset, p, version)
    write_yaml(out / "diagnostics.yaml", {"dataset": dataset, "partition": p, "version": version,
                                          "validation_diagnostics": diag, "border_share_beyond_1pc": beyond,
                                          "grid": grid, "timings": timings})
    write_yaml(out / "manifest.yaml", manifest(script="train_members.py", dataset=dataset, partition=p, version=version))
    print("grid:", grid, "timings:", timings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the members and calibrate the pools.")
    ap.add_argument("dataset", nargs="?", choices=("carclaims", "ulb"))
    ap.add_argument("partition", nargs="?", type=int)
    ap.add_argument("--unconstrained", action="store_true",
                    help="rebuild the LightGBM members of S2 without the constraint on the Hessian of the leaves")
    a = ap.parse_args()
    version = "unconstrained" if a.unconstrained else "final"
    jobs = ([(a.dataset, a.partition)] if a.partition is not None else
            [(d, p) for d in ((a.dataset,) if a.dataset else ("carclaims", "ulb")) for p in range(N_PARTITIONS[d])])
    for d, p in jobs:
        main(d, p, version)
