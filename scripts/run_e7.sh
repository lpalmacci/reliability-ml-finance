#!/bin/sh
# E7: robustness of the case-study verdicts and independent-null check of L1. Resumable.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.e7_robust --config configs/full.yaml --jobs "${1:-6}" > results/logs/e7.log 2>&1 && echo done > results/logs/e7.done
