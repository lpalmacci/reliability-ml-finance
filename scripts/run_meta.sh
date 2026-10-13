#!/bin/sh
# E1 then E2 (full pre-registered sizes). Logs in results/logs/.
set -e
# one BLAS thread per worker: parallelism comes from joblib processes
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
mkdir -p results/logs
uv run python -m tavp.experiments.e1_null --config configs/full.yaml > results/logs/e1.log 2>&1
uv run python -m tavp.experiments.e2_defects --config configs/full.yaml > results/logs/e2.log 2>&1
echo done > results/logs/meta.done
