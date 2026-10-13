#!/bin/sh
# Wait for the running E1 (pid given as $1), then run E2 with one BLAS thread per worker.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run python -m tavp.experiments.e2_defects --config configs/full.yaml > results/logs/e2.log 2>&1
echo done > results/logs/e2.done
