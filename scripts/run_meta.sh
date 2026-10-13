#!/bin/sh
# Meta-validation: E2 (planted defects) then E6 meta part (L4 in pseudo-real mode).
# Resumable: completed tasks are cached under results/<experiment>/tasks/.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.e2_defects --config configs/full.yaml --jobs "${1:-4}" > results/logs/e2.log 2>&1 && echo done > results/logs/e2.done
uv run python -m tavp.experiments.e6_calibration --config configs/full.yaml --part meta --jobs "${1:-4}" > results/logs/e6meta.log 2>&1 && echo done > results/logs/e6meta.done
