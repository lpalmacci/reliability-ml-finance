#!/bin/sh
# E1 (rerun with N = 100,000 for L2), then E2 and E6 (meta part). Logs in results/logs/.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.e1_null --config configs/full.yaml > results/logs/e1.log 2>&1 && echo done > results/logs/e1.done
uv run python -m tavp.experiments.e2_defects --config configs/full.yaml > results/logs/e2.log 2>&1 && echo done > results/logs/e2.done
uv run python -m tavp.experiments.e6_calibration --config configs/full.yaml --part meta > results/logs/e6meta.log 2>&1 && echo done > results/logs/e6meta.done
