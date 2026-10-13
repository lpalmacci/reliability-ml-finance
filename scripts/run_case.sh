#!/bin/sh
# Case study: E4 (TC-VAE), E5 (comparison), E6 case part (L4 on held-out histories). Resumable.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.e4_case --config configs/full.yaml --jobs "${1:-4}" > results/logs/e4.log 2>&1 && echo done > results/logs/e4.done
uv run python -m tavp.experiments.e5_compare --config configs/full.yaml --jobs "${1:-4}" > results/logs/e5.log 2>&1 && echo done > results/logs/e5.done
uv run python -m tavp.experiments.e6_calibration --config configs/full.yaml --part case --jobs "${1:-4}" > results/logs/e6case.log 2>&1 && echo done > results/logs/e6case.done
