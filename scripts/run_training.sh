#!/bin/sh
# Train all neural generators (G1 TC-VAE: 4 processes x 2 regimes x 5 seeds; G2: 4 x 3 seeds).
# Usage: scripts/run_training.sh [jobs]; logs in results/logs/train.log.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.train --config configs/full.yaml --jobs "${1:-4}" > results/logs/train.log 2>&1
echo done > results/logs/train.done
