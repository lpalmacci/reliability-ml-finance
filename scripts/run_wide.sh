#!/bin/sh
# Exploratory X1 (deviation D2): train the wide non-causal ablation, then validate it.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.train --config configs/exploratory.yaml --jobs "${1:-2}" > results/logs/train_wide.log 2>&1 && echo done > results/logs/train_wide.done
uv run python -m tavp.experiments.x1_wide --config configs/exploratory.yaml --jobs "${1:-2}" > results/logs/x1.log 2>&1 && echo done > results/logs/x1.done
