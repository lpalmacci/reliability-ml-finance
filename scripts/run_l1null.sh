#!/bin/sh
# Deviation D3: extend the L1 null to B pairs and recompute the L1 tables of every experiment.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
uv run python -m tavp.experiments.l1_null --config configs/full.yaml --jobs "${1:-6}" > results/logs/l1null.log 2>&1 && echo done > results/logs/l1null.done
