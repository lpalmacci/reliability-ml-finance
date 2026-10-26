#!/usr/bin/env bash
# Full replication of Chapter 4: from the two datasets to the tables, the figures and the check of every number.
#
#   ./run_all.sh            # all the steps
#   ./run_all.sh 4          # from step 4 (the steps already done are kept: every step resumes where it stopped)
#   JOBS=4 ./run_all.sh     # number of parallel processes of the evaluation (default 10)
#   PYTHON=/path/to/python ./run_all.sh
#
# Step  Script                                     Output                                          Time (12-core laptop)
#  1    make_partitions.py                         results/partitions/                             seconds
#  2    train_members.py                           results/members/                                ~40 min
#  3    train_members.py --unconstrained           results/members_unconstrained_lightgbm/         a few minutes
#  4    run_evaluation.py (freeze, run)            results/evaluation/                             ~2 h 40 min with 10 processes
#  5    run_evaluation.py --unconstrained          results/evaluation_unconstrained_lightgbm/      ~1.5 h with 10 processes
#  6    summarise.py (both evaluations)            results/chapter4_results*.json, summary/        seconds
#  7    keff_checks.py                             results/keff_checks.parquet                     seconds
#  8    export_diagnostics.py                      results/chapter4_diagnostics.json               ~75 min with 10 processes
#  9    make_tables.py                             results/tables/                                 seconds
# 10    make_figures.py                            figures/                                        seconds
# 11    check_values.py                            (prints the checks; exit status 1 on failure)   seconds
#
# The TabPFN members are not recomputed: their scores are in results/tabpfn/ (query_tabpfn.py queries the
# remote service again, with an account; see README.md). Decompress ULB first: gunzip -k data/ulb_creditcard.csv.gz
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
JOBS=${JOBS:-10}
FIRST=${1:-1}

step() {
    local n=$1; shift
    if (( n >= FIRST )); then
        echo "=== step $n: $* ($(date '+%F %T'))"
        "$PY" -u "$@"
    fi
}

freeze_once() {    # fingerprints of the code, written once so that a resumed run keeps the first ones
    local dir=$1; shift
    [[ -f "$dir/frozen_code.yaml" ]] || "$PY" -u run_evaluation.py freeze "$@"
}

[[ -f data/ulb_creditcard.csv ]] || gunzip -k data/ulb_creditcard.csv.gz

step 1 make_partitions.py
step 2 train_members.py
step 3 train_members.py --unconstrained
if (( 4 >= FIRST )); then freeze_once results/evaluation; fi
step 4 run_evaluation.py run --jobs "$JOBS"
if (( 5 >= FIRST )); then freeze_once results/evaluation_unconstrained_lightgbm --unconstrained; fi
step 5 run_evaluation.py run --unconstrained --jobs "$JOBS"
step 6 summarise.py
step 6 summarise.py --unconstrained
step 7 keff_checks.py
step 8 export_diagnostics.py "$JOBS"
step 9 make_tables.py
step 10 make_figures.py
step 11 check_values.py
echo "=== done ($(date '+%F %T'))"
