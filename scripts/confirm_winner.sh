#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
PANEL="11 23 12 43 53 59 36"
COMMON="--dataset-source openml --dataset-ids $PANEL --require-autogluon --autogluon-profile best_quality --seed 42"
MP="outputs/val_metric.pt"
echo "### BQ baseline"
rm -rf outputs/bq_baseline
python3 scripts/run_recommend.py $COMMON --baseline-only no_preprocessing --output-dir outputs/bq_baseline >outputs/bq_baseline.log 2>&1
echo "### BQ winner C (topk5)"
rm -rf outputs/bq_C
python3 scripts/run_recommend.py $COMMON --metric-path "$MP" --n-ants 6 --n-iterations 4 \
  --aco-mmas-bounds --aco-weight-method linear --hybrid-select --final-autogluon-topk 5 \
  --use-aco --optimizer aco --output-dir outputs/bq_C >outputs/bq_C.log 2>&1
echo "### BQ CONFIRM COMPLETE"
