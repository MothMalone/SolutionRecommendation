#!/usr/bin/env bash
# LOCAL TEST MODE (Path A) — pre-flight smoke, NOT final numbers.
#
# Serial, CPU, small search budget, a handful of local datasets. Confirms the pipeline runs
# end-to-end, the leakage holdout fires, and operator distributions are non-degenerate BEFORE
# spending Kaggle quota. AutoGluon is optional locally: if absent, the proxy evaluator is used
# (so the accuracies here are proxy, not the AutoGluon numbers reported in the paper).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"

# A few local eval IDs (present in test_data_local/). These are TEST datasets — the run holds
# all 24 eval IDs out of the reference (leakage holdout) and excludes the query at retrieval.
IDS="${1:-2 18 1047}"

python3 scripts/run_recommend.py \
  --dataset-source local \
  --dataset-ids $IDS \
  --paper-faithful \
  --local-test-mode \
  --use-aco --optimizer aco \
  --allow-autogluon-fallback \
  --output-dir outputs/local_smoke \
  --verbose

echo
echo "Local smoke complete. Inspect outputs/local_smoke/dataset_*/recommendation.json"
echo "Check: leakage_holdout populated, chosen operators vary across steps (non-degenerate)."
