#!/usr/bin/env bash
# Evaluate the no-search (transfer-only) pipeline vs no-preprocessing for all 23 eval datasets.
# Each dataset runs in its OWN process so a hard crash (e.g. 1164's Ray segfault) cannot kill the
# batch. Resumable: skips a dataset whose recommendation.json already exists.
set -u
cd "$(dirname "$0")/.."

OUT="${1:?usage: run_no_search_all.sh <output_dir> [time_limit] [profile]}"
TL="${2:-120}"
PROFILE="${3:-best_quality}"
IDS="248 1066 1164 1047 862 2 40663 1054 1387 876 18 1520 1548 184 378 381 382 993 1485 14 27 29 31"

mkdir -p "$OUT"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for ID in $IDS; do
  if [ -f "$OUT/dataset_${ID}/recommendation.json" ]; then
    echo "[skip] $ID (already done)"; continue
  fi
  echo "[run ] $ID  ($(date +%H:%M:%S))"
  .venv/bin/python scripts/run_recommend.py \
    --dataset-source openml --dataset-ids "$ID" \
    --baseline-only no_search_retrieval --baseline-fit-include-val \
    --train-metric-inline --metric-loss pearson --metric-weight-decay 1e-4 --metric-objective embedding_cosine \
    --require-autogluon --autogluon-profile "$PROFILE" --time-limit "$TL" --seed 42 \
    --output-dir "$OUT/dataset_${ID}" >> "$OUT/run_${ID}.log" 2>&1
  echo "[done] $ID exit=$?  ($(date +%H:%M:%S))"
done

echo "=== building report ==="
.venv/bin/python scripts/report_no_search.py --input-dir "$OUT" --out "$OUT/NO_SEARCH_REPORT.md" || true
echo "=== ALL DONE -> $OUT/NO_SEARCH_REPORT.md ==="
