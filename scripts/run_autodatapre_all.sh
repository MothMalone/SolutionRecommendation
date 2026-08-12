#!/usr/bin/env bash
# Full AutoDP baseline over all 23 evaluation datasets, both protocols, end to end.
#
#   stage 1  export_eval_datasets.py   main env    OpenML -> data/eval_datasets/<id>.csv
#   stage 2  run_autodatapre.py        .venv-autodp  AutoDP MCTS -> prepared.csv
#   stage 3  eval_autodatapre.py       main env    AutoGluon on our split -> autodp_eval.json
#   stage 4  report_autodatapre.py     main env    accuracy + runtime table
#
# Resumable: a dataset whose autodp_eval.json exists is skipped, so re-running after a Kaggle
# session times out picks up where it stopped. Every dataset runs in its own process, so one
# crash cannot take down the batch.
#
# First time:
#   bash scripts/setup_autodp_env.sh
#
# Then:
#   bash scripts/run_autodatapre_all.sh outputs/autodp
#   bash scripts/run_autodatapre_all.sh outputs/autodp 1800 300 "native fair"
#
# Args: <output_dir> [cap_seconds] [autogluon_time_limit] [modes] [ids]
#   cap_seconds           wall-clock watchdog per dataset. AutoDP runs to ITS OWN convergence rule
#                         (its default, strongest setting); the cap only rescues a dataset whose
#                         convergence never fires, by retrying it with an explicit runTime budget.
#   autogluon_time_limit  passed to AutoGluon in stage 3. Use the SAME value your own runs used,
#                         or the comparison is not compute-matched.
set -u
cd "$(dirname "$0")/.."

OUT="${1:?usage: run_autodatapre_all.sh <output_dir> [cap_seconds] [ag_time_limit] [modes] [ids]}"
CAP="${2:-1800}"
TL="${3:-300}"
MODES="${4:-native fair}"
IDS="${5:-248 1066 1164 1047 862 2 40663 1054 1387 876 18 1520 1548 184 378 381 382 993 1485 14 27 29 31}"

MAIN_PY="${MAIN_PY:-.venv/bin/python}"
ADP_PY="${ADP_PY:-.venv-autodp/bin/python}"
DATA_DIR="${DATA_DIR:-data/eval_datasets}"

command -v "$MAIN_PY" >/dev/null 2>&1 || { echo "FATAL: no main python at '$MAIN_PY'. On Kaggle: MAIN_PY=python bash $0 ..."; exit 1; }
command -v "$ADP_PY"  >/dev/null 2>&1 || { echo "FATAL: no AutoDP python at '$ADP_PY' -- run: bash scripts/setup_autodp_env.sh"; exit 1; }

mkdir -p "$OUT"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

echo "=== stage 1: exporting datasets ==="
# Without internet the OpenML API is unreachable and the loader returns nothing. Point
# OPENML_LOCAL_FOLDER at a directory of <id>.csv files to use the loader's local fallback, e.g. on
# Kaggle: OPENML_LOCAL_FOLDER=/kaggle/input/datasets/mathurinache/openml
EXPORT_ARGS=""
if [ -n "${OPENML_LOCAL_FOLDER:-}" ]; then
  echo "  (local OpenML fallback: $OPENML_LOCAL_FOLDER)"
  EXPORT_ARGS="--openml-local-folder $OPENML_LOCAL_FOLDER"
fi
"$MAIN_PY" scripts/export_eval_datasets.py --ids "$IDS" --out-dir "$DATA_DIR" --verbose \
  $EXPORT_ARGS 2>&1 | tee -a "$OUT/export.log"

for MODE in $MODES; do
  for ID in $IDS; do
    CSV="$DATA_DIR/$ID.csv"
    DDIR="$OUT/$MODE/dataset_$ID"
    LOG="$OUT/${MODE}_${ID}.log"
    if [ ! -f "$CSV" ]; then
      echo "[skip] $ID ($MODE): no export at $CSV"; continue
    fi
    if [ -f "$DDIR/autodp_eval.json" ]; then
      echo "[skip] $ID ($MODE): already scored"; continue
    fi

    if [ ! -f "$DDIR/autodp_meta.json" ]; then
      echo "[prep] $ID ($MODE)  ($(date +%H:%M:%S))"
      "$ADP_PY" scripts/run_autodatapre.py \
        --dataset-csv "$CSV" --dataset-id "$ID" --mode "$MODE" \
        --cap-seconds "$CAP" --seed 42 --out-dir "$OUT" >> "$LOG" 2>&1
      if [ $? -ne 0 ]; then
        echo "[FAIL] $ID ($MODE): AutoDP stage failed, see $LOG"; continue
      fi
    fi

    echo "[eval] $ID ($MODE)  ($(date +%H:%M:%S))"
    "$MAIN_PY" scripts/eval_autodatapre.py \
      --dataset-csv "$CSV" --prepared-dir "$DDIR" \
      --time-limit "$TL" --autogluon-profile best_quality --seed 42 >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "[FAIL] $ID ($MODE): AutoGluon stage failed, see $LOG"; continue
    fi
    tail -1 "$LOG"
  done
done

echo "=== stage 4: report ==="
"$MAIN_PY" scripts/report_autodatapre.py --input-dir "$OUT" || true
echo "=== ALL DONE -> $OUT/AUTODP_REPORT.md ==="
