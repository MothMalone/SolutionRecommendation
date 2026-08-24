#!/usr/bin/env bash
# Full AutoDP baseline over all 30 evaluation datasets (EVAL_IDS), fair protocol, end to end.
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
#   bash scripts/run_autodatapre_all.sh outputs/autodp 1800 300 "fair" "$(python3 -c 'import sys; sys.path.insert(0,"src"); from automl_aco.eval_ids import EVAL_IDS; print(" ".join(EVAL_IDS))')"
#
# Args: <output_dir> [cap_seconds] [autogluon_time_limit] [modes] [ids]
#   cap_seconds           wall-clock watchdog per dataset. AutoDP runs to ITS OWN convergence rule
#                         (its default, strongest setting); the cap only rescues a dataset whose
#                         convergence never fires, by retrying it with an explicit runTime budget.
#   autogluon_time_limit  passed to AutoGluon in stage 3. Use the SAME value your own runs used,
#                         or the comparison is not compute-matched.
#   modes                 default "fair" -- the only REPORTED protocol (docs/AUTODP_BASELINE.md).
#                         "native" is AutoDP's literal published API, deliberately unpatched and
#                         NOT a reported number for either method; pass it explicitly only for a
#                         disclosure column.
set -u
cd "$(dirname "$0")/.."

OUT="${1:?usage: run_autodatapre_all.sh <output_dir> [cap_seconds] [ag_time_limit] [modes] [ids]}"
CAP="${2:-1800}"
TL="${3:-300}"
MODES="${4:-fair}"
# The current 30-id EVAL_IDS (src/automl_aco/eval_ids.py). The old default here was the LEGACY
# 23-id set; results against it are not comparable to anything reported against the 30.
IDS="${5:-1066 1047 862 40663 1054 876 18 1520 1548 378 1485 14 27 44956 1037 42932 40668 1471 100000 42165 41001 41671 1046 46597 30 802 722 40922 1119 1497}"

AG_PROFILE="${AG_PROFILE:-best_quality}"   # match whatever your own runs used
MAIN_PY="${MAIN_PY:-.venv/bin/python}"
ADP_PY="${ADP_PY:-.venv-autodp/bin/python}"
DATA_DIR="${DATA_DIR:-data/eval_datasets}"

command -v "$MAIN_PY" >/dev/null 2>&1 || { echo "FATAL: no main python at '$MAIN_PY'. On Kaggle: MAIN_PY=python bash $0 ..."; exit 1; }
command -v "$ADP_PY"  >/dev/null 2>&1 || { echo "FATAL: no AutoDP python at '$ADP_PY' -- run: bash scripts/setup_autodp_env.sh"; exit 1; }

# PYTHONPATH is set PER STAGE, never exported globally. On Kaggle the main env typically gets
# AutoGluon via `pip install --target=/kaggle/working/acorec_deps` + PYTHONPATH, and PYTHONPATH
# takes precedence over a venv's own site-packages -- so exporting that path would drag numpy>=1.26
# and pandas 2.x into .venv-autodp and break AutoDP, which needs numpy<1.24 / pandas<2.0.
USER_PYTHONPATH="${PYTHONPATH:-}"
MAIN_PYTHONPATH="src${USER_PYTHONPATH:+:$USER_PYTHONPATH}"   # our package + whatever you provide
ADP_PYTHONPATH="src"                                          # our package only; deps come from the venv
unset PYTHONPATH

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

# Preflight: stage 3 needs AutoGluon in the MAIN env. Without this the batch happily burns hours
# preparing datasets it can never score, failing identically on every one of them.
if ! PYTHONPATH="$MAIN_PYTHONPATH" "$MAIN_PY" -c "import autogluon.tabular" >/dev/null 2>&1; then
  echo "FATAL: '$MAIN_PY' cannot import autogluon.tabular, so stage 3 would fail on every dataset."
  echo "       PYTHONPATH for main-env stages is: ${MAIN_PYTHONPATH}"
  echo "       If you install AutoGluon with --target, export that dir first, e.g.:"
  echo "         pip install --target=/kaggle/working/acorec_deps 'numpy<2' 'pandas<3' autogluon.tabular==1.5.0"
  echo "         PYTHONPATH=/kaggle/working/acorec_deps MAIN_PY=python bash $0 $*"
  PYTHONPATH="$MAIN_PYTHONPATH" "$MAIN_PY" -c "import autogluon.tabular" 2>&1 | tail -3
  exit 1
fi

mkdir -p "$OUT"

echo "=== stage 1: exporting datasets ==="
# Without internet the OpenML API is unreachable and the loader returns nothing. Point
# OPENML_LOCAL_FOLDER at a directory of <id>.csv files to use the loader's local fallback, e.g. on
# Kaggle: OPENML_LOCAL_FOLDER=/kaggle/input/datasets/mathurinache/openml
EXPORT_ARGS=""
if [ -n "${OPENML_LOCAL_FOLDER:-}" ]; then
  echo "  (local OpenML fallback: $OPENML_LOCAL_FOLDER)"
  EXPORT_ARGS="--openml-local-folder $OPENML_LOCAL_FOLDER"
fi
PYTHONPATH="$MAIN_PYTHONPATH" "$MAIN_PY" scripts/export_eval_datasets.py \
  --ids "$IDS" --out-dir "$DATA_DIR" --verbose $EXPORT_ARGS 2>&1 | tee -a "$OUT/export.log"

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
    if [ -f "$DDIR/autodp_failed.json" ] && [ "${RETRY_FAILED:-0}" != "1" ]; then
      echo "[skip] $ID ($MODE): timed out on an earlier run (RETRY_FAILED=1 to try again)"; continue
    fi

    if [ ! -f "$DDIR/autodp_meta.json" ]; then
      echo "[prep] $ID ($MODE)  ($(date +%H:%M:%S))"
      PYTHONPATH="$ADP_PYTHONPATH" "$ADP_PY" scripts/run_autodatapre.py \
        --dataset-csv "$CSV" --dataset-id "$ID" --mode "$MODE" \
        --cap-seconds "$CAP" --seed 42 --out-dir "$OUT" >> "$LOG" 2>&1
      if [ $? -ne 0 ]; then
        echo "[FAIL] $ID ($MODE): AutoDP stage failed, see $LOG"; continue
      fi
    fi

    echo "[eval] $ID ($MODE)  ($(date +%H:%M:%S))"
    PYTHONPATH="$MAIN_PYTHONPATH" "$MAIN_PY" scripts/eval_autodatapre.py \
      --dataset-csv "$CSV" --prepared-dir "$DDIR" \
      --time-limit "$TL" --autogluon-profile "$AG_PROFILE" --seed 42 >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "[FAIL] $ID ($MODE): AutoGluon stage failed, see $LOG"; continue
    fi
    tail -1 "$LOG"
  done
done

echo "=== stage 4: report ==="
PYTHONPATH="$MAIN_PYTHONPATH" "$MAIN_PY" scripts/report_autodatapre.py --input-dir "$OUT" || true
echo "=== ALL DONE -> $OUT/AUTODP_REPORT.md ==="
