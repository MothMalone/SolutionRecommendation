#!/usr/bin/env bash
# KAGGLE FULL RUN (Path B) — the deliverable launched on Kaggle.
#
# Mirrors the notebook cell structure: env pinning, vendored deps via --target, PYTHONPATH wiring,
# sharding across sessions, a worker pool within a session, and tar of outputs. CPU-only.
#
# Usage in a Kaggle cell:
#   !bash /kaggle/input/datasets/mothmalone123/acorec/scripts/run_kaggle.sh "1/4" 2
# where "1/4" is this session's shard (i/n) and 2 is the worker count (~2 cores/session).
set -euo pipefail

SHARD="${1:-1/1}"     # i/n — split the 24 eval IDs across N notebook sessions
WORKERS="${2:-2}"     # concurrent datasets within this session

ROOT="/kaggle/input/datasets/mothmalone123/acorec"
DEPS="/kaggle/working/deps"
OUT="/kaggle/working/acorec_out"
mkdir -p "$DEPS" "$OUT"

# --- Env pinning (vendored into $DEPS via --target so the base image is untouched) ---
pip install --target="$DEPS" --no-warn-script-location \
  "numpy<2" "pandas<3" "autogluon.tabular==1.5.0" "openml==0.15.1" >/dev/null 2>&1 || true

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$DEPS:$ROOT/src:$ROOT:${PYTHONPATH:-}"

# All 24 evaluation IDs (the workhorse holds these out of the reference automatically).
EVAL_IDS="248 1066 1164 1047 862 2 40663 1054 1387 876 18 1520 1548 184 378 381 382 993 1485 14 27 29 31"

python3 "$ROOT/scripts/run_recommend.py" \
  --kaggle-root "$ROOT" \
  --dataset-source openml \
  --openml-local-folder "/kaggle/input/openml" \
  --dataset-ids $EVAL_IDS \
  --shard "$SHARD" \
  --workers "$WORKERS" \
  --paper-faithful \
  --use-aco --optimizer aco \
  --require-autogluon \
  --autogluon-profile best_quality \
  --seed 42 \
  --output-dir "$OUT" \
  --tar-outputs \
  --verbose

echo "Done. Tarball at ${OUT}.tar.gz — download from /kaggle/working."
