#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/ctxpipe}"
RUN_SUFFIX="${RUN_SUFFIX:-solrec-r32000}"
RUN_NAME="${RUN_NAME:-ctxpipe-3linear-${RUN_SUFFIX}}"
CKPT="${CKPT:-32000}"

DATASET_PREFIX="${DATASET_PREFIX:-${ROOT}/data/openml_solrec}"
DATASET_IDS_FILE="${DATASET_IDS_FILE:-${ROOT}/data/meta/solrec_openml_ids.txt}"

AIPIPE_PREFIX="${AIPIPE_PREFIX:-${ROOT}/exp/solrec_openml/solrec_space_all/aipipe}"
RESULT_PREFIX="${RESULT_PREFIX:-${ROOT}/exp/solrec_openml/solrec_space_all/result}"
AG_OUT_DIR="${AG_OUT_DIR:-${ROOT}/exp/solrec_openml/solrec_space_all/autogluon_eval}"

AG_SELECTION="${AG_SELECTION:-best_ctx_reward}"
AG_TIME_LIMIT="${AG_TIME_LIMIT:-300}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"
SPLIT_SEED="${SPLIT_SEED:-42}"

INFER_MAX_RETRY="${INFER_MAX_RETRY:-24}"
CLEAR_PIPELINES="${CLEAR_PIPELINES:-1}"

echo "[1/5] Validate paths"
test -d "${ROOT}"
test -d "${DATASET_PREFIX}"
test -f "${DATASET_IDS_FILE}"

mkdir -p "${AIPIPE_PREFIX}" "${RESULT_PREFIX}" "${AG_OUT_DIR}" "${ROOT}/logs/${RUN_NAME}"

PIPELINES_TSV="${ROOT}/exp/${RUN_NAME}/pipelines.tsv"
if [[ "${CLEAR_PIPELINES}" == "1" ]]; then
  echo "[2/5] Clear old pipelines.tsv"
  rm -f "${PIPELINES_TSV}"
fi

echo "[3/5] Inference (CKPT=${CKPT})"
CTXPIPE_OPERATOR_SPACE=solrec \
CTXPIPE_RUN_SUFFIX="${RUN_SUFFIX}" \
CTXPIPE_SPLIT_MODE=solrec \
CTXPIPE_INFER_MAX_RETRY="${INFER_MAX_RETRY}" \
CTXPIPE_EVAL_DATASET_PREFIX="${DATASET_PREFIX}" \
CTXPIPE_EVAL_AIPIPE_PREFIX="${AIPIPE_PREFIX}" \
CTXPIPE_EVAL_RESULT_PREFIX="${RESULT_PREFIX}" \
python "${ROOT}/test.py" "${CKPT}" "${CKPT}" 2>&1 | tee "${ROOT}/logs/${RUN_NAME}/infer_${CKPT}.log"

test -f "${PIPELINES_TSV}"
cp "${PIPELINES_TSV}" "${ROOT}/exp/${RUN_NAME}/pipelines_ckpt_${CKPT}.tsv"

echo "[4/5] AutoGluon re-eval"
python "${ROOT}/scripts/evaluate_ctxpipe_pipelines_autogluon.py" \
  --pipelines-tsv "${PIPELINES_TSV}" \
  --dataset-prefix "${DATASET_PREFIX}" \
  --dataset-ids-file "${DATASET_IDS_FILE}" \
  --selection "${AG_SELECTION}" \
  --time-limit-per-model "${AG_TIME_LIMIT}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --split-seed "${SPLIT_SEED}" \
  --output-csv "${AG_OUT_DIR}/results_ckpt_${CKPT}.csv" \
  --output-json "${AG_OUT_DIR}/results_ckpt_${CKPT}.json" \
  --verbose 2>&1 | tee "${ROOT}/logs/${RUN_NAME}/ag_eval_${CKPT}.log"

echo "[5/5] Failure analysis artifacts"
python "${ROOT}/scripts/analyze_ctxpipe_inference.py" \
  --pipelines-tsv "${PIPELINES_TSV}" \
  --ag-results-csv "${AG_OUT_DIR}/results_ckpt_${CKPT}.csv" \
  --output-dir "${AG_OUT_DIR}/analysis_ckpt_${CKPT}" \
  --title "CtxPipe SoluRec ckpt=${CKPT}" 2>&1 | tee "${ROOT}/logs/${RUN_NAME}/analysis_${CKPT}.log"

echo
echo "Done."
echo "Pipelines: ${PIPELINES_TSV}"
echo "AG CSV:    ${AG_OUT_DIR}/results_ckpt_${CKPT}.csv"
echo "Analysis:  ${AG_OUT_DIR}/analysis_ckpt_${CKPT}"
