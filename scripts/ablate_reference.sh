#!/usr/bin/env bash
# Off-test lever ablation on a fixed REFERENCE panel (never the 24 eval IDs).
# Trains the metric once, reuses it; each config changes ONE lever. Reports mean AutoGluon accuracy.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"

PANEL="11 23 12 43 53 59 36"
AGP="local_rf_xt"   # medium_quality natively crashes on this dev install; local_rf_xt is fast+robust
COMMON="--dataset-source openml --dataset-ids $PANEL --require-autogluon --autogluon-profile $AGP --seed 42"
BUD="--n-ants 6 --n-iterations 4"
METRIC="--metric-loss pearson --metric-weight-decay 1e-4 --metric-objective embedding_cosine"
MP="outputs/val_metric.pt"
mkdir -p outputs

run() { # name, extra-flags...
  local name="$1"; shift
  rm -rf "outputs/val_$name"
  python3 scripts/run_recommend.py $COMMON "$@" --output-dir "outputs/val_$name" >"outputs/val_$name.log" 2>&1
  python3 - "$name" "outputs/val_$name" <<'PY'
import json,sys,glob,os
name=sys.argv[1]; d=sys.argv[2]
scores={}
for p in glob.glob(os.path.join(d,"**","recommendation.json"),recursive=True)+glob.glob(os.path.join(d,"recommendation.json")):
    try:
        r=json.load(open(p)); did=str(r.get("dataset_id")); fe=r.get("final_evaluation",{})
        sc=fe.get("score") if fe.get("score") is not None else fe.get("test_accuracy")
        if sc is not None: scores[did]=float(sc)
    except Exception: pass
if scores:
    import statistics
    print(f"RESULT {name}: mean={statistics.mean(scores.values()):.4f}  n={len(scores)}  per={ {k:round(v,3) for k,v in sorted(scores.items())} }")
else:
    print(f"RESULT {name}: NO-RESULTS (all datasets failed)")
PY
}

echo "### baseline (no preprocessing)"
run baseline --baseline-only no_preprocessing

echo "### A: base method (Siamese-pearson, hybrid-select, no prior, topk1, margin0) [+ train/save metric]"
run A --train-metric-inline $METRIC --save-trained-metric "$MP" $BUD \
      --aco-mmas-bounds --aco-weight-method linear --hybrid-select --use-aco --optimizer aco

echo "### B: A + global-prior 0.3"
run B --metric-path "$MP" $BUD --aco-mmas-bounds --aco-weight-method linear --hybrid-select \
      --use-aco --optimizer aco --global-prior-weight 0.3

echo "### C: A + final-autogluon-topk 5"
run C --metric-path "$MP" $BUD --aco-mmas-bounds --aco-weight-method linear --hybrid-select \
      --use-aco --optimizer aco --final-autogluon-topk 5

echo "### D: A + hybrid-select-margin 0.01"
run D --metric-path "$MP" $BUD --aco-mmas-bounds --aco-weight-method linear --hybrid-select \
      --use-aco --optimizer aco --hybrid-select-margin 0.01

echo "### E: A + all three (prior0.3 + topk5 + margin0.01)"
run E --metric-path "$MP" $BUD --aco-mmas-bounds --aco-weight-method linear --hybrid-select \
      --use-aco --optimizer aco --global-prior-weight 0.3 --final-autogluon-topk 5 --hybrid-select-margin 0.01

echo "### ABLATION COMPLETE"
