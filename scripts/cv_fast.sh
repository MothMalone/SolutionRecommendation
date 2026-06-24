#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
PANEL="181 337 54 377"   # wins + catastrophe-prone
COMMON="--dataset-source openml --dataset-ids $PANEL --require-autogluon --autogluon-profile local_rf_xt --seed 42"
MP="outputs/hard_metric.pt"
SHARED="--metric-path $MP --aco-mmas-bounds --aco-weight-method linear --hybrid-select --final-autogluon-topk 1 --proxy-seeds 42,52,62 --use-aco --optimizer aco --n-ants 4 --n-iterations 3"
run(){ local n="$1"; shift; rm -rf "outputs/cvf_$n"; python3 scripts/run_recommend.py $COMMON "$@" --output-dir "outputs/cvf_$n" >"outputs/cvf_$n.log" 2>&1; echo "DONE $n"; }
run baseline --baseline-only no_preprocessing
run single $SHARED --hybrid-select-margin 0.02
run cv     $SHARED --cv-select-folds 3
echo "### CVFAST COMPLETE"
python3 - <<'PY'
import json,glob,os,statistics
panel=["20","181","337","54","377"]
def C(d):
    s={}
    for p in glob.glob(f"outputs/cvf_{d}/dataset_*/recommendation.json"):
        try:
            r=json.load(open(p));v=r["final_evaluation"].get("score") or r["final_evaluation"].get("test_accuracy")
            did=os.path.basename(os.path.dirname(p)).replace("dataset_","")
            if v is not None: s[did]=float(v)
        except: pass
    return s
B=C("baseline");S=C("single");V=C("cv")
print(f"{'ds':>5} | {'base':>6} | {'single':>6} | {'cv':>6}")
for d in panel: print(f"{d:>5} | {B.get(d,float('nan')):6.3f} | {S.get(d,float('nan')):6.3f} | {V.get(d,float('nan')):6.3f}")
for n,R in [("single",S),("cv",V)]:
    ds=[d for d in panel if d in R and d in B]
    if ds: print(f"{n}: mean={statistics.mean(R[d] for d in ds):.4f} base={statistics.mean(B[d] for d in ds):.4f} worst={min(R[d]-B[d] for d in ds):+.3f} W/L={sum(R[d]>B[d]+1e-4 for d in ds)}/{sum(R[d]<B[d]-1e-4 for d in ds)}")
PY
