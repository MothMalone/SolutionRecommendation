#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
PANEL="20 54 181 312 337 377"
COMMON="--dataset-source openml --dataset-ids $PANEL --require-autogluon --autogluon-profile best_quality --time-limit 90 --seed 42"
rm -rf outputs/hard_cv
python3 scripts/run_recommend.py $COMMON --metric-path outputs/hard_metric.pt \
  --aco-mmas-bounds --aco-weight-method linear \
  --hybrid-select --final-autogluon-topk 1 --proxy-seeds 42,52,62 --cv-select-folds 3 \
  --use-aco --optimizer aco --output-dir outputs/hard_cv >outputs/hard_cv.log 2>&1
echo "### CV VALIDATION COMPLETE"
python3 - <<'PY'
import json,glob,os,statistics
panel=["20","54","181","312","337","377"]
def collect(d):
    sc={}
    for p in glob.glob(f"outputs/hard_{d}/dataset_*/recommendation.json"):
        try:
            r=json.load(open(p));fe=r.get("final_evaluation",{})
            v=fe.get("score") or fe.get("test_accuracy")
            did=os.path.basename(os.path.dirname(p)).replace("dataset_","")
            if v is not None: sc[did]=float(v)
        except Exception: pass
    return sc
B=collect("baseline");C=collect("cheapfix");V=collect("cv")
print(f"{'ds':>5} | {'base':>6} | {'cheapfix':>8} | {'cv':>6}")
for d in panel:
    print(f"{d:>5} | {B.get(d,float('nan')):6.3f} | {C.get(d,float('nan')):8.3f} | {V.get(d,float('nan')) if d in V else float('nan'):6.3f}")
for n,R in [("cheapfix",C),("cv",V)]:
    ds=[d for d in panel if d in R and d in B]
    if ds: print(f"{n}: mean={statistics.mean(R[d] for d in ds):.4f} base={statistics.mean(B[d] for d in ds):.4f} worst={min(R[d]-B[d] for d in ds):+.3f} W/L={sum(R[d]>B[d]+1e-4 for d in ds)}/{sum(R[d]<B[d]-1e-4 for d in ds)} (n={len(ds)})")
PY
