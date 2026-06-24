#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"
PANEL="20 54 181 312 337 377"
COMMON="--dataset-source openml --dataset-ids $PANEL --require-autogluon --autogluon-profile best_quality --time-limit 90 --seed 42"
BASEY="--metric-path outputs/hard_metric.pt --aco-mmas-bounds --aco-weight-method linear --hybrid-select --use-aco --optimizer aco --final-autogluon-topk 1"
run(){ local name="$1"; shift; rm -rf "outputs/hard_$name"; \
  python3 scripts/run_recommend.py $COMMON "$@" --output-dir "outputs/hard_$name" >"outputs/hard_$name.log" 2>&1; echo "DONE $name"; }
echo "### oldstyle: topk1 single-seed margin0"; run oldstyle $BASEY
echo "### cheapfix: topk1 multiseed margin0.02"; run cheapfix $BASEY --proxy-seeds 42,52,62 --hybrid-select-margin 0.02
echo "### FAST HARD ABLATION COMPLETE"
python3 - <<'PY'
import json,glob,os,statistics
panel=["20","54","181","312","337","377"]
def collect(d):
    sc={}
    for p in glob.glob(f"outputs/hard_{d}/dataset_*/recommendation.json")+glob.glob(f"outputs/hard_{d}/recommendation.json"):
        try:
            r=json.load(open(p));fe=r.get("final_evaluation",{})
            v=fe.get("score") if fe.get("score") is not None else fe.get("test_accuracy")
            did=os.path.basename(os.path.dirname(p)).replace("dataset_","")
            if did.startswith("hard_"): did=str(r.get("dataset_id"))
            if v is not None: sc[did]=float(v)
        except Exception: pass
    return sc
B=collect("baseline"); rows={c:collect(c) for c in ["oldstyle","cheapfix"]}
print(f"{'dataset':>8} | {'base':>6} | {'oldstyle':>8} | {'cheapfix':>8}")
for d in panel:
    print(f"{d:>8} | {B.get(d,float('nan')):6.3f} | {rows['oldstyle'].get(d,float('nan')):8.3f} | {rows['cheapfix'].get(d,float('nan')):8.3f}")
print("-"*44)
for c in ["oldstyle","cheapfix"]:
    ds=[d for d in panel if d in rows[c] and d in B]
    if not ds: print(f"{c}: none"); continue
    m=statistics.mean(rows[c][d] for d in ds); bm=statistics.mean(B[d] for d in ds)
    worst=min(rows[c][d]-B[d] for d in ds); w=sum(rows[c][d]>B[d]+1e-4 for d in ds); l=sum(rows[c][d]<B[d]-1e-4 for d in ds)
    print(f"{c:>9}: mean={m:.4f} (base={bm:.4f} on n={len(ds)})  WORST-loss={worst:+.3f}  W/L={w}/{l}")
PY
