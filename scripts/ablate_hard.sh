#!/usr/bin/env bash
# Robustness ablation on a HARD reference panel (high spread + high downside) with best_quality
# (needed to expose the winner's curse). Reports mean AND worst-case single-dataset loss vs baseline.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD"

PANEL="20 54 181 312 337 377"
COMMON="--dataset-source openml --dataset-ids $PANEL --require-autogluon --autogluon-profile best_quality --time-limit 90 --seed 42"
METRIC="--metric-loss pearson --metric-weight-decay 1e-4 --metric-objective embedding_cosine"
MP="outputs/hard_metric.pt"
BASEY="--aco-mmas-bounds --aco-weight-method linear --hybrid-select --use-aco --optimizer aco"
mkdir -p outputs

run(){ local name="$1"; shift; rm -rf "outputs/hard_$name"; \
  python3 scripts/run_recommend.py $COMMON "$@" --output-dir "outputs/hard_$name" >"outputs/hard_$name.log" 2>&1; \
  echo "DONE $name"; }

echo "### baseline"; run baseline --baseline-only no_preprocessing
echo "### PROBLEM topk5 (+train/save metric)"; run topk5 --train-metric-inline $METRIC --save-trained-metric "$MP" $BASEY --final-autogluon-topk 5
echo "### CHEAPFIX topk1 + multiseed proxy + margin0.02"; run cheapfix --metric-path "$MP" $BASEY --final-autogluon-topk 1 --proxy-seeds 42,52,62 --hybrid-select-margin 0.02
echo "### HARD ABLATION COMPLETE"

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
B=collect("baseline")
print(f"{'dataset':>8} | {'base':>6} | {'topk5':>6} | {'cheapfix':>8}")
rows={c:collect(c) for c in ["topk5","cheapfix"]}
for d in panel:
    print(f"{d:>8} | {B.get(d,float('nan')):6.3f} | {rows['topk5'].get(d,float('nan')):6.3f} | {rows['cheapfix'].get(d,float('nan')):8.3f}")
print("-"*44)
for c in ["topk5","cheapfix"]:
    ds=[d for d in panel if d in rows[c] and d in B]
    if not ds: print(f"{c}: no results"); continue
    m=statistics.mean(rows[c][d] for d in ds); bm=statistics.mean(B[d] for d in ds)
    worst=min(rows[c][d]-B[d] for d in ds); w=sum(rows[c][d]>B[d]+1e-4 for d in ds); l=sum(rows[c][d]<B[d]-1e-4 for d in ds)
    print(f"{c:>9}: mean={m:.4f} (base={bm:.4f})  WORST-loss-vs-base={worst:+.3f}  W/L={w}/{l}")
PY
