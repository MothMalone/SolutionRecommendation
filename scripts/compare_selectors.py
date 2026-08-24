"""Which selector ranks candidates better: the screening rung's single split, or the CV gate?

The ladder narrows 20 -> 5 with a SINGLE validation split, then hands the survivors to a 3-fold CV
gate whose own docstring calls it "a far less noisy estimate than a single validation split".
If that is true, the ladder is filtering with the WEAKER judge before the stronger one ever sees
the candidates -- which would discard candidates the gate would have preferred.
"""
import json, os, sys, time
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from measure_gate_fidelity import candidate_grid
from automl_aco.search.evaluation import (evaluate_candidates_autogluon,
                                          evaluate_candidates_autogluon_cv)

CORPUS = "data/adp_ourops_corpus/datasets"
ids = sys.argv[1].split(",")
cands = candidate_grid(10, seed=42)
out = []
for did in ids:
    df = pd.read_csv(f"{CORPUS}/{did}.csv")
    row = {"dataset_id": did, "n_rows": len(df)}
    for label, fn in (("single_split", "val"), ("cv3", "cv")):
        t0 = time.time()
        try:
            if fn == "val":
                _b,_s,ranked,_u = evaluate_candidates_autogluon(
                    dataset=df, target_column="target", candidate_configs=cands,
                    time_limit_per_model=25, autogluon_profile="local_rf_xt",
                    select_on_val=True, prepare_mode="leakfree")
            else:
                _b,_s,ranked,_u = evaluate_candidates_autogluon_cv(
                    dataset=df, target_column="target", candidate_configs=cands,
                    n_folds=3, time_limit_per_model=25, autogluon_profile="local_rf_xt",
                    seed=42, prepare_mode="leakfree")
            tests=[s for _c,s in ranked]
            rho = float(spearmanr([-i for i in range(len(tests))], tests).statistic) if len(tests)>=3 else None
            row[f"{label}_rho"]=rho
            row[f"{label}_regret"]=(max(tests)-tests[0]) if tests else None
            row[f"{label}_secs"]=round(time.time()-t0,1)
        except Exception as e:
            row[f"{label}_error"]=f"{type(e).__name__}: {e}"[:200]
    print(json.dumps(row), flush=True)
    out.append(row)
with open("outputs/selector_compare.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
ok=[r for r in out if r.get("single_split_rho") is not None and r.get("cv3_rho") is not None]
if ok:
    a=np.array([r["single_split_rho"] for r in ok]); b=np.array([r["cv3_rho"] for r in ok])
    ra=np.array([r["single_split_regret"] for r in ok]); rb=np.array([r["cv3_regret"] for r in ok])
    print(f"\nn={len(ok)}")
    print(f"  single-split (screening rung): rho={a.mean():+.3f}  top1_regret={ra.mean():.4f}")
    print(f"  cv3          (the real gate) : rho={b.mean():+.3f}  top1_regret={rb.mean():.4f}")
    print(f"  -> the gate is {'STRONGER' if b.mean()>a.mean() else 'WEAKER'} than the rung filtering for it")
