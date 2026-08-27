# Estimator-only TPOT scoring of frozen ACORec recommendations — the ACORec arm of the
# ACORec-vs-AutoDP-under-TPOT comparison.
#
# RUNS IN THE TPOT ENVIRONMENT (requirements-tpot-kaggle.txt). It reads:
#   * one ACORec recommendation.json per dataset (the frozen pipeline_config), and
#   * the exported <id>.csv — the SAME file the AutoDP arm reads, so both arms share row order and
#     the seed-42 0.6/0.2/0.2 split lands on identical rows.
# ACORec's search/recommender/matrix are untouched; only its top-1 pipeline is re-scored by TPOT.
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/SolutionRecommendation")

# Per-dataset ACORec output dir holding recommendation.json (from scripts/run_recommend.py).
# dataset_<id>/recommendation.json when a run sharded by dataset, else a single dir.
RECOMMENDATION_ROOT = Path("/kaggle/working/acorec_recommendations")

# The exported <id>.csv dir — the SAME files the AutoDP arm reads.
CACHE_DIR = Path("/kaggle/working/eval_all")

OUTPUT_DIR = Path("/kaggle/working/acorec_tpot_shard_00")

RUN_IDS = [1066]

TPOT_SPLIT_SEED = 42
TPOT_RANDOM_STATE = 1
TPOT_MAX_TIME_MINS = 5
TPOT_MAX_EVAL_TIME_MINS = 1
TPOT_N_JOBS = 2
TPOT_MEMORY_LIMIT = "5GB"
TPOT_POPULATION_SIZE = 20
TPOT_MAX_CV_FOLDS = 5

env = os.environ.copy()
env.update({
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

for dataset_id in RUN_IDS:
    rec_dir = (
        RECOMMENDATION_ROOT
        if (RECOMMENDATION_ROOT / "recommendation.json").exists()
        else RECOMMENDATION_ROOT / f"dataset_{dataset_id}"
    )
    recommendation_path = rec_dir / "recommendation.json"
    dataset_csv = CACHE_DIR / f"{dataset_id}.csv"
    output_path = (
        OUTPUT_DIR / "tpot_evaluation.json"
        if len(RUN_IDS) == 1
        else OUTPUT_DIR / f"dataset_{dataset_id}" / "tpot_evaluation.json"
    )

    if not recommendation_path.exists():
        print(f"SKIP {dataset_id}: missing {recommendation_path}")
        continue
    if not dataset_csv.exists():
        print(f"SKIP {dataset_id}: missing {dataset_csv}")
        continue

    command = [
        sys.executable,
        str(REPO_DIR / "scripts/evaluate_acorec_tpot.py"),
        "--recommendation-json", str(recommendation_path),
        "--dataset-csv", str(dataset_csv),
        "--dataset-id", str(dataset_id),
        "--output-json", str(output_path),
        "--split-seed", str(TPOT_SPLIT_SEED),
        "--tpot-seed", str(TPOT_RANDOM_STATE),
        "--max-time-mins", str(TPOT_MAX_TIME_MINS),
        "--max-eval-time-mins", str(TPOT_MAX_EVAL_TIME_MINS),
        "--n-jobs", str(TPOT_N_JOBS),
        "--memory-limit", TPOT_MEMORY_LIMIT,
        "--population-size", str(TPOT_POPULATION_SIZE),
        "--max-cv-folds", str(TPOT_MAX_CV_FOLDS),
        "--verbose", "2",
    ]

    print("\nACORec+TPOT command:")
    print(" ".join(shlex.quote(str(x)) for x in command))

    result = subprocess.run(command, cwd=REPO_DIR, env=env, check=False)
    print(f"Dataset {dataset_id} return code: {result.returncode}")

    if output_path.exists():
        row = json.loads(output_path.read_text())
        print(
            f"  {dataset_id}: status={row.get('status')} "
            f"score={row.get('score')} metric={row.get('primary_metric')}"
        )
