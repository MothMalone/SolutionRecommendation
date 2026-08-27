# Estimator-only TPOT scoring of AutoDP-prepared frames — the AutoDP arm of the
# ACORec-vs-AutoDP-under-TPOT comparison.
#
# RUNS IN THE TPOT ENVIRONMENT (requirements-tpot-kaggle.txt). It does NOT run the AutoDP search:
# it reads the prepared.csv + autodp_meta.json that scripts/run_autodatapre.py already wrote in the
# pinned .venv-autodp environment (stage 2), and scores that frame with the same estimator-only
# TPOT the ACORec arm uses (scripts/_tpot_eval.py).
#
# Mirror of the ACORec runner: same RUN_IDS, same env, same output-per-dataset layout.
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/SolutionRecommendation")

# Where scripts/run_autodatapre.py wrote its stage-2 output. For --mode fair --operator-space ours
# that is  <PREPARED_ROOT>/fair_ourops/dataset_<id>/prepared.csv
PREPARED_ROOT = Path("/kaggle/working/adp_prepared")
PREPARED_SPACE_TAG = "fair_ourops"          # "fair" for theirs-ops, "fair_ourops" for ours-ops

# The exported <id>.csv dir — the SAME files the ACORec arm reads (source of the original target).
CACHE_DIR = Path("/kaggle/working/eval_all")

OUTPUT_DIR = Path("/kaggle/working/autodp_tpot_shard_00")

RUN_IDS = [1066]  # the dataset ids this shard scores

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
    prepared_dir = PREPARED_ROOT / PREPARED_SPACE_TAG / f"dataset_{dataset_id}"
    dataset_csv = CACHE_DIR / f"{dataset_id}.csv"
    output_path = (
        OUTPUT_DIR / "tpot_evaluation.json"
        if len(RUN_IDS) == 1
        else OUTPUT_DIR / f"dataset_{dataset_id}" / "tpot_evaluation.json"
    )

    if not (prepared_dir / "prepared.csv").exists():
        print(f"SKIP {dataset_id}: missing {prepared_dir / 'prepared.csv'}")
        continue
    if not dataset_csv.exists():
        print(f"SKIP {dataset_id}: missing {dataset_csv}")
        continue

    command = [
        sys.executable,
        str(REPO_DIR / "scripts/evaluate_autodp_tpot.py"),
        "--dataset-csv", str(dataset_csv),
        "--prepared-dir", str(prepared_dir),
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

    print("\nAutoDP+TPOT command:")
    print(" ".join(shlex.quote(str(x)) for x in command))

    result = subprocess.run(command, cwd=REPO_DIR, env=env, check=False)
    print(f"Dataset {dataset_id} return code: {result.returncode}")

    if output_path.exists():
        row = json.loads(output_path.read_text())
        print(
            f"  {dataset_id}: status={row.get('status')} "
            f"score_full={row.get('score_full')} score_kept={row.get('score_kept')} "
            f"coverage={row.get('test_coverage')} dead_search={row.get('dead_search')}"
        )
