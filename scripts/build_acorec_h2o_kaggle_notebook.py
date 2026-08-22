"""Build Kaggle notebook for ACORec + H2O AutoML."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "run-acorec-tpot-kaggle.ipynb"
OUTPUT = ROOT / "notebooks" / "run-acorec-h2o-kaggle.ipynb"


def _source(value: str) -> list[str]:
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def _code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(value)}


def _markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
cells = notebook["cells"]
cells[0] = _markdown(
    """
    # ACORec + H2O AutoML evaluator

    ACORec searches the original operator space and H2O AutoML evaluates each
    frozen recommendation. The H2O target-encoding option is disabled because
    ACORec already supplies the preprocessing pipeline.

    Protocol: ACORec receives only the fixed outer train+validation 80% while
    recommending. H2O trains on processed train, selects its model on
    validation, and scores the untouched outer test once.
    Use ten Save-Version jobs with `SHARD_INDEX=0..9`.
    """
)
cells[1] = _code(
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPO_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
    BRANCH = "feature/acorec-autodp-space"
    REPO_DIR = Path("/kaggle/working/SolutionRecommendation")
    if (REPO_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "switch", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
    else:
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h2o>=3.46.0.11", "pyarrow>=15", "requests"], check=True)
    subprocess.run([sys.executable, "-c", "import h2o; print('H2O:', h2o.__version__)"], check=True)
    os.chdir(REPO_DIR)
    print("Repo commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
    """
)
cells[2] = _code(
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(REPO_DIR / "src"))
    from automl_aco.eval_ids import EVAL_DATASETS

    RUN_MODE = "smoke"       # change to final after smoke succeeds
    NUM_SHARDS = 10
    SHARD_INDEX = 0
    WORKERS = 1
    ACO_SEED = 42
    FINAL_N_ANTS, FINAL_N_ITERATIONS = 10, 10
    FINAL_METRIC_EPOCHS = 100
    SPLIT_SEED = 42
    MAX_SAMPLES = 100_000
    H2O_MAX_RUNTIME_SECS = 120 if RUN_MODE == "smoke" else 300
    H2O_MAX_RUNTIME_SECS_PER_MODEL = 60
    H2O_NFOLDS, H2O_NTHREADS, H2O_MAX_MEM_SIZE = 5, 1, "6G"
    if RUN_MODE not in {"smoke", "final"} or not 0 <= SHARD_INDEX < NUM_SHARDS:
        raise ValueError("Invalid RUN_MODE or SHARD_INDEX")
    all_ids = [int(dataset_id) for dataset_id in EVAL_DATASETS.values()]
    shard_ids = all_ids[SHARD_INDEX::NUM_SHARDS]
    run_ids = shard_ids[:1] if RUN_MODE == "smoke" else shard_ids
    CACHE_DIR = Path("/kaggle/working/acorec_h2o_eval30_data")
    OUTPUT_DIR = Path(f"/kaggle/working/acorec_h2o_{RUN_MODE}_shard_{SHARD_INDEX:02d}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Mode={RUN_MODE}; shard={SHARD_INDEX}/{NUM_SHARDS - 1}; IDs={run_ids}")
    """
)
# Keep cell 3: it exports the exact cached DiffPrep/OpenML snapshots used by
# the previous ACORec notebooks.
cells[4] = _code(
    """
    import shlex
    aco_command = [
        sys.executable, str(REPO_DIR / "scripts" / "run_recommend.py"),
        "--operator-space", "ours",
        "--performance-matrix", str(REPO_DIR / "data" / "openml" / "training_performance_matrix_autogluon.csv"),
        "--metafeatures", str(REPO_DIR / "data" / "openml" / "dataset_feats.csv"),
        "--pipeline-configs", str(REPO_DIR / "aco" / "pipeline_configs.json"),
        "--dataset-source", "openml", "--openml-backend", "gitlab",
        "--openml-local-folder", str(CACHE_DIR), "--dataset-ids",
        *[str(dataset_id) for dataset_id in run_ids], "--optimizer", "aco",
        "--seed", str(ACO_SEED), "--workers", str(WORKERS),
        "--output-dir", str(OUTPUT_DIR), "--skip-aco-plot", "--no-autogluon",
        "--recommend-on-train-val", "--recommend-split-seed", str(SPLIT_SEED), "--verbose",
    ]
    if RUN_MODE == "smoke":
        aco_command += ["--n-ants", "1", "--n-iterations", "1", "--no-train-metric-inline"]
    else:
        aco_command += ["--n-ants", str(FINAL_N_ANTS), "--n-iterations", str(FINAL_N_ITERATIONS), "--train-metric-inline", "--metric-epochs", str(FINAL_METRIC_EPOCHS)]
    print("ACORec command:\\n", " ".join(shlex.quote(part) for part in aco_command))
    """
)
cells[5] = _code(
    """
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    subprocess.run(aco_command, cwd=REPO_DIR, env=env, check=True)
    """
)
cells[6] = _code(
    """
    import json
    import shlex
    def dataset_output_dir(dataset_id):
        return OUTPUT_DIR if len(run_ids) == 1 else OUTPUT_DIR / f"dataset_{dataset_id}"

    for dataset_id in run_ids:
        dataset_dir = dataset_output_dir(dataset_id)
        recommendation_path = dataset_dir / "recommendation.json"
        if not recommendation_path.exists():
            raise FileNotFoundError(recommendation_path)
        output_path = dataset_dir / "h2o_evaluation.json"
        command = [
            sys.executable, str(REPO_DIR / "scripts" / "evaluate_acorec_h2o.py"),
            "--recommendation-json", str(recommendation_path), "--dataset-id", str(dataset_id),
            "--data-dir", str(CACHE_DIR), "--output-json", str(output_path),
            "--max-samples", str(MAX_SAMPLES), "--split-seed", str(SPLIT_SEED),
            "--h2o-preprocessing", "none", "--max-runtime-secs", str(H2O_MAX_RUNTIME_SECS),
            "--max-runtime-secs-per-model", str(H2O_MAX_RUNTIME_SECS_PER_MODEL),
            "--nfolds", str(H2O_NFOLDS), "--seed", "42", "--nthreads", str(H2O_NTHREADS),
            "--max-mem-size", H2O_MAX_MEM_SIZE,
        ]
        print("H2O evaluation:", " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, cwd=REPO_DIR, env=env, check=True)
    """
)
cells[7] = _code(
    """
    import json
    import pandas as pd
    import shutil
    rows = []
    for dataset_id in run_ids:
        result = json.loads((dataset_output_dir(dataset_id) / "h2o_evaluation.json").read_text(encoding="utf-8"))
        rows.append({key: result.get(key) for key in ("dataset_id", "status", "evaluator", "score", "accuracy", "balanced_accuracy", "f1_macro", "selected_model_id", "selected_model_algo", "validation_score", "h2o_preprocessing", "test_rows")})
    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "acorec_h2o_summary.csv"
    summary.to_csv(summary_path, index=False)
    display(summary)
    archive = shutil.make_archive(str(Path("/kaggle/working") / OUTPUT_DIR.name), "gztar", root_dir=OUTPUT_DIR)
    print("Summary:", summary_path)
    print("Archive:", archive)
    """
)
cells[8] = _markdown(
    """
    ## Final protocol

    Run `RUN_MODE="smoke"` first. Then run `RUN_MODE="final"` with
    `SHARD_INDEX=0..9`. H2O target encoding is disabled for ACORec; H2O's
    native categorical/missing-value handling remains part of its models.
    """
)

for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
