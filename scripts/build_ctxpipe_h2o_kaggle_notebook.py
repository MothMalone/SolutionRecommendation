#!/usr/bin/env python3
"""Build the native CtxPipe + H2O Kaggle notebook from the audited TPOT flow."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "reproduce-ctxpipe-tpot.ipynb"
OUTPUT = ROOT / "notebooks" / "reproduce-ctxpipe-h2o.ipynb"


def lines(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
cells = notebook["cells"]
cells[0] = markdown(
    """
    # Native CtxPipe + H2O AutoML on the 30 evaluation datasets

    This notebook runs the official CtxPipe operator space and official
    `ctx_32000` checkpoint. CtxPipe recommends a six-step preprocessing
    sequence; H2O then evaluates that frozen sequence. H2O's optional target
    encoding is disabled here, so the evaluator uses `preprocessing=None`.
    TPOT is not installed or called in this notebook.

    Protocol:

    - fixed outer split: 60% train / 20% validation / 20% untouched test, seed 42;
    - native CtxPipe sees only outer train+validation (80%);
    - the selected sequence is fitted on outer train only and transforms validation/test;
    - H2O trains on train, selects the model on validation, and scores test once;
    - run five Save-Version jobs with `SHARD_INDEX=0..4`.

    Use a Kaggle GPU accelerator for native CtxPipe and enable Internet for the
    GTE-large checkpoint unless it is attached as a Kaggle model/input.
    """
)

cells[1] = code(
    """
    # Clone pinned experiment sources and install CtxPipe + H2O dependencies.
    import os
    import subprocess
    import sys
    from pathlib import Path

    SOLUTION_URL = "https://github.com/MothMalone/SolutionRecommendation.git"
    SOLUTION_BRANCH = "feature/acorec-autodp-space"
    SOLUTION_DIR = Path("/kaggle/working/SolutionRecommendation")

    CTXPIPE_URL = "https://github.com/ctxpipe/ctxpipe.git"
    CTXPIPE_COMMIT = "79caaa17f17ebdeeac6ba549abe150c5b3f1381d"
    CTXPIPE_DIR = Path("/kaggle/working/ctxpipe")

    if (SOLUTION_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(SOLUTION_DIR), "fetch", "origin", SOLUTION_BRANCH], check=True)
        subprocess.run(["git", "-C", str(SOLUTION_DIR), "switch", SOLUTION_BRANCH], check=True)
        subprocess.run(["git", "-C", str(SOLUTION_DIR), "pull", "--ff-only", "origin", SOLUTION_BRANCH], check=True)
    else:
        subprocess.run(["git", "clone", "--branch", SOLUTION_BRANCH, "--single-branch", SOLUTION_URL, str(SOLUTION_DIR)], check=True)

    if not (CTXPIPE_DIR / ".git").exists():
        subprocess.run(["git", "clone", CTXPIPE_URL, str(CTXPIPE_DIR)], check=True)
    subprocess.run(["git", "-C", str(CTXPIPE_DIR), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(CTXPIPE_DIR), "checkout", "--detach", CTXPIPE_COMMIT], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-r",
        str(SOLUTION_DIR / "requirements-ctxpipe-h2o-kaggle.txt"),
    ], check=True)
    subprocess.run([
        sys.executable, "-c",
        "import numpy,pandas,sklearn,torch,h2o,transformers; "
        "print('health',numpy.__version__,pandas.__version__,sklearn.__version__,"
        "torch.__version__,h2o.__version__,transformers.__version__)",
    ], check=True)
    print("Solution commit:", subprocess.check_output(["git", "-C", str(SOLUTION_DIR), "rev-parse", "--short", "HEAD"], text=True).strip())
    print("CtxPipe commit:", subprocess.check_output(["git", "-C", str(CTXPIPE_DIR), "rev-parse", "HEAD"], text=True).strip())
    """
)

controls = "".join(cells[2]["source"])
controls = controls.replace(
    "# Experiment controls. Run five Save-Version jobs with SHARD_INDEX=0..4.",
    "# Experiment controls. Run five Save-Version jobs with SHARD_INDEX=0..4.",
)
controls = controls.replace("TPOT_RANDOM_STATE = 1\n", "")
controls = controls.replace("TPOT_MAX_TIME_MINS = 5\n", "")
controls = controls.replace("TPOT_MAX_EVAL_TIME_MINS = 1\n", "")
controls = controls.replace("TPOT_N_JOBS = 2\n", "")
controls = controls.replace('TPOT_WORKER_MEMORY = "5GB"\n', "")
controls = controls.replace("TPOT_POPULATION_SIZE = 20\n", "")
controls = controls.replace("TPOT_MAX_CV_FOLDS = 5\n", "")
controls = controls.replace(
    "CTXPIPE_TIMEOUT_MINS_PER_DATASET = 45\n",
    "CTXPIPE_TIMEOUT_MINS_PER_DATASET = 45\n\n"
    "H2O_MAX_RUNTIME_SECS = 300\n"
    "H2O_MAX_RUNTIME_SECS_PER_MODEL = 60\n"
    "H2O_NFOLDS = 5\n"
    "H2O_NTHREADS = 1\n"
    'H2O_MAX_MEM_SIZE = "6G"\n',
)
controls = controls.replace("ctxpipe_tpot_data", "ctxpipe_h2o_data")
controls = controls.replace("ctxpipe_tpot_", "ctxpipe_h2o_")
cells[2] = code(controls)

cells[8] = code(
    """
    # Replay each frozen sequence leak-free and evaluate it with H2O only.
    import shlex

    for spec in run_specs:
        dataset_id = int(spec["dataset_id"])
        dataset_dir = OUTPUT_DIR / f"dataset_{dataset_id}"
        recommendation_path = dataset_dir / "ctxpipe_recommendation.json"
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        if recommendation.get("status") != "ok":
            print(f"SKIP H2O {dataset_id}: native CtxPipe failed")
            continue
        output_path = dataset_dir / "h2o_evaluation.json"
        command = [
            sys.executable,
            str(SOLUTION_DIR / "scripts" / "evaluate_ctxpipe_h2o.py"),
            "--ctxpipe-result-json", str(recommendation_path),
            "--dataset-id", str(dataset_id),
            "--data-dir", str(CACHE_DIR),
            "--output-json", str(output_path),
            "--max-samples", str(MAX_SAMPLES),
            "--split-seed", str(SPLIT_SEED),
            "--max-runtime-secs", str(H2O_MAX_RUNTIME_SECS),
            "--max-runtime-secs-per-model", str(H2O_MAX_RUNTIME_SECS_PER_MODEL),
            "--nfolds", str(H2O_NFOLDS),
            "--nthreads", str(H2O_NTHREADS),
            "--max-mem-size", H2O_MAX_MEM_SIZE,
        ]
        print("\\n", " ".join(shlex.quote(value) for value in command))
        completed = subprocess.run(command, cwd=SOLUTION_DIR, env=env, check=False)
        if completed.returncode != 0:
            print(f"H2O failed for {dataset_id}; failure JSON was retained")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    """
)

cells[9] = code(
    """
    # Summarize and archive all recommendations, H2O scores, traces, and failures.
    rows = []
    for spec in run_specs:
        dataset_id = int(spec["dataset_id"])
        dataset_dir = OUTPUT_DIR / f"dataset_{dataset_id}"
        recommendation = json.loads(
            (dataset_dir / "ctxpipe_recommendation.json").read_text(encoding="utf-8")
        )
        evaluation_path = dataset_dir / "h2o_evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
        rows.append({
            "dataset_id": dataset_id,
            "dataset_name": spec["name"],
            "ctxpipe_status": recommendation.get("status"),
            "ctxpipe_sequence": " -> ".join(recommendation.get("sequence", [])),
            "native_reward": recommendation.get("native_reward"),
            "h2o_status": evaluation.get("status", "not_run"),
            "accuracy": evaluation.get("accuracy"),
            "validation_score": evaluation.get("validation_score"),
            "selected_model": evaluation.get("selected_model_algo"),
            "error": evaluation.get("error", recommendation.get("error")),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    display(summary)
    archive = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
    print("Download:", archive)
    """
)

for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # The template contains only normal Python cells; unlike %pip cells,
        # this assertion catches accidental corruption before Kaggle import.
        ast.parse(source, filename=f"cell_{index}")
    cell["id"] = f"cell-{index:02d}"

notebook["metadata"]["accelerator"] = "GPU"
notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}")
