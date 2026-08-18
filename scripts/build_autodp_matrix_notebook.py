"""Generate the Kaggle notebook for the AutoDP-space performance matrix."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "build-performance-matrix-autodp.ipynb"
PIPELINES_OUTPUT = ROOT / "aco" / "pipeline_configs_autodp36.json"
sys.path.insert(0, str(ROOT / "src"))

from automl_aco.preprocessing.autodp import build_autodp_reference_pipelines  # noqa: E402


def extract_historical_ids() -> list[int]:
    notebook = json.loads(
        (ROOT / "solrec-aco-our-valid-data.ipynb").read_text(encoding="utf-8")
    )
    candidates = []
    for cell in notebook.get("cells", []):
        text = "".join(cell.get("source", []))
        for match in re.finditer(r"(?ms)^train_dataset_ids\s*=\s*(\[[^\]]*\])", text):
            try:
                value = ast.literal_eval(match.group(1))
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, list) and all(isinstance(item, int) for item in value):
                candidates.append(value)
    if not candidates:
        raise RuntimeError("Could not extract train_dataset_ids from the ACORec notebook")
    return list(dict.fromkeys(max(candidates, key=len)))


HISTORICAL_IDS = extract_historical_ids()
AUTODP_MODULE_SOURCE = (ROOT / "src" / "automl_aco" / "preprocessing" / "autodp.py").read_text(
    encoding="utf-8"
)


def source(text: str):
    cleaned = textwrap.dedent(text).strip("\n") + "\n"
    return cleaned.splitlines(keepends=True)


def markdown(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    markdown(
        """
        # Build a Logistic/Linear performance matrix on the AutoDP operator space

        This notebook has one purpose: build the training performance matrix. It uses
        Logistic Regression for classification and Linear Regression for
        regression, with the 60/20/20 split used by the supplied notebook. The
        preprocessing space is replaced by the six AutoDP families from the paper
        (8,400 combinations including `none`) and 36 deterministic human-designed
        reference pipelines are evaluated.

        Methodological safeguards:

        - preprocessing is fit on the train split only;
        - validation and test rows are never removed;
        - the full historical matrix may contain AutoDP test datasets, but the
          `reference` matrix always removes all 60 AutoDP IDs for later leakage-safe use;
        - work is checkpointed and can optionally be sharded by `(dataset, pipeline)`.

        This file is self-contained for Kaggle: the complete AutoDP implementation,
        36 pipelines, 912 historical IDs and 60 holdout IDs are embedded directly.
        Dataset tables and metadata are downloaded only from OpenML's read-only
        GitLab/DataGit mirror; no OpenML API call or repository clone is required.
        """
    ),
    code(
        """
        # Kaggle setup. Re-running this cell is safe; no repository clone is needed.
        from pathlib import Path
        import os

        if Path("/kaggle/working").exists():
            os.chdir("/kaggle/working")

        print(f"Working directory: {Path.cwd()}")
        """
    ),
    code(AUTODP_MODULE_SOURCE),
    code(
        """
        # Imports and experiment controls
        from __future__ import annotations

        import json
        import math
        import os
        import time
        import warnings
        from collections import defaultdict
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import requests
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import accuracy_score, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.utils import shuffle

        warnings.filterwarnings("ignore")
        RANDOM_STATE = 42

        LINEAR_MODEL_CONFIG = {
            "logistic_C": 1.0,
            "logistic_max_iter": 1_000,
        }

        GITLAB_OPENML_ROOT = "https://gitlab.com/data/d/openml"
        GITLAB_MAX_RETRIES = 5
        GITLAB_RETRY_BASE_SECONDS = 3
        DOWNLOAD_FAILURES = []

        # Kaggle execution controls. Use ONLY_JOB_SHARDS to rescue specific shards.
        RUN_MATRIX = True
        CORPUS = "historical"          # "historical" or "autodp60"
        NUM_JOB_SHARDS = 32
        SHARDS_PER_RUN = 4
        RUN_INDEX = 0                    # 0 <= index < ceil(NUM_JOB_SHARDS / SHARDS_PER_RUN)
        ONLY_JOB_SHARDS = None            # e.g. [20] to resume only shard 20
        RESUME_FROM_DATASET_ID = None      # e.g. 1080; skip earlier datasets in that shard
        AUTO_IMPORT_RESUME_PARTS = True   # discover matching .csv/.txt in /kaggle/input

        NUM_RUNS = math.ceil(NUM_JOB_SHARDS / SHARDS_PER_RUN)
        if ONLY_JOB_SHARDS is None:
            if not 0 <= RUN_INDEX < NUM_RUNS:
                raise ValueError(f"RUN_INDEX must be between 0 and {NUM_RUNS - 1}")
            start_shard = RUN_INDEX * SHARDS_PER_RUN
            JOB_SHARD_INDICES = range(
                start_shard,
                min(start_shard + SHARDS_PER_RUN, NUM_JOB_SHARDS),
            )
        else:
            JOB_SHARD_INDICES = tuple(int(index) for index in ONLY_JOB_SHARDS)
            if not JOB_SHARD_INDICES or any(
                index < 0 or index >= NUM_JOB_SHARDS for index in JOB_SHARD_INDICES
            ):
                raise ValueError("ONLY_JOB_SHARDS contains an invalid shard index")

        MAX_SAMPLES_HISTORICAL = 5_000
        MAX_SAMPLES_AUTODP60 = 100_000
        OUTPUT_DIR = Path("/kaggle/working/autodp_matrix") if Path("/kaggle/working").exists() else Path("outputs/autodp_matrix")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_CACHE_DIR = (
            Path("/kaggle/temp/openml_gitlab_cache")
            if Path("/kaggle/temp").exists()
            else OUTPUT_DIR / "openml_gitlab_cache"
        )
        DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        pipeline_configs = build_autodp_reference_pipelines()
        options = AUTODP_OPTIONS  # same variable name as the supplied notebook

        print(f"AutoDP space size: {autodp_space_size():,}")
        print(f"Reference pipelines: {len(pipeline_configs)}")
        print(f"AutoDP holdout IDs: {len(AUTODP_60_IDS)}")
        print(f"Run {RUN_INDEX + 1}/{NUM_RUNS}; shards: {list(JOB_SHARD_INDICES)}")
        print(f"Dataset backend: GitLab/DataGit mirror ({GITLAB_OPENML_ROOT})")
        """
    ),
    code(
        """
        # Exact historical OpenML IDs are embedded so this notebook is standalone.
        historical_dataset_ids = __HISTORICAL_IDS__
        autodp60_dataset_ids = list(AUTODP_60_IDS)
        overlap_ids = sorted(set(historical_dataset_ids).intersection(AUTODP_60_IDS))
        print(f"Embedded historical IDs: {len(historical_dataset_ids)} unique IDs")
        print(f"Historical/AutoDP60 overlap: {len(overlap_ids)} IDs -> {overlap_ids}")

        coverage = []
        for stage, allowed in AUTODP_OPTIONS.items():
            counts = pd.Series([config[stage] for config in pipeline_configs]).value_counts()
            coverage.append({"stage": stage, **{value: int(counts.get(value, 0)) for value in allowed}})
        display(pd.DataFrame(coverage).fillna("-"))
        display(pd.DataFrame(pipeline_configs))
        """.replace("__HISTORICAL_IDS__", repr(HISTORICAL_IDS))
    ),
    code(
        """
        # OpenML datasets are read only from the GitLab/DataGit mirror.
        _GITLAB_SESSION = requests.Session()
        _GITLAB_SESSION.headers.update({"User-Agent": "ACORec-AutoDP-matrix/1.0"})


        def _gitlab_raw_url(dataset_id, relative_path):
            return (
                f"{GITLAB_OPENML_ROOT}/{int(dataset_id)}/-/raw/master/"
                f"{relative_path}"
            )


        def _download_gitlab_file(dataset_id, relative_path, destination):
            destination = Path(destination)
            if destination.exists() and destination.stat().st_size > 0:
                return destination

            temporary = destination.with_suffix(destination.suffix + ".part")
            url = _gitlab_raw_url(dataset_id, relative_path)
            errors = []
            for attempt in range(1, GITLAB_MAX_RETRIES + 1):
                try:
                    with _GITLAB_SESSION.get(
                        url,
                        stream=True,
                        timeout=(20, 300),
                        allow_redirects=True,
                    ) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as output:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    output.write(chunk)
                    if not temporary.exists() or temporary.stat().st_size == 0:
                        raise IOError(f"empty response from {url}")
                    os.replace(temporary, destination)
                    return destination
                except Exception as error:
                    errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                    temporary.unlink(missing_ok=True)
                    if attempt < GITLAB_MAX_RETRIES:
                        wait_seconds = GITLAB_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                        print(
                            f"  GitLab retry {attempt}/{GITLAB_MAX_RETRIES} "
                            f"for D_{dataset_id} in {wait_seconds}s"
                        )
                        time.sleep(wait_seconds)
            raise RuntimeError(" | ".join(errors))


        def _metadata_attributes(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
            return [item.strip() for item in str(value).split(",") if item.strip()]


        def download_gitlab_dataset(dataset_id):
            dataset_id = int(dataset_id)
            dataset_cache = DATASET_CACHE_DIR / str(dataset_id)
            dataset_cache.mkdir(parents=True, exist_ok=True)
            metadata_path = _download_gitlab_file(
                dataset_id,
                "dataset/metadata.json",
                dataset_cache / "metadata.json",
            )
            parquet_path = _download_gitlab_file(
                dataset_id,
                "dataset/tables/data.pq",
                dataset_cache / "data.pq",
            )

            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            description = metadata.get("data_set_description", metadata)

            frame = pd.read_parquet(parquet_path)
            frame.columns = frame.columns.astype(str)
            targets = _metadata_attributes(description.get("default_target_attribute"))
            if len(targets) != 1:
                raise ValueError(
                    f"expected one default target in GitLab metadata, got {targets}"
                )
            target = targets[0]
            if target not in frame.columns:
                raise KeyError(f"target {target!r} is absent from the Parquet columns")

            excluded = {target}
            excluded.update(_metadata_attributes(description.get("ignore_attribute")))
            excluded.update(_metadata_attributes(description.get("row_id_attribute")))
            feature_columns = [column for column in frame.columns if column not in excluded]
            return frame[feature_columns], frame[target], "gitlab-parquet"


        def load_gitlab_dataset(dataset_id, *, is_autodp60=False):
            try:
                X, y, download_backend = download_gitlab_dataset(dataset_id)
                X, y = X.copy(), y.copy()
                X.columns = X.columns.astype(str)
                X = X.dropna(axis=1, how="all")
                valid_target = ~y.isna()
                X = X.loc[valid_target].reset_index(drop=True)
                y = y.loc[valid_target].reset_index(drop=True)

                known_regression = int(dataset_id) in set(AUTODP_REGRESSION_IDS)
                numeric_target = pd.api.types.is_numeric_dtype(y)
                task_type = "regression" if known_regression or (numeric_target and y.nunique() > 50) else "classification"

                if task_type == "classification":
                    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))
                    counts = y.value_counts()
                    keep_classes = counts[counts >= 5].index
                    keep = y.isin(keep_classes)
                    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
                    if y.nunique() < 2:
                        raise ValueError("fewer than two classes remain after rare-class filtering")
                else:
                    y = pd.to_numeric(y, errors="coerce")
                    keep = y.notna()
                    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

                max_samples = MAX_SAMPLES_AUTODP60 if is_autodp60 else MAX_SAMPLES_HISTORICAL
                if len(X) > max_samples:
                    X, y = shuffle(X, y, n_samples=max_samples, random_state=RANDOM_STATE)
                    X, y = X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)
                if len(X) < 10:
                    raise ValueError(f"only {len(X)} valid rows")

                print(f"Loaded D_{dataset_id}: shape={X.shape}, task={task_type}, via={download_backend}")
                return {"id": int(dataset_id), "name": f"D_{int(dataset_id)}", "X": X, "y": y, "task_type": task_type}
            except Exception as error:
                print(f"FAILED D_{dataset_id}: {type(error).__name__}: {error}")
                DOWNLOAD_FAILURES.append({
                    "dataset_id": int(dataset_id),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                pd.DataFrame(DOWNLOAD_FAILURES).to_csv(
                    OUTPUT_DIR / "failed_gitlab_downloads.csv",
                    index=False,
                )
                return None
        """
    ),
    code(
        """
        # Leakage-safe split and evaluation.
        def split_train_val_test(X, y, task_type):
            stratify = y if task_type == "classification" else None
            try:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=stratify
                )
                temp_stratify = y_temp if task_type == "classification" else None
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_stratify
                )
            except ValueError:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.40, random_state=RANDOM_STATE
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
                )
            return tuple(item.reset_index(drop=True) for item in (X_train, X_val, X_test, y_train, y_val, y_test))


        def _linear_adapter(X_train):
            numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical = [column for column in X_train.columns if column not in numeric]
            transformers = []
            if numeric:
                transformers.append(("num", SimpleImputer(strategy="median"), numeric))
            if categorical:
                transformers.append(("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]), categorical))
            return ColumnTransformer(transformers=transformers, remainder="drop")


        def _evaluate_pipeline_linear(dataset, pipeline_config):
            X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                dataset["X"], dataset["y"], dataset["task_type"]
            )
            preprocessor = AutoDPPreprocessor(
                pipeline_config, task_type=dataset["task_type"], random_state=RANDOM_STATE
            )
            X_train, y_train = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)
            if len(X_train) < 2 or X_train.shape[1] == 0:
                return np.nan
            adapter = _linear_adapter(X_train)
            model = LinearRegression() if dataset["task_type"] == "regression" else LogisticRegression(
                C=LINEAR_MODEL_CONFIG["logistic_C"],
                max_iter=LINEAR_MODEL_CONFIG["logistic_max_iter"],
                random_state=RANDOM_STATE,
            )
            estimator = Pipeline([("adapter", adapter), ("model", model)])
            estimator.fit(X_train, y_train)
            prediction = estimator.predict(X_test)
            return float(r2_score(y_test, prediction) if dataset["task_type"] == "regression" else accuracy_score(y_test, prediction))


        def evaluate_pipeline(dataset, pipeline_config):
            try:
                return _evaluate_pipeline_linear(dataset, pipeline_config)
            except Exception as error:
                print(f"  FAILED {pipeline_config['name']} on {dataset['name']}: {type(error).__name__}: {error}")
                return np.nan
        """
    ),
    code(
        """
        # Resumable work planning, part-matrix writing, and shard merging.
        def make_job_shard(dataset_ids, configs, shard_index, num_shards):
            if not 0 <= shard_index < num_shards:
                raise ValueError("JOB_SHARD_INDEX must satisfy 0 <= index < NUM_JOB_SHARDS")
            jobs = [(dataset_id, config["name"]) for dataset_id in dataset_ids for config in configs]
            start = math.floor(len(jobs) * shard_index / num_shards)
            stop = math.floor(len(jobs) * (shard_index + 1) / num_shards)
            selected = jobs[start:stop]
            print(f"Shard {shard_index}/{num_shards}: jobs [{start}, {stop}) = {len(selected)} evaluations")
            grouped = defaultdict(list)
            for dataset_id, pipeline_name in selected:
                grouped[dataset_id].append(pipeline_name)
            return grouped


        def _atomic_csv(frame, path):
            path = Path(path)
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary)
            os.replace(temporary, path)


        def _import_resume_part_from_kaggle(part_path):
            # Seed a working checkpoint from an attached Kaggle .csv/.txt file.
            part_path = Path(part_path)
            if part_path.exists() or not AUTO_IMPORT_RESUME_PARTS:
                return
            input_root = Path("/kaggle/input")
            if not input_root.exists():
                return

            candidate_names = {part_path.name, part_path.with_suffix(".txt").name}
            candidates = [
                candidate
                for name in candidate_names
                for candidate in input_root.rglob(name)
                if candidate.is_file()
            ]
            valid = []
            for candidate in candidates:
                try:
                    saved = pd.read_csv(candidate, index_col=0)
                    completed_cells = int(saved.notna().sum().sum())
                    valid.append((completed_cells, candidate, saved))
                except Exception as error:
                    print(f"Ignoring invalid resume file {candidate}: {error}")
            if not valid:
                return

            completed_cells, source, saved = max(valid, key=lambda item: item[0])
            _atomic_csv(saved, part_path)
            print(
                f"Imported resume checkpoint {source} -> {part_path} "
                f"({completed_cells} completed cells)"
            )


        def run_job_shard(dataset_ids, configs, *, corpus, shard_index, num_shards):
            grouped = make_job_shard(dataset_ids, configs, shard_index, num_shards)
            part_path = OUTPUT_DIR / f"{corpus}_autodp36.part_{shard_index:04d}_of_{num_shards:04d}.csv"
            _import_resume_part_from_kaggle(part_path)
            columns = [f"D_{dataset_id}" for dataset_id in grouped]
            matrix = pd.DataFrame(index=[config["name"] for config in configs], columns=columns, dtype=float)
            if part_path.exists():
                saved = pd.read_csv(part_path, index_col=0)
                matrix.update(saved)
                print(f"Resuming {part_path}")

            config_by_name = {config["name"]: config for config in configs}
            resume_reached = RESUME_FROM_DATASET_ID is None
            if not resume_reached and int(RESUME_FROM_DATASET_ID) not in grouped:
                raise ValueError(
                    f"RESUME_FROM_DATASET_ID={RESUME_FROM_DATASET_ID} is not in shard {shard_index}"
                )
            for dataset_id, pipeline_names in grouped.items():
                if not resume_reached:
                    if int(dataset_id) == int(RESUME_FROM_DATASET_ID):
                        resume_reached = True
                        print(f"Resume boundary reached at D_{dataset_id}")
                    else:
                        print(f"Skipping D_{dataset_id} before resume boundary")
                        continue
                column = f"D_{dataset_id}"
                pending = [name for name in pipeline_names if pd.isna(matrix.loc[name, column])]
                if not pending:
                    continue
                dataset = load_gitlab_dataset(dataset_id, is_autodp60=(corpus == "autodp60"))
                if dataset is None:
                    _atomic_csv(matrix, part_path)
                    continue
                for pipeline_name in pending:
                    print(f"{column} | {pipeline_name}")
                    score = evaluate_pipeline(dataset, config_by_name[pipeline_name])
                    matrix.loc[pipeline_name, column] = score
                    print(f"  score={score}")
                    _atomic_csv(matrix, part_path)
            return matrix, part_path


        def merge_matrix_shards(part_paths, *, full_output, reference_output=None):
            merged = None
            conflicts = []
            for path in sorted(map(Path, part_paths)):
                part = pd.read_csv(path, index_col=0)
                if merged is None:
                    merged = part.copy()
                    continue
                merged = merged.reindex(index=merged.index.union(part.index), columns=merged.columns.union(part.columns))
                for row in part.index:
                    for column in part.columns:
                        value = part.loc[row, column]
                        if pd.isna(value):
                            continue
                        existing = merged.loc[row, column]
                        if pd.notna(existing) and not np.isclose(float(existing), float(value), equal_nan=True):
                            conflicts.append((row, column, existing, value, str(path)))
                        merged.loc[row, column] = value
            if merged is None:
                raise ValueError("No shard files supplied")
            if conflicts:
                raise ValueError(f"Conflicting shard values (first five): {conflicts[:5]}")
            merged = merged.reindex(index=[config["name"] for config in pipeline_configs])
            _atomic_csv(merged, full_output)
            if reference_output is not None:
                reference, removed = exclude_holdout_columns(merged)
                forbidden = {f"D_{dataset_id}" for dataset_id in AUTODP_60_IDS}
                assert forbidden.isdisjoint(reference.columns)
                _atomic_csv(reference, reference_output)
                print(f"Leakage guard removed {len(removed)} present AutoDP60 columns: {removed}")
            print(f"Merged matrix: {merged.shape}, missing cells={int(merged.isna().sum().sum())}")
            return merged
        """
    ),
    code(
        """
        # Execute several consecutive shards in this Kaggle run.
        corpus_ids = historical_dataset_ids if CORPUS == "historical" else autodp60_dataset_ids
        completed_paths = []

        if RUN_MATRIX:
            for shard_index in JOB_SHARD_INDICES:
                print(f"\\n{'=' * 70}")
                print(
                    f"Running shard {shard_index}/{NUM_JOB_SHARDS - 1} "
                    f"for RUN_INDEX={RUN_INDEX}"
                )
                print(f"{'=' * 70}")
                shard_matrix, shard_path = run_job_shard(
                    corpus_ids,
                    pipeline_configs,
                    corpus=CORPUS,
                    shard_index=shard_index,
                    num_shards=NUM_JOB_SHARDS,
                )
                completed_paths.append(shard_path)
                print(f"Completed shard {shard_index}: {shard_path}")

            print("\\nCompleted files in this Kaggle run:")
            for path in completed_paths:
                print(path)
        else:
            for shard_index in JOB_SHARD_INDICES:
                make_job_shard(
                    corpus_ids,
                    pipeline_configs,
                    shard_index,
                    NUM_JOB_SHARDS,
                )
            print("Dry run only. Set RUN_MATRIX=True to execute.")
        """
    ),
    code(
        """
        # Merge after all shards have been copied into OUTPUT_DIR.
        # This cell is also safe as a dry run while shards are incomplete.
        expected_pattern = f"{CORPUS}_autodp36.part_*_of_{NUM_JOB_SHARDS:04d}.csv"
        part_paths = sorted(OUTPUT_DIR.glob(expected_pattern))
        print(f"Found {len(part_paths)}/{NUM_JOB_SHARDS} shard files")

        if len(part_paths) == NUM_JOB_SHARDS:
            full_path = OUTPUT_DIR / f"{CORPUS}_performance_matrix_autodp36_full.csv"
            reference_path = OUTPUT_DIR / "training_performance_matrix_autodp36_reference.csv" if CORPUS == "historical" else None
            merged_matrix = merge_matrix_shards(
                part_paths,
                full_output=full_path,
                reference_output=reference_path,
            )
            print(f"Full matrix: {full_path}")
            if reference_path:
                print(f"Leakage-safe reference matrix: {reference_path}")
        else:
            print("Merge skipped until every shard is present.")
        """
    ),
    markdown(
        """
        ## Required downstream rule

        Keep both outputs: `*_full.csv` is the archival result over the whole historical
        corpus, while `training_performance_matrix_autodp36_reference.csv` has every
        present AutoDP test ID removed and is the safe input for later experiments.

        The JSON representation of the same 36 pipelines is available at
        `aco/pipeline_configs_autodp36.json`.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {
            "accelerator": "none",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)

PIPELINES_OUTPUT.write_text(
    json.dumps(build_autodp_reference_pipelines(), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(PIPELINES_OUTPUT)
