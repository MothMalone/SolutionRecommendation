"""Generate a self-contained Kaggle notebook for the two TPOT baselines."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "tpot-test-baselines-kaggle.ipynb"


def source(text: str) -> list[str]:
    cleaned = textwrap.dedent(text).strip("\n") + "\n"
    return cleaned.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
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
        # TPOT baselines on the 30 ACORec test datasets

        This notebook runs **only TPOT training and outer-test evaluation**. It does
        not contain ACORec, DiffPrep, CtxPipe, AutoDP, or performance-matrix code.

        It evaluates two baselines with deliberately different TPOT behavior:

        1. `no_preprocessing`: only the train-fitted compatibility adapter needed by
           candidate estimators (median/mode imputation plus one-hot encoding), then
           TPOT receives numeric data with `preprocessing=False` and searches only
           over estimators. No preprocessing choice is optimized.
        2. `tpot_default`: raw mixed-type data is passed to normal TPOT with its
           default `linear` evolutionary search space and `preprocessing=True`.
           TPOT first applies its fixed missing-value/one-hot compatibility layer,
           then evolves selectors, transformers, and estimators.

        Thus `no_preprocessing` measures TPOT model search after only unavoidable
        data conversion, while `tpot_default` measures TPOT's ordinary end-to-end
        pipeline optimization.

        Data loading:

        - 29 OpenML datasets are downloaded from the GitLab/DataGit mirror, not the
          OpenML API;
        - `google` has no OpenML ID in the supplied experiment and is read from the
          Kaggle input `diffprep-dataset/google/data.csv` with target `Rating>4.2`;
        - enable Internet in Kaggle and attach the `diffprep-dataset` input if the
          `google` row must be evaluated.

        The notebook uses the repository's 60/20/20 outer split. The validation split
        is retained for compatibility but is not used by TPOT. Results are checkpointed
        after every `(dataset, mode)` evaluation.
        """
    ),
    code(
        """
        # Kaggle installation. Run once at the beginning of a fresh session.
        %pip install -q "TPOT==1.1.0" "pyarrow>=15"
        """
    ),
    code(
        """
        # Imports and experiment controls
        from __future__ import annotations

        import gc
        import json
        import math
        import os
        import time
        import traceback
        import warnings
        from pathlib import Path

        # Prevent nested BLAS/OpenMP parallelism inside TPOT/Dask workers.
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import requests
        import sklearn
        import tpot
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            mean_squared_error,
            r2_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from tpot import TPOTClassifier, TPOTRegressor
        from tpot.config import get_search_space

        warnings.filterwarnings("ignore")
        RANDOM_STATE = 42

        # Five Kaggle Save-Version runs cover all 30 datasets (6 datasets each).
        NUM_DATASET_SHARDS = 5
        DATASET_SHARD_INDEX = 0       # change to 0, 1, 2, 3, or 4
        RUN_MODES = ["no_preprocessing", "tpot_default"]

        # Evaluator budget. A shard has 7-8 datasets x 2 modes.
        TPOT_MAX_TIME_MINS = 5
        TPOT_MAX_EVAL_TIME_MINS = 1
        TPOT_N_JOBS = 2
        TPOT_WORKER_MEMORY = "5GB"
        TPOT_VERBOSE = 2
        TPOT_POPULATION_SIZE = 20
        MAX_CV_FOLDS = 5

        # This matches the practical 20k-row cap used by the current raw AutoGluon
        # evaluator. Set to 100_000 only if the final protocol requires it and the
        # Kaggle runtime/memory budget has been validated first.
        MAX_SAMPLES = 20_000
        MAX_RAW_FEATURES = 2_000
        PARQUET_BATCH_SIZE = 4_096

        OUTPUT_DIR = (
            Path("/kaggle/working/tpot_baselines")
            if Path("/kaggle/working").exists()
            else Path("outputs/tpot_baselines")
        )
        CACHE_DIR = (
            Path("/kaggle/temp/tpot_openml_cache")
            if Path("/kaggle/temp").exists()
            else OUTPUT_DIR / "openml_cache"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
            raise ValueError(
                f"DATASET_SHARD_INDEX must be in [0, {NUM_DATASET_SHARDS - 1}]"
            )

        print("TPOT:", tpot.__version__)
        print("scikit-learn:", sklearn.__version__)
        print("Output:", OUTPUT_DIR)
        """
    ),
    code(
        """
        # Exact 30-dataset ACORec test corpus supplied for this experiment.
        # google=100000 is the synthetic ID already used in the project notebook.
        DATASETS = [
            {"dataset_id": 1066, "name": "kc1-binary"},
            {"dataset_id": 1047, "name": "usp05"},
            {"dataset_id": 862, "name": "sleuth-ex2016"},
            {"dataset_id": 40663, "name": "calendarDOW"},
            {"dataset_id": 1054, "name": "mc2"},
            {"dataset_id": 876, "name": "fri-c1"},
            {"dataset_id": 18, "name": "mfeat-morphological"},
            {"dataset_id": 1520, "name": "robot-failures-lp5"},
            {"dataset_id": 1548, "name": "autoUniv-au4"},
            {"dataset_id": 378, "name": "ipums-la-99"},
            {"dataset_id": 1485, "name": "madelon"},
            {"dataset_id": 14, "name": "mfeat-fourier"},
            {"dataset_id": 27, "name": "colic"},
            {"dataset_id": 44956, "name": "abalone"},
            {"dataset_id": 1037, "name": "ada_prior"},
            {"dataset_id": 42932, "name": "avila"},
            {"dataset_id": 40668, "name": "connect-4"},
            {"dataset_id": 1471, "name": "eeg"},
            {"dataset_id": 100000, "name": "google", "source": "kaggle_csv"},
            {"dataset_id": 42165, "name": "house"},
            {"dataset_id": 41001, "name": "jungle_chess"},
            {"dataset_id": 41671, "name": "micro"},
            {"dataset_id": 1046, "name": "mozilla4"},
            {"dataset_id": 46597, "name": "obesity"},
            {"dataset_id": 30, "name": "page-blocks"},
            {"dataset_id": 802, "name": "pbcseq"},
            {"dataset_id": 722, "name": "pol"},
            {"dataset_id": 40922, "name": "run_or_walk"},
            {"dataset_id": 1119, "name": "uscensus"},
            {"dataset_id": 1497, "name": "wall-robot-nav"},
        ]

        # OpenML 42932 has no default target in its metadata. Its Parquet table
        # stores the Avila class in column 10 and two original-split indicators
        # in train/test. Google is not an OpenML dataset here.
        TARGET_OVERRIDES = {42932: "10", 100000: "Rating>4.2"}
        IGNORE_OVERRIDES = {42932: ["train", "test"]}

        positions = np.array_split(np.arange(len(DATASETS)), NUM_DATASET_SHARDS)
        SHARD_DATASETS = [DATASETS[int(i)] for i in positions[DATASET_SHARD_INDEX]]
        print(
            f"Dataset shard {DATASET_SHARD_INDEX}/{NUM_DATASET_SHARDS - 1}: "
            f"{len(SHARD_DATASETS)} datasets, {len(SHARD_DATASETS) * len(RUN_MODES)} jobs"
        )
        display(pd.DataFrame(SHARD_DATASETS))
        """
    ),
    code(
        """
        # Memory-bounded GitLab/DataGit loader. No openml package or API is used.
        GITLAB_OPENML_ROOT = "https://gitlab.com/data/d/openml"
        GITLAB_MAX_RETRIES = 5
        GITLAB_RETRY_BASE_SECONDS = 3
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": "ACORec-TPOT-baselines/1.0"})


        def _attributes(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
            return [item.strip() for item in str(value).split(",") if item.strip()]


        def _download(dataset_id, relative_path, destination):
            destination = Path(destination)
            if destination.exists() and destination.stat().st_size > 0:
                return destination

            url = (
                f"{GITLAB_OPENML_ROOT}/{int(dataset_id)}/-/raw/master/"
                f"{relative_path}"
            )
            partial = destination.with_suffix(destination.suffix + ".part")
            errors = []
            for attempt in range(1, GITLAB_MAX_RETRIES + 1):
                try:
                    with _SESSION.get(
                        url, stream=True, timeout=(20, 300), allow_redirects=True
                    ) as response:
                        response.raise_for_status()
                        with partial.open("wb") as output:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    output.write(chunk)
                    if partial.stat().st_size == 0:
                        raise IOError(f"Empty response from {url}")
                    os.replace(partial, destination)
                    return destination
                except Exception as error:
                    partial.unlink(missing_ok=True)
                    errors.append(f"{type(error).__name__}: {error}")
                    if attempt < GITLAB_MAX_RETRIES:
                        wait = GITLAB_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                        print(f"  download retry {attempt}/{GITLAB_MAX_RETRIES} in {wait}s")
                        time.sleep(wait)
            raise RuntimeError(" | ".join(errors))


        def _read_parquet_sample(path, max_rows, seed):
            # Read a deterministic row sample without materializing the full table.
            parquet_file = pq.ParquetFile(path)
            total_rows = parquet_file.metadata.num_rows
            sample_size = total_rows if max_rows is None else min(total_rows, max_rows)

            if sample_size == total_rows:
                selected = None
            else:
                rng = np.random.default_rng(seed)
                selected = np.sort(rng.choice(total_rows, sample_size, replace=False))

            pieces = []
            offset = 0
            for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE):
                batch_table = pa.Table.from_batches([batch])
                end = offset + batch.num_rows
                if selected is None:
                    pieces.append(batch_table)
                else:
                    left = np.searchsorted(selected, offset, side="left")
                    right = np.searchsorted(selected, end, side="left")
                    if right > left:
                        local_indices = pa.array(selected[left:right] - offset, type=pa.int64())
                        pieces.append(batch_table.take(local_indices))
                offset = end

            if not pieces:
                raise ValueError("Parquet table contains no rows")
            table = pa.concat_tables(pieces)
            return table.to_pandas(split_blocks=True, self_destruct=True), total_rows


        def _load_gitlab(spec):
            dataset_id = int(spec["dataset_id"])
            directory = CACHE_DIR / str(dataset_id)
            directory.mkdir(parents=True, exist_ok=True)
            metadata_path = _download(
                dataset_id, "dataset/metadata.json", directory / "metadata.json"
            )
            parquet_path = _download(
                dataset_id, "dataset/tables/data.pq", directory / "data.pq"
            )
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            description = metadata.get("data_set_description", metadata)
            target_candidates = _attributes(
                TARGET_OVERRIDES.get(dataset_id)
                or description.get("default_target_attribute")
            )
            if len(target_candidates) != 1:
                raise ValueError(f"Expected exactly one target, got {target_candidates}")

            frame, original_rows = _read_parquet_sample(
                parquet_path, MAX_SAMPLES, RANDOM_STATE
            )
            frame.columns = frame.columns.astype(str)
            target = target_candidates[0]
            if target not in frame.columns:
                raise KeyError(f"Target {target!r} is absent from the Parquet table")

            excluded = {target}
            excluded.update(_attributes(description.get("ignore_attribute")))
            excluded.update(_attributes(description.get("row_id_attribute")))
            excluded.update(IGNORE_OVERRIDES.get(dataset_id, []))
            features = [column for column in frame.columns if column not in excluded]
            return frame[features].copy(), frame[target].copy(), target, original_rows


        def _find_google_csv():
            configured = Path("/kaggle/input/diffprep-dataset/google/data.csv")
            if configured.exists():
                return configured
            root = Path("/kaggle/input")
            matches = sorted(root.glob("**/google/data.csv")) if root.exists() else []
            if matches:
                return matches[0]
            raise FileNotFoundError(
                "google/data.csv was not found. Attach the Kaggle input "
                "diffprep-dataset (expected google/data.csv)."
            )


        def _load_google(spec):
            path = _find_google_csv()
            frame = pd.read_csv(path)
            original_rows = len(frame)
            if len(frame) > MAX_SAMPLES:
                frame = frame.sample(n=MAX_SAMPLES, random_state=RANDOM_STATE).sort_index()
            target = TARGET_OVERRIDES[100000]
            if target not in frame.columns:
                info_path = path.with_name("info.json")
                if info_path.exists():
                    with info_path.open("r", encoding="utf-8") as handle:
                        target = json.load(handle)["label"]
            if target not in frame.columns:
                raise KeyError(f"Target {target!r} is absent from {path}")
            return frame.drop(columns=[target]).copy(), frame[target].copy(), target, original_rows


        def load_dataset(spec):
            if spec.get("source") == "kaggle_csv":
                X, y, target, original_rows = _load_google(spec)
                backend = "kaggle_csv"
            else:
                X, y, target, original_rows = _load_gitlab(spec)
                backend = "gitlab_datagit"

            valid_target = ~pd.isna(y)
            X = X.loc[valid_target].reset_index(drop=True)
            y = y.loc[valid_target].reset_index(drop=True)
            X = X.dropna(axis=1, how="all")

            if X.shape[1] > MAX_RAW_FEATURES:
                rng = np.random.default_rng(RANDOM_STATE)
                selected = np.sort(
                    rng.choice(X.shape[1], MAX_RAW_FEATURES, replace=False)
                )
                X = X.iloc[:, selected].copy()

            # Preserve the task convention of the supplied ACORec notebook:
            # numeric target with >50 distinct values => regression; otherwise class.
            numeric_target = pd.api.types.is_numeric_dtype(y)
            task_type = (
                "regression" if numeric_target and y.nunique(dropna=True) > 50
                else "classification"
            )
            if task_type == "regression":
                y = pd.to_numeric(y, errors="coerce")
                keep = y.notna()
                X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
            else:
                y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name=target)
                class_counts = y.value_counts()
                keep_classes = class_counts[class_counts >= 5].index
                keep = y.isin(keep_classes)
                X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
                if y.nunique() < 2:
                    raise ValueError("Fewer than two classes remain after filtering")

            # sklearn estimators cannot consume infinities.
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns):
                X.loc[:, numeric_columns] = X.loc[:, numeric_columns].replace(
                    [np.inf, -np.inf], np.nan
                )
            if len(X) < 20 or X.shape[1] == 0:
                raise ValueError(f"Insufficient usable data: shape={X.shape}")

            print(
                f"Loaded {spec['name']} ({spec['dataset_id']}): {X.shape}, "
                f"task={task_type}, source={backend}, original_rows={original_rows}"
            )
            return {
                **spec,
                "X": X,
                "y": y,
                "target": target,
                "task_type": task_type,
                "backend": backend,
                "original_rows": int(original_rows),
            }
        """
    ),
    code(
        """
        # Repo-consistent split and the minimal compatibility adapter.
        def split_60_20_20(X, y):
            n_rows = len(y)
            n_val = int(n_rows * 0.20)
            n_test = int(n_rows * 0.20)
            rng = np.random.RandomState(RANDOM_STATE)
            indices = rng.permutation(n_rows)
            test_indices = indices[:n_test]
            val_indices = indices[n_test:n_test + n_val]
            train_indices = indices[n_test + n_val:]
            return {
                "X_train": X.iloc[train_indices].reset_index(drop=True),
                "y_train": y.iloc[train_indices].reset_index(drop=True),
                "X_val": X.iloc[val_indices].reset_index(drop=True),
                "y_val": y.iloc[val_indices].reset_index(drop=True),
                "X_test": X.iloc[test_indices].reset_index(drop=True),
                "y_test": y.iloc[test_indices].reset_index(drop=True),
            }


        def tpot_ready_raw_frame(X):
            # Normalize categorical dtypes without learning any data statistics.
            X = X.copy()
            categorical = list(X.select_dtypes(exclude=[np.number]).columns)
            for column in categorical:
                missing = X[column].isna()
                X[column] = X[column].astype(str).astype(object)
                X.loc[missing, column] = np.nan
            return X, categorical


        def build_minimal_adapter(X_train):
            numeric = list(X_train.select_dtypes(include=[np.number]).columns)
            categorical = [column for column in X_train.columns if column not in numeric]
            transformers = []
            if numeric:
                transformers.append(
                    ("numeric", SimpleImputer(strategy="median"), numeric)
                )
            if categorical:
                transformers.append(
                    (
                        "categorical",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                (
                                    "onehot",
                                    OneHotEncoder(
                                        handle_unknown="ignore",
                                        sparse_output=False,
                                        dtype=np.float32,
                                    ),
                                ),
                            ]
                        ),
                        categorical,
                    )
                )
            return ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=0.0,
                verbose_feature_names_out=False,
            )


        def safe_cv_folds(y_train, task_type):
            if task_type == "regression":
                return min(MAX_CV_FOLDS, len(y_train))
            counts = pd.Series(y_train).value_counts()
            # TPOT removes classes below the chosen fold count from its internal
            # CV search. Require at least two represented classes to remain.
            for folds in range(MAX_CV_FOLDS, 1, -1):
                if int((counts >= folds).sum()) >= 2:
                    return folds
            raise ValueError("Fewer than two classes have enough rows for 2-fold CV")
        """
    ),
    code(
        """
        # TPOT search configuration and one outer-test evaluation.
        def make_estimator_search_space(task_type, n_classes, n_samples, n_features):
            group = "classifiers" if task_type == "classification" else "regressors"
            return get_search_space(
                group,
                n_classes=n_classes,
                n_samples=n_samples,
                n_features=n_features,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )


        def evaluate_mode(dataset, mode):
            split = split_60_20_20(dataset["X"], dataset["y"])
            task_type = dataset["task_type"]
            cv_folds = safe_cv_folds(split["y_train"], task_type)

            adapter = None
            categorical_features = None
            if mode == "no_preprocessing":
                minimal_train, _ = tpot_ready_raw_frame(split["X_train"])
                minimal_test, _ = tpot_ready_raw_frame(split["X_test"])
                adapter = build_minimal_adapter(minimal_train)
                X_train = adapter.fit_transform(minimal_train, split["y_train"])
                X_test = adapter.transform(minimal_test)
                X_train = np.asarray(X_train, dtype=np.float32)
                X_test = np.asarray(X_test, dtype=np.float32)
                tpot_preprocessing = False
                model_feature_count = int(X_train.shape[1])
            elif mode == "tpot_default":
                X_train, categorical_features = tpot_ready_raw_frame(split["X_train"])
                X_test, _ = tpot_ready_raw_frame(split["X_test"])
                tpot_preprocessing = True
                model_feature_count = None
            else:
                raise ValueError(f"Unknown mode: {mode}")

            n_classes = (
                int(pd.Series(split["y_train"]).nunique())
                if task_type == "classification" else 1
            )
            if mode == "no_preprocessing":
                search_space = make_estimator_search_space(
                    task_type=task_type,
                    n_classes=n_classes,
                    n_samples=len(split["y_train"]),
                    n_features=dataset["X"].shape[1],
                )
            else:
                # TPOT's normal linear space evolves selectors, transformers,
                # optional inner predictors, and the final estimator.
                search_space = "linear"

            common = dict(
                search_space=search_space,
                scorers=["accuracy" if task_type == "classification" else "r2"],
                scorers_weights=[1],
                cv=cv_folds,
                preprocessing=tpot_preprocessing,
                categorical_features=categorical_features,
                max_time_mins=TPOT_MAX_TIME_MINS,
                max_eval_time_mins=TPOT_MAX_EVAL_TIME_MINS,
                n_jobs=TPOT_N_JOBS,
                memory_limit=TPOT_WORKER_MEMORY,
                validation_strategy="none",
                early_stop=5,
                verbose=TPOT_VERBOSE,
                random_state=RANDOM_STATE,
                population_size=TPOT_POPULATION_SIZE,
                initial_population_size=TPOT_POPULATION_SIZE,
            )
            model = TPOTClassifier(**common) if task_type == "classification" else TPOTRegressor(**common)

            started = time.perf_counter()
            model.fit(X_train, split["y_train"])
            prediction = model.predict(X_test)
            fit_seconds = time.perf_counter() - started

            if model_feature_count is None and getattr(model, "_preprocessing_pipeline", None):
                one_row = model._preprocessing_pipeline.transform(X_train.iloc[:1])
                model_feature_count = int(one_row.shape[1])

            result = {
                "dataset_id": dataset["dataset_id"],
                "dataset": dataset["name"],
                "mode": mode,
                "status": "ok",
                "source": dataset["backend"],
                "task_type": task_type,
                "primary_metric": "accuracy" if task_type == "classification" else "r2",
                "score": np.nan,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "f1_macro": np.nan,
                "r2": np.nan,
                "rmse": np.nan,
                "original_rows": dataset["original_rows"],
                "used_rows": len(dataset["y"]),
                "train_rows": len(split["y_train"]),
                "validation_rows_unused": len(split["y_val"]),
                "test_rows": len(split["y_test"]),
                "raw_features": dataset["X"].shape[1],
                "model_features": model_feature_count,
                "cv_folds": cv_folds,
                "fit_seconds": fit_seconds,
                "evaluated_individuals": len(model.evaluated_individuals),
                "pipeline": repr(model.fitted_pipeline_),
                "error_type": "",
                "error": "",
            }
            if task_type == "classification":
                result["accuracy"] = accuracy_score(split["y_test"], prediction)
                result["balanced_accuracy"] = balanced_accuracy_score(
                    split["y_test"], prediction
                )
                result["f1_macro"] = f1_score(
                    split["y_test"], prediction, average="macro", zero_division=0
                )
                result["score"] = result["accuracy"]
            else:
                result["r2"] = r2_score(split["y_test"], prediction)
                result["rmse"] = math.sqrt(
                    mean_squared_error(split["y_test"], prediction)
                )
                result["score"] = result["r2"]
            return result, model, adapter
        """
    ),
    code(
        """
        # Run this dataset shard and checkpoint after every mode.
        RESULT_PATH = OUTPUT_DIR / (
            f"tpot_results_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.csv"
        )


        def read_results():
            if RESULT_PATH.exists():
                return pd.read_csv(RESULT_PATH).to_dict("records")
            return []


        def upsert_and_save(rows, new_row):
            key = (str(new_row["dataset_id"]), new_row["mode"])
            rows[:] = [
                row for row in rows
                if (str(row.get("dataset_id")), row.get("mode")) != key
            ]
            rows.append(new_row)
            frame = pd.DataFrame(rows)
            frame.to_csv(RESULT_PATH, index=False)
            return frame


        results = read_results()
        successful = {
            (str(row.get("dataset_id")), row.get("mode"))
            for row in results if row.get("status") == "ok"
        }

        for dataset_number, spec in enumerate(SHARD_DATASETS, start=1):
            pending_modes = [
                mode for mode in RUN_MODES
                if (str(spec["dataset_id"]), mode) not in successful
            ]
            if not pending_modes:
                print(f"SKIP {spec['name']}: both modes already successful")
                continue

            print("\\n" + "=" * 78)
            print(
                f"Dataset {dataset_number}/{len(SHARD_DATASETS)}: "
                f"{spec['name']} ({spec['dataset_id']})"
            )
            print("=" * 78)
            try:
                dataset = load_dataset(spec)
            except Exception as error:
                print(f"LOAD FAILED: {type(error).__name__}: {error}")
                for mode in pending_modes:
                    failure = {
                        "dataset_id": spec["dataset_id"],
                        "dataset": spec["name"],
                        "mode": mode,
                        "status": "load_failed",
                        "error_type": type(error).__name__,
                        "error": str(error).replace("\\n", " ")[:2000],
                    }
                    upsert_and_save(results, failure)
                gc.collect()
                continue

            for mode in pending_modes:
                print(f"\\n--- {mode} ---")
                model = adapter = None
                try:
                    result, model, adapter = evaluate_mode(dataset, mode)
                    print(
                        f"OK score={result['score']:.6f}, "
                        f"time={result['fit_seconds'] / 60:.1f} min"
                    )
                except Exception as error:
                    print(f"FAILED: {type(error).__name__}: {error}")
                    traceback.print_exc(limit=2)
                    result = {
                        "dataset_id": spec["dataset_id"],
                        "dataset": spec["name"],
                        "mode": mode,
                        "status": "failed",
                        "source": dataset["backend"],
                        "task_type": dataset["task_type"],
                        "error_type": type(error).__name__,
                        "error": str(error).replace("\\n", " ")[:2000],
                    }
                finally:
                    upsert_and_save(results, result)
                    del model, adapter
                    gc.collect()

            del dataset
            gc.collect()

        final_results = pd.read_csv(RESULT_PATH)
        print(f"\\nSaved {len(final_results)} rows to {RESULT_PATH}")
        display(final_results)
        """
    ),
    code(
        """
        # Summary for this run. If several shard CSVs are present, merge them too.
        shard_files = sorted(OUTPUT_DIR.glob("tpot_results_shard_*_of_*.csv"))
        merged = pd.concat([pd.read_csv(path) for path in shard_files], ignore_index=True)
        merged = merged.drop_duplicates(["dataset_id", "mode"], keep="last")
        merged_path = OUTPUT_DIR / "tpot_results_merged_available_shards.csv"
        merged.to_csv(merged_path, index=False)

        print(f"Available shard files: {len(shard_files)}/{NUM_DATASET_SHARDS}")
        print(f"Merged rows: {len(merged)}/{len(DATASETS) * len(RUN_MODES)}")
        display(merged.groupby(["mode", "status"], dropna=False).size().rename("count"))
        if "score" in merged.columns:
            display(
                merged.pivot_table(
                    index=["dataset_id", "dataset"],
                    columns="mode",
                    values="score",
                    aggfunc="last",
                )
            )
        print("Merged output:", merged_path)
        """
    ),
]

for cell_index, cell in enumerate(cells):
    cell["id"] = f"cell-{cell_index:02d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def validate_python_cells() -> None:
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if any(line.lstrip().startswith("%") for line in text.splitlines()):
            continue
        try:
            ast.parse(text)
        except SyntaxError as error:
            raise SyntaxError(f"Invalid Python in notebook cell {index}: {error}") from error


validate_python_cells()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUTPUT}")
