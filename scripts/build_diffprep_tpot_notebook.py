"""Build the Kaggle notebook that evaluates DiffPrep pipelines with TPOT."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "reproduce-diffprep-tpot.ipynb"


def _source(value: str) -> list[str]:
    return (textwrap.dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


cells = [
    markdown(
        """
        # DiffPrep + TPOT evaluator on the 30 ACORec test datasets

        This notebook preserves **DiffPrep's original preprocessing search space**.
        `main.py` imports `space` from the upstream `prep_space.py`; the fork does
        not replace it with ACORec operators. The learned deterministic DiffPrep
        pipeline is applied first, and TPOT then searches **estimators only** with
        `preprocessing=False`. Consequently, TPOT is the robust final evaluator,
        not a second preprocessing recommender.

        Protocol:

        - DiffPrep method: `diffprep_fix`, logistic differentiable evaluator;
        - outer split: the repository's 60/20/20 split with seed 42;
        - TPOT fits only the 60% training split and is scored once on the 20% test;
        - the 20% validation split remains reserved/unused by TPOT;
        - 30 datasets are divided into five Kaggle shards of six datasets;
        - results are checkpointed after every dataset.

        Every DiffPrep input is now materialized through the same canonical
        ACORec loader used by the other methods. This applies the same target
        coercion, rare-class filtering, row order, and 100k cap before DiffPrep
        creates its own fitted pipeline. The fork's original data is used only
        as the frozen Google (100000) fallback.
        """
    ),
    code(
        """
        # Install only DiffPrep's missing dependency plus TPOT. AutoGluon is not used.
        %pip install -q "TPOT==1.1.0" "impyute>=0.0.8" "pyarrow>=15"
        """
    ),
    code(
        """
        from __future__ import annotations

        import gc
        import json
        import math
        import os
        import pickle
        import shutil
        import subprocess
        import sys
        import time
        import traceback
        import warnings
        from pathlib import Path

        for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[variable] = "1"

        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import requests
        import sklearn
        import torch
        import tpot
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        from tpot import TPOTClassifier
        from tpot.config import get_search_space

        warnings.filterwarnings("ignore")

        # -------- Controls changed between Kaggle Save-Version runs --------
        NUM_DATASET_SHARDS = 5
        DATASET_SHARD_INDEX = 0       # run 0, 1, 2, 3, 4

        # -------- Reproduction protocol --------
        METHOD = "diffprep_fix"
        SPLIT_SEED = 42
        TRAIN_SEED = 1
        MAX_SAMPLES = 100_000
        PARQUET_BATCH_SIZE = 4_096

        # TPOT is an estimator-only evaluator after DiffPrep preprocessing.
        TPOT_MAX_TIME_MINS = 5
        TPOT_MAX_EVAL_TIME_MINS = 1
        TPOT_N_JOBS = 2
        TPOT_WORKER_MEMORY = "5GB"
        TPOT_POPULATION_SIZE = 20
        TPOT_VERBOSE = 2
        MAX_CV_FOLDS = 5

        KAGGLE = Path("/kaggle/working").exists()
        OUTPUT_DIR = Path("/kaggle/working/diffprep_tpot") if KAGGLE else Path("outputs/diffprep_tpot")
        TEMP_ROOT = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else OUTPUT_DIR / "temp"
        REPO_DIR = TEMP_ROOT / "DiffPrep"
        CACHE_DIR = TEMP_ROOT / "openml_datagit_cache"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if not 0 <= DATASET_SHARD_INDEX < NUM_DATASET_SHARDS:
            raise ValueError(f"DATASET_SHARD_INDEX must be in [0, {NUM_DATASET_SHARDS - 1}]")

        print("Python:", sys.version.split()[0])
        print("PyTorch:", torch.__version__)
        print("scikit-learn:", sklearn.__version__)
        print("TPOT:", tpot.__version__)
        print("Output:", OUTPUT_DIR)
        """
    ),
    code(
        """
        # Clone the same fork/branch used by reproduce-diffprep.ipynb.
        # /kaggle/temp avoids including the whole repository in notebook outputs.
        if not (REPO_DIR / ".git").exists():
            subprocess.run(
                [
                    "git", "clone", "--branch", "kaggle-experiments", "--single-branch",
                    "https://github.com/dangvu53/DiffPrep.git", str(REPO_DIR),
                ],
                check=True,
            )
        subprocess.run(["git", "checkout", "kaggle-experiments"], cwd=REPO_DIR, check=True)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip()
        print("DiffPrep commit:", commit)

        SOLUTION_DIR = TEMP_ROOT / "SolutionRecommendation"
        if not (SOLUTION_DIR / ".git").exists():
            subprocess.run(
                [
                    "git", "clone", "--branch", "feature/acorec-autodp-space",
                    "--single-branch", "https://github.com/MothMalone/SolutionRecommendation.git",
                    str(SOLUTION_DIR),
                ],
                check=True,
            )
        sys.path.insert(0, str(SOLUTION_DIR / "src"))
        from automl_aco.data.loaders import load_gitlab_openml_dataset
        from automl_aco.eval_ids import EVAL_IDS

        # The upstream trainer evaluates X_test every epoch and passes it into
        # pipeline initialization. Patch that behavior in this reproduction:
        # DiffPrep may use train/validation only; outer test is reserved for TPOT.
        def patch_exact(path, old, new):
            path = Path(path)
            source = path.read_text(encoding="utf-8")
            if old in source:
                path.write_text(source.replace(old, new), encoding="utf-8")
            elif new not in source:
                raise RuntimeError(f"DiffPrep leakage patch target not found: {path}")

        for pipeline_name in ("diffprep_fix_pipeline.py", "diffprep_flex_pipeline.py"):
            pipeline_path = REPO_DIR / "pipeline" / pipeline_name
            patch_exact(
                pipeline_path,
                "return df.isnull().values.sum() > 0",
                "return df is not None and df.isnull().values.sum() > 0",
            )
            patch_exact(
                pipeline_path,
                '        first_transformer.pre_cache(X_test, "test")',
                '        if X_test is not None:' + chr(92) + 'n            first_transformer.pre_cache(X_test, "test")',
            )
        patch_exact(
            REPO_DIR / "experiment" / "diffprep_experiment.py",
            "prep_pipeline.init_parameters(X_train, X_val, X_test)",
            "prep_pipeline.init_parameters(X_train, X_val, None)",
        )
        patch_exact(
            REPO_DIR / "experiment" / "diffprep_experiment.py",
            "result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, X_test, y_test)",
            "result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, None, None)",
        )
        patch_exact(
            REPO_DIR / "trainer" / "diffprep_trainer.py",
            "            test_loss, test_acc = self.evaluate(X_test, y_test, X_type='test', max_only=False)",
            "            if X_test is None or y_test is None:" + chr(92) + "n                test_loss, test_acc = float('nan'), float('nan')" + chr(92) + "n            else:" + chr(92) + "n                test_loss, test_acc = self.evaluate(X_test, y_test, X_type='test', max_only=False)",
        )
        patch_exact(
            REPO_DIR / "extract_and_save_pipeline.py",
            "prep_pipeline.init_parameters(X_train, X_val, X_test)",
            "prep_pipeline.init_parameters(X_train, X_val, None)",
        )
        patch_exact(
            REPO_DIR / "extract_and_save_pipeline.py",
            "'original_test_acc': result['best_test_acc'],",
            "'original_test_acc': None,",
        )
        print("Patched DiffPrep: no outer-test access during pipeline search")

        # Ensure repository modules are importable when unpickling the learned pipeline.
        os.chdir(REPO_DIR)
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        """
    ),
    code(
        """
        # Exact 30-dataset ACORec test corpus supplied for the experiment.
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
            {"dataset_id": 44956, "name": "abalone", "dataset_key": "abalone"},
            {"dataset_id": 1037, "name": "ada_prior", "dataset_key": "ada_prior"},
            {"dataset_id": 42932, "name": "avila", "dataset_key": "avila"},
            {"dataset_id": 40668, "name": "connect-4", "dataset_key": "connect-4"},
            {"dataset_id": 1471, "name": "eeg", "dataset_key": "eeg"},
            {"dataset_id": 100000, "name": "google", "dataset_key": "google", "source": "kaggle_csv"},
            {"dataset_id": 42165, "name": "house", "dataset_key": "house_prices"},
            {"dataset_id": 41001, "name": "jungle_chess", "dataset_key": "jungle_chess_2pcs_raw_endgame_complete"},
            {"dataset_id": 41671, "name": "micro", "dataset_key": "microaggregation2"},
            {"dataset_id": 1046, "name": "mozilla4", "dataset_key": "mozilla4"},
            {"dataset_id": 46597, "name": "obesity", "dataset_key": "obesity"},
            {"dataset_id": 30, "name": "page-blocks", "dataset_key": "page-blocks"},
            {"dataset_id": 802, "name": "pbcseq", "dataset_key": "pbcseq"},
            {"dataset_id": 722, "name": "pol", "dataset_key": "pol"},
            {"dataset_id": 40922, "name": "run_or_walk", "dataset_key": "Run_or_walk_information"},
            {"dataset_id": 1119, "name": "uscensus", "dataset_key": "USCensus"},
            {"dataset_id": 1497, "name": "wall-robot-nav", "dataset_key": "wall-robot-navigation"},
        ]

        TARGET_OVERRIDES = {42932: "10", 100000: "Rating>4.2"}
        IGNORE_OVERRIDES = {42932: ["train", "test"]}

        positions = np.array_split(np.arange(len(DATASETS)), NUM_DATASET_SHARDS)
        SHARD_DATASETS = [DATASETS[int(i)] for i in positions[DATASET_SHARD_INDEX]]
        print(
            f"Shard {DATASET_SHARD_INDEX}/{NUM_DATASET_SHARDS - 1}: "
            f"{len(SHARD_DATASETS)} datasets"
        )
        display(pd.DataFrame(SHARD_DATASETS))
        """
    ),
    code(
        """
        # Prefer the exact data directories committed in the fork. Download only a
        # missing numeric OpenML dataset from GitLab/DataGit.
        GITLAB_ROOT = "https://gitlab.com/data/d/openml"
        SESSION = requests.Session()
        SESSION.headers.update({"User-Agent": "DiffPrep-TPOT-reproduction/1.0"})


        def attributes(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
            return [item.strip() for item in str(value).split(",") if item.strip()]


        def download(dataset_id, relative_path, destination, retries=5):
            destination = Path(destination)
            if destination.exists() and destination.stat().st_size > 0:
                return destination
            url = f"{GITLAB_ROOT}/{int(dataset_id)}/-/raw/master/{relative_path}"
            partial = destination.with_suffix(destination.suffix + ".part")
            errors = []
            for attempt in range(retries):
                try:
                    with SESSION.get(url, stream=True, timeout=(20, 300)) as response:
                        response.raise_for_status()
                        with partial.open("wb") as handle:
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    if partial.stat().st_size == 0:
                        raise IOError("Empty response")
                    os.replace(partial, destination)
                    return destination
                except Exception as error:
                    partial.unlink(missing_ok=True)
                    errors.append(f"{type(error).__name__}: {error}")
                    if attempt + 1 < retries:
                        time.sleep(3 * (2 ** attempt))
            raise RuntimeError(" | ".join(errors))


        def read_parquet_sample(path, max_rows, seed):
            parquet = pq.ParquetFile(path)
            total_rows = parquet.metadata.num_rows
            sample_size = min(total_rows, max_rows)
            if sample_size == total_rows:
                selected = None
            else:
                selected = np.sort(
                    np.random.default_rng(seed).choice(total_rows, sample_size, replace=False)
                )
            pieces, offset = [], 0
            for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE):
                table = pa.Table.from_batches([batch])
                end = offset + batch.num_rows
                if selected is None:
                    pieces.append(table)
                else:
                    left = np.searchsorted(selected, offset, side="left")
                    right = np.searchsorted(selected, end, side="left")
                    if right > left:
                        local = pa.array(selected[left:right] - offset, type=pa.int64())
                        pieces.append(table.take(local))
                offset = end
            if not pieces:
                raise ValueError("Parquet table contains no rows")
            return pa.concat_tables(pieces).to_pandas(split_blocks=True, self_destruct=True), total_rows


        def find_google_csv():
            expected = Path("/kaggle/input/diffprep-dataset/google/data.csv")
            if expected.exists():
                return expected
            root = Path("/kaggle/input")
            matches = sorted(root.glob("**/google/data.csv")) if root.exists() else []
            if matches:
                return matches[0]
            raise FileNotFoundError(
                "Attach a Kaggle input containing google/data.csv (normally diffprep-dataset)."
            )


        def load_raw_dataset(spec):
            dataset_id = int(spec["dataset_id"])
            if spec.get("source") == "kaggle_csv":
                path = find_google_csv()
                frame = pd.read_csv(path)
                original_rows = len(frame)
                if len(frame) > MAX_SAMPLES:
                    frame = frame.sample(MAX_SAMPLES, random_state=SPLIT_SEED).sort_index()
                target = TARGET_OVERRIDES[dataset_id]
                if target not in frame.columns and path.with_name("info.json").exists():
                    target = json.loads(path.with_name("info.json").read_text())["label"]
                return frame, target, original_rows, "kaggle_csv"

            cache = CACHE_DIR / str(dataset_id)
            cache.mkdir(parents=True, exist_ok=True)
            metadata_path = download(dataset_id, "dataset/metadata.json", cache / "metadata.json")
            parquet_path = download(dataset_id, "dataset/tables/data.pq", cache / "data.pq")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            description = metadata.get("data_set_description", metadata)
            targets = attributes(
                TARGET_OVERRIDES.get(dataset_id) or description.get("default_target_attribute")
            )
            if len(targets) != 1:
                raise ValueError(f"Expected exactly one target, got {targets}")
            frame, original_rows = read_parquet_sample(parquet_path, MAX_SAMPLES, SPLIT_SEED)
            frame.columns = frame.columns.astype(str)
            target = targets[0]
            excluded = set(attributes(description.get("ignore_attribute")))
            excluded.update(attributes(description.get("row_id_attribute")))
            excluded.update(IGNORE_OVERRIDES.get(dataset_id, []))
            frame = frame.drop(columns=[c for c in excluded if c in frame.columns])
            return frame, target, original_rows, "gitlab_datagit"


        def materialize_for_diffprep(spec):
            dataset_key = spec.get("dataset_key", str(spec["dataset_id"]))
            dataset_dir = REPO_DIR / "data" / dataset_key
            data_path = dataset_dir / "data.csv"
            info_path = dataset_dir / "info.json"
            # Use the same canonical loader as ACORec/CtxPipe. Existing
            # snapshots are overwritten so rare-class filtering, row order,
            # target encoding, and sample caps cannot diverge.
            if spec.get("source") == "kaggle_csv":
                canonical_csv = CACHE_DIR / f"{int(spec['dataset_id'])}.csv"
                source_google = REPO_DIR / "data" / dataset_key / "data.csv"
                if not canonical_csv.exists() and source_google.exists():
                    shutil.copyfile(source_google, canonical_csv)
            dataset = load_gitlab_openml_dataset(
                int(spec["dataset_id"]),
                cache_dir=str(CACHE_DIR),
                test_dataset_ids=[int(value) for value in EVAL_IDS],
                verbose=True,
                max_samples_if_test=MAX_SAMPLES,
            )
            if dataset is None:
                raise RuntimeError(f"Canonical loader could not load {spec['name']}")
            frame = pd.DataFrame(dataset["X"]).copy()
            frame["target"] = pd.Series(dataset["y"]).reset_index(drop=True)
            original_rows = int(dataset.get("original_rows", len(frame)))
            backend = str(dataset.get("download_backend", "canonical_loader"))
            if len(frame) < 20 or frame.shape[1] < 2:
                raise ValueError(f"Insufficient usable data: {frame.shape}")

            # DiffPrep's build_data() label-encodes every target; canonical
            # loader emits contiguous integer labels, so this is idempotent.
            target = "target"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(data_path, index=False)
            info = {
                "label": target,
                "dataset_id": int(spec["dataset_id"]),
                "dataset_name": spec["name"],
                "source": backend,
                "original_rows": int(original_rows),
                "used_rows": int(len(frame)),
                "raw_features": int(frame.shape[1] - 1),
            }
            info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"Canonical DiffPrep input {spec['name']}: {frame.shape}, source={backend}")
            del dataset
            del frame
            gc.collect()
            return dataset_key, info
        """
    ),
    code(
        """
        # TPOT evaluator: DiffPrep transforms first; TPOT gets classifiers only.
        def as_numpy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)


        def load_saved_pipeline(dataset_key):
            directory = REPO_DIR / "saved_pipelines" / METHOD / dataset_key
            with (directory / "pipeline.pkl").open("rb") as handle:
                pipeline = pickle.load(handle)
            with (directory / "data_splits.pkl").open("rb") as handle:
                split = pickle.load(handle)
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            return pipeline, split, metadata, directory


        def transform_with_diffprep(pipeline, split):
            if not pipeline.is_fitted:
                pipeline.fit(split["X_train"])
            # Test rows are cached only now, after the DiffPrep search is over.
            if "test" not in pipeline.pipeline[0].cache:
                pipeline.pipeline[0].pre_cache(split["X_test"], "test")
            transformed = {}
            with torch.no_grad():
                for part, x_type in (("train", "train"), ("val", "val"), ("test", "test")):
                    value = pipeline.transform(
                        split[f"X_{part}"], X_type=x_type, max_only=True, resample=False
                    )
                    array = as_numpy(value).astype(np.float32, copy=False)
                    if not np.isfinite(array).all():
                        raise ValueError(f"DiffPrep produced NaN/inf in transformed {part} data")
                    transformed[part] = array
            return transformed


        def safe_cv_folds(y):
            counts = pd.Series(y).value_counts()
            if len(counts) < 2:
                raise ValueError("Training split has fewer than two classes")
            folds = min(MAX_CV_FOLDS, int(counts.min()))
            if folds < 2:
                raise ValueError("A training class has fewer than two samples")
            return folds


        def evaluate_with_tpot(dataset_key, spec, data_info):
            pipeline, split, metadata, pipeline_dir = load_saved_pipeline(dataset_key)
            transformed = transform_with_diffprep(pipeline, split)
            y_train = as_numpy(split["y_train"]).ravel()
            y_test = as_numpy(split["y_test"]).ravel()
            cv_folds = safe_cv_folds(y_train)

            # This is intentionally not TPOT's default 'linear' pipeline space.
            # Only classifier choice/hyperparameters evolve after external preprocessing.
            search_space = get_search_space(
                "classifiers",
                n_classes=int(np.unique(y_train).size),
                n_samples=int(transformed["train"].shape[0]),
                n_features=int(transformed["train"].shape[1]),
                random_state=TRAIN_SEED,
                n_jobs=1,
            )
            model = TPOTClassifier(
                search_space=search_space,
                scorers=["accuracy"],
                scorers_weights=[1],
                cv=cv_folds,
                preprocessing=False,
                max_time_mins=TPOT_MAX_TIME_MINS,
                max_eval_time_mins=TPOT_MAX_EVAL_TIME_MINS,
                n_jobs=TPOT_N_JOBS,
                memory_limit=TPOT_WORKER_MEMORY,
                validation_strategy="none",
                early_stop=5,
                verbose=TPOT_VERBOSE,
                random_state=TRAIN_SEED,
                population_size=TPOT_POPULATION_SIZE,
                initial_population_size=TPOT_POPULATION_SIZE,
            )
            started = time.time()
            model.fit(transformed["train"], y_train)
            fit_seconds = time.time() - started
            prediction = model.predict(transformed["test"])

            config_path = pipeline_dir / "pipeline_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            return {
                "dataset_id": int(spec["dataset_id"]),
                "dataset": spec["name"],
                "dataset_key": dataset_key,
                "status": "ok",
                "method": METHOD,
                "evaluator": "TPOTClassifier",
                "tpot_space": "classifiers",
                "tpot_preprocessing": False,
                "accuracy": float(accuracy_score(y_test, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
                "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
                "diffprep_internal_test_accuracy": float(metadata["original_test_acc"]),
                "source": data_info.get("source", "diffprep_fork_snapshot"),
                "original_rows": data_info.get("original_rows"),
                "used_rows": int(len(y_train) + len(split["y_val"]) + len(y_test)),
                "train_rows": int(len(y_train)),
                "validation_rows_unused": int(len(split["y_val"])),
                "test_rows": int(len(y_test)),
                "raw_features": int(split["X_train"].shape[1]),
                "transformed_features": int(transformed["train"].shape[1]),
                "cv_folds": int(cv_folds),
                "fit_seconds": float(fit_seconds),
                "evaluated_individuals": int(len(model.evaluated_individuals)),
                "tpot_pipeline": repr(model.fitted_pipeline_),
                "diffprep_pipeline_config": json.dumps(config, separators=(",", ":")),
                "split_seed": SPLIT_SEED,
                "train_seed": TRAIN_SEED,
                "diffprep_commit": commit,
                "error_type": "",
                "error": "",
            }, model, pipeline, transformed
        """
    ),
    code(
        """
        # Run one shard. Re-running the cell skips successful dataset IDs in the checkpoint.
        RESULT_PATH = OUTPUT_DIR / (
            f"diffprep_tpot_shard_{DATASET_SHARD_INDEX:02d}_of_{NUM_DATASET_SHARDS:02d}.csv"
        )


        def read_rows():
            return pd.read_csv(RESULT_PATH).to_dict("records") if RESULT_PATH.exists() else []


        def upsert(rows, row):
            dataset_id = str(row["dataset_id"])
            rows[:] = [old for old in rows if str(old.get("dataset_id")) != dataset_id]
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULT_PATH, index=False)


        rows = read_rows()
        completed = {
            str(row["dataset_id"]) for row in rows if str(row.get("status")) == "ok"
        }

        for position, spec in enumerate(SHARD_DATASETS, start=1):
            if str(spec["dataset_id"]) in completed:
                print(f"SKIP successful: {spec['name']} ({spec['dataset_id']})")
                continue

            print("\\n" + "=" * 78)
            print(f"[{position}/{len(SHARD_DATASETS)}] {spec['name']} ({spec['dataset_id']})")
            print("=" * 78)
            started = time.time()
            model = pipeline = transformed = None
            try:
                dataset_key, data_info = materialize_for_diffprep(spec)

                subprocess.run(
                    [
                        sys.executable, "main.py", "--dataset", dataset_key,
                        "--method", METHOD, "--model", "log",
                        "--split_seed", str(SPLIT_SEED),
                        "--train_seed", str(TRAIN_SEED),
                    ],
                    cwd=REPO_DIR,
                    check=True,
                )
                # Important: use the same split seed as main.py. The old notebook
                # omitted this argument and silently reconstructed seed-1 splits.
                subprocess.run(
                    [
                        sys.executable, "extract_and_save_pipeline.py",
                        "--dataset", dataset_key, "--method", METHOD,
                        "--split_seed", str(SPLIT_SEED),
                    ],
                    cwd=REPO_DIR,
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable, "extract_pipeline_config.py",
                        "--dataset", dataset_key, "--method", METHOD,
                    ],
                    cwd=REPO_DIR,
                    check=True,
                )

                row, model, pipeline, transformed = evaluate_with_tpot(
                    dataset_key, spec, data_info
                )
                row["total_seconds"] = float(time.time() - started)
                print(f"TPOT test accuracy: {row['accuracy']:.6f}")
            except Exception as error:
                traceback.print_exc()
                row = {
                    "dataset_id": int(spec["dataset_id"]),
                    "dataset": spec["name"],
                    "status": "failed",
                    "method": METHOD,
                    "evaluator": "TPOTClassifier",
                    "split_seed": SPLIT_SEED,
                    "train_seed": TRAIN_SEED,
                    "diffprep_test_seen_during_search": False,
                    "diffprep_commit": commit,
                    "total_seconds": float(time.time() - started),
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                }
            finally:
                upsert(rows, row)
                del model, pipeline, transformed
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("Saved:", RESULT_PATH)
        display(pd.DataFrame(rows).sort_values("dataset_id"))
        """
    ),
    markdown(
        """
        ## Outputs

        Download `diffprep_tpot_shard_XX_of_05.csv` from
        `/kaggle/working/diffprep_tpot/`. Run five Save-Version jobs with
        `DATASET_SHARD_INDEX = 0..4`, then concatenate the five CSV files.

        A failed row is intentionally retried when the run cell is executed again;
        only rows with `status == "ok"` are skipped.
        """
    ),
]


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


for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index:02d}"


for index, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    value = "".join(cell["source"])
    parseable = "\n".join(
        line for line in value.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    ast.parse(parseable, filename=f"cell-{index}")

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUTPUT}")
