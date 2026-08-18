import json

import pandas as pd

from automl_aco.data.loaders import load_gitlab_openml_dataset


def _write_cached_dataset(tmp_path, dataset_id, frame, metadata):
    folder = tmp_path / "openml_gitlab" / str(dataset_id)
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    frame.to_parquet(folder / "data.pq", index=False)


def test_gitlab_loader_reads_cached_parquet_and_metadata(tmp_path):
    dataset_id = 123
    frame = pd.DataFrame(
        {
            "row_id": range(12),
            "number": range(12),
            "category": ["a", "b"] * 6,
            "ignored": [1] * 12,
            "target": ["no", "yes"] * 6,
        }
    )
    metadata = {
        "data_set_description": {
            "default_target_attribute": "target",
            "ignore_attribute": "ignored",
            "row_id_attribute": "row_id",
        }
    }
    _write_cached_dataset(tmp_path, dataset_id, frame, metadata)

    loaded = load_gitlab_openml_dataset(
        dataset_id,
        cache_dir=str(tmp_path),
        test_dataset_ids=[dataset_id],
    )

    assert loaded is not None
    assert loaded["X"].columns.tolist() == ["number", "category"]
    assert loaded["task_type"] == "classification"
    assert loaded["download_backend"] == "gitlab-parquet"


def test_gitlab_loader_can_force_known_regression_task(tmp_path):
    dataset_id = 189
    frame = pd.DataFrame({"feature": range(12), "target": range(12)})
    metadata = {"data_set_description": {"default_target_attribute": "target"}}
    _write_cached_dataset(tmp_path, dataset_id, frame, metadata)

    loaded = load_gitlab_openml_dataset(
        dataset_id,
        cache_dir=str(tmp_path),
        test_dataset_ids=[dataset_id],
        regression_dataset_ids=[dataset_id],
    )

    assert loaded is not None
    assert loaded["task_type"] == "regression"
    assert loaded["y"].dtype.kind in "iufc"


def test_gitlab_backend_prefers_synthetic_local_csv(tmp_path):
    dataset_id = 100000
    frame = pd.DataFrame(
        {
            "Category": ["TOOLS", "GAME"] * 6,
            "Reviews": range(12),
            "Rating>4.2": [0, 1] * 6,
        }
    )
    frame.to_csv(tmp_path / f"{dataset_id}.csv", index=False)

    loaded = load_gitlab_openml_dataset(
        dataset_id,
        cache_dir=str(tmp_path),
        test_dataset_ids=[dataset_id],
    )

    assert loaded is not None
    assert loaded["X"].columns.tolist() == ["Category", "Reviews"]
    assert loaded["task_type"] == "classification"
    assert loaded["download_backend"] == "local-csv"
