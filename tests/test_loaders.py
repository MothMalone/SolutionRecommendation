from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from automl_aco.data import loaders


def _make_classification_df(n_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        rows.append({"f1": float(i), "f2": float(i % 3), "target": 0})
    for i in range(n_per_class):
        rows.append({"f1": float(i + 10), "f2": float((i + 1) % 3), "target": 1})
    return pd.DataFrame(rows)


def test_openml_local_fallback_csv(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(loaders, "fetch_openml", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    dataset_id = 1520
    df = _make_classification_df()
    csv_path = tmp_path / f"{dataset_id}.csv"
    df.to_csv(csv_path, index=False)

    loaded = loaders.load_openml_dataset(
        dataset_id=dataset_id,
        verbose=False,
        local_data_folder=str(tmp_path),
    )
    assert loaded is not None
    assert loaded["id"] == dataset_id
    assert len(loaded["X"]) == len(df)
    assert len(loaded["y"]) == len(df)


def test_openml_local_fallback_csv_zip(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(loaders, "fetch_openml", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    dataset_id = 381
    df = _make_classification_df()
    csv_inner = tmp_path / f"{dataset_id}.csv"
    zip_path = tmp_path / f"{dataset_id}.csv.zip"
    df.to_csv(csv_inner, index=False)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_inner, arcname=csv_inner.name)
    csv_inner.unlink()

    loaded = loaders.load_openml_dataset(
        dataset_id=dataset_id,
        verbose=False,
        local_data_folder=str(tmp_path),
    )
    assert loaded is not None
    assert loaded["id"] == dataset_id
    assert len(loaded["X"]) == len(df)
    assert len(loaded["y"]) == len(df)


def test_openml_direct_api_path(monkeypatch):
    class _FakeDataset:
        default_target_attribute = "target"

        def __init__(self, frame: pd.DataFrame):
            self._frame = frame

        def get_data(self, target=None, dataset_format="dataframe"):
            assert dataset_format == "dataframe"
            X = self._frame.drop(columns=[target]).copy()
            y = self._frame[target].copy()
            return X, y, None, list(X.columns)

    class _FakeOpenMLDatasets:
        @staticmethod
        def get_dataset(dataset_id, download_data=True):
            assert dataset_id == 1520
            assert download_data is True
            return _FakeDataset(_make_classification_df())

    class _FakeOpenML:
        datasets = _FakeOpenMLDatasets()

    monkeypatch.setattr(loaders, "openml_api", _FakeOpenML())
    monkeypatch.setattr(
        loaders,
        "fetch_openml",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fetch_openml should not run")),
    )

    loaded = loaders.load_openml_dataset(dataset_id=1520, verbose=False, use_direct_api=True)
    assert loaded is not None
    assert loaded["id"] == 1520
    assert len(loaded["X"]) == 12
    assert len(loaded["y"]) == 12


def test_openml_direct_api_failure_falls_back_to_sklearn(monkeypatch):
    class _FailingOpenMLDatasets:
        @staticmethod
        def get_dataset(dataset_id, download_data=True):
            raise RuntimeError("direct api failed")

    class _FailingOpenML:
        datasets = _FailingOpenMLDatasets()

    class _Fetched:
        def __init__(self, frame: pd.DataFrame):
            self.data = frame.drop(columns=["target"]).copy()
            self.target = frame["target"].copy()

    monkeypatch.setattr(loaders, "openml_api", _FailingOpenML())
    monkeypatch.setattr(loaders, "fetch_openml", lambda *args, **kwargs: _Fetched(_make_classification_df()))

    loaded = loaders.load_openml_dataset(dataset_id=381, verbose=False, use_direct_api=True)
    assert loaded is not None
    assert loaded["id"] == 381
    assert len(loaded["X"]) == 12
    assert len(loaded["y"]) == 12
