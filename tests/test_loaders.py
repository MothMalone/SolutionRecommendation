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
