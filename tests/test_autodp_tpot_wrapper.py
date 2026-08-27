"""The AutoDP+TPOT evaluator shares split / coverage accounting with eval_autodatapre."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_autodp_tpot_test_module",
    ROOT / "scripts" / "evaluate_autodp_tpot.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeTPOTClassifier:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted_pipeline_ = "fake-majority-classifier"
        FakeTPOTClassifier.last_instance = self

    def fit(self, X, y):
        self.fit_shape = X.shape
        values, counts = np.unique(y, return_counts=True)
        self.majority = values[int(np.argmax(counts))]
        return self

    def predict(self, X):
        self.predict_shape = X.shape
        return np.repeat(self.majority, len(X))


def fake_search_space(group, **kwargs):
    return {"group": group, **kwargs}


def _write_case(tmp_path: Path, *, drop_test_rows=0, inject_nan=False):
    """Build a matched <id>.csv + prepared/ dir the way run_autodatapre would (fair mode)."""
    n = 100
    rng = np.random.RandomState(0)
    orig = pd.DataFrame({
        "f0": rng.rand(n),
        "f1": rng.rand(n),
        "target": ([0] * 50) + ([1] * 50),
    })
    csv_path = tmp_path / "999.csv"
    orig.to_csv(csv_path, index=False)

    # seed-42 positional split, same helper the evaluator uses
    tr, val, te = MODULE._split_positions(n, seed=42)
    keep_test = list(te)[drop_test_rows:]
    kept_rows = list(tr) + list(val) + keep_test

    feat = orig.loc[kept_rows, ["f0", "f1"]].reset_index(drop=True)
    if inject_nan:
        feat.loc[0, "f0"] = np.nan
    prepared = feat.copy()
    prepared["__adp_row__"] = kept_rows
    prepared["__adp_split__"] = (
        ["train"] * (len(tr) + len(val)) + ["test"] * len(keep_test)
    )
    pdir = tmp_path / "prepared"
    pdir.mkdir()
    prepared.to_csv(pdir / "prepared.csv", index=False)
    (pdir / "autodp_meta.json").write_text(json.dumps({
        "mode": "fair",
        "status": "prepared",
        "pipeline": ["scale_standard"],
        "search_seconds": 12.0,
        "dead_search": False,
    }))
    return str(csv_path), str(pdir)


def test_full_coverage_scores_and_carries_meta(tmp_path):
    csv_path, pdir = _write_case(tmp_path)
    result, _model = MODULE.score_prepared_tpot(
        csv_path, pdir,
        estimator_factory=FakeTPOTClassifier,
        search_space_factory=fake_search_space,
        verbose=0,
    )
    assert result["status"] == "ok"
    assert result["evaluator"] == "tpot"
    assert result["test_coverage"] == 1.0
    assert result["score"] == result["score_full"]
    assert result["score_full"] == result["score_kept"]
    assert result["autodp_search_seconds"] == 12.0
    assert result["dead_search"] is False
    assert result["tpot_space"] == "classifiers"
    inst = FakeTPOTClassifier.last_instance
    assert inst.kwargs["preprocessing"] is False
    assert inst.kwargs["validation_strategy"] == "none"


def test_dropped_test_rows_make_full_below_kept(tmp_path):
    csv_path, pdir = _write_case(tmp_path, drop_test_rows=5)
    result, _model = MODULE.score_prepared_tpot(
        csv_path, pdir,
        estimator_factory=FakeTPOTClassifier,
        search_space_factory=fake_search_space,
        verbose=0,
    )
    assert result["test_coverage"] < 1.0
    assert result["n_test_rows_kept"] < result["n_test_rows_expected"]
    # rows AutoDP dropped count as wrong -> score_full <= score_kept
    assert result["score_full"] <= result["score_kept"] + 1e-9


def test_nan_frame_raises_rather_than_imputing(tmp_path):
    csv_path, pdir = _write_case(tmp_path, inject_nan=True)
    with pytest.raises(ValueError, match="NaN"):
        MODULE.score_prepared_tpot(
            csv_path, pdir,
            estimator_factory=FakeTPOTClassifier,
            search_space_factory=fake_search_space,
            verbose=0,
        )
