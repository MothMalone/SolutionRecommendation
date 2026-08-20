"""The meta-corpus must carry signal, not artifacts of how it was sampled.

Two defects found on the first real Kaggle build, both silent:
  1. `imputation: none` was sampled on frames WITH missing values. The proxy rejects those
     ("missing values require non-'none' imputation"), so ~50% of evaluations were wasted -- and
     each rejection was written to label.csv as EvaluationMetric 0.0, i.e. as though a valid
     pipeline were the worst available. Their 1-NN takes idxmax of a neighbour's block, so those
     zeros actively distorted the task order it learned.
  2. Datasets whose pipelines all tie (dataset 51: five scores, every one 0.8275862) contribute an
     arbitrary "best" pipeline -- idxmax over equal values is just the first row.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_adp_meta_corpus", REPO / "scripts" / "build_adp_meta_corpus.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_never_skips_imputation_when_data_has_missing():
    mod = _mod()
    rng = random.Random(0)
    for _ in range(200):
        _, cfg = mod.sample_pipeline(rng, has_missing=True, has_categorical=False)
        assert cfg["imputation"] != "none", "sampled imputation:none on a frame with missing values"


def test_never_skips_encoding_when_data_has_categoricals():
    mod = _mod()
    rng = random.Random(0)
    for _ in range(200):
        _, cfg = mod.sample_pipeline(rng, has_missing=False, has_categorical=True)
        assert cfg["encoding"] != "none", "sampled encoding:none on a frame with categoricals"


def test_still_skips_families_when_data_permits():
    """The *_null slots must remain reachable on clean numeric data, or every pipeline is full."""
    mod = _mod()
    rng = random.Random(0)
    skipped = set()
    for _ in range(400):
        _, cfg = mod.sample_pipeline(rng, has_missing=False, has_categorical=False)
        skipped |= {f for f, op in cfg.items() if op == "none"}
    assert "imputation" in skipped and "encoding" in skipped, (
        f"families never switched off on clean data: {skipped}")


def test_describe_frame_detects_missing_and_categorical():
    mod = _mod()
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"], "target": [0, 1, 0]})
    assert mod.describe_frame(df, "target") == {"has_missing": True, "has_categorical": True}

    clean = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "target": [0, 1]})
    assert mod.describe_frame(clean, "target") == {"has_missing": False, "has_categorical": False}


def test_describe_frame_ignores_target_column():
    """A missing/categorical TARGET must not constrain preprocessing of the features."""
    mod = _mod()
    df = pd.DataFrame({"a": [1.0, 2.0], "target": ["yes", "no"]})
    assert mod.describe_frame(df, "target")["has_categorical"] is False


def test_time_budget_stops_starting_new_datasets(tmp_path, monkeypatch):
    """A 12h Kaggle session must be able to bound the build; overrunning loses everything."""
    mod = _mod()
    import argparse

    # Two ids, a budget already exhausted -> nothing should be attempted.
    argv = ["--out-dir", str(tmp_path / "c"), "--ids", "2,29",
            "--time-budget", "0.0001", "--local-dir", str(REPO / "data" / "eval_datasets")]
    monkeypatch.setattr("sys.argv", ["build_adp_meta_corpus.py"] + argv)
    rc = mod.main()
    # No dataset finished, so no CSVs are written and it reports that rather than crashing.
    assert rc == 1
    assert not (tmp_path / "c" / "label.csv").exists()


def test_score_max_rows_does_not_shrink_the_cached_frame(tmp_path):
    """Metafeature #1 is row count. Subsampling for scoring must not reach the cached CSV."""
    mod = _mod()
    import pandas as pd
    import numpy as np

    rng = np.random.RandomState(0)
    big = pd.DataFrame(rng.randn(3000, 4), columns=list("abcd"))
    big["target"] = rng.randint(0, 2, 3000)
    src = tmp_path / "datasets"
    src.mkdir()
    big.to_csv(src / "999.csv", index=False)

    loaded = mod.load_table("999", [src], "target")
    assert len(loaded) == 3000, "load_table must return the full frame; subsampling is scoring-only"
