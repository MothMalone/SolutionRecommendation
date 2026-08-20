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


def test_subsample_keeps_every_class_above_proxy_minimum():
    """Unstratified sampling starved rare classes and cost us whole datasets (184: 0/10 scored)."""
    mod = _mod()
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    # 18 classes, heavily imbalanced -- several classes have only 4 rows in 20k.
    y = np.concatenate([rng.randint(0, 5, 19_000), np.repeat(np.arange(5, 18), 4)])
    df = pd.DataFrame(rng.randn(len(y), 6), columns=list("abcdef"))
    df["target"] = y

    sub = mod.subsample_preserving_classes(df, "target", 1500, seed=42)

    assert len(sub) <= 1500 + 18, f"subsample overshot: {len(sub)}"
    counts = sub["target"].value_counts()
    assert set(counts.index) == set(np.unique(y)), "a class disappeared from the subsample"
    assert counts.min() >= 3, f"class starved below the proxy minimum: {counts.min()}"


def test_subsample_is_a_noop_when_frame_is_small():
    mod = _mod()
    import pandas as pd
    df = pd.DataFrame({"a": range(10), "target": [0, 1] * 5})
    assert mod.subsample_preserving_classes(df, "target", 1500, 42) is df
    assert mod.subsample_preserving_classes(df, "target", 0, 42) is df


def test_subsample_survives_more_classes_than_budget():
    """Floor of 3/class can exceed n; must return something valid rather than crash."""
    mod = _mod()
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(0)
    y = np.repeat(np.arange(100), 5)
    df = pd.DataFrame(rng.randn(len(y), 3), columns=list("abc"))
    df["target"] = y
    sub = mod.subsample_preserving_classes(df, "target", 50, seed=42)
    assert sub["target"].nunique() == 100
    assert sub["target"].value_counts().min() >= 3


def test_continuous_target_detected_by_sklearn_not_by_heuristic():
    """A float target with few distinct values is still 'continuous' to sklearn.

    The first guard here checked dtype + cardinality, which let such datasets through; every
    sampled pipeline then failed with "Unknown label type: continuous", burning two full sampling
    rounds per dataset to rediscover what the target column already said.
    """
    import numpy as np
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target

    y = pd.Series(np.tile(np.arange(12) + 0.5, 100))   # 12 distinct, float
    assert type_of_target(y) == "continuous"
    # the discarded heuristic would NOT have flagged this
    assert not (y.dtype.kind == "f" and y.nunique() > max(20, 0.05 * len(y)))


def test_integer_multiclass_target_is_not_treated_as_regression():
    """Guard must not throw away legitimate multiclass datasets."""
    import numpy as np
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target

    y = pd.Series(np.tile(np.arange(12), 100))          # 12 distinct, integer
    assert type_of_target(y) == "multiclass"


def test_pool_is_restricted_to_classification_datasets():
    """The corpus feeds their CLASSIFICATION meta-learner; 57% of the library is regression.

    Sampling blind spent well over half of every build downloading and loading datasets only to
    discover the target was continuous. Observed yield went from 40% to 77% once the pool was
    filtered on NumberOfClasses.
    """
    import pandas as pd

    feats = pd.read_csv(REPO / "data" / "openml" / "dataset_feats.csv", index_col=0)
    assert "NumberOfClasses" in feats.columns, "metafeature table lost NumberOfClasses"
    n_reg = int((feats["NumberOfClasses"] == 0).sum())
    n_cls = int((feats["NumberOfClasses"] >= 2).sum())
    assert n_reg > 0 and n_cls > 0
    # the filter must actually remove a large share, or it is not doing its job
    assert n_reg / len(feats) > 0.3, f"expected many regression rows, got {n_reg}/{len(feats)}"

    src = (REPO / "scripts" / "build_adp_meta_corpus.py").read_text()
    assert 'feats["NumberOfClasses"] >= 2' in src, "classification filter missing from choose_ids"


def test_zero_scoring_round_stops_instead_of_retrying():
    """A round that scores nothing fails for structural reasons; a second round repeats it."""
    src = (REPO / "scripts" / "build_adp_meta_corpus.py").read_text()
    assert "if gained == 0:" in src, "fail-fast on a zero-scoring round is missing"


def test_pool_excludes_frames_far_wider_than_any_eval_dataset():
    """Proxy cost scales with COLUMNS, not just rows, and --score-max-rows only caps rows.

    Dataset 4136 (dexter, ~20k features) burned 1444s and scored nothing. In a 120-dataset sample
    11 sat above 5k features, up to 54,614 -- hours of guaranteed waste. They also cannot be
    retrieved: the widest evaluation dataset is madelon at 500 features, so a 50k-feature corpus
    row is never anyone's nearest neighbour.
    """
    import pandas as pd

    src = (REPO / "scripts" / "build_adp_meta_corpus.py").read_text()
    assert 'feats["NumberOfFeatures"] <= args.max_features' in src, "width filter missing"
    assert '"--max-features"' in src

    feats = pd.read_csv(REPO / "data" / "openml" / "dataset_feats.csv", index_col=0)
    wide = int((feats["NumberOfFeatures"] > 1000).sum())
    assert wide > 0, "expected some very wide datasets in the library"


def test_width_filter_keeps_the_widest_eval_dataset_in_range():
    """The default must not exclude datasets comparable to what we actually evaluate on."""
    import pandas as pd

    feats = pd.read_csv(REPO / "data" / "openml" / "dataset_feats.csv", index_col=0)
    # madelon (1485) is the widest evaluation dataset at ~500 features
    if 1485 in feats.index:
        assert feats.loc[1485, "NumberOfFeatures"] <= 1000, (
            "default --max-features would exclude datasets as wide as our own eval set")
