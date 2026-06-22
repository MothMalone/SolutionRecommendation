"""Leakage-prevention tests: the 24 eval IDs must never enter the reference fit set."""
import numpy as np
import pandas as pd
import pytest

from automl_aco.eval_ids import (
    EVAL_IDS,
    EVAL_ID_SET,
    assert_disjoint,
    holdout_reference,
    is_eval_id,
    normalize_id,
)


def test_normalize_id_canonicalizes_variants():
    assert normalize_id("D_248") == "248"
    assert normalize_id("248.0") == "248"
    assert normalize_id(248) == "248"
    assert normalize_id(248.0) == "248"
    assert normalize_id("dataset_18") == "18"
    assert normalize_id("openml-31") == "31"


def test_is_eval_id():
    assert is_eval_id("D_378") and is_eval_id(378) and is_eval_id("378.0")
    assert not is_eval_id("D_999999")


def _toy_reference():
    # 4 reference datasets + 2 eval datasets (248, 18) mixed in.
    cols = ["D_3", "D_5", "248", "D_7", "18", "D_9"]
    perf = pd.DataFrame(
        np.random.RandomState(0).rand(3, len(cols)),
        index=["baseline", "simple", "robust"],
        columns=cols,
    )
    meta = pd.DataFrame(
        np.random.RandomState(1).rand(len(cols), 4),
        index=cols,
        columns=[f"mf{i}" for i in range(4)],
    )
    return perf, meta


def test_holdout_removes_all_eval_ids():
    perf, meta = _toy_reference()
    perf_c, meta_c, report = holdout_reference(perf, meta)
    # eval IDs gone from both
    assert "248" not in [normalize_id(c) for c in perf_c.columns]
    assert "18" not in [normalize_id(c) for c in perf_c.columns]
    assert "248" not in [normalize_id(i) for i in meta_c.index]
    assert "18" not in [normalize_id(i) for i in meta_c.index]
    # non-eval reference datasets preserved
    assert perf_c.shape[1] == 4 and meta_c.shape[0] == 4
    assert set(report["perf_cols_dropped"]) == {"248", "18"}
    assert set(report["meta_rows_dropped"]) == {"248", "18"}


def test_assert_disjoint_raises_on_contamination():
    # clean passes
    assert_disjoint(["D_3", "D_5", "D_7"], context="clean")
    # dirty raises loudly
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_disjoint(["D_3", "248", "D_7"], context="dirty")


def test_holdout_postcondition_is_disjoint():
    perf, meta = _toy_reference()
    perf_c, meta_c, _ = holdout_reference(perf, meta)
    # The cleaned reference must satisfy the fit-boundary invariant.
    assert_disjoint(perf_c.columns, context="perf after holdout")
    assert_disjoint(meta_c.index, context="meta after holdout")


def test_eval_id_set_has_24_unique():
    assert len(EVAL_IDS) == 23  # 23 distinct IDs in the declared list
    assert len(EVAL_ID_SET) == len(set(EVAL_IDS))
