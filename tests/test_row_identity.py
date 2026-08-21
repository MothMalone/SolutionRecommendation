"""Row identity must survive a row-dropping operator.

A pipeline that drops rows used to leave no record of WHICH rows survived: _fit_outlier_removal
dropped inline and bypassed _apply_keep_mask, so row_drop_log_ stayed empty and kept_positions_
was never set. scripts/autodp_our_space.py then renumbered the frame 0..m-1, run_autodatapre wrote
that as __adp_row__, and eval_autodatapre used it to attach each prepared row to a source label --
so labels landed on the wrong features. Measured effect: run_or_walk (binary) scored 0.4944, pure
chance, with all 6 features and 97.8% of rows present. After the fix, 0.9923 against arm 0's 0.993.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ROW_DROPPING = ["iqr", "zscore", "lof", "isolation_forest"]


def _frame(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 5), columns=list("abcde"))
    y = pd.Series(rng.randint(0, 2, n))
    return X, y


def _cfg(**kw):
    c = {k: "none" for k in ["imputation", "encoding", "scaling",
                             "feature_selection", "outlier_removal", "dimensionality_reduction"]}
    c.update(kw)
    return c


@pytest.mark.parametrize("op", ROW_DROPPING)
def test_kept_positions_reported_for_every_row_dropping_operator(op):
    from automl_aco.preprocessing.preprocessor import Preprocessor

    X, y = _frame()
    pre = Preprocessor(_cfg(outlier_removal=op), step_order=["outlier_removal"])
    res = pre.fit_transform(X.copy(), y.copy())
    Xp = res[0] if isinstance(res, tuple) else res

    kept = getattr(pre, "kept_positions_", None)
    if len(Xp) == len(X):
        return                                  # nothing dropped: tracking not required
    assert kept is not None, f"{op} dropped rows without recording which"
    assert len(kept) == len(Xp), f"{op}: {len(kept)} tracked vs {len(Xp)} returned"
    assert np.all(np.asarray(kept) < len(X)), f"{op}: positions out of range"
    assert len(set(np.asarray(kept).tolist())) == len(kept), f"{op}: duplicate positions"


@pytest.mark.parametrize("op", ROW_DROPPING)
def test_tracked_positions_select_the_rows_actually_returned(op):
    """kept_positions_ must name the surviving rows, not merely count them."""
    from automl_aco.preprocessing.preprocessor import Preprocessor

    X, y = _frame()
    marker = X.copy()
    marker["__id__"] = np.arange(len(X))        # ride along so we can identify survivors
    pre = Preprocessor(_cfg(outlier_removal=op), step_order=["outlier_removal"])
    res = pre.fit_transform(marker.copy(), y.copy())
    Xp = res[0] if isinstance(res, tuple) else res
    if len(Xp) == len(X) or "__id__" not in Xp.columns:
        pytest.skip("operator dropped nothing, or the marker column did not survive")

    kept = np.asarray(getattr(pre, "kept_positions_"))
    np.testing.assert_array_equal(np.asarray(Xp["__id__"], dtype=int), kept)


@pytest.mark.parametrize("op", ROW_DROPPING)
def test_row_drop_log_records_the_drop(op):
    """The inline path bypassed _apply_keep_mask, leaving row_drop_log_ silently empty."""
    from automl_aco.preprocessing.preprocessor import Preprocessor

    X, y = _frame()
    pre = Preprocessor(_cfg(outlier_removal=op), step_order=["outlier_removal"])
    res = pre.fit_transform(X.copy(), y.copy())
    Xp = res[0] if isinstance(res, tuple) else res
    log = getattr(pre, "row_drop_log_", None)
    assert log, f"{op}: row_drop_log_ empty despite going through outlier removal"
    assert log[-1]["kept"] == len(Xp)


def test_features_and_labels_stay_aligned_after_dropping():
    """X and y must still describe the same rows -- the invariant the classifier depends on."""
    from automl_aco.preprocessing.preprocessor import Preprocessor

    X, y = _frame()
    X = X.copy()
    X["__id__"] = np.arange(len(X))
    y = pd.Series(np.arange(len(X)))            # label == row id, so misalignment is visible
    pre = Preprocessor(_cfg(outlier_removal="iqr"), step_order=["outlier_removal"])
    res = pre.fit_transform(X.copy(), y.copy())
    Xp, yp = res if isinstance(res, tuple) else (res, y)
    if "__id__" not in Xp.columns or len(Xp) == len(X):
        pytest.skip("nothing dropped or marker not preserved")
    np.testing.assert_array_equal(np.asarray(Xp["__id__"], dtype=int),
                                  np.asarray(yp, dtype=int))
