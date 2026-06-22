"""Canonical evaluation-dataset IDs and leakage-prevention helpers.

The 24 evaluation datasets are TEST data. They must never enter any step that fits,
normalizes, trains, or selects on their values: Siamese training, similarity-target
normalization, metafeature imputer/scaler fitting, the neighbor pool, or Eq-7 heuristic
aggregation.

This module is the single source of truth for those IDs plus two helpers:

- ``holdout_reference(perf, meta)``  -> cleaned reference copies with all eval IDs removed
  (used to *construct* the recommender). The *full* metafeatures table is kept separately by
  the caller for target-row lookup, since a new dataset's metafeatures are read from the same
  precomputed file (see ``data/metafeatures.extract_enhanced_metafeatures``).
- ``assert_disjoint(ids)`` -> raise loudly if any eval ID is present in a set meant to be
  reference-only.

Leakage policy (decided with the user): a single shared, frozen Siamese is trained once. To
keep that shared model genuinely leak-free without per-fold retraining, the reference used for
training / normalization / neighbor retrieval excludes ALL 24 eval IDs. At inference the current
query is additionally excluded via ``query_dataset_id`` (defense in depth).
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Tuple

import pandas as pd

# All 24 evaluation dataset OpenML IDs (pre-declared, fixed).
EVAL_IDS: Tuple[str, ...] = (
    "248", "1066", "1164", "1047", "862", "2", "40663", "1054", "1387", "876",
    "18", "1520", "1548", "184", "378", "381", "382", "993", "1485", "14",
    "27", "29", "31",
)
EVAL_ID_SET = frozenset(EVAL_IDS)


def normalize_id(val: Any) -> str:
    """Normalize a dataset identifier to its bare integer string.

    Mirrors the recommender's ``_normalize_id`` so that ``D_248``, ``248``, ``248.0`` and
    ``dataset_248`` all map to ``"248"``.
    """
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == val and abs(val - round(val)) <= 1e-9:  # finite & integral
            return str(int(round(val)))
        return str(val).strip()
    s = str(val).strip()
    m = re.fullmatch(r"([0-9]+)\.0+", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+)", s)
    if m:
        return m.group(1)
    return s


def is_eval_id(val: Any) -> bool:
    return normalize_id(val) in EVAL_ID_SET


def assert_disjoint(ids: Iterable[Any], *, context: str = "reference set") -> None:
    """Raise if any eval ID appears in ``ids``. Use before every fit boundary."""
    contaminated = sorted({normalize_id(x) for x in ids} & EVAL_ID_SET)
    if contaminated:
        raise AssertionError(
            f"LEAKAGE: {len(contaminated)} evaluation ID(s) found in {context}: "
            f"{contaminated}. Eval datasets must be held out of every fit/normalize/train/"
            f"neighbor step. Run holdout_reference() before constructing the recommender."
        )


def holdout_reference(
    performance_matrix: pd.DataFrame,
    metafeatures_df: pd.DataFrame,
    *,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return copies of (perf, meta) with all 24 eval IDs removed.

    - ``performance_matrix``: pipelines (rows) x datasets (columns). Eval-ID columns dropped.
    - ``metafeatures_df``: datasets (rows) x metafeatures (columns). Eval-ID rows dropped.

    The returned frames are for *constructing the recommender* (training, normalization,
    neighbor pool). Callers should retain the ORIGINAL full metafeatures table for target-row
    lookup of the dataset under evaluation.
    """
    perf_drop = [c for c in performance_matrix.columns if normalize_id(c) in EVAL_ID_SET]
    meta_drop = [i for i in metafeatures_df.index if normalize_id(i) in EVAL_ID_SET]

    perf_clean = performance_matrix.drop(columns=perf_drop, errors="ignore").copy()
    meta_clean = metafeatures_df.drop(index=meta_drop, errors="ignore").copy()

    # Post-condition: the cleaned reference is disjoint from EVAL_IDS.
    assert_disjoint(perf_clean.columns, context="performance_matrix columns after holdout")
    assert_disjoint(meta_clean.index, context="metafeatures index after holdout")

    report = {
        "eval_ids_total": len(EVAL_IDS),
        "perf_cols_dropped": sorted({normalize_id(c) for c in perf_drop}),
        "meta_rows_dropped": sorted({normalize_id(i) for i in meta_drop}),
        "perf_cols_before": int(performance_matrix.shape[1]),
        "perf_cols_after": int(perf_clean.shape[1]),
        "meta_rows_before": int(metafeatures_df.shape[0]),
        "meta_rows_after": int(meta_clean.shape[0]),
    }
    if verbose:
        print(
            f"[leakage-holdout] dropped {len(report['perf_cols_dropped'])} eval cols from perf "
            f"({report['perf_cols_before']}->{report['perf_cols_after']}), "
            f"{len(report['meta_rows_dropped'])} eval rows from metafeatures "
            f"({report['meta_rows_before']}->{report['meta_rows_after']})."
        )
    return perf_clean, meta_clean, report
