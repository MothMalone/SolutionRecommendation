"""Canonical evaluation-dataset IDs and leakage-prevention helpers.

The 30 evaluation datasets are TEST data. They must never enter any step that fits,
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

# ---------------------------------------------------------------------------
# The 30 evaluation datasets (paper Table 2), name -> id. Fixed by the supervisor:
# our 13, plus DiffPrep's 18 minus `shuttle`.
#
# The previous 23-ID set is recorded in EVAL_IDS_LEGACY_23 below. Ten of it were dropped
# (248 1164 2 1387 184 381 382 993 29 31) and 17 DiffPrep datasets added, so any result
# produced against the old set is not comparable -- see docs/DATASET_CHANGE_AND_RQ3.md.
# ---------------------------------------------------------------------------
EVAL_DATASETS: "dict[str, str]" = {
    # --- our 13 (OpenML) ---
    "kc1-binary": "1066",
    "usp05": "1047",
    "sleuth-ex2016": "862",
    "calendarDOW": "40663",
    "mc2": "1054",
    "fri-c1": "876",
    "mfeat-morphological": "18",
    "robot-failures-lp5": "1520",
    "autoUniv-au4": "1548",
    "ipums-la-99": "378",
    "madelon": "1485",
    "mfeat-fourier": "14",
    "colic": "27",
    # --- DiffPrep's 17 (loaded from the DiffPrep CSV folder, github.com/chu-data-lab/DiffPrep) ---
    "abalone": "44956",
    "ada_prior": "1037",
    "avila": "42932",
    "connect-4": "40668",
    "eeg": "1471",
    "google": "100000",          # SYNTHETIC -- not on OpenML, see DIFFPREP_SYNTHETIC_IDS
    "house": "42165",
    "jungle_chess": "41001",
    "micro": "41671",            # microaggregation2 -- corroborated, see note below
    "mozilla4": "1046",
    "obesity": "46597",
    "page-blocks": "30",
    "pbcseq": "802",
    "pol": "722",
    "run_or_walk": "40922",
    "uscensus": "1119",
    "wall-robot-nav": "1497",
}

# `google` is the Kaggle Google-Play-Store apps table with a derived binary target
# (`Rating>4.2`); it has no OpenML entry. DiffPrep's claim that all 18 come from OpenML is
# loose. The id below is arbitrary-but-frozen and follows the notebook's convention; it must
# never collide with a real OpenML id.
DIFFPREP_SYNTHETIC_IDS = frozenset({"100000"})

# `micro` = DiffPrep's `microaggregation2`. Not confirmed against the OpenML API (openml.org was
# unreachable), but corroborated locally on four independent quantities: dataset_feats.csv row
# 41671 reads 20,000 instances / 21 features / 20 numeric + 1 symbolic / 5 classes, and DiffPrep's
# microaggregation2/data.csv is exactly `a1..a20 + class` with 5 classes. Nothing else in the
# reference library has that signature. Treated as confirmed.
DIFFPREP_UNVERIFIED_IDS: frozenset = frozenset()

EVAL_IDS: Tuple[str, ...] = tuple(EVAL_DATASETS.values())
EVAL_ID_SET = frozenset(EVAL_IDS)
assert len(EVAL_ID_SET) == len(EVAL_IDS) == 30, "duplicate or missing evaluation id"

# The pre-2026-08 evaluation set, kept only so old outputs can be identified as stale.
EVAL_IDS_LEGACY_23: Tuple[str, ...] = (
    "248", "1066", "1164", "1047", "862", "2", "40663", "1054", "1387", "876",
    "18", "1520", "1548", "184", "378", "381", "382", "993", "1485", "14",
    "27", "29", "31",
)


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
    # Strip an optional D_/dataset_/openml_ prefix, then an optional decimal suffix.
    #
    # The decimal suffix matters for leakage, not cosmetics. When a dataset appears twice in the
    # performance matrix, pandas de-duplicates the column names as ``D_1037`` / ``D_1037.1``. The
    # old pattern only matched ``.0+`` (so ``248.0`` -> ``248``) and left ``D_1037.1`` untouched,
    # which meant such a column survived holdout_reference() AND passed assert_disjoint(), since
    # both go through this function. Six columns in the shipped matrix are exactly this shape
    # (D_1037.1 D_1471.1 D_1046.1 D_802.1 D_722.1 D_40685.1); five are evaluation datasets and the
    # sixth, 40685/shuttle, is legitimately reference-only and must stay.
    # Dataset ids are integers, so collapsing any ``<int>.<int>`` to ``<int>`` is safe here.
    m = re.fullmatch(r"(?i)(?:d|dataset|openml)[_\-: ]*([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        s = m.group(1)
    m = re.fullmatch(r"([0-9]+)\.[0-9]+", s)
    if m:
        return m.group(1)
    return s


def is_eval_id(val: Any) -> bool:
    return normalize_id(val) in EVAL_ID_SET


def normalize_ids(ids: Iterable[Any]) -> "frozenset[str]":
    """Normalize an iterable of dataset identifiers to a set of bare integer strings."""
    return frozenset(normalize_id(x) for x in ids if normalize_id(x))


def assert_disjoint(
    ids: Iterable[Any],
    *,
    context: str = "reference set",
    extra_ids: Iterable[Any] = (),
) -> None:
    """Raise if any held-out ID appears in ``ids``. Use before every fit boundary.

    ``extra_ids`` extends the forbidden set beyond ``EVAL_IDS``. It exists for the
    cross-comparison arms that evaluate on a *different* dataset list (AutoDP's own ten,
    ``run_arms.THEIR_DATASETS``): those ids are not in ``EVAL_IDS``, so the default holdout
    does not touch them, yet five of them are columns of the shipped performance matrix.
    Without this, ACORec retrieves the target dataset's own best pipeline.
    """
    forbidden = EVAL_ID_SET | normalize_ids(extra_ids)
    contaminated = sorted({normalize_id(x) for x in ids} & forbidden)
    if contaminated:
        raise AssertionError(
            f"LEAKAGE: {len(contaminated)} held-out ID(s) found in {context}: "
            f"{contaminated}. Held-out datasets must be excluded from every fit/normalize/train/"
            f"neighbor step. Run holdout_reference() before constructing the recommender."
        )


def holdout_reference(
    performance_matrix: pd.DataFrame,
    metafeatures_df: pd.DataFrame,
    *,
    verbose: bool = False,
    extra_ids: Iterable[Any] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return copies of (perf, meta) with all 30 eval IDs -- plus ``extra_ids`` -- removed.

    - ``performance_matrix``: pipelines (rows) x datasets (columns). Held-out columns dropped.
    - ``metafeatures_df``: datasets (rows) x metafeatures (columns). Held-out rows dropped.

    The returned frames are for *constructing the recommender* (training, normalization,
    neighbor pool). Callers should retain the ORIGINAL full metafeatures table for target-row
    lookup of the dataset under evaluation.

    ``extra_ids`` is for arms whose evaluation set is not ``EVAL_IDS`` -- see ``assert_disjoint``.
    """
    extra = normalize_ids(extra_ids)
    forbidden = EVAL_ID_SET | extra

    perf_drop = [c for c in performance_matrix.columns if normalize_id(c) in forbidden]
    meta_drop = [i for i in metafeatures_df.index if normalize_id(i) in forbidden]

    perf_clean = performance_matrix.drop(columns=perf_drop, errors="ignore").copy()
    meta_clean = metafeatures_df.drop(index=meta_drop, errors="ignore").copy()

    # Post-condition: the cleaned reference is disjoint from every held-out id.
    assert_disjoint(perf_clean.columns, context="performance_matrix columns after holdout",
                    extra_ids=extra)
    assert_disjoint(meta_clean.index, context="metafeatures index after holdout", extra_ids=extra)

    dropped_perf = sorted({normalize_id(c) for c in perf_drop})
    dropped_meta = sorted({normalize_id(i) for i in meta_drop})
    report = {
        "eval_ids_total": len(EVAL_IDS),
        "extra_ids_requested": sorted(extra),
        "perf_cols_dropped": dropped_perf,
        "meta_rows_dropped": dropped_meta,
        # Broken out so a run log shows what the arm-specific holdout actually caught, rather
        # than burying it in the eval-ID total.
        "extra_perf_cols_dropped": sorted(set(dropped_perf) & extra),
        "extra_meta_rows_dropped": sorted(set(dropped_meta) & extra),
        "perf_cols_before": int(performance_matrix.shape[1]),
        "perf_cols_after": int(perf_clean.shape[1]),
        "meta_rows_before": int(metafeatures_df.shape[0]),
        "meta_rows_after": int(meta_clean.shape[0]),
    }
    if verbose:
        print(
            f"[leakage-holdout] dropped {len(report['perf_cols_dropped'])} held-out cols from perf "
            f"({report['perf_cols_before']}->{report['perf_cols_after']}), "
            f"{len(report['meta_rows_dropped'])} held-out rows from metafeatures "
            f"({report['meta_rows_before']}->{report['meta_rows_after']})."
        )
        if extra:
            print(
                f"[leakage-holdout] of those, {len(report['extra_perf_cols_dropped'])} perf col(s) "
                f"{report['extra_perf_cols_dropped']} and "
                f"{len(report['extra_meta_rows_dropped'])} meta row(s) "
                f"{report['extra_meta_rows_dropped']} came from --holdout-ids."
            )
    return perf_clean, meta_clean, report
