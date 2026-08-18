"""Leakage-prevention tests: the 30 eval IDs must never enter the reference fit set."""
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
    # 4 reference datasets + 2 eval datasets (1066, 18) mixed in.
    cols = ["D_3", "D_5", "1066", "D_7", "18", "D_9"]
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
    assert "1066" not in [normalize_id(c) for c in perf_c.columns]
    assert "18" not in [normalize_id(c) for c in perf_c.columns]
    assert "1066" not in [normalize_id(i) for i in meta_c.index]
    assert "18" not in [normalize_id(i) for i in meta_c.index]
    # non-eval reference datasets preserved
    assert perf_c.shape[1] == 4 and meta_c.shape[0] == 4
    assert set(report["perf_cols_dropped"]) == {"1066", "18"}
    assert set(report["meta_rows_dropped"]) == {"1066", "18"}


def test_assert_disjoint_raises_on_contamination():
    # clean passes
    assert_disjoint(["D_3", "D_5", "D_7"], context="clean")
    # dirty raises loudly
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_disjoint(["D_3", "1066", "D_7"], context="dirty")


def test_holdout_postcondition_is_disjoint():
    perf, meta = _toy_reference()
    perf_c, meta_c, _ = holdout_reference(perf, meta)
    # The cleaned reference must satisfy the fit-boundary invariant.
    assert_disjoint(perf_c.columns, context="perf after holdout")
    assert_disjoint(meta_c.index, context="meta after holdout")


def test_eval_id_set_is_30_unique():
    assert len(EVAL_IDS) == 30  # paper Table 2: our 13 + DiffPrep's 18 minus shuttle
    assert len(EVAL_ID_SET) == len(set(EVAL_IDS))


# --- regression: pandas duplicate-column suffixes must not evade the holdout ---------------
#
# A dataset present twice in the performance matrix is de-duplicated by pandas into
# ``D_1037`` / ``D_1037.1``. normalize_id used to strip only ``.0+``, so the ``.1`` copy
# survived holdout_reference() *and* passed assert_disjoint() (both call normalize_id), making
# the leak invisible to its own guard. Six such columns exist in the shipped matrix and every
# one of them is an evaluation dataset.


def test_normalize_id_strips_duplicate_column_suffix():
    assert normalize_id("D_1037.1") == "1037"
    assert normalize_id("1037.1") == "1037"
    assert normalize_id("D_40685.12") == "40685"


def test_holdout_drops_duplicate_suffixed_eval_columns():
    cols = ["D_3", "D_1037", "D_1037.1", "D_5", "D_722.1"]
    perf = pd.DataFrame(
        np.random.RandomState(0).rand(2, len(cols)),
        index=["baseline", "simple"],
        columns=cols,
    )
    meta = pd.DataFrame(
        np.random.RandomState(1).rand(len(cols), 3),
        index=cols,
        columns=[f"mf{i}" for i in range(3)],
    )
    perf_c, meta_c, report = holdout_reference(perf, meta)
    assert list(perf_c.columns) == ["D_3", "D_5"]
    assert list(meta_c.index) == ["D_3", "D_5"]
    assert set(report["perf_cols_dropped"]) == {"1037", "722"}


def test_assert_disjoint_catches_duplicate_suffixed_id():
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_disjoint(["D_3", "D_1471.1"], context="dup-suffixed")


def test_arm_dataset_list_matches_eval_ids():
    """run_arms.OUR_DATASETS had silently drifted from EVAL_IDS; it must not drift again."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_arms.py"
    spec = importlib.util.spec_from_file_location("run_arms", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(mod.OUR_DATASETS) == list(EVAL_IDS)


def test_normalize_id_only_collapses_integer_decimals():
    """The .N fix must not mangle anything that is not an <int>.<int> dataset id."""
    for s in ["abc", "D_x", "1.2.3", "v1.2", "", "D_", "kc1-binary"]:
        assert normalize_id(s) == s


# --- metafeature computation for datasets absent from the OpenML table ---------------------


def test_metafeature_miss_no_longer_returns_empty_dict():
    """An empty dict became an all-ZERO metafeature vector downstream, silently."""
    from automl_aco.data.metafeatures import extract_enhanced_metafeatures

    table = pd.DataFrame(np.zeros((1, 3)), index=[7], columns=["a", "b", "c"])
    with pytest.raises(KeyError, match="no row in the metafeature table"):
        extract_enhanced_metafeatures({"id": 999999}, meta_features_df=table)


def test_metafeatures_computed_match_the_openml_table_conventions():
    """Computed rows must be on the SAME scale as the reference rows, or retrieval is garbage.

    Guards the three scale bugs inherited from the notebook: ErrRate held accuracy, ClassEntropy
    was in nats not bits, and the Percentage* columns were 0-1 rather than 0-100.
    """
    from automl_aco.data.metafeatures import compute_metafeatures_from_data

    rng = np.random.RandomState(0)
    n = 120
    X = pd.DataFrame({"num": rng.rand(n), "cat": rng.choice(list("abc"), n)})
    y = pd.Series(rng.choice([0, 1], n))
    m = compute_metafeatures_from_data(X, y)

    assert 0.0 <= m["ClassEntropy"] <= 1.0000001          # bits, 2 classes -> max 1.0
    assert 40.0 <= m["MajorityClassPercentage"] <= 100.0  # 0-100, not 0-1
    assert m["PercentageOfNumericFeatures"] + m["PercentageOfSymbolicFeatures"] == pytest.approx(100.0)
    assert 0.0 <= m["DecisionStumpErrRate"] <= 1.0
    # OpenML counts the target among the features: 2 predictors + target.
    assert m["NumberOfFeatures"] == 3
    assert m["NumberOfInstances"] == n


def test_acorec_arms_use_the_trained_siamese_not_cosine():
    """Without --train-metric-inline run_recommend silently falls back to COSINE similarity,
    which is the RQ2.1 ablation rather than ACORec. The arms must not run the ablation."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_arms.py"
    spec = importlib.util.spec_from_file_location("run_arms_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = type("A", (), dict(
        target_column="target", n_ants=4, n_iterations=3, protocol="native", time_limit=300,
        seed=42, acorec_config="ref", extra="", acorec_extra="",
    ))()
    cmd = mod.acorec_cmd("1066", pathlib.Path("/x.csv"), "ours", pathlib.Path("/w"), args)
    for required in ("--train-metric-inline", "--require-autogluon", "--aco-mmas-bounds"):
        assert required in cmd, f"{required} missing from the ACORec arm command"


# --------------------------------------------------------------------------- AutoDP meta-corpus

def test_adp_meta_corpus_excludes_eval_ids():
    """The retrained meta-learner corpus is a fitting step, so no eval dataset may enter it."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "build_adp_meta_corpus",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "build_adp_meta_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Args:
        ids = ""
        n_datasets = 500
        seed = 42
        shard = ""

    chosen = mod.choose_ids(_Args())
    assert chosen, "corpus selection returned nothing"
    leaked = [d for d in chosen if normalize_id(d) in EVAL_ID_SET]
    assert not leaked, f"eval datasets leaked into the AutoDP meta-corpus: {leaked}"


def test_adp_meta_corpus_rejects_eval_ids_passed_explicitly():
    """--ids must not be an escape hatch around the holdout."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "build_adp_meta_corpus",
        pathlib.Path(__file__).resolve().parent.parent / "scripts" / "build_adp_meta_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Args:
        ids = "1066,2,29"          # 1066 is an eval dataset
        n_datasets = 10
        seed = 42
        shard = ""

    with pytest.raises(Exception) as excinfo:
        mod.choose_ids(_Args())
    assert "1066" in str(excinfo.value) or "eval" in str(excinfo.value).lower()
