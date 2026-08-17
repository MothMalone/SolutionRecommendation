"""Metafeature extraction interface and helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Landmarking columns the OpenML table carries but that we do not recompute. Left as NaN so the
# recommender's imputer treats them as missing rather than as a real value of 0.
_UNCOMPUTED_COLUMNS = (
    "CfsSubsetEval_DecisionStumpAUC", "CfsSubsetEval_DecisionStumpErrRate",
    "CfsSubsetEval_DecisionStumpKappa", "CfsSubsetEval_NaiveBayesAUC",
    "CfsSubsetEval_NaiveBayesErrRate", "CfsSubsetEval_NaiveBayesKappa",
    "CfsSubsetEval_kNN1NAUC", "CfsSubsetEval_kNN1NErrRate", "CfsSubsetEval_kNN1NKappa",
    "J48.00001.AUC", "J48.00001.ErrRate", "J48.00001.Kappa",
    "J48.0001.AUC", "J48.0001.ErrRate", "J48.0001.Kappa",
    "J48.001.AUC", "J48.001.ErrRate", "J48.001.Kappa",
    "REPTreeDepth1AUC", "REPTreeDepth1ErrRate", "REPTreeDepth1Kappa",
    "REPTreeDepth2AUC", "REPTreeDepth2ErrRate", "REPTreeDepth2Kappa",
    "REPTreeDepth3AUC", "REPTreeDepth3ErrRate", "REPTreeDepth3Kappa",
    "RandomTreeDepth1AUC", "RandomTreeDepth1ErrRate", "RandomTreeDepth1Kappa",
    "RandomTreeDepth2AUC", "RandomTreeDepth2ErrRate", "RandomTreeDepth2Kappa",
    "RandomTreeDepth3AUC", "RandomTreeDepth3ErrRate", "RandomTreeDepth3Kappa",
    "AutoCorrelation",
)


def compute_metafeatures_from_data(X, y, *, seed: int = 42) -> Dict[str, Any]:
    """Compute OpenML-style metafeatures from raw data, for datasets absent from the table.

    Needed because 8 of the 30 evaluation datasets (the DiffPrep ones -- avila, google, house,
    jungle_chess, micro, uscensus, abalone, obesity) have no row in ``dataset_feats.csv``.

    Ported from the Kaggle notebook's ``extract_69_metafeatures`` with **one deliberate
    correction**. The notebook computed ``err = 1 - mean(accuracy)`` and then stored ``1 - err``
    -- i.e. *accuracy* -- into the ``*ErrRate`` fields. The reference table's ErrRate columns are
    genuine error rates (measured: DecisionStumpErrRate mean 0.321, median 0.278), so the
    notebook's values were on the opposite scale from every dataset they would be compared
    against, which would systematically misplace these datasets in the retrieval space. We store
    the error rate. The rationale is consistency with the reference table and is checkable
    without reference to any test accuracy.
    """
    from scipy.stats import entropy
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()
    meta: Dict[str, Any] = {}

    n_instances, n_predictors = X.shape
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    symbolic_cols = X.select_dtypes(exclude=[np.number]).columns

    # OpenML counts the TARGET among the features: on dataset 27 the table reads 23 features /
    # 16 symbolic where X alone has 22 / 15. Matching that convention keeps the counts and every
    # percentage denominator on the same basis as the reference rows. Landmarking below still
    # uses X only.
    target_is_numeric = pd.api.types.is_numeric_dtype(y)
    n_features = n_predictors + 1
    n_numeric = len(numeric_cols) + (1 if target_is_numeric else 0)
    n_symbolic = len(symbolic_cols) + (0 if target_is_numeric else 1)

    meta["NumberOfInstances"] = n_instances
    meta["NumberOfFeatures"] = n_features
    meta["NumberOfNumericFeatures"] = n_numeric
    meta["NumberOfSymbolicFeatures"] = n_symbolic
    meta["NumberOfBinaryFeatures"] = int(
        sum(X[c].nunique(dropna=True) == 2 for c in X.columns)
        + (1 if y.nunique(dropna=True) == 2 else 0)
    )
    # The OpenML table stores every Percentage* column on a 0-100 scale (verified: all seven have
    # max ~100 and means like 57.0 / 83.3). The notebook emitted 0-1 fractions, a 100x error that
    # would dominate any distance computed against the reference rows.
    meta["PercentageOfNumericFeatures"] = 100.0 * n_numeric / max(n_features, 1)
    meta["PercentageOfSymbolicFeatures"] = 100.0 * n_symbolic / max(n_features, 1)
    meta["PercentageOfBinaryFeatures"] = 100.0 * meta["NumberOfBinaryFeatures"] / max(n_features, 1)
    # The OpenML table defines Dimensionality as features per instance; the notebook left it NaN.
    meta["Dimensionality"] = n_features / max(n_instances, 1)

    n_missing = int(X.isna().sum().sum())
    n_inst_missing = int(X.isna().any(axis=1).sum())
    meta["NumberOfMissingValues"] = n_missing
    meta["NumberOfInstancesWithMissingValues"] = n_inst_missing
    meta["PercentageOfMissingValues"] = 100.0 * n_missing / max(n_instances * n_predictors, 1)
    meta["PercentageOfInstancesWithMissingValues"] = 100.0 * n_inst_missing / max(n_instances, 1)

    y_enc = LabelEncoder().fit_transform(y)
    class_counts = np.bincount(y_enc)
    probs = class_counts / class_counts.sum()
    meta["NumberOfClasses"] = len(class_counts)
    meta["MajorityClassSize"] = int(class_counts.max())
    meta["MinorityClassSize"] = int(class_counts.min())
    meta["MajorityClassPercentage"] = 100.0 * float(probs.max())
    meta["MinorityClassPercentage"] = 100.0 * float(probs.min())
    # base 2, not nats. The notebook used scipy's default (natural log); on dataset 27 that gives
    # 0.659 against OpenML's 0.950, and 0.659 / ln(2) = 0.951 -- the table is in bits.
    meta["ClassEntropy"] = float(entropy(probs, base=2))

    if len(symbolic_cols) > 0:
        distinct = X[symbolic_cols].nunique(dropna=True)
        meta["MaxNominalAttDistinctValues"] = float(distinct.max())
        meta["MinNominalAttDistinctValues"] = float(distinct.min())
        meta["MeanNominalAttDistinctValues"] = float(distinct.mean())
        meta["StdvNominalAttDistinctValues"] = float(distinct.std())
    else:
        for k in ("Max", "Min", "Mean", "Stdv"):
            meta[f"{k}NominalAttDistinctValues"] = np.nan

    # --- landmarking ---
    n_splits = int(min(3, np.min(class_counts))) if len(class_counts) else 0
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_cols),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), symbolic_cols),
            ],
            remainder="drop",
        )

        def eval_model(model):
            aucs, accs, kappas = [], [], []
            for tr, te in cv.split(X, y_enc):
                pipe = Pipeline([("prep", preprocessor), ("mdl", model)])
                pipe.fit(X.iloc[tr], y_enc[tr])
                preds = pipe.predict(X.iloc[te])
                accs.append(accuracy_score(y_enc[te], preds))
                kappas.append(cohen_kappa_score(y_enc[te], preds))
                if len(np.unique(y_enc)) == 2:
                    aucs.append(roc_auc_score(y_enc[te], pipe.predict_proba(X.iloc[te])[:, 1]))
                else:
                    aucs.append(np.nan)
            with np.errstate(invalid="ignore"):
                auc = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else np.nan
            # ErrRate is an ERROR rate, matching the reference table. See the docstring.
            return auc, float(1.0 - np.nanmean(accs)), float(np.nanmean(kappas))

        for prefix, model in (
            ("DecisionStump", DecisionTreeClassifier(max_depth=1, random_state=seed)),
            ("NaiveBayes", GaussianNB()),
            ("kNN1N", KNeighborsClassifier(n_neighbors=1)),
        ):
            try:
                auc, err, kap = eval_model(model)
            except Exception as exc:  # a landmark failing must not lose the whole vector
                logger.warning("landmark %s failed: %s", prefix, exc)
                auc = err = kap = np.nan
            meta[f"{prefix}AUC"] = auc
            meta[f"{prefix}ErrRate"] = err
            meta[f"{prefix}Kappa"] = kap
    else:
        logger.warning(
            "smallest class has %s member(s); skipping landmarking metafeatures",
            int(np.min(class_counts)) if len(class_counts) else 0,
        )
        for prefix in ("DecisionStump", "NaiveBayes", "kNN1N"):
            for suffix in ("AUC", "ErrRate", "Kappa"):
                meta[f"{prefix}{suffix}"] = np.nan

    for col in _UNCOMPUTED_COLUMNS:
        meta.setdefault(col, np.nan)
    return meta


def load_metafeatures_csv(path: str, index_col: int = 0) -> pd.DataFrame:
    """Load metafeatures from CSV with index column."""
    df = pd.read_csv(path, index_col=index_col)
    return df


def extract_enhanced_metafeatures(
    dataset: Dict[str, Any],
    meta_features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Fetch precomputed metafeatures for a dataset by id.

    Matches the notebook behavior: lookup by dataset['id'] and return row as dict.
    """
    if meta_features_df is None:
        raise ValueError("meta_features_df must be provided for extract_enhanced_metafeatures")

    dataset_id = dataset.get("id") if isinstance(dataset, dict) else None
    if dataset_id is None:
        raise ValueError("Dataset does not have an 'id' field")

    try:
        row = meta_features_df.loc[[dataset_id]]
    except KeyError:
        row = pd.DataFrame()

    if not row.empty:
        return row.iloc[0].to_dict()

    # MISS. This used to `return {}`, which the recommender turned into an all-zero metafeature
    # vector via `pd.DataFrame([{}]).reindex(columns=..., fill_value=0)` -- silently, with no
    # exception and no warning. The dataset was then embedded at the origin and its "behaviourally
    # similar" neighbours were meaningless. Eight of the 30 evaluation datasets are absent from
    # dataset_feats.csv, so this was not a rare path.
    if "X" in dataset and "y" in dataset:
        logger.warning(
            "dataset %s is not in the metafeature table; computing metafeatures from raw data",
            dataset_id,
        )
        return compute_metafeatures_from_data(dataset["X"], dataset["y"])

    raise KeyError(
        f"dataset {dataset_id!r} has no row in the metafeature table and carries no 'X'/'y' to "
        "compute one from. Returning an empty dict here would silently embed it at the origin. "
        "Either add it to dataset_feats.csv or pass the loaded data."
    )


def build_metafeatures_matrix(
    datasets: Iterable[Dict[str, Any]],
    meta_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build metafeatures matrix for a list of datasets."""
    metafeatures_list = []
    dataset_names = []

    for dataset in datasets:
        metafeatures = extract_enhanced_metafeatures(dataset, meta_features_df=meta_features_df)
        if metafeatures:
            metafeatures_list.append(metafeatures)
            dataset_names.append(dataset.get("name", str(dataset.get("id"))))

    if metafeatures_list:
        return pd.DataFrame(metafeatures_list, index=dataset_names)
    return pd.DataFrame()
