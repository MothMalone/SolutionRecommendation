"""Metafeature extraction interface and helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import pandas as pd


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

    if row.empty:
        return {}

    return row.iloc[0].to_dict()


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
