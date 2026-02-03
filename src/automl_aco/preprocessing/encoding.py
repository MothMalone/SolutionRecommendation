"""Encoding helpers."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

try:
    import category_encoders as ce
except Exception:  # pragma: no cover - optional dependency
    ce = None


def build_encoder(method: str) -> Optional[Any]:
    if method == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if method == "frequency":
        if ce is None:
            raise RuntimeError("category_encoders is required for frequency encoding")
        return ce.CountEncoder(normalize=True)
    if method == "count":
        if ce is None:
            raise RuntimeError("category_encoders is required for count encoding")
        return ce.CountEncoder(normalize=False)
    if method == "ordinal":
        if ce is None:
            raise RuntimeError("category_encoders is required for ordinal encoding")
        return ce.OrdinalEncoder()
    if method == "binary":
        if ce is None:
            raise RuntimeError("category_encoders is required for binary encoding")
        return ce.BinaryEncoder()
    if method == "none":
        return None
    if ce is None:
        raise RuntimeError("category_encoders is required for non-onehot encoding")
    return ce.OrdinalEncoder()
