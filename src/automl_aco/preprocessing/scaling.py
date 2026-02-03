"""Scaling helpers."""
from __future__ import annotations

from typing import Optional

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


def build_scaler(method: str) -> Optional[object]:
    return {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "maxabs": MaxAbsScaler(),
    }.get(method)
