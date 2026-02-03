"""Dataset schemas and validation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import pandas as pd


@dataclass
class Dataset:
    id: Any
    name: str
    X: pd.DataFrame
    y: pd.Series
    task_type: Optional[str] = None


def ensure_dataset(obj: Any) -> Dataset:
    """Normalize input into Dataset dataclass."""
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, Mapping):
        if "X" in obj and "y" in obj:
            dataset_id = obj.get("id")
            name = obj.get("name") or (f"D_{dataset_id}" if dataset_id is not None else "dataset")
            X = pd.DataFrame(obj["X"]).copy()
            y = pd.Series(obj["y"]).copy()
            task_type = obj.get("task_type")
            return Dataset(id=dataset_id, name=name, X=X, y=y, task_type=task_type)
    raise ValueError("dataset must be Dataset or dict with keys 'X' and 'y'")
