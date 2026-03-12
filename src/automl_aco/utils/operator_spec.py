"""Utilities for parsing parameterized operator tokens.

Token format:
    "<operator>@k1=v1,k2=v2"

Examples:
    "knn@k=7"
    "pca@n=20"
    "zscore@z=2.5"
"""
from __future__ import annotations

from typing import Dict, Tuple


def parse_operator_spec(value: object) -> Tuple[str, Dict[str, str]]:
    """Return base operator name and parsed key/value params."""
    raw = str(value).strip()
    if "@" not in raw:
        return raw, {}

    base, payload = raw.split("@", 1)
    base = base.strip()
    params: Dict[str, str] = {}

    for chunk in payload.split(","):
        item = chunk.strip()
        if not item or "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key:
            params[key] = val
    return base, params


def base_operator_name(value: object) -> str:
    """Return base operator token without parameter payload."""
    return parse_operator_spec(value)[0]
