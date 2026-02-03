"""I/O utilities for performance matrices and configs."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_performance_matrix_available(
    csv_path: str,
    remove_ratio: float = 0.0,
    random_state: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    try:
        performance_matrix = pd.read_csv(csv_path, index_col=0)
        logger.info("Loaded performance matrix from %s", csv_path)

        current_missing_ratio = performance_matrix.isna().sum().sum() / performance_matrix.size
        logger.info("Current missing ratio: %.2f%%", current_missing_ratio * 100)

        if remove_ratio > 0 and current_missing_ratio < remove_ratio:
            np.random.seed(random_state)
            total_cells = performance_matrix.size
            target_missing_cells = int(total_cells * remove_ratio)
            current_missing_cells = int(total_cells * current_missing_ratio)
            n_remove = target_missing_cells - current_missing_cells

            if n_remove > 0:
                non_nan_positions = np.argwhere(~performance_matrix.isna().values)
                chosen_indices = non_nan_positions[
                    np.random.choice(len(non_nan_positions), n_remove, replace=False)
                ]
                for r, c in chosen_indices:
                    performance_matrix.iat[r, c] = np.nan

                new_missing_ratio = performance_matrix.isna().sum().sum() / total_cells
                logger.info("Added %s NaN cells; new missing ratio: %.2f%%", n_remove, new_missing_ratio * 100)
            else:
                logger.info("Already above target missing ratio; no cells removed")
        else:
            logger.info("No additional NaNs introduced")

        return performance_matrix
    except Exception as exc:
        logger.exception("Failed to load performance matrix from %s: %s", csv_path, exc)
        return None


def encode_pipeline_config(pipeline_config: Mapping[str, Any], options: Mapping[str, list]) -> np.ndarray:
    """One-hot encode a pipeline configuration dict into a flat vector."""
    encoding = []
    for step, choices in options.items():
        vec = [0] * len(choices)
        if step in pipeline_config:
            try:
                idx = choices.index(pipeline_config[step])
                vec[idx] = 1
            except ValueError:
                pass
        encoding.extend(vec)
    return np.array(encoding, dtype=int)
