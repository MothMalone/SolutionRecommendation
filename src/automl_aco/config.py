"""Configuration, operator space, and dataclass defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping
from collections import OrderedDict

# Domain model ordering (from project description)
DOMAIN_LAYER_ORDER: List[str] = [
    "imputation",
    "encoding",
    "scaling",
    "outlier_removal",
    "feature_selection",
    "dimensionality_reduction",
]

# Operator space as specified in project instructions (order preserved)
OPERATORS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", ["none", "mean", "median", "most_frequent", "constant", "knn"]),
        ("encoding", ["none", "onehot"]),
        ("scaling", ["none", "standard", "minmax", "robust", "maxabs"]),
        ("outlier_removal", ["none", "iqr", "zscore", "lof", "isolation_forest"]),
        ("feature_selection", ["none", "variance_threshold", "k_best", "mutual_info"]),
        ("dimensionality_reduction", ["none", "pca", "svd"]),
    ]
)

# Kaggle repo root (uploaded dataset path)
KAGGLE_REPO_ROOT = "/kaggle/input/acorec"

# Notebook-sourced default paths (kept for parity with Kaggle runs)
KAGGLE_TRAIN_PERF_PATHS = [
    f"{KAGGLE_REPO_ROOT}/aco/training_performance_matrix_autogluon.csv",
    "/kaggle/input/disable-autogluon/training_performance_matrix_autogluon.csv",
    "/kaggle/input/diffprep-setting/training_performance_matrix_autogluon_diffprep.csv",
    "/kaggle/input/quick-test-regression/training_performance_matrix_autogluon.csv",
]
KAGGLE_METAFEATURES_PATH = f"{KAGGLE_REPO_ROOT}/aco/dataset_feats.csv"
KAGGLE_PIPELINES_PATH = f"{KAGGLE_REPO_ROOT}/aco/pipeline_configs.json"
KAGGLE_DATA_FOLDER = "/kaggle/input/openml"

# Local-friendly defaults (repo-relative)
LOCAL_TRAIN_PERF_PATH = "aco/training_performance_matrix_autogluon.csv"
LOCAL_METAFEATURES_PATH = "aco/dataset_feats.csv"
LOCAL_PIPELINES_PATH = "aco/pipeline_configs.json"
LOCAL_PIPELINES_PATH_ALT = "Data/openml/pipelines.json"

# Notebook default search ordering (preserves behavior)
DEFAULT_PIPELINE_OPTIONS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", OPERATORS["imputation"]),
        ("scaling", OPERATORS["scaling"]),
        ("encoding", ["onehot"]),
        ("feature_selection", OPERATORS["feature_selection"]),
        ("outlier_removal", OPERATORS["outlier_removal"]),
        ("dimensionality_reduction", OPERATORS["dimensionality_reduction"]),
    ]
)

# Preprocessor execution order used in the notebook
DEFAULT_PREPROCESSOR_ORDER: List[str] = [
    "imputation",
    "scaling",
    "encoding",
    "outlier_removal",
    "feature_selection",
    "dimensionality_reduction",
]


@dataclass
class PipelineConfig:
    """Typed pipeline configuration container."""

    name: str
    imputation: str = "none"
    scaling: str = "none"
    encoding: str = "onehot"
    feature_selection: str = "none"
    outlier_removal: str = "none"
    dimensionality_reduction: str = "none"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "imputation": self.imputation,
            "scaling": self.scaling,
            "encoding": self.encoding,
            "feature_selection": self.feature_selection,
            "outlier_removal": self.outlier_removal,
            "dimensionality_reduction": self.dimensionality_reduction,
        }
        data.update(self.extra)
        return data


@dataclass
class ACOParams:
    n_pipelines: int = 3
    n_ants: int = 3
    n_iterations: int = 5
    seed: int = 42
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.2
    dataset_weighting: str = "equality"
    time_limit_per_model: int = 120
    local_search: bool = False
    top_k_pheromone: int = 3
    average_pheromone_update: bool = False
    use_all_iter_pipelines: bool = False
    weight_method: str = "rank"
    markov_order: int = 2
    lambda_smooth: float = 0.7


@dataclass
class MetricParams:
    hidden_dim: int = 64
    embed_dim: int = 64
    epochs: int = 100
    lr: float = 1e-3
    seed: int = 42


@dataclass
class AutoGluonConfig:
    eval_metric: str = "accuracy"
    time_limit: int = 300
    presets: str = "best_quality"
    verbosity: int = 0
    hyperparameter_tune_kwargs: Any = None
    ag_args_fit: Dict[str, Any] = field(default_factory=lambda: {"ag.max_memory_usage_ratio": 0.9})
    seed: int = 42


def ensure_pipeline_name(cfg: MutableMapping[str, Any]) -> None:
    if "name" not in cfg or cfg["name"] is None:
        cfg["name"] = str(cfg)
