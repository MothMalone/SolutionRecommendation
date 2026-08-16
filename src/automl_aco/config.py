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
    f"{KAGGLE_REPO_ROOT}/data/openml/training_performance_matrix_autogluon.csv",
    f"{KAGGLE_REPO_ROOT}/aco/training_performance_matrix_autogluon.csv",
    "/kaggle/input/disable-autogluon/training_performance_matrix_autogluon.csv",
    "/kaggle/input/diffprep-setting/training_performance_matrix_autogluon_diffprep.csv",
    "/kaggle/input/quick-test-regression/training_performance_matrix_autogluon.csv",
]
KAGGLE_METAFEATURES_PATH = f"{KAGGLE_REPO_ROOT}/data/openml/dataset_feats.csv"
KAGGLE_PIPELINES_PATH = f"{KAGGLE_REPO_ROOT}/aco/pipeline_configs.json"
KAGGLE_DATA_FOLDER = "/kaggle/input/openml"

# Local-friendly defaults (repo-relative)
LOCAL_TRAIN_PERF_PATH = "data/openml/training_performance_matrix_autogluon.csv"
LOCAL_METAFEATURES_PATH = "data/openml/dataset_feats.csv"
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

# Active operator space used by the legacy notebook. This is intentionally
# separate from DEFAULT_PIPELINE_OPTIONS because label/frequency encoding makes
# old notebook comparisons less directly comparable to the current onehot-only
# RQ3 protocol.
NOTEBOOK_LEGACY_PIPELINE_OPTIONS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", ["none", "mean", "median", "most_frequent", "knn", "constant"]),
        ("scaling", ["none", "standard", "minmax", "robust", "maxabs"]),
        ("encoding", ["onehot", "label", "frequency"]),
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

# Default precedence constraints for ordering search.
# (a, b) means a must appear before b.
DEFAULT_ORDERING_CONSTRAINTS: List[tuple[str, str]] = [
    ("imputation", "encoding"),
    ("imputation", "scaling"),
    ("imputation", "outlier_removal"),
    ("imputation", "feature_selection"),
    ("imputation", "dimensionality_reduction"),
    ("encoding", "feature_selection"),
    ("encoding", "dimensionality_reduction"),
    ("scaling", "dimensionality_reduction"),
    ("outlier_removal", "feature_selection"),
    ("outlier_removal", "dimensionality_reduction"),
]


# =============================================================================================
# AutoDP's operator space, reimplemented leak-free (see preprocessing/autodp_ops.py for the
# complete deviation table). Used by the cross-comparison arms so that "same operator space" is a
# real claim rather than a family-level approximation.
#
# Codes are UPPERCASE; ACORec's are lowercase. A config is therefore self-describing and the two
# spaces cannot be mixed up by accident.
#
# Differences in COVERAGE, which must be disclosed wherever these arms are reported:
#   * 22 of their 24 operators. EM and AD are dropped -- see autodp_ops.DROPPED for the reasons.
#   * duplicate_removal is a step ACORec does not otherwise have; it exists only in this space.
#   * dimensionality_reduction is absent here, because their space has no such family.
# =============================================================================================
AUTODP_OPERATORS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", ["none", "MEAN", "MEDIAN", "MF", "KNN", "MICE", "RAND", "DROP"]),
        ("encoding", ["OE", "BE", "FE", "CBE"]),
        ("scaling", ["none", "ZS", "MM", "DS"]),
        ("feature_selection", ["none", "MR", "WR", "LC", "TB"]),
        ("outlier_removal", ["none", "ZSB", "IQR", "LOF"]),
        ("duplicate_removal", ["none", "ED"]),
    ]
)

AUTODP_PIPELINE_OPTIONS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("imputation", AUTODP_OPERATORS["imputation"]),
        ("scaling", AUTODP_OPERATORS["scaling"]),
        ("encoding", AUTODP_OPERATORS["encoding"]),
        ("duplicate_removal", AUTODP_OPERATORS["duplicate_removal"]),
        ("outlier_removal", AUTODP_OPERATORS["outlier_removal"]),
        ("feature_selection", AUTODP_OPERATORS["feature_selection"]),
    ]
)

AUTODP_PREPROCESSOR_ORDER: List[str] = [
    "imputation",
    "scaling",
    "encoding",
    "duplicate_removal",
    "outlier_removal",
    "feature_selection",
]

AUTODP_ORDERING_CONSTRAINTS: List[tuple[str, str]] = [
    ("imputation", "encoding"),
    ("imputation", "scaling"),
    ("imputation", "outlier_removal"),
    ("imputation", "feature_selection"),
    ("imputation", "duplicate_removal"),
    ("encoding", "feature_selection"),
    ("outlier_removal", "feature_selection"),
    ("duplicate_removal", "outlier_removal"),
    ("duplicate_removal", "feature_selection"),
]

def constraints_for(step_names: Iterable[str]) -> List[tuple[str, str]]:
    """Precedence constraints appropriate to whichever operator space these steps came from.

    Selected by shape rather than by a flag, so every call site picks the right set without having
    to thread the space name through: ``duplicate_removal`` exists only in AutoDP's space.
    """
    steps = set(step_names)
    base = AUTODP_ORDERING_CONSTRAINTS if "duplicate_removal" in steps else DEFAULT_ORDERING_CONSTRAINTS
    return [(a, b) for a, b in base if a in steps and b in steps]


#: Registry so callers can select a space by name without importing each constant.
OPERATOR_SPACES: Dict[str, Dict[str, Any]] = {
    "ours": {
        "options": DEFAULT_PIPELINE_OPTIONS,
        "order": DEFAULT_PREPROCESSOR_ORDER,
        "constraints": DEFAULT_ORDERING_CONSTRAINTS,
    },
    "theirs": {
        "options": AUTODP_PIPELINE_OPTIONS,
        "order": AUTODP_PREPROCESSOR_ORDER,
        "constraints": AUTODP_ORDERING_CONSTRAINTS,
    },
}


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
    dataset_weighting: str = "similarity"
    heuristic_transfer_method: str = "weighted_topk_topl"
    heuristic_top_k: int = 10
    heuristic_top_l: int = 3
    heuristic_similarity_temperature: float = 1.0
    heuristic_eta_floor: float = 0.05
    score_direction: str = "higher_is_better"
    time_limit_per_model: int = 120
    local_search: bool = False
    top_k_pheromone: int = 3
    average_pheromone_update: bool = False
    use_all_iter_pipelines: bool = False
    weight_method: str = "rank"
    markov_order: int = 2
    lambda_smooth: float = 0.0


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
