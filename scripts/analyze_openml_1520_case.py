#!/usr/bin/env python3
"""Case-study analysis for OpenML data_id 1520 (robot-failures-lp5).

Compares:
- CtxPipe-like discovered pipeline (mostly blank + Robust scaling)
- ACORec pipeline
- One historical reference pipeline from the training matrix

Outputs:
- numeric summaries (CSV/JSON)
- stage-by-stage preprocessing traces
- publication-ready plots for slides
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "mplconfig"))

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automl_aco.data.loaders import load_openml_dataset
from automl_aco.data.loaders import load_csv_dataset
from automl_aco.data.splits import split_train_val_test
from automl_aco.preprocessing.preprocessor import Preprocessor
from automl_aco.config import DEFAULT_PREPROCESSOR_ORDER


@dataclass
class PipelineSpec:
    label: str
    config: Dict[str, str]
    step_order: List[str]
    source: str


class TracePreprocessor(Preprocessor):
    """Preprocessor with stage-wise trace snapshots."""

    def __init__(self, config: Dict[str, str], step_order: Optional[List[str]] = None):
        super().__init__(config=config, step_order=step_order)
        self.trace: List[Dict[str, Any]] = []

    def _snapshot(self, stage: str, X_num: Optional[pd.DataFrame], X_cat: Optional[pd.DataFrame]) -> None:
        n_num = 0 if X_num is None else int(X_num.shape[1])
        n_cat = 0 if X_cat is None else int(X_cat.shape[1])
        n_rows = 0
        if X_num is not None:
            n_rows = int(len(X_num))
        elif X_cat is not None:
            n_rows = int(len(X_cat))

        num_outlier_cell_rate_z = outlier_cell_rate_zscore(X_num)
        num_outlier_row_rate_iqr = outlier_row_rate_iqr(X_num)

        self.trace.append(
            {
                "stage": stage,
                "n_rows": n_rows,
                "n_features_total": n_num + n_cat,
                "n_features_numeric": n_num,
                "n_features_categorical": n_cat,
                "outlier_cell_rate_zscore_num": num_outlier_cell_rate_z,
                "outlier_row_rate_iqr_num": num_outlier_row_rate_iqr,
            }
        )

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")

        X = X.copy()
        X.columns = X.columns.astype(str)

        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        X_num = X[self.num_cols].copy() if self.num_cols else None
        X_cat = X[self.cat_cols].copy() if self.cat_cols else None
        self._snapshot("input", X_num, X_cat)

        for step in self.step_order:
            if step == "imputation":
                X_num, X_cat = self._fit_imputation(X_num, X_cat)
            elif step == "outlier_removal":
                X_num, X_cat, y = self._fit_outlier_removal(X_num, X_cat, y)
            elif step == "outlier_cleaning":
                X_num, X_cat = self._fit_outlier_cleaning(X_num, X_cat)
            elif step == "encoding":
                X_cat = self._fit_encoding(X_cat)
            elif step == "feature_selection":
                X_num, X_cat = self._fit_feature_selection(X_num, X_cat, y)
            elif step == "scaling":
                X_num = self._fit_scaling(X_num)
            elif step == "dimensionality_reduction":
                X_num = self._fit_dim_reduction(X_num)
            self._snapshot(step, X_num, X_cat)

        if X_num is not None:
            X_num.columns = X_num.columns.astype(str)
        if X_cat is not None:
            X_cat.columns = X_cat.columns.astype(str)

        X_out = None
        if X_cat is not None and X_num is not None:
            X_out = pd.concat([X_num, X_cat], axis=1)
        elif X_num is not None:
            X_out = X_num
        elif X_cat is not None:
            X_out = X_cat

        self.fitted = True
        return X_out, y


def outlier_cell_rate_zscore(df_num: Optional[pd.DataFrame], threshold: float = 3.0) -> float:
    if df_num is None or df_num.shape[1] == 0:
        return float("nan")
    x = df_num.select_dtypes(include=[np.number]).astype(float)
    if x.empty:
        return float("nan")
    std = x.std(ddof=0).replace(0, np.nan)
    z = (x - x.mean()) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    return float(np.nanmean((np.abs(z) > threshold).to_numpy()))


def outlier_row_rate_iqr(df_num: Optional[pd.DataFrame], k: float = 1.5) -> float:
    if df_num is None or df_num.shape[1] == 0:
        return float("nan")
    x = df_num.select_dtypes(include=[np.number]).astype(float)
    if x.empty:
        return float("nan")
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (x.lt(lower) | x.gt(upper)).fillna(False)
    return float(mask.any(axis=1).mean())


def infer_high_dimensionality_ratio(X: pd.DataFrame) -> Dict[str, float]:
    X_copy = X.copy()
    X_copy.columns = X_copy.columns.astype(str)
    raw = int(X_copy.shape[1])
    onehot = int(pd.get_dummies(X_copy, dummy_na=True).shape[1])
    ratio = float(onehot / raw) if raw > 0 else float("nan")
    return {"raw_features": raw, "onehot_features": onehot, "onehot_expansion_ratio": ratio}


def make_model_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xtr = pd.get_dummies(train_df, dummy_na=True)
    xte = pd.get_dummies(test_df, dummy_na=True)
    xtr, xte = xtr.align(xte, join="left", axis=1, fill_value=0)
    return xtr.astype(float), xte.astype(float)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    ca = pred_a == y_true
    cb = pred_b == y_true
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2_stat = (abs(b - c) - 1.0) ** 2 / float(n)
    if chi2 is None:
        p = float("nan")
    else:
        p = float(chi2.sf(chi2_stat, df=1))
    return {"b": b, "c": c, "chi2": float(chi2_stat), "p_value": p}


def rowname_to_pipeline_config(row_name: str) -> Optional[Dict[str, str]]:
    common = {
        "imputation": "none",
        "encoding": "onehot",
        "scaling": "none",
        "outlier_removal": "none",
        "feature_selection": "none",
        "dimensionality_reduction": "none",
    }
    mapping: Dict[str, Dict[str, str]] = {
        "baseline": {},
        "mean_standard": {"imputation": "mean", "scaling": "standard"},
        "median_robust": {"imputation": "median", "scaling": "robust"},
        "mostfreq_minmax": {"imputation": "most_frequent", "scaling": "minmax"},
        "constant_maxabs": {"imputation": "constant", "scaling": "maxabs"},
        "standard_kbest": {"scaling": "standard", "feature_selection": "k_best"},
        "robust_mutualinfo": {"scaling": "robust", "feature_selection": "mutual_info"},
        "knn_standard": {"imputation": "knn", "scaling": "standard"},
        "knn_robust_outlier": {"imputation": "knn", "scaling": "robust", "outlier_removal": "zscore"},
        "pca_pipeline": {"imputation": "mean", "scaling": "standard", "dimensionality_reduction": "pca"},
        "svd_pipeline": {"imputation": "mean", "scaling": "standard", "dimensionality_reduction": "svd"},
        "incremental_pca_large": {"imputation": "mean", "scaling": "standard", "dimensionality_reduction": "pca"},
        "pca_feature_select": {"imputation": "mean", "scaling": "standard", "feature_selection": "k_best", "dimensionality_reduction": "pca"},
        "svd_variance": {"scaling": "standard", "feature_selection": "variance_threshold", "dimensionality_reduction": "svd"},
    }
    if row_name not in mapping:
        return None
    cfg = dict(common)
    cfg.update(mapping[row_name])
    return cfg


def pick_reference_pipeline(
    perf_matrix_path: Path,
    dataset_id: int,
    preferred_row_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[float]]:
    perf = pd.read_csv(perf_matrix_path, index_col=0)
    col_candidates = [f"D_{dataset_id}", f"openml_{dataset_id}", str(dataset_id)]
    col_name = None
    for c in col_candidates:
        if c in perf.columns:
            col_name = c
            break
    if col_name is None:
        return None, None, None

    if preferred_row_name:
        if preferred_row_name in perf.index:
            cfg = rowname_to_pipeline_config(str(preferred_row_name))
            if cfg is not None:
                sc = float(perf.loc[preferred_row_name, col_name]) if pd.notna(perf.loc[preferred_row_name, col_name]) else float("nan")
                return str(preferred_row_name), cfg, sc

    ranking = perf[col_name].dropna().sort_values(ascending=False)
    for row_name, sc in ranking.items():
        cfg = rowname_to_pipeline_config(str(row_name))
        if cfg is not None:
            return str(row_name), cfg, float(sc)
    return None, None, None


def evaluate_pipeline(
    spec: PipelineSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> Dict[str, Any]:
    prep = TracePreprocessor(config=spec.config, step_order=spec.step_order)
    X_train_p, y_train_p = prep.fit_transform(X_train, y_train)
    if X_train_p is None or len(X_train_p) == 0:
        raise RuntimeError(f"{spec.label}: preprocessing produced empty train set")
    X_test_p = prep.transform(X_test)
    if X_test_p is None:
        raise RuntimeError(f"{spec.label}: preprocessing produced None test set")

    xtr_model, xte_model = make_model_matrix(X_train_p, X_test_p)

    clf = LogisticRegression(max_iter=3000, random_state=random_state)
    clf.fit(xtr_model, y_train_p.astype(int))
    pred = clf.predict(xte_model)

    metrics = {
        "pipeline": spec.label,
        "source": spec.source,
        "step_order": ",".join(spec.step_order),
        "accuracy": float(accuracy_score(y_test.astype(int), pred.astype(int))),
        "balanced_accuracy": float(balanced_accuracy_score(y_test.astype(int), pred.astype(int))),
        "f1_macro": float(f1_score(y_test.astype(int), pred.astype(int), average="macro")),
        "train_rows_before": int(len(X_train)),
        "train_rows_after": int(len(X_train_p)),
        "train_row_drop_ratio": float(1.0 - len(X_train_p) / max(1, len(X_train))),
        "features_before_raw": int(X_train.shape[1]),
        "features_after_pipeline": int(X_train_p.shape[1]),
        "features_after_model_matrix": int(xtr_model.shape[1]),
        "raw_num_outlier_cell_rate_z": outlier_cell_rate_zscore(X_train.select_dtypes(include=[np.number])),
        "post_num_outlier_cell_rate_z": outlier_cell_rate_zscore(X_train_p.select_dtypes(include=[np.number])),
        "raw_num_outlier_row_rate_iqr": outlier_row_rate_iqr(X_train.select_dtypes(include=[np.number])),
        "post_num_outlier_row_rate_iqr": outlier_row_rate_iqr(X_train_p.select_dtypes(include=[np.number])),
    }
    return {
        "metrics": metrics,
        "trace": prep.trace,
        "y_test": y_test.astype(int).to_numpy(),
        "pred": pred.astype(int),
        "x_test_model": xte_model,
        "x_train_model": xtr_model,
    }


def save_plot_dataset_characteristics(X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    if not HAS_MPL:
        return
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    out_rate_feature = {}
    if len(num_cols) > 0:
        xnum = X[num_cols].astype(float)
        q1 = xnum.quantile(0.25)
        q3 = xnum.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (xnum.lt(lower) | xnum.gt(upper)).fillna(False)
        out_rate_feature = mask.mean().sort_values(ascending=False).head(15).to_dict()

    onehot_info = infer_high_dimensionality_ratio(X)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(["numeric", "categorical"], [len(num_cols), len(cat_cols)], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Feature Type Counts")
    axes[0].set_ylabel("Count")

    if out_rate_feature:
        axes[1].barh(list(out_rate_feature.keys())[::-1], list(out_rate_feature.values())[::-1], color="#54A24B")
        axes[1].set_title("Top Numeric Features by IQR Outlier Rate")
        axes[1].set_xlabel("Outlier row fraction")
    else:
        axes[1].text(0.5, 0.5, "No numeric features", ha="center", va="center")
        axes[1].set_axis_off()

    cls = y.value_counts().sort_index()
    axes[2].bar(cls.index.astype(str), cls.values, color="#E45756")
    axes[2].set_title(
        f"Class Distribution\nRaw={onehot_info['raw_features']}, One-hot={onehot_info['onehot_features']}, "
        f"Expand x{onehot_info['onehot_expansion_ratio']:.1f}"
    )
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Samples")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_feature_flow(trace_df: pd.DataFrame, out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for pipe, sub in trace_df.groupby("pipeline"):
        ax.plot(sub["stage"], sub["n_features_total"], marker="o", label=pipe)
    ax.set_title("Feature Dimensionality Across Preprocessing Stages")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Number of features")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_outlier_and_accuracy(metrics_df: pd.DataFrame, out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(metrics_df))
    w = 0.35
    axes[0].bar(x - w / 2, metrics_df["raw_num_outlier_row_rate_iqr"], width=w, label="Raw", color="#9C755F")
    axes[0].bar(x + w / 2, metrics_df["post_num_outlier_row_rate_iqr"], width=w, label="Post-pipeline", color="#59A14F")
    axes[0].set_title("Row Outlier Rate (IQR) Before vs After")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df["pipeline"], rotation=20)
    axes[0].set_ylabel("Outlier row fraction")
    axes[0].legend()

    axes[1].bar(metrics_df["pipeline"], metrics_df["accuracy"], color="#4C78A8")
    axes[1].set_title("Test Accuracy (LogisticRegression)")
    axes[1].set_ylabel("Accuracy")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0.0, 1.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_projection_and_confusion(
    results: Dict[str, Dict[str, Any]],
    out_projection: Path,
    out_confusion: Path,
) -> None:
    if not HAS_MPL:
        return
    labels = list(results.keys())
    n = len(labels)

    # 2D projection
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, label in enumerate(labels):
        ax = axes[0, i]
        xte = results[label]["x_test_model"]
        y = results[label]["y_test"]
        if xte.shape[1] < 2 or xte.shape[0] < 3:
            ax.text(0.5, 0.5, "Not enough dimensions", ha="center", va="center")
            ax.set_axis_off()
            continue
        pca2 = PCA(n_components=2, random_state=42)
        emb = pca2.fit_transform(xte)
        for cls in np.unique(y):
            m = y == cls
            ax.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.7, label=str(cls))
        ax.set_title(f"{label}\nPCA(2) of processed test features")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="upper right", title="Class")
    plt.tight_layout()
    fig.savefig(out_projection, dpi=180)
    plt.close(fig)

    # confusion matrices
    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, label in enumerate(labels):
        ax = axes2[0, i]
        y = results[label]["y_test"]
        pred = results[label]["pred"]
        cm = confusion_matrix(y, pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix: {label}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig2.savefig(out_confusion, dpi=180)
    plt.close(fig2)


def build_report_md(
    out_dir: Path,
    dataset_id: int,
    dataset_name: str,
    profile: Dict[str, Any],
    metrics_df: pd.DataFrame,
    ref_note: str,
    mcnemar: Optional[Dict[str, float]],
) -> None:
    lines: List[str] = []
    lines.append(f"# OpenML {dataset_id} Case Study ({dataset_name})")
    lines.append("")
    lines.append("## Dataset Characteristics")
    lines.append(
        f"- Samples: `{profile['n_samples']}` | Raw features: `{profile['n_features']}` "
        f"(numeric `{profile['n_numeric']}`, categorical `{profile['n_categorical']}`)"
    )
    lines.append(
        f"- One-hot expansion potential: `{profile['onehot_features']}` "
        f"(x`{profile['onehot_expansion_ratio']:.2f}` of raw)"
    )
    lines.append(
        f"- Raw numeric outlier rates: z-score cell `{profile['raw_outlier_cell_z']:.4f}`, "
        f"IQR row `{profile['raw_outlier_row_iqr']:.4f}`"
    )
    lines.append("")
    lines.append("## Pipeline Comparison (same train/test split, LogisticRegression)")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    if mcnemar is not None:
        lines.append("## CtxPipe vs ACORec Statistical Comparison")
        lines.append(
            f"- McNemar b={mcnemar['b']}, c={mcnemar['c']}, chi2={mcnemar['chi2']:.4f}, p-value={mcnemar['p_value']:.6f}"
        )
        lines.append("")
    lines.append("## Historical Reference Pipeline")
    lines.append(f"- {ref_note}")
    lines.append("")
    lines.append("## Figures")
    lines.append("- `01_dataset_characteristics.png`")
    lines.append("- `02_feature_flow_by_stage.png`")
    lines.append("- `03_outlier_and_accuracy.png`")
    lines.append("- `04_projection_pca2d.png`")
    lines.append("- `05_confusion_matrices.png`")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_step_order(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OpenML 1520 case: CtxPipe vs ACORec pipeline behavior.")
    parser.add_argument("--dataset-id", type=int, default=1520)
    parser.add_argument("--dataset-csv", type=str, default=None, help="Optional local CSV path. If set, bypass OpenML download.")
    parser.add_argument("--target-column", type=str, default="target", help="Target column for --dataset-csv mode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "openml_1520_case_analysis"),
    )
    parser.add_argument(
        "--reference-row-name",
        type=str,
        default=None,
        help="Optional explicit row from training_performance_matrix_autogluon.csv to use as historical reference.",
    )
    parser.add_argument(
        "--sklearn-data-home",
        type=str,
        default=str(REPO_ROOT / "outputs" / ".sklearn_data"),
        help="Directory for OpenML cache (SCIKIT_LEARN_DATA).",
    )
    parser.add_argument(
        "--mpl-config-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / ".mplconfig"),
        help="Directory for matplotlib cache (MPLCONFIGDIR).",
    )
    parser.add_argument(
        "--ctxpipe-step-order",
        type=str,
        default="imputation,encoding,scaling,feature_selection,dimensionality_reduction",
    )
    parser.add_argument(
        "--acorec-step-order",
        type=str,
        default="imputation,encoding,outlier_removal,scaling,feature_selection,dimensionality_reduction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sklearn_data_home = Path(args.sklearn_data_home).resolve()
    sklearn_data_home.mkdir(parents=True, exist_ok=True)
    os.environ["SCIKIT_LEARN_DATA"] = str(sklearn_data_home)

    mpl_config_dir = Path(args.mpl_config_dir).resolve()
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    if args.dataset_csv:
        ds = load_csv_dataset(
            csv_path=args.dataset_csv,
            target_column=args.target_column,
            dataset_id=args.dataset_id,
            verbose=args.verbose,
        )
        if "name" not in ds or not ds["name"]:
            ds["name"] = f"csv_{Path(args.dataset_csv).stem}"
    else:
        ds = load_openml_dataset(args.dataset_id, verbose=args.verbose)
    if ds is None:
        raise RuntimeError(f"Failed to load OpenML dataset {args.dataset_id}")
    X = ds["X"].copy()
    y = ds["y"].copy()
    X.columns = X.columns.astype(str)

    X_train, y_train, _X_val, _y_val, X_test, y_test = split_train_val_test(X, y, seed=args.seed)

    ctxpipe_spec = PipelineSpec(
        label="CtxPipe-like",
        source="ctxpipe_result",
        config={
            "imputation": "none",
            "encoding": "none",
            "scaling": "robust",
            "outlier_removal": "none",
            "feature_selection": "none",
            "dimensionality_reduction": "none",
        },
        step_order=parse_step_order(args.ctxpipe_step_order),
    )

    acorec_spec = PipelineSpec(
        label="ACORec",
        source="acorec_result",
        config={
            "imputation": "none",
            "encoding": "onehot",
            "outlier_removal": "zscore",
            "scaling": "robust",
            "feature_selection": "mutual_info",
            "dimensionality_reduction": "pca",
        },
        step_order=parse_step_order(args.acorec_step_order),
    )

    ref_name, ref_cfg, ref_score = pick_reference_pipeline(
        perf_matrix_path=REPO_ROOT / "aco" / "training_performance_matrix_autogluon.csv",
        dataset_id=args.dataset_id,
        preferred_row_name=args.reference_row_name,
    )
    ref_spec = None
    ref_note = "No mappable reference pipeline found in training matrix."
    if ref_cfg is not None and ref_name is not None:
        ref_spec = PipelineSpec(
            label=f"HistoricalRef({ref_name})",
            source="training_matrix",
            config=ref_cfg,
            step_order=list(DEFAULT_PREPROCESSOR_ORDER),
        )
        ref_note = f"Selected `{ref_name}` with matrix score `{ref_score:.4f}` on D_{args.dataset_id}."

    pipeline_specs: List[PipelineSpec] = [ctxpipe_spec, acorec_spec]
    if ref_spec is not None:
        pipeline_specs.append(ref_spec)
    (out_dir / "pipelines_used.json").write_text(
        json.dumps(
            [
                {
                    "label": p.label,
                    "source": p.source,
                    "config": p.config,
                    "step_order": p.step_order,
                }
                for p in pipeline_specs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    results: Dict[str, Dict[str, Any]] = {}
    traces: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    for spec in pipeline_specs:
        r = evaluate_pipeline(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=args.seed,
        )
        results[spec.label] = r
        metric_rows.append(r["metrics"])
        tdf = pd.DataFrame(r["trace"])
        tdf["pipeline"] = spec.label
        traces.append(tdf)

    metrics_df = pd.DataFrame(metric_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    trace_df = pd.concat(traces, ignore_index=True)
    metrics_df.to_csv(out_dir / "metrics_pipeline.csv", index=False)
    trace_df.to_csv(out_dir / "trace_stages.csv", index=False)

    profile = infer_high_dimensionality_ratio(X_train)
    profile.update(
        {
            "dataset_id": int(args.dataset_id),
            "dataset_name": str(ds.get("name", f"D_{args.dataset_id}")),
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_numeric": int(len(X.select_dtypes(include=[np.number]).columns)),
            "n_categorical": int(len(X.select_dtypes(exclude=[np.number]).columns)),
            "raw_outlier_cell_z": outlier_cell_rate_zscore(X_train.select_dtypes(include=[np.number])),
            "raw_outlier_row_iqr": outlier_row_rate_iqr(X_train.select_dtypes(include=[np.number])),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        }
    )
    (out_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    mcnemar_result = None
    if "CtxPipe-like" in results and "ACORec" in results:
        mcnemar_result = mcnemar_test(
            y_true=results["ACORec"]["y_test"],
            pred_a=results["ACORec"]["pred"],
            pred_b=results["CtxPipe-like"]["pred"],
        )
        (out_dir / "mcnemar_ctxpipe_vs_acorec.json").write_text(
            json.dumps(mcnemar_result, indent=2),
            encoding="utf-8",
        )

    save_plot_dataset_characteristics(X_train, y_train, out_dir / "01_dataset_characteristics.png")
    save_plot_feature_flow(trace_df, out_dir / "02_feature_flow_by_stage.png")
    save_plot_outlier_and_accuracy(metrics_df, out_dir / "03_outlier_and_accuracy.png")
    save_plot_projection_and_confusion(
        results,
        out_projection=out_dir / "04_projection_pca2d.png",
        out_confusion=out_dir / "05_confusion_matrices.png",
    )

    build_report_md(
        out_dir=out_dir,
        dataset_id=args.dataset_id,
        dataset_name=str(ds.get("name", f"D_{args.dataset_id}")),
        profile=profile,
        metrics_df=metrics_df,
        ref_note=ref_note,
        mcnemar=mcnemar_result,
    )

    print(f"Saved analysis to: {out_dir}")
    print(f"- {out_dir / 'REPORT.md'}")
    print(f"- {out_dir / 'metrics_pipeline.csv'}")
    print(f"- {out_dir / 'trace_stages.csv'}")


if __name__ == "__main__":
    main()
