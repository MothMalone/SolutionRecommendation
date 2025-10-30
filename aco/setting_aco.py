"""
Simplified Setting for ACO Pipeline Optimization
Only includes what's needed for ACO - no recommender code
"""

# --- Core Python ---
import os
import warnings

# --- Data & Utilities ---
import numpy as np
import pandas as pd

# --- scikit-learn ---
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline

# --- Config ---
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Test dataset IDs (from your original code)
test_dataset_ids = [
    1503, 23517, 1551, 1552,
    154, 255,
    920, 475, 481, 516, 6, 801, 10, 12, 14, 9, 11, 987,
    # 3
]

# Preprocessing options (search space for ACO)
options = {
    'imputation': ['none', 'mean', 'median', 'most_frequent', 'knn', 'constant'],
    'scaling': ['none', 'standard', 'minmax', 'robust', 'maxabs'],
    'encoding': ['none', 'onehot'],
    'feature_selection': ['none', 'variance_threshold', 'k_best', 'mutual_info'],
    'outlier_removal': ['none', 'iqr', 'zscore', 'lof', 'isolation_forest'],
    'dimensionality_reduction': ['none', 'pca', 'svd']
}

# AutoGluon configuration
AUTOGLUON_CONFIG = {
    "eval_metric": "accuracy",
    "time_limit": 5,  # 5 minutes per evaluation
    "presets": "medium_quality",
    "verbosity": 2,
    "hyperparameter_tune_kwargs": None,
    "ag_args_fit": {
        "ag.max_memory_usage_ratio": 0.9,
    },
    "seed": 42
}



def create_preprocessing_pipeline(config):
    """Create a scikit-learn preprocessing pipeline for numeric data only"""
    def get_column_transformer(X):
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        transformers = []

        if numeric_cols:
            numeric_steps = []

            # Imputation
            if config['imputation'] != 'none':
                if config['imputation'] in ['mean', 'median', 'most_frequent', 'constant']:
                    numeric_steps.append(('imputer', SimpleImputer(strategy=config['imputation'])))
                elif config['imputation'] == 'knn':
                    numeric_steps.append(('imputer', KNNImputer(n_neighbors=min(5, len(X) - 1))))
                else:
                    numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
            
            # Scaling
            if config['scaling'] != 'none':
                if config['scaling'] == 'standard':
                    numeric_steps.append(('scaler', StandardScaler()))
                elif config['scaling'] == 'minmax':
                    numeric_steps.append(('scaler', MinMaxScaler()))
                elif config['scaling'] == 'robust':
                    numeric_steps.append(('scaler', RobustScaler())) 
                elif config['scaling'] == 'maxabs':
                    numeric_steps.append(('scaler', MaxAbsScaler()))
            
            if numeric_steps:
                numeric_pipeline = Pipeline(numeric_steps)
                transformers.append(('num', numeric_pipeline, numeric_cols))
            else:
                transformers.append(('num', 'passthrough', numeric_cols))

        if not transformers:
            return None
        return ColumnTransformer(transformers, remainder='drop')
    
    return get_column_transformer


def apply_preprocessing(X, y, config):
    """
    Apply preprocessing pipeline based on configuration.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        config: Configuration dict with preprocessing steps
        
    Returns:
        X_transformed, y_transformed: Preprocessed data
    """
    try:
        X_processed = X.copy().reset_index(drop=True)
        y_processed = pd.Series(y).reset_index(drop=True)

        # --- Baseline ---
        if config.get('name') == 'baseline':
            X_baseline = X_processed.copy()
            for col in X_baseline.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_baseline[col] = le.fit_transform(X_baseline[col].fillna('missing'))
            return X_baseline, y_processed

        # --- Numeric preprocessing ---
        preprocessor_func = create_preprocessing_pipeline(config)
        preprocessor = preprocessor_func(X_processed)
        if preprocessor is not None:
            X_transformed = preprocessor.fit_transform(X_processed)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        else:
            X_transformed = X_processed.copy()

        X_transformed = X_transformed.reset_index(drop=True)
        y_processed = y_processed.reset_index(drop=True)

        # --- Encoding Stage ---
        cat_cols = X_processed.select_dtypes(exclude=['number']).columns.tolist()
        if cat_cols and config.get('encoding') == 'onehot':
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            enc_df = pd.DataFrame(enc.fit_transform(X_processed[cat_cols]))
            enc_df.columns = enc.get_feature_names_out(cat_cols)
            
            # Drop old categorical cols and join encoded
            X_transformed = pd.concat(
                [X_transformed.drop(columns=[c for c in cat_cols if c in X_transformed.columns], errors="ignore"),
                 enc_df.reset_index(drop=True)],
                axis=1
            )

        # --- Outlier removal ---
        if config.get('outlier_removal') == 'iqr':
            for col in X_transformed.select_dtypes(include=['number']):
                Q1, Q3 = X_transformed[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    mask = (X_transformed[col] >= Q1 - 1.5 * IQR) & (X_transformed[col] <= Q3 + 1.5 * IQR)
                    X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        elif config.get('outlier_removal') == 'zscore':
            from scipy.stats import zscore
            z_scores = np.abs(zscore(X_transformed.select_dtypes(include=['number'])))
            mask = (z_scores < 3).all(axis=1)
            X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        elif config.get('outlier_removal') == 'lof':
            lof = LocalOutlierFactor(n_neighbors=20)
            y_pred = lof.fit_predict(X_transformed.select_dtypes(include=['number']))
            mask = y_pred == 1
            X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        elif config.get('outlier_removal') == 'isolation_forest':
            iso = IsolationForest(contamination=0.05, random_state=42)
            y_pred = iso.fit_predict(X_transformed.select_dtypes(include=['number']))
            mask = y_pred == 1
            X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        # --- Feature selection ---
        if config.get('feature_selection') == 'variance_threshold':
            selector = VarianceThreshold(threshold=0.01)
            X_transformed = pd.DataFrame(
                selector.fit_transform(X_transformed),
                columns=X_transformed.columns[selector.get_support()]
            )

        elif config.get('feature_selection') == 'k_best':
            k = min(20, X_transformed.shape[1])
            if k > 0 and len(X_transformed) > k:
                selector = SelectKBest(f_classif, k=k)
                X_transformed = pd.DataFrame(
                    selector.fit_transform(X_transformed, y_processed),
                    columns=X_transformed.columns[selector.get_support()]
                )

        elif config.get('feature_selection') == 'mutual_info':
            k = min(20, X_transformed.shape[1])
            if k > 0 and len(X_transformed) > k:
                selector = SelectKBest(mutual_info_classif, k=k)
                X_transformed = pd.DataFrame(
                    selector.fit_transform(X_transformed, y_processed),
                    columns=X_transformed.columns[selector.get_support()]
                )

        # --- Dimensionality reduction ---
        if config.get('dimensionality_reduction') in ['pca', 'svd']:
            n_components = min(10, X_transformed.shape[1], len(X_transformed) - 1)
            if n_components > 0:
                reducer = PCA(n_components=n_components) if config['dimensionality_reduction'] == 'pca' else TruncatedSVD(n_components=n_components)
                reducer.fit(X_transformed)
                X_transformed = pd.DataFrame(
                    reducer.transform(X_transformed),
                    index=X_transformed.index
                )

        # --- Final cleanup ---
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
        y_processed = y_processed.reset_index(drop=True)

        return X_transformed, y_processed

    except Exception as e:
        print(f"Error in preprocessing {config.get('name', 'unknown')}: {e}")
        X_fallback = X.copy()
        for col in X_fallback.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_fallback[col] = le.fit_transform(X_fallback[col].fillna('missing'))
        return X_fallback.reset_index(drop=True), pd.Series(y).reset_index(drop=True)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_openml_dataset(dataset_id, test_dataset_ids=None):
    """Load OpenML dataset with error handling and automatic problem type detection"""
    try:
        try:
            dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        except ValueError as e:
            if "Sparse ARFF" in str(e):
                print(f"Retrying dataset {dataset_id} with as_frame=False...")
                dataset = fetch_openml(data_id=dataset_id, as_frame=False, parser='auto')
            else:
                raise e

        X = dataset.data.copy()
        y = dataset.target

        # Handle categorical features properly
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype(str)

        # Handle target encoding
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)

        # Drop invalid samples
        X = X.dropna(axis=1, how='all')
        mask = ~pd.isna(y)
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)

        # Detect problem type
        if y.nunique() > 20 and y.dtype.kind in "iufc":
            task_type = 'regression'
        else:
            task_type = 'classification'
            y = y.astype(int)

        # Remove rare classes (<5 samples)
        if task_type == 'classification':
            class_counts = y.value_counts()
            valid_classes = class_counts[class_counts >= 5].index
            mask = y.isin(valid_classes)
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)

        # Limit dataset size for efficiency
        max_samples = 8000 if (test_dataset_ids and dataset_id in test_dataset_ids) else 5000
        if len(X) > max_samples:
            X, y = shuffle(X, y, n_samples=max_samples, random_state=42)
            X = X.reset_index(drop=True)
            y = pd.Series(y).reset_index(drop=True)

        print(f"Loaded dataset {dataset_id}: Shape={X.shape}, Task={task_type}, Classes={len(np.unique(y))}")

        return {
            'id': dataset_id,
            'name': f"D_{dataset_id}",
            'X': X,
            'y': y,
            'task_type': task_type
        }
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}")
        return None


if __name__ == "__main__":
    print("="*80)
    print("ACO PIPELINE OPTIMIZATION - SETTINGS MODULE")
    print("="*80)
    print(f"\n✓ Preprocessing options configured:")
    for step, ops in options.items():
        print(f"  - {step}: {len(ops)} options")
    print(f"\n✓ Total possible pipelines: {np.prod([len(ops) for ops in options.values()])}")
    print(f"\n✓ Test datasets configured: {len(test_dataset_ids)}")
    print("\nReady for ACO optimization!")
