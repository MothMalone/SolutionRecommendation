# --- Core Python ---
import os
import shutil
import tempfile
import warnings

# --- Data & Utilities ---
import numpy as np
import pandas as pd

# --- scikit-learn ---
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity

# --- Optimization ---
from scipy.optimize import minimize

# --- External Libraries ---
from autogluon.tabular import TabularPredictor
import openml

# --- Config ---
warnings.filterwarnings('ignore')
predictor_path = '/kaggle/working/autogluon_models'

from autogluon.features.generators import IdentityFeatureGenerator

# AUTOGLUON_CONFIG = {
#     "eval_metric": "accuracy",
#     "time_limit": 600,  # 5 minutes per dataset
#     "presets": "medium_quality",
#     "verbosity": 1,
#     "hyperparameter_tune_kwargs": None,
#     "ag_args_fit": {
#         "ag.max_memory_usage_ratio": 0.9,
#     },
#     "seed": 42
# }

train_dataset_ids = [
    22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
    37, 39, 40, 41, 42, 43, 48, 49, 50, 53, 54, 55,
    56, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
    187, 188, 275, 276,
    277, 278, 285, 300, 301, 307, 308,
    310, 311, 312, 313, 316, 327, 328, 329, 
    333, 334, 335, 336,
    337, 338, 339, 340, 342, 343, 346, 372, 375,
    378, 443, 444, 446, 448, 450, 451, 452, 453, 454, 455, 457, 458, 459, 461,
    462, 463, 464, 465, 467, 468, 469, 2009, 2804, 2309, 1907
]
test_dataset_ids = [
    1503, 23517, 1551, 1552,
    154, 255,
    920, 475, 481, 516, 3, 6, 801, 10, 12, 14, 9, 11, 987
]


pipeline_configs = [
    {'name': 'baseline', 'imputation': 'none', 'scaling': 'none', 'encoding': 'none', 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'simple_preprocess', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'robust_preprocess', 'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'},
    {'name': 'feature_selection', 'imputation': 'median', 'scaling': 'standard', 'encoding': 'onehot', 'feature_selection': 'k_best', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'dimension_reduction', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
    {'name': 'conservative', 'imputation': 'median', 'scaling': 'minmax', 'encoding': 'onehot', 'feature_selection': 'variance_threshold', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    {'name': 'aggressive', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 'feature_selection': 'k_best', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'pca'},
    {'name': 'knn_impute_pca', 'imputation': 'knn', 'scaling': 'standard', 'encoding': 'onehot', 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
    {'name': 'mutual_info_zscore', 'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 'feature_selection': 'mutual_info', 'outlier_removal': 'zscore', 'dimensionality_reduction': 'none'},
    {'name': 'constant_maxabs_iforest', 'imputation': 'constant', 'scaling': 'maxabs', 'encoding': 'onehot', 'feature_selection': 'variance_threshold', 'outlier_removal': 'isolation_forest', 'dimensionality_reduction': 'none'},
    {'name': 'mean_minmax_lof_svd', 'imputation': 'mean', 'scaling': 'minmax', 'encoding': 'onehot', 'feature_selection': 'k_best', 'outlier_removal': 'lof', 'dimensionality_reduction': 'svd'},
    {'name': 'mostfreq_standard_iqr', 'imputation': 'most_frequent', 'scaling': 'standard', 'encoding': 'onehot', 'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'}
]

options = {
    'imputation': ['none', 'mean', 'median', 'most_frequent', 'knn', 'constant'],
    'scaling': ['none', 'standard', 'minmax', 'robust', 'maxabs'],
    'encoding': ['none', 'onehot'],
    'feature_selection': ['none', 'variance_threshold', 'k_best', 'mutual_info'],
    'outlier_removal': ['none', 'iqr', 'zscore', 'lof', 'isolation_forest'],
    'dimensionality_reduction': ['none', 'pca', 'svd']
}



import os
os.environ["OMP_NUM_THREADS"] = "1"

from xgboost import XGBRanker 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, OneHotEncoder, LabelEncoder
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Try to import category_encoders, but it's optional
try:
    import category_encoders as ce
    HAS_CATEGORY_ENCODERS = True
except ImportError:
    HAS_CATEGORY_ENCODERS = False
    print("Warning: category_encoders not installed. Only 'onehot' and 'none' encoding will be available.")


# --------------------------
# Preprocessing pipeline (numeric only)
# --------------------------
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


# --------------------------
# Apply preprocessing
# --------------------------
def apply_preprocessing(X, y, config):
    """Apply preprocessing pipeline based on configuration with extended methods"""
    try:
        X_processed = X.copy().reset_index(drop=True)
        y_processed = pd.Series(y).reset_index(drop=True)

        # --- Baseline ---
        if config['name'] == 'baseline':
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

        # --- Encoding Stage (explicit) ---
        cat_cols = X_processed.select_dtypes(exclude=['number']).columns.tolist()
        if cat_cols:
            if config["encoding"] == "onehot":
                enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                enc_df = pd.DataFrame(enc.fit_transform(X_processed[cat_cols]))
                enc_df.columns = enc.get_feature_names_out(cat_cols)
            elif config["encoding"] == "none":
                # Label encode for compatibility
                enc_df = X_processed[cat_cols].copy()
                for col in enc_df.columns:
                    le = LabelEncoder()
                    enc_df[col] = le.fit_transform(enc_df[col].fillna('missing'))
            elif HAS_CATEGORY_ENCODERS:
                # Use category_encoders if available
                if config["encoding"] == "frequency":
                    enc = ce.CountEncoder(normalize=True)
                    enc_df = enc.fit_transform(X_processed[cat_cols])
                elif config["encoding"] == "count":
                    enc = ce.CountEncoder(normalize=False)
                    enc_df = enc.fit_transform(X_processed[cat_cols])
                elif config["encoding"] == "ordinal":
                    enc = ce.OrdinalEncoder()
                    enc_df = enc.fit_transform(X_processed[cat_cols])
                elif config["encoding"] == "binary":
                    enc = ce.BinaryEncoder()
                    enc_df = enc.fit_transform(X_processed[cat_cols])
                else:
                    enc = ce.OrdinalEncoder()
                    enc_df = enc.fit_transform(X_processed[cat_cols])
            else:
                # Fallback to label encoding
                enc_df = X_processed[cat_cols].copy()
                for col in enc_df.columns:
                    le = LabelEncoder()
                    enc_df[col] = le.fit_transform(enc_df[col].fillna('missing'))

            # Drop old categorical cols and join encoded
            X_transformed = pd.concat(
                [X_transformed.drop(columns=[c for c in cat_cols if c in X_transformed.columns], errors="ignore"),
                 enc_df.reset_index(drop=True)],
                axis=1
            )

        # --- Outlier removal ---
        if config['outlier_removal'] == 'iqr':
            for col in X_transformed.select_dtypes(include=['number']):
                Q1, Q3 = X_transformed[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    mask = (X_transformed[col] >= Q1 - 1.5 * IQR) & (X_transformed[col] <= Q3 + 1.5 * IQR)
                    X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        elif config['outlier_removal'] == 'zscore':
            from scipy.stats import zscore
            z_scores = np.abs(zscore(X_transformed.select_dtypes(include=['number'])))
            mask = (z_scores < 3).all(axis=1)
            X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        elif config['outlier_removal'] == 'lof':
            lof = LocalOutlierFactor(n_neighbors=20)
            y_pred = lof.fit_predict(X_transformed.select_dtypes(include=['number']))
            mask = y_pred == 1
            X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        elif config['outlier_removal'] == 'isolation_forest':
            iso = IsolationForest(contamination=0.05, random_state=42)
            y_pred = iso.fit_predict(X_transformed.select_dtypes(include=['number']))
            mask = y_pred == 1
            X_transformed, y_processed = X_transformed[mask], y_processed[mask]

        # --- Feature selection ---
        if config['feature_selection'] == 'variance_threshold':
            selector = VarianceThreshold(threshold=0.01)
            X_transformed = pd.DataFrame(
                selector.fit_transform(X_transformed),
                columns=X_transformed.columns[selector.get_support()]
            )

        elif config['feature_selection'] == 'k_best':
            k = min(20, X_transformed.shape[1])
            if k > 0 and len(X_transformed) > k:
                selector = SelectKBest(f_classif, k=k)
                X_transformed = pd.DataFrame(
                    selector.fit_transform(X_transformed, y_processed),
                    columns=X_transformed.columns[selector.get_support()]
                )

        elif config['feature_selection'] == 'mutual_info':
            k = min(20, X_transformed.shape[1])
            if k > 0 and len(X_transformed) > k:
                selector = SelectKBest(mutual_info_classif, k=k)
                X_transformed = pd.DataFrame(
                    selector.fit_transform(X_transformed, y_processed),
                    columns=X_transformed.columns[selector.get_support()]
                )

        # --- Dimensionality reduction ---
        if config['dimensionality_reduction'] in ['pca', 'svd']:
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
        print(f"Error in preprocessing {config['name']}: {e}")
        X_fallback = X.copy()
        for col in X_fallback.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_fallback[col] = le.fit_transform(X_fallback[col].fillna('missing'))
        return X_fallback.reset_index(drop=True), pd.Series(y).reset_index(drop=True)



import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

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

        print(f"Loaded dataset {dataset_id}")
        print(f"  Shape: {X.shape}")
        print(f"  Task: {task_type}")
        print(f"  Target classes: {len(np.unique(y)) if task_type=='classification' else 'N/A'}")

        return {
            'id': dataset_id,
            'name': f"D_{dataset_id}",
            #'name': f"Dataset_{dataset_id}",
            'X': X,
            'y': y,
            'task_type': task_type
        }
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {e}")
        return None


def load_kaggle_dataset(dataset_id, data_folder="/kaggle/input/openml", test_dataset_ids=None):
    """Load dataset from Kaggle input folder with error handling and automatic problem type detection"""
    try:
        file_path = os.path.join(data_folder, f"{dataset_id}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read CSV
        dataset = pd.read_csv(file_path)

        if "target" not in dataset.columns:
            raise ValueError(f"No 'target' column found in dataset {dataset_id}")

        # Split features and target
        X = dataset.drop(columns=["target"]).copy()
        y = dataset["target"].copy()

        # Handle categorical features properly
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

        print(f"Loaded dataset {dataset_id}")
        print(f"  Shape: {X.shape}")
        print(f"  Task: {task_type}")
        print(f"  Target classes: {len(np.unique(y)) if task_type=='classification' else 'N/A'}")

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

def evaluate_pipeline_with_autogluon(dataset, pipeline_config):
    """Evaluate a preprocessing pipeline using AutoGluon"""
    try:
        X, y = dataset['X'], dataset['y']

        # Apply preprocessing
        X_processed, y_processed = apply_preprocessing(X, y, pipeline_config)

        if X_processed.empty or len(y_processed) == 0:
            print(f"Empty dataset after preprocessing for {pipeline_config['name']}")
            return np.nan

        # --- Detect problem type ---
        unique_classes = np.unique(y_processed)
        if np.issubdtype(y_processed.dtype, np.number) and len(unique_classes) > 20:
            problem_type = "regression"
        elif len(unique_classes) == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"

        
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42,
                stratify=y_processed if problem_type != "regression" else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.5, random_state=42
            )

        # --- Prepare data for AutoGluon ---
        train_data = X_train.copy()
        train_data['target'] = y_train
        test_data = X_test.copy()

        # === AutoGluon temp dir: unique + clean ===
        import uuid, tempfile, os, shutil, warnings
        from sklearn.metrics import accuracy_score, r2_score

        temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

        # Suppress "path already exists" warning
        warnings.filterwarnings("ignore", message="path already exists! This predictor may overwrite")

        try:
            predictor = TabularPredictor(
                label="target",
                path=temp_dir,
                problem_type=problem_type,
                eval_metric=("r2" if problem_type == "regression" else "accuracy"),
                verbosity=AUTOGLUON_CONFIG["verbosity"]
            )

            predictor.fit(
                train_data=train_data,
                time_limit=AUTOGLUON_CONFIG["time_limit"],
                presets=AUTOGLUON_CONFIG["presets"],
                hyperparameter_tune_kwargs=AUTOGLUON_CONFIG["hyperparameter_tune_kwargs"],
                ag_args_fit=AUTOGLUON_CONFIG["ag_args_fit"],
                feature_generator=IdentityFeatureGenerator()
            )

            predictions = predictor.predict(test_data)
            if problem_type == "regression":
                score = r2_score(y_test, predictions)
            else:
                score = accuracy_score(y_test, predictions)

            return score

        except Exception as e:
            print(f"AutoGluon error for {pipeline_config['name']}: {e}")
            print("Fallback: using RandomForestClassifier/Regressor")

            if problem_type == "regression":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == "regression":
                return r2_score(y_test, y_pred)
            else:
                return accuracy_score(y_test, y_pred)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"Error evaluating pipeline {pipeline_config['name']} on {dataset['name']}: {e}")
        return np.nan



def extract_enhanced_metafeatures(dataset, meta_features_df=None):
    """
    Fetch precomputed meta-features for a dataset from a CSV/Excel file.
    dataset: dict with 'id' field
    """
    try:
        if meta_features_df is None:
            # Meta-features not available
            return {}
            
        dataset_id = dataset.get('id', None)
        if dataset_id is None:
            raise ValueError("Dataset does not have an 'id' field")

        # Look up by id
        row = meta_features_df.loc[[dataset_id]]
        if row.empty:
            print(f"No meta-features found for dataset id={dataset_id}")
            return {}

        # Convert row to dict (excluding 'id' column)
        metafeatures = row.iloc[0].to_dict()
        return metafeatures

    except Exception as e:
        print(f"Error fetching meta-features for dataset {dataset.get('id', 'unknown')}: {e}")
        return {}


def build_metafeatures_matrix(datasets):
    """Build metafeatures matrix for all datasets"""
    metafeatures_list = []
    dataset_names = []
    
    for dataset in datasets:
        print(f"\nExtracting meta-features for {dataset['name']}")
        metafeatures = extract_enhanced_metafeatures(dataset)
        if metafeatures:
            metafeatures_list.append(metafeatures)
            dataset_names.append(f"{dataset['name']}")
    
    if metafeatures_list:
        metafeatures_df = pd.DataFrame(metafeatures_list, index=dataset_names)
        return metafeatures_df
    else:
        return pd.DataFrame()

def build_performance_matrix(datasets, pipeline_configs, use_autogluon=True):
    """Build performance matrix by evaluating each pipeline on each dataset"""
    performance_matrix = pd.DataFrame(
        index=[config['name'] for config in pipeline_configs],
        columns=[f"{dataset['name']}" for dataset in datasets]
    )
    
    eval_func = evaluate_pipeline_with_autogluon if use_autogluon else evaluate_pipeline_fallback
    
    for config in pipeline_configs:
        print(f"\nEvaluating pipeline: {config['name']}")
        for dataset in datasets:
            print(f"  Dataset: {dataset['name']} (ID: {dataset['id']})")
            performance = eval_func(dataset, config)
            performance_matrix.loc[config['name'], f"{dataset['name']}"] = performance
            if not np.isnan(performance):
                print(f"    Performance: {performance:.4f}")
    
    return performance_matrix

def build_performance_matrix_available(csv_path, remove_ratio=0.0, random_state=None):
    try:
        performance_matrix = pd.read_csv(csv_path, index_col=0)
        print(f"✅ Loaded performance matrix from {csv_path}")
        print("Matrix shape:", performance_matrix.shape)

        # --- Step 1: Compute current missing ratio ---
        current_missing_ratio = performance_matrix.isna().sum().sum() / performance_matrix.size
        print(f"📊 Current missing ratio: {current_missing_ratio*100:.2f}%")

        # --- Step 2: Add more missing cells only if below threshold ---
        if remove_ratio > 0 and current_missing_ratio < remove_ratio:
            np.random.seed(random_state)

            # Calculate how many additional cells to remove
            total_cells = performance_matrix.size
            target_missing_cells = int(total_cells * remove_ratio)
            current_missing_cells = int(total_cells * current_missing_ratio)
            n_remove = target_missing_cells - current_missing_cells

            if n_remove > 0:
                # Randomly pick cells that are currently not NaN
                non_nan_positions = np.argwhere(~performance_matrix.isna().values)
                chosen_indices = non_nan_positions[
                    np.random.choice(len(non_nan_positions), n_remove, replace=False)
                ]

                # Set selected cells to NaN
                for r, c in chosen_indices:
                    performance_matrix.iat[r, c] = np.nan

                new_missing_ratio = performance_matrix.isna().sum().sum() / total_cells
                print(f"Added {n_remove} NaN cells.")
                print(f"New missing ratio: {new_missing_ratio*100:.2f}%")
            else:
                print("Already above target missing ratio; no cells removed.")
        else:
            print("No additional NaNs introduced (already at or above threshold).")

        return performance_matrix

    except Exception as e:
        print(f"Failed to load performance matrix from {csv_path}: {e}")
        return None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

def evaluate_pipeline_fallback(dataset, pipeline_config):
    """Fallback evaluation that auto-detects problem type (classification or regression)"""
    try:
        X, y = dataset['X'], dataset['y']
        X_processed, y_processed = apply_preprocessing(X, y, pipeline_config)
        
        if X_processed.empty or len(y_processed) == 0:
            return np.nan

        # --- Detect problem type ---
        unique_classes = np.unique(y_processed)
        if np.issubdtype(y_processed.dtype, np.number) and len(unique_classes) > 20:
            problem_type = "regression"
        elif len(unique_classes) == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"

        # --- Check if class too small for classification ---
        if problem_type != "regression":
            _, class_counts = np.unique(y_processed, return_counts=True)
            if class_counts.min() < 3:
                return np.nan

        # --- Split data ---
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42,
                stratify=y_processed if problem_type != "regression" else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )

        # --- Choose models ---
        if problem_type == "regression":
            models = [
                RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
                LinearRegression()
            ]
        else:
            models = [
                RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
                LogisticRegression(random_state=42, max_iter=500, solver='liblinear')
            ]

        # --- Evaluate ---
        scores = []
        for model in models:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if problem_type == "regression":
                    score = r2_score(y_test, y_pred)
                else:
                    score = accuracy_score(y_test, y_pred)
                scores.append(score)
            except Exception as e:
                continue

        return np.mean(scores) if scores else np.nan

    except Exception as e:
        print(f"Error evaluating pipeline {pipeline_config['name']} on {dataset['name']}: {e}")
        return np.nan



def encode_pipeline_config(pipeline_config, options):
    """
    One-hot encode a pipeline configuration dict into a flat vector.
    """
    encoding = []
    for step, choices in options.items():
        vec = [0] * len(choices)
        if step in pipeline_config:
            try:
                idx = choices.index(pipeline_config[step])
                vec[idx] = 1
            except ValueError:
                pass  # unknown value, keep as all zero
        encoding.extend(vec)
    return np.array(encoding, dtype=int)


def split_dataset(dataset, seed=42):
    """Split a dataset into train/val/test (70/15/15)."""
    X = dataset["X"]
    y = dataset["y"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    train_set = {"id": dataset["id"], "name": dataset["name"], "X": X_train, "y": y_train}
    val_set   = {"id": dataset["id"], "name": dataset["name"], "X": X_val, "y": y_val}
    test_set  = {"id": dataset["id"], "name": dataset["name"], "X": X_test, "y": y_test}
    return train_set, val_set, test_set


print("="*80)
print("ENHANCED PREPROCESSING RECOMMENDER WITH AUTOGLUON")
print("="*80)
     
# Load datasets
print("\nLoading datasets...")
np.random.seed(42)

train_datasets = []

for dataset_id in train_dataset_ids:
    dataset = load_openml_dataset(dataset_id)
    if dataset:
        train_datasets.append(dataset)
    if len(train_datasets) >= 1000:  # Limit for computational efficiency
        break

print(f"\nLoaded {len(train_datasets)} training datasets")

if len(train_datasets) < 3:
    print("Need at least 3 training datasets to proceed")

# Load test datasets
print("\nLoading test datasets...")
test_datasets = []
for dataset_id in test_dataset_ids:
    dataset = load_openml_dataset(dataset_id)
    if dataset:
        test_datasets.append(dataset)

print(f"\nLoaded {len(test_datasets)} test datasets")

# === Train phase ===
print("\nBuilding training performance matrix with AutoGluon...")
train_performance_matrix = build_performance_matrix_available("/drive1/nammt/gfacs/solurec/training_performance_matrix_autogluon.csv")
test_performance_matrix = build_performance_matrix_available("/drive1/nammt/gfacs/solurec/testing_performance_matrix_autogluon.csv")


valid_pipeline_names = [cfg['name'] for cfg in pipeline_configs]

train_performance_matrix = train_performance_matrix.loc[
    train_performance_matrix.index.intersection(valid_pipeline_names)
]

print(f"Filtered performance matrix shape: {train_performance_matrix.shape}")

print("\nTraining Performance Matrix:")
train_performance_matrix.to_csv("training_performance_matrix_autogluon.csv")

print("\nComputing average performance of pipelines across training datasets...")
pipeline_avg_perf = train_performance_matrix.mean(axis=1).sort_values(ascending=False)
pipeline_avg_perf_df = pipeline_avg_perf.reset_index()
pipeline_avg_perf_df.columns = ["pipeline", "average_performance"]
pipeline_avg_perf_df.to_csv("pipelines_rank.csv", index=False)
print("\nSaved pipeline ranking to 'pipelines_rank.csv'")

print("\nBuilding training metafeatures matrix...")
train_metafeatures_df = build_metafeatures_matrix(train_datasets)
if not train_metafeatures_df.empty:
    train_metafeatures_df.to_csv("training_metafeatures.csv")

print("\nTesting on test datasets...")
test_results = []
for test_dataset in test_datasets:
    print(f"\n{'='*60}")
    print(f"TESTING ON {test_dataset['name']} (ID: {test_dataset['id']})")
    print(f"{'='*60}")


    # Debug: Check the structure of test_dataset
    print("Dataset type:", type(test_dataset))
    if isinstance(test_dataset, dict):
        print("Dataset keys:", list(test_dataset.keys()))
        for key, value in test_dataset.items():
            print(f"  {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
    
    # If it's in X, y format, convert properly:
    # Convert test_dataset to proper format
    if isinstance(test_dataset, dict) and 'X' in test_dataset:
        import pandas as pd
        X = test_dataset['X']  # Already a DataFrame
        y = test_dataset['y']   # Already a Series - NOTE: it's 'y', not 'target'
        
        # Combine X and y
        test_dataset_df = X.copy()
        test_dataset_df['target'] = y
        
        print(f"Combined dataset shape: {test_dataset_df.shape}")
        print(f"Columns: {test_dataset_df.columns.tolist()}")
        
        
        # Evaluate all pipelines
        test_performances = {}
        for config in pipeline_configs:
          
            status = "⭐ RECOMMENDED" if config["name"] == recommendation["pipeline_config"]["name"] else "" 
            #value = test_performance_matrix.loc[config['name'], f"{test_dataset['id']}_{test_dataset['name']}"]
            value = test_performance_matrix.loc[config['name'], f"{test_dataset['name']}"]
    
            if pd.notna(value):
                print(f"  {config['name']:20s}: {value:.4f} {status}")
            else:
                print(f"  {config['name']:20s}: NaN {status}")
    
        #dataset_key = f"{test_dataset['id']}_{test_dataset['name']}"
        dataset_key = f"{test_dataset['name']}"
        test_performances = test_performance_matrix[dataset_key].to_dict()
        
        # Analysis
        valid_performances = [(name, perf) for name, perf in test_performances.items() if not np.isnan(perf)]
        
        if valid_performances:
            # --- Sort pipelines by performance (descending) ---
            sorted_pipelines = sorted(valid_performances, key=lambda x: x[1], reverse=True)
            pipeline_names = [name for name, _ in sorted_pipelines]
            performances = np.array([score for _, score in sorted_pipelines])
    
            # --- Compute ranks with ties (method="min" means all tied scores get best rank) ---
            ranks = rankdata(-performances, method="min")  # negate to rank higher scores better
            rank_map = dict(zip(pipeline_names, ranks))
        
            # --- Identify best pipeline ---
            best_pipeline, best_performance = sorted_pipelines[0]
        
            # --- Recommendation details ---
            rec_name = recommendation['pipeline_config']['name']
            rec_performance = test_performances.get(rec_name, np.nan)
            rec_rank = rank_map.get(rec_name, len(pipeline_names))
        
            # --- Baseline pipeline ---
            baseline_rank = rank_map.get('baseline', len(pipeline_names))
            baseline_perf = test_performances.get('baseline', np.nan)
        
            # --- Performance difference ---
            perf_diff = 100 * (rec_performance - best_performance) if not np.isnan(rec_performance) else np.nan
            gap_scale = 100 * np.abs(rec_performance - best_performance) / rec_performance if not np.isnan(rec_performance) else np.nan
            gap_scale = round(gap_scale, 2)
            
            # --- Print analysis ---
            print(f"\nResults Analysis:")
            print(f"  Best actual pipeline: {best_pipeline} ({best_performance:.4f})")
            print(f"  Recommended pipeline: {rec_name} ({rec_performance:.4f})")
            print(f"  Recommendation rank: {rec_rank}/{len(pipeline_names)}")
            print(f"  Performance gap: {perf_diff:.2f}")
        
            # --- Success metrics ---
            top3 = rec_rank <= 3
            top_half = rec_rank <= len(pipeline_names) // 2
            better_than_baseline = (
                not np.isnan(rec_performance)
                and not np.isnan(baseline_perf)
                and rec_performance > baseline_perf
            )
        
            print(f"  Success metrics:")
            print(f"    Top 3: {'✓' if top3 else '✗'}")
            print(f"    Top half: {'✓' if top_half else '✗'}")
            print(f"    Better than baseline: {'✓' if better_than_baseline else '✗'}")
        
            # --- Store results ---
            test_results.append({
                'dataset_id': test_dataset['id'],
                'dataset_name': test_dataset['name'],
                'recommended_pipeline': rec_name,
                'best_pipeline': best_pipeline,
                'rank': rec_rank,
                'baseline_rank': baseline_rank,
                'baseline_performance': baseline_perf,
                'recommended_performance': rec_performance,
                'best_performance': best_performance,
                'score_gap_btw_best_and_recommended': perf_diff,
                'gap_scale': gap_scale,
                'better_than_baseline': better_than_baseline,
                'total_pipelines': len(pipeline_names),
                'top3': top3,
                'top_half': top_half,
                'all_performances': test_performances
            })
        
            # --- Confidence assessment ---
            if recommendation['confidence'] == 'high' and top3:
                print(f"  🎯 High confidence recommendation was successful!")
            elif recommendation['confidence'] == 'low' and not top_half:
                print(f"  ⚠️ Low confidence recommendation performed as expected")
            elif top3:
                print(f"  ✅ Good recommendation despite {recommendation['confidence']} confidence")
            else:
                print(f"  ❌ Recommendation could be improved")
        
        else:
            print("  ❌ No valid performance results")
            test_results.append({
                'dataset_id': test_dataset['id'],
                'dataset_name': test_dataset['name'],
                'error': 'No valid performances'
            })
    
    print(test_performance_matrix.round(4))
    test_performance_matrix.to_csv("testing_performance_matrix_autogluon.csv")
    
    # Overall summary
    print(f"\n" + "="*80)
    print("OVERALL EVALUATION SUMMARY")
    print("="*80)
    
    # Filter valid results
    valid_results = [r for r in test_results if 'error' not in r]
    
    if valid_results:
        # Calculate aggregate metrics
        total_datasets = len(valid_results)
        top3_count = sum(1 for r in valid_results if r['top3'])
        top_half_count = sum(1 for r in valid_results if r['top_half'])
        better_than_baseline_count = sum(1 for r in valid_results if r['better_than_baseline'])
        
        avg_rank = np.mean([r['rank'] for r in valid_results])
        
        print(f"\nAggregate Performance:")
        print(f"  Total test datasets: {total_datasets}")
        print(f"  Top 3 recommendations: {top3_count}/{total_datasets} ({top3_count/total_datasets*100:.1f}%)")
        print(f"  Top half recommendations: {top_half_count}/{total_datasets} ({top_half_count/total_datasets*100:.1f}%)")
        print(f"  Better than baseline: {better_than_baseline_count}/{total_datasets} ({better_than_baseline_count/total_datasets*100:.1f}%)")
        print(f"  Average rank: {avg_rank:.2f}")
        
        # Pipeline analysis
        pipeline_recommendations = {}
        pipeline_performances = {}
        
        for result in valid_results:
            rec_pipeline = result['recommended_pipeline']
            pipeline_recommendations[rec_pipeline] = pipeline_recommendations.get(rec_pipeline, 0) + 1
            
            for pipeline, perf in result['all_performances'].items():
                if not np.isnan(perf):
                    if pipeline not in pipeline_performances:
                        pipeline_performances[pipeline] = []
                    pipeline_performances[pipeline].append(perf)
        
        print(f"\nPipeline Recommendation Frequency:")
        for pipeline, count in sorted(pipeline_recommendations.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pipeline}: {count} times ({count/total_datasets*100:.1f}%)")
        
        print(f"\nAverage Pipeline Performance Across Test Sets:")
        avg_performances = {p: np.mean(perfs) for p, perfs in pipeline_performances.items()}
        for pipeline, avg_perf in sorted(avg_performances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pipeline:20s}: {avg_perf:.4f}")
    
    performance_gap = None
    # Save detailed results
    if test_results:
        # Drop all_performances dict and round numeric values
        results_df = pd.DataFrame([
            {k: (round(v, 4) if isinstance(v, (int, float)) and not np.isnan(v) else v) 
             for k, v in r.items() if k != 'all_performances'}
            for r in test_results
        ])

        # Filter valid results
        #valid_results = [r for r in test_results if 'error' not in r]
        
        # Now compute averages safely (no NaNs left)
        avg_baseline_rank = np.nanmean([r['baseline_rank'] for r in valid_results]) if valid_results else np.nan
        avg_baseline_perf = np.nanmean([r['baseline_performance'] for r in valid_results]) if valid_results else np.nan
        avg_recommended_perf = np.nanmean([r['recommended_performance'] for r in valid_results]) if valid_results else np.nan
        avg_recommended_rank = np.nanmean([r['rank'] for r in valid_results]) if valid_results else np.nan
        avg_best_pref = np.nanmean([r['best_performance'] for r in valid_results]) if valid_results else np.nan
    
    
        # Compute performance gap (assuming this column exists)
        if 'score_gap_btw_best_and_recommended' in results_df.columns:
            performance_gap = np.nanmean(results_df['score_gap_btw_best_and_recommended'])
        else:
            performance_gap = np.nan
    
        # Compute percentage gap relative to average recommended performance
        if not np.isnan(performance_gap) and not np.isnan(avg_recommended_perf) and avg_recommended_perf != 0:
            performance_percentage = np.abs(avg_recommended_perf / (avg_best_pref)) * 100
        else:
            performance_percentage = np.nan

        # Insert summary values only in the first row
        if not results_df.empty:
            results_df.loc[0, 'average_baseline_rank'] = round(avg_baseline_rank, 2) if not np.isnan(avg_baseline_rank) else np.nan
            results_df.loc[0, 'average_recommended_rank'] = round(avg_recommended_rank, 2) if not np.isnan(avg_recommended_rank) else np.nan
            results_df.loc[0, 'average_baseline_performance'] = round(avg_baseline_perf, 4) if not np.isnan(avg_baseline_perf) else np.nan
            results_df.loc[0, 'average_recommended_performance'] = round(avg_recommended_perf, 4) if not np.isnan(avg_recommended_perf) else np.nan
            results_df.loc[0, 'average_perf_gap'] = round(performance_gap, 2) if not np.isnan(performance_gap) else np.nan
            results_df.loc[0, 'average_perf_percentage'] = round(performance_percentage, 2) if not np.isnan(performance_percentage) else np.nan

        # Save to CSV
        results_df.to_csv(f'test_evaluation_results_cluster_{cluster_num}.csv', index=False)
        print(f"\nDetailed results saved to 'test_evaluation_results_cluster_{cluster_num}.csv'")
    
        p_gap = results_df['score_gap_btw_best_and_recommended']
        gap_s = results_df['gap_scale']
        print("Performance gap:")
        print(p_gap)
        print("Percentage Scale:")
        print(gap_s)
        # Add cluster summary
    

        cluster_summaries.append({
            "cluster_num": cluster_num,
            "avg_baseline_rank": avg_baseline_rank,
            "avg_recommended_rank": avg_recommended_rank,
            "avg_baseline_perf": avg_baseline_perf,
            "avg_recommended_perf": avg_recommended_perf,
            "performance_gap": performance_gap,
            "performance_gap_percentage": round(performance_percentage, 2) if not np.isnan(performance_percentage) else np.nan
        })
        

# === Final summary across all cluster_num ===
print("\n" + "="*80)
print("COMPARISON ACROSS CLUSTER NUMBERS")
print("="*80)

summary_df = pd.DataFrame(cluster_summaries).round(4)
print(summary_df.to_string(index=False))

summary_df.to_csv("cluster_comparison_summary.csv", index=False)
print("\n📊 Saved cluster comparison summary to 'cluster_comparison_summary.csv'")