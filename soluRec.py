import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import math
import warnings
import tempfile
import os
import shutil
from autogluon.tabular import TabularPredictor
import openml

warnings.filterwarnings('ignore')

predictor_path = '/kaggle/working/autogluon_models'

AUTOGLUON_CONFIG = {
    "problem_type": "multiclass",
    "eval_metric": "accuracy",
    "time_limit": 600,  # 5 minutes per dataset
    "presets": "medium_quality",
    "verbosity": 1,
    "included_model_types": [
            "GBM",       # Gradient boosting (LightGBM variants)
            "CAT",       # CatBoost
            "XGB",       # XGBoost
            "RF",        # Random Forest
            "XT",        # Extra Trees
            "KNN",       # K-Nearest Neighbors
            "LR",        # Logistic Regression
            "NN_TORCH",  # Neural Networks (PyTorch)
            "FASTAI",    # FastAI Neural Networks
            "NN_MXNET",  # MXNet Neural Networks
            "TABPFN",    # TabPFN (if available)
            "DUMMY",     # Dummy classifier
            "NB"         # Naive Bayes
    ],
    "hyperparameter_tune_kwargs": None,
    "ag_args_fit": {
        "ag.max_memory_usage_ratio": 0.9,
        'num_gpus': 0  # Use CPU only for broader compatibility
    },
    "seed": 42
}

def preprocess_consistent_across_splits(X_train_val, y_train_val, X_test, y_test, config):
    """
    Apply preprocessing with consistent feature names across all splits
    while ensuring no duplicate indices.
    """
    # Make sure indices are reset at the beginning
    X_train_val = X_train_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train_val = y_train_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Store the original size for splitting back later
    train_val_size = len(X_train_val)
    
    # Combine datasets
    X_combined = pd.concat([X_train_val, X_test], axis=0).reset_index(drop=True)
    
    # We need to create a dummy y for preprocessing that won't affect the actual data
    dummy_y = pd.Series([0] * len(X_combined))
    
    # Apply preprocessing to combined data
    X_processed, _ = apply_preprocessing(X_combined, dummy_y, config)
    
    # If preprocessing failed or returned empty dataframe
    if X_processed is None or X_processed.empty:
        print(f"  Error: Preprocessing returned empty dataset for {config['name']}")
        return None, y_train_val, None, y_test
    
    # After preprocessing, split back based on the original indices
    X_train_val_processed = X_processed.iloc[:train_val_size].copy()
    X_test_processed = X_processed.iloc[train_val_size:].copy()
    
    # Make sure indices are reset again
    X_train_val_processed = X_train_val_processed.reset_index(drop=True)
    X_test_processed = X_test_processed.reset_index(drop=True)
    
    return X_train_val_processed, y_train_val, X_test_processed, y_test

def preprocess_consistent_features(X_train, y_train, X_val, y_val, config):
    try:
        # Create copies to avoid modifying originals
        X_train_copy = X_train.copy()
        X_val_copy = X_val.copy()
        
        # Handle categorical features first to ensure consistent encoding
        for col in X_train_copy.select_dtypes(include=['object', 'category']).columns:
            if col in X_val_copy.columns:
                # Create a combined series of all unique values from both datasets
                combined_values = pd.concat([
                    X_train_copy[col].fillna('missing'),
                    X_val_copy[col].fillna('missing')
                ]).unique()
                
                # Create a consistent mapping for both datasets
                value_map = {val: idx for idx, val in enumerate(combined_values)}
                
                # Apply the same mapping to both datasets
                X_train_copy[col] = X_train_copy[col].fillna('missing').map(value_map)
                X_val_copy[col] = X_val_copy[col].fillna('missing').map(value_map)
        
        # Now apply other preprocessing steps from the config
        preprocessor_func = create_preprocessing_pipeline(config)
        
        # Check if we have a valid preprocessor
        if preprocessor_func is None:
            return X_train_copy, y_train, X_val_copy, y_val
            
        # Create and fit preprocessor on training data only
        preprocessor = preprocessor_func(X_train_copy)
        if preprocessor is None:
            return X_train_copy, y_train, X_val_copy, y_val
            
        # Transform both datasets using fitted preprocessor
        try:
            X_train_transformed = preprocessor.fit_transform(X_train_copy)
            feature_names = preprocessor.get_feature_names_out()
            X_train_processed = pd.DataFrame(
                X_train_transformed,
                columns=feature_names
            )
            
            # Apply same transformation to validation data
            X_val_processed = pd.DataFrame(
                preprocessor.transform(X_val_copy),
                columns=feature_names
            )
            
        except Exception as transform_error:
            print(f"  Transformation error: {transform_error}")
            return X_train_copy, y_train, X_val_copy, y_val
            
        return X_train_processed, y_train, X_val_processed, y_val
        
    except Exception as e:
        print(f"  Error in consistent preprocessing: {e}")
        # Apply minimal processing (handle categoricals with label encoding)
        X_train_simple = X_train.copy()
        X_val_simple = X_val.copy()
        
        # Minimal preprocessing - just handle categoricals with LabelEncoder
        for col in X_train_simple.select_dtypes(include=['object', 'category']).columns:
            if col in X_val_simple.columns:
                le = LabelEncoder()
                combined = pd.concat([
                    X_train_simple[col].fillna('missing'),
                    X_val_simple[col].fillna('missing')
                ])
                le.fit(combined)
                X_train_simple[col] = le.transform(X_train_simple[col].fillna('missing'))
                X_val_simple[col] = le.transform(X_val_simple[col].fillna('missing'))
            
        return X_train_simple, y_train, X_val_simple, y_val

def fix_autogluon_predict(splits, pipeline_config):
    """Train on splits['X_train','y_train'] and evaluate on val splits with better error handling"""
    try:
        X_train = splits['X_train'].copy()
        y_train = splits['y_train']
        X_val = splits['X_val'].copy()
        y_val = splits['y_val']
        
        # Apply consistent preprocessing across train and validation
        X_train_processed, y_train_processed, X_val_processed, y_val_processed = preprocess_consistent_features(
            X_train, y_train, X_val, y_val, pipeline_config
        )
        
        if X_train_processed.empty or X_val_processed.empty:
            print(f"  Empty dataset after preprocessing")
            return np.nan
        
        # Prepare data format for AutoGluon
        train_data = X_train_processed.copy()
        train_data['target'] = y_train_processed
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Determine problem type automatically
            num_unique_classes = len(np.unique(y_train_processed))
            problem_type = 'multiclass' if num_unique_classes > 2 else 'binary'
            print(f"  Training with problem_type={problem_type}, {num_unique_classes} classes")
            
            # Define custom ag_args_fit to control resource usage
            ag_args_fit = {
                "ag.max_memory_usage_ratio": 0.3,  # Use at most 30% of available memory
                'num_gpus': 0,                     # Use CPU only for stability
                'num_cpus': min(10, os.cpu_count()) # Limit CPU usage
            }
            
            # Initialize predictor with specific problem type
            predictor = TabularPredictor(
                label='target',
                path=temp_dir,
                problem_type=problem_type,
                eval_metric='accuracy',
                verbosity=2  # Increase verbosity to debug issues
            )
            
            # Use a smaller subset of model types for better stability
            stable_models = [
                "GBM",       # Gradient boosting (LightGBM variants)
                "CAT",       # CatBoost
                "XGB",       # XGBoost
                "RF",        # Random Forest
                "XT",        # Extra Trees
                "KNN",       # K-Nearest Neighbors
                "LR",        # Logistic Regression
                "NN_TORCH",  # Neural Networks (PyTorch)
                "FASTAI",    # FastAI Neural Networks
                "NN_MXNET",  # MXNet Neural Networks
                "TABPFN",    # TabPFN (if available)
                "DUMMY",     # Dummy classifier
                "NB"         # Naive Bayes
            ] 
            
            # Fit with more controlled parameters
            predictor.fit(
                train_data,
                time_limit=600,
                presets='medium_quality',
                included_model_types=stable_models,
                hyperparameter_tune_kwargs=None,  # Disable HPO for stability
                feature_generator=None,  # Skip AutoGluon's feature generation
                ag_args_fit=ag_args_fit,
                raise_on_no_models_fitted=False  # Don't raise error if no models fit
            )
            
            # Better way to check if models were trained
            try:
                # First try the safer API approach
                model_names = predictor.get_model_names()
                if not model_names:
                    raise ValueError("No models were trained successfully")
            except Exception:
                # If the above fails, try accessing internal attributes
                try:
                    if not hasattr(predictor, '_trainer') or not predictor._trainer.models:
                        raise ValueError("No models were trained successfully")
                except:
                    # Last resort, just assume models were trained if no error
                    pass
            
            # Try to predict and handle various failure cases
            try:
                preds = predictor.predict(X_val_processed)
                score = accuracy_score(y_val_processed, preds)
                print(f"  AutoGluon score: {score:.4f}")
                return score
            except Exception as pred_error:
                print(f"  Prediction error: {str(pred_error)}")
                raise ValueError("Failed to generate predictions")
            
        except Exception as e:
            print(f"  AutoGluon error: {str(e)}")
            print("  Falling back to RandomForest")
            try:
                # Simple and robust fallback
                rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                rf.fit(X_train_processed, y_train_processed)
                preds = rf.predict(X_val_processed)
                score = accuracy_score(y_val_processed, preds)
                print(f"  RandomForest score: {score:.4f}")
                return score
            except Exception as inner_e:
                print(f"  Fallback also failed: {str(inner_e)}")
                return np.nan
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"  Unexpected error: {str(e)}")
        return np.nan

# Enhanced preprocessing pipeline configurations


# Test dataset IDs (holdout sets)
test_dataset_ids = [1503, 23517, 1551, 1552, 183, 255, 545, 546,
                    475, 481, 516]

# Training dataset IDs (larger pool for training the recommender)
# train_dataset_ids = [
#     3, 6, 8, 10, 12, 14, 16, 18, 22, 28,
#     32, 37, 44, 46, 50, 54, 182, 188, 300
#     # 307, 312, 333, 334, 335, 336, 337, 338, 5, 7, 
#     # 9, 11, 20, 23, 24, 26, 29, 30, 31, 33, 34
# ]
train_dataset_ids = [
                     22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
    #                  37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 53, 54, 55,
    #                  56, 57, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
                     187, 188, 275, 276,
    #                  277, 278, 285, 300, 301, 307, 308,
    #                  310, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336,
    #                  337, 338, 339, 340, 342, 343, 346, 372, 375, 377,
                     378, 443,
                     # 444, 446, 448, 449, 450, 451, 452, 453, 454, 455, 457, 458, 459, 461,
                     462, 463, 464, 465, 467, 468, 469, 471
                    ]

import os
os.environ["OMP_NUM_THREADS"] = "1"
from xgboost import XGBRanker 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
    FunctionTransformer, OneHotEncoder, LabelEncoder
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd

pipeline_configs = [
    # No preprocessing (baseline)
    {'name': 'baseline', 'imputation': 'none', 'scaling': 'none', 'encoding': 'none', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    
    # Simple imputation and scaling
    {'name': 'simple_preprocess', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    
    # Robust preprocessing for noisy data
    {'name': 'robust_preprocess', 'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'},
    
    # Feature selection focused
    {'name': 'feature_selection', 'imputation': 'median', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'k_best', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    
    # Dimensionality reduction focused
    {'name': 'dimension_reduction', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
   
    # Conservative preprocessing
    {'name': 'conservative', 'imputation': 'median', 'scaling': 'minmax', 'encoding': 'onehot', 
     'feature_selection': 'variance_threshold', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
    
    # Aggressive preprocessing
    {'name': 'aggressive', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
     'feature_selection': 'k_best', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'pca'},

    {
        'name': 'knn_impute_pca',
        'imputation': 'knn',
        'scaling': 'standard',
        'encoding': 'onehot',
        'feature_selection': 'none',
        'outlier_removal': 'none',
        'dimensionality_reduction': 'pca'
    },
    {
        'name': 'mutual_info_zscore',
        'imputation': 'median',
        'scaling': 'robust',
        'encoding': 'onehot',
        'feature_selection': 'mutual_info',
        'outlier_removal': 'zscore',
        'dimensionality_reduction': 'none'
    },
    {
        'name': 'constant_maxabs_iforest',
        'imputation': 'constant',
        'scaling': 'maxabs',
        'encoding': 'onehot',
        'feature_selection': 'variance_threshold',
        'outlier_removal': 'isolation_forest',
        'dimensionality_reduction': 'none'
    },
    {
        'name': 'mean_minmax_lof_svd',
        'imputation': 'mean',
        'scaling': 'minmax',
        'encoding': 'onehot',
        'feature_selection': 'k_best',
        'outlier_removal': 'lof',
        'dimensionality_reduction': 'svd'
    },
    {
        'name': 'mostfreq_standard_iqr',
        'imputation': 'most_frequent',
        'scaling': 'standard',
        'encoding': 'onehot',
        'feature_selection': 'none',
        'outlier_removal': 'iqr',
        'dimensionality_reduction': 'none'
    }
]

def split_dataset_stratified(dataset, train_frac=0.6, val_frac=0.2, test_frac=0.2, random_state=42):
    """
    Split a single dataset dict into stratified train/val/test splits.
    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = dataset['X']
    y = dataset['y']

    # Determine stratification (only for classification)
    stratify = y if (dataset.get('task_type') != 'regression' and len(np.unique(y)) > 1) else None

    # First split: train vs. (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_frac, random_state=random_state, stratify=stratify
    )

    # Second split: val vs test (from the remaining portion)
    remaining = 1.0 - train_frac
    val_prop = val_frac / remaining if remaining > 0 else 0.5

    stratify_temp = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_prop, random_state=random_state, stratify=stratify_temp
    )

    return {
        'X_train': X_train.reset_index(drop=True),
        'y_train': pd.Series(y_train).reset_index(drop=True),
        'X_val': X_val.reset_index(drop=True),
        'y_val': pd.Series(y_val).reset_index(drop=True),
        'X_test': X_test.reset_index(drop=True),
        'y_test': pd.Series(y_test).reset_index(drop=True)
    }

def create_preprocessing_pipeline(config):
    """Create a scikit-learn preprocessing pipeline based on configuration"""
    def get_column_transformer(X):
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        
        transformers = []
        
        # Numeric preprocessing
        if numeric_cols:
            numeric_steps = []

            # Imputation
            if config['imputation'] != 'none':
                if config['imputation'] in ['mean', 'median', 'most_frequent', 'constant']:
                    numeric_steps.append(('imputer', SimpleImputer(strategy=config['imputation'])))
                elif config['imputation'] == 'knn':
                    numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
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
        
        # Categorical preprocessing
        if categorical_cols:
            categorical_steps = []
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            
            if config['encoding'] == 'onehot':
                categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20)))
            
            categorical_pipeline = Pipeline(categorical_steps)
            transformers.append(('cat', categorical_pipeline, categorical_cols))
        
        if not transformers:
            return None
            
        return ColumnTransformer(transformers, remainder='drop')
    
    return get_column_transformer


# --------------------------
# Apply preprocessing
# --------------------------
def adaptive_dimensionality_reduction(X, method='pca', target_components=20):
    """Apply dimensionality reduction with adaptive component selection"""
    # Calculate max possible components based on data dimensions
    max_components = min(X.shape[0]-1, X.shape[1]) 
    n_components = min(target_components, max_components)
    
    # Safety check
    if n_components < 1:
        return X  # Return original data if reduction not possible
        
    # Apply reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    else:
        return X
        
    try:
        reduced = reducer.fit_transform(X)
        return pd.DataFrame(
            reduced, 
            index=X.index,
            columns=[f"{method.upper()}{i+1}" for i in range(n_components)]
        )
    except Exception as e:
        print(f"  Reduction error: {e}")
        return X

def apply_preprocessing(X, y, config):
    """Apply preprocessing pipeline with proper index tracking to ensure X and y stay aligned"""
    try:
        # Create copies to avoid modifying originals
        X_processed = X.copy().reset_index(drop=True)
        y_processed = pd.Series(y).reset_index(drop=True)
        
        # Skip if baseline (but still handle categorical features)
        if config['name'] == 'baseline':
            X_baseline = X_processed.copy()
            for col in X_baseline.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_baseline[col] = le.fit_transform(X_baseline[col].fillna('missing'))
            return X_baseline, y_processed
        
        # Build preprocessing pipeline
        preprocessor_func = create_preprocessing_pipeline(config)
        preprocessor = preprocessor_func(X_processed)
        if preprocessor is None:
            return X_processed, y_processed
        
        # Apply pipeline
        X_transformed = preprocessor.fit_transform(X_processed)
        
        # Convert back to DataFrame
        if not isinstance(X_transformed, pd.DataFrame):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        
        X_transformed = X_transformed.reset_index(drop=True)
        y_processed = y_processed.reset_index(drop=True)
        
        # Create a row tracking column before any filtering
        X_transformed['__row_id__'] = np.arange(len(X_transformed))
        
        # Outlier removal - CRITICAL: keep track of which rows are removed
        if config['outlier_removal'] == 'iqr':
            for col in X_transformed.select_dtypes(include=['number']).columns:
                if col == '__row_id__':
                    continue
                Q1, Q3 = X_transformed[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                if IQR > 0:
                    mask = (X_transformed[col] >= Q1 - 1.5 * IQR) & (X_transformed[col] <= Q3 + 1.5 * IQR)
                    X_transformed = X_transformed[mask]
            
            # Update y based on remaining row_ids
            remaining_rows = X_transformed['__row_id__'].values
            y_processed = y_processed.iloc[remaining_rows].reset_index(drop=True)
        
        elif config['outlier_removal'] == 'zscore':
            from scipy.stats import zscore
            # Only apply zscore to numeric columns that aren't the row ID
            numeric_cols = [col for col in X_transformed.select_dtypes(include=['number']).columns 
                           if col != '__row_id__']
            
            if numeric_cols:  # Make sure there are numeric columns to process
                z_scores = np.abs(zscore(X_transformed[numeric_cols], nan_policy='omit'))
                mask = (z_scores < 3).all(axis=1)
                X_transformed = X_transformed[mask]
                
                # Update y based on remaining row_ids
                remaining_rows = X_transformed['__row_id__'].values
                y_processed = y_processed.iloc[remaining_rows].reset_index(drop=True)
        
        elif config['outlier_removal'] == 'lof':
            # Only apply LOF to numeric columns that aren't the row ID
            numeric_cols = [col for col in X_transformed.select_dtypes(include=['number']).columns 
                           if col != '__row_id__']
            
            if numeric_cols and len(X_transformed) > 20:  # Need enough samples for LOF
                lof = LocalOutlierFactor(n_neighbors=min(20, len(X_transformed) - 1))
                y_pred = lof.fit_predict(X_transformed[numeric_cols])
                mask = y_pred == 1
                X_transformed = X_transformed[mask]
                
                # Update y based on remaining row_ids
                remaining_rows = X_transformed['__row_id__'].values
                y_processed = y_processed.iloc[remaining_rows].reset_index(drop=True)
        
        elif config['outlier_removal'] == 'isolation_forest':
            # Only apply Isolation Forest to numeric columns that aren't the row ID
            numeric_cols = [col for col in X_transformed.select_dtypes(include=['number']).columns 
                           if col != '__row_id__']
            
            if numeric_cols and len(X_transformed) > 20:  # Need enough samples
                iso = IsolationForest(contamination=0.05, random_state=42)
                y_pred = iso.fit_predict(X_transformed[numeric_cols])
                mask = y_pred == 1
                X_transformed = X_transformed[mask]
                
                # Update y based on remaining row_ids
                remaining_rows = X_transformed['__row_id__'].values
                y_processed = y_processed.iloc[remaining_rows].reset_index(drop=True)
        
        # Remove the row tracking column before continuing
        X_transformed = X_transformed.drop('__row_id__', axis=1)
        
        # Feature selection
        if config['feature_selection'] == 'variance_threshold':
            selector = VarianceThreshold(threshold=0.01)
            X_transformed = pd.DataFrame(
                selector.fit_transform(X_transformed),
                columns=X_transformed.columns[selector.get_support()]
            )
        
        elif config['feature_selection'] == 'k_best':
            k = min(max(5, X_transformed.shape[1] // 4), X_transformed.shape[1] - 1)
            selector = SelectKBest(f_classif, k=k)
            X_transformed = pd.DataFrame(
                selector.fit_transform(X_transformed, y_processed),
                columns=X_transformed.columns[selector.get_support()]
            )
        
        elif config['feature_selection'] == 'mutual_info':
            k = min(max(5, X_transformed.shape[1] // 4), X_transformed.shape[1] - 1)
            selector = SelectKBest(mutual_info_classif, k=k)
            X_transformed = pd.DataFrame(
                selector.fit_transform(X_transformed, y_processed),
                columns=X_transformed.columns[selector.get_support()]
            )
        
        # Dimensionality reduction
        if config['dimensionality_reduction'] == 'pca':
            X_transformed = adaptive_dimensionality_reduction(
                X_transformed, 
                method='pca',
                target_components=min(20, X_transformed.shape[1] // 2)
            )
        
        elif config['dimensionality_reduction'] == 'svd':
            X_transformed = adaptive_dimensionality_reduction(
                X_transformed, 
                method='svd',
                target_components=min(20, X_transformed.shape[1] // 2)
            )
        
        # Final cleanup
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
        y_processed = y_processed.reset_index(drop=True)
        
        return X_transformed, y_processed
    
    except Exception as e:
        print(f"Error in preprocessing {config['name']}: {e}")
        X_fallback = X.copy()
        for col in X_fallback.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_fallback[col] = le.fit_transform(X_fallback[col].fillna('missing'))
        return X_fallback.reset_index(drop=True), pd.Series(y).reset_index(drop=True)

meta_features_df = pd.read_csv("dataset_feats.csv", index_col=0)

print(meta_features_df.head())
print(meta_features_df.loc[2])
print(meta_features_df.shape)


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
            'name': f"Dataset_{dataset_id}",
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
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y_processed, return_counts=True)
        min_class_count = class_counts.min()
        
        if min_class_count < 3:
            print(f"Skipping {pipeline_config['name']} on {dataset['name']}: class with only {min_class_count} sample(s)")
            return np.nan
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42
            )
        
        # Prepare data for AutoGluon
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        test_data = X_test.copy()
        
        # Create temporary directory for AutoGluon models
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Configure AutoGluon
            config = AUTOGLUON_CONFIG.copy()
            config['path'] = temp_dir
            
            # Train AutoGluon predictor
                        # Initialize predictor with specific problem type
            predictor = TabularPredictor(
                label='target',
                path=temp_dir,
                problem_type=problem_type,
                eval_metric='accuracy',
                verbosity=2  # Increase verbosity to debug issues
            )
            
            # Use a smaller subset of model types for better stability
            stable_models = [
                "GBM",       # Gradient boosting (LightGBM variants)
                "CAT",       # CatBoost
                "XGB",       # XGBoost
                "RF",        # Random Forest
                "XT",        # Extra Trees
                "KNN",       # K-Nearest Neighbors
                "LR",        # Logistic Regression
                "NN_TORCH",  # Neural Networks (PyTorch)
                "FASTAI",    # FastAI Neural Networks
                "NN_MXNET",  # MXNet Neural Networks
                "TABPFN",    # TabPFN (if available)
                "DUMMY",     # Dummy classifier
                "NB"         # Naive Bayes
            ] 
            
            # Fit with more controlled parameters
            predictor.fit(
                train_data,
                time_limit=600,
                presets='medium_quality',
                included_model_types=stable_models,
                hyperparameter_tune_kwargs=None,  # Disable HPO for stability
                feature_generator=None,  # Skip AutoGluon's feature generation
                ag_args_fit=ag_args_fit,
                raise_on_no_models_fitted=False  # Don't raise error if no models fit
            )
            predictions = predictor.predict(test_data)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, predictions)
            
            return accuracy
            
        except Exception as e:
            print(f"AutoGluon error for {pipeline_config['name']}: {e}")
            # Fallback to simple evaluation
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return accuracy_score(y_test, y_pred)
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"Error evaluating pipeline {pipeline_config['name']} on {dataset['name']}: {e}")
        return np.nan

# def extract_enhanced_metafeatures(dataset):
#     """Extract enhanced meta-features from a dataset"""
#     try:
#         X, y = dataset['X'], dataset['y']
        
#         # Ensure y is numeric
#         if hasattr(y, 'dtype') and (y.dtype == 'object' or y.dtype.name == 'category'):
#             le = LabelEncoder()
#             y_numeric = le.fit_transform(y)
#         else:
#             y_numeric = np.array(y)
        
#         # Sample if dataset is too large
        # if len(X) > 3000:
        #     idx = np.random.choice(len(X), 3000, replace=False)
        #     X_sample = X.iloc[idx]
        #     y_sample = y_numeric[idx]
        # else:
        #     X_sample = X
        #     y_sample = y_numeric
        
        # metafeatures = {}
        
        # # Basic dataset characteristics
        # metafeatures['n_instances'] = len(X_sample)
        # metafeatures['n_features'] = X_sample.shape[1]
        # metafeatures['n_classes'] = len(np.unique(y_sample))
        # metafeatures['n_numeric_features'] = len(X_sample.select_dtypes(include=['number']).columns)
        # metafeatures['n_categorical_features'] = len(X_sample.select_dtypes(exclude=['number']).columns)
        
        # # Dataset ratios
        # metafeatures['instances_to_features'] = metafeatures['n_instances'] / max(1, metafeatures['n_features'])
        # metafeatures['categorical_ratio'] = metafeatures['n_categorical_features'] / max(1, metafeatures['n_features'])
        
        # # Class characteristics
        # class_counts = np.bincount(y_sample.astype(int))
        # metafeatures['class_imbalance'] = (class_counts.max() - class_counts.min()) / max(1, class_counts.max())
        # metafeatures['minority_class_ratio'] = class_counts.min() / max(1, class_counts.sum())
        
        # # Missing value characteristics
        # metafeatures['missing_ratio'] = X_sample.isnull().sum().sum() / (X_sample.shape[0] * X_sample.shape[1])
        
        # # Statistical features for numeric data
        # numeric_cols = X_sample.select_dtypes(include=['number'])
        # if len(numeric_cols.columns) > 0:
        #     means = numeric_cols.mean()
        #     metafeatures['mean_of_means'] = means.mean() if not means.isna().all() else 0
            
    #         stds = numeric_cols.std()
    #         metafeatures['mean_std'] = stds.mean() if not stds.isna().all() else 0
    #         metafeatures['std_of_stds'] = stds.std() if not stds.isna().all() else 0
            
    #         skews = numeric_cols.skew()
    #         metafeatures['mean_abs_skew'] = abs(skews).mean() if not skews.isna().all() else 0
            
    #         kurtoses = numeric_cols.kurtosis()
    #         metafeatures['mean_abs_kurtosis'] = abs(kurtoses).mean() if not kurtoses.isna().all() else 0
            
    #         corr_matrix = numeric_cols.corr()
    #         upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    #         correlations = upper_tri.stack()
    #         metafeatures['mean_abs_correlation'] = abs(correlations).mean() if len(correlations) > 0 else 0
            
    #     else:
    #         metafeatures.update({
    #             'mean_of_means': 0, 'mean_std': 0, 'std_of_stds': 0,
    #             'mean_abs_skew': 0, 'mean_abs_kurtosis': 0, 'mean_abs_correlation': 0
    #         })
        
    #     return metafeatures
    # except Exception as e:
    #     print(f"Error extracting meta-features from {dataset['name']}: {e}")
    #     return {}

def extract_enhanced_metafeatures(dataset, meta_features_df=meta_features_df):
    """
    Fetch precomputed meta-features for a dataset from a CSV/Excel file.
    dataset: dict with 'id' field
    """
    try:
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
            dataset_names.append(f"{dataset['id']}_{dataset['name']}")
    
    if metafeatures_list:
        metafeatures_df = pd.DataFrame(metafeatures_list, index=dataset_names)
        return metafeatures_df
    else:
        return pd.DataFrame()

def build_performance_matrix(datasets, pipeline_configs, use_autogluon=True):
    """Build performance matrix by evaluating each pipeline on each dataset"""
    performance_matrix = pd.DataFrame(
        index=[config['name'] for config in pipeline_configs],
        columns=[f"{dataset['id']}_{dataset['name']}" for dataset in datasets]
    )
    
    eval_func = evaluate_pipeline_with_autogluon if use_autogluon else evaluate_pipeline_fallback
    
    for config in pipeline_configs:
        print(f"\nEvaluating pipeline: {config['name']}")
        for dataset in datasets:
            print(f"  Dataset: {dataset['name']} (ID: {dataset['id']})")
            performance = eval_func(dataset, config)
            performance_matrix.loc[config['name'], f"{dataset['id']}_{dataset['name']}"] = performance
            if not np.isnan(performance):
                print(f"    Performance: {performance:.4f}")
    
    return performance_matrix


def build_performance_matrix_with_splits(datasets, pipeline_configs, use_autogluon=True, train_frac=0.6, val_frac=0.2, test_frac=0.2):
    dataset_names = [f"{d['id']}_{d['name']}" for d in datasets]
    performance_matrix = pd.DataFrame(index=[cfg['name'] for cfg in pipeline_configs], columns=dataset_names, dtype=float)
    dataset_splits = {}
    
    for ds in datasets:
        ds_name = f"{ds['id']}_{ds['name']}"
        print(f"\nPreparing splits for dataset {ds_name}")
        
        try:
            splits = split_dataset_stratified(ds, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
            dataset_splits[ds_name] = splits
            
            for cfg in pipeline_configs:
                print(f"  Evaluating pipeline {cfg['name']} on validation split")
                
                # Use our fixed evaluation function
                if use_autogluon:
                    score = fix_autogluon_predict(splits, cfg)
                else:
                    score = evaluate_pipeline_fallback_on_splits(splits, cfg)
                    
                performance_matrix.loc[cfg['name'], ds_name] = score
                if not np.isnan(score):
                    print(f"    Validation score: {score:.4f}")
        except Exception as e:
            print(f"  Error processing dataset {ds_name}: {e}")
            continue
    
    return performance_matrix, dataset_splits


def evaluate_pipeline_fallback(dataset, pipeline_config):
    """Fallback evaluation using sklearn classifiers"""
    try:
        X, y = dataset['X'], dataset['y']
        X_processed, y_processed = apply_preprocessing(X, y, pipeline_config)
        
        if X_processed.empty or len(y_processed) == 0:
            return np.nan
        
        unique_classes, class_counts = np.unique(y_processed, return_counts=True)
        if class_counts.min() < 3:
            return np.nan
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42
            )
        
        classifiers = [
            RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
            LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
        ]
        
        accuracies = []
        for clf in classifiers:
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            except:
                continue
        
        return np.mean(accuracies) if accuracies else np.nan
        
    except Exception as e:
        print(f"Error evaluating pipeline {pipeline_config['name']} on {dataset['name']}: {e}")
        return np.nan


class EnhancedPreprocessingRecommender:
    def __init__(self, performance_matrix, metafeatures_df):
        self.performance_matrix = performance_matrix.astype(float)
        self.metafeatures_df = metafeatures_df
        self.pipeline_configs = pipeline_configs
        self.ranker = None
        
        if len(metafeatures_df) == 0:
            print("Warning: No metafeatures available")
            return
        
        # Prepare meta-features for similarity search
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.metafeatures_imputed = self.imputer.fit_transform(metafeatures_df)
        self.metafeatures_scaled = self.scaler.fit_transform(self.metafeatures_imputed)
        
        # Fit nearest neighbors model
        k_neighbors = min(len(metafeatures_df), 5)
        if k_neighbors > 0:
            self.nn_euclidean = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
            self.nn_euclidean.fit(self.metafeatures_scaled)
            
            self.nn_manhattan = NearestNeighbors(n_neighbors=k_neighbors, metric='manhattan')
            self.nn_manhattan.fit(self.metafeatures_scaled)
        else:
            self.nn_euclidean = None
            self.nn_manhattan = None
    
    # def recommend(self, new_dataset, k=3):
    #     """Recommend preprocessing pipeline for a new dataset"""
    #     if self.nn_euclidean is None or len(self.metafeatures_df) == 0:
    #         # Fallback: return the best overall pipeline
    #         valid_performances = self.performance_matrix.mean(axis=1, skipna=True).dropna()
    #         if len(valid_performances) > 0:
    #             top_pipeline_name = valid_performances.idxmax()
    #             top_pipeline_config = next(config for config in pipeline_configs if config['name'] == top_pipeline_name)
    #             return {
    #                 'pipeline_config': top_pipeline_config,
    #                 'expected_performance': valid_performances[top_pipeline_name],
    #                 'similar_datasets': [],
    #                 'pipeline_ranking': valid_performances.sort_values(ascending=False).index.tolist(),
    #                 'confidence': 'low'
    #             }
    #         return None
        
    #     # Extract meta-features for new dataset
    #     new_metafeatures = extract_enhanced_metafeatures(new_dataset)
        
    #     if not new_metafeatures:
    #         return None
        
    #     # Convert to DataFrame and align columns
    #     new_mf_df = pd.DataFrame([new_metafeatures])
    #     new_mf_df = new_mf_df.reindex(columns=self.metafeatures_df.columns, fill_value=0)
        
    #     # Impute and scale
    #     new_mf_imputed = self.imputer.transform(new_mf_df)
    #     new_mf_scaled = self.scaler.transform(new_mf_imputed)
        
    #     # Find nearest neighbors
    #     k = min(k, len(self.metafeatures_df))
        
    #     distances_euc, indices_euc = self.nn_euclidean.kneighbors(new_mf_scaled, n_neighbors=k)
    #     similar_datasets_euc = self.metafeatures_df.iloc[indices_euc[0]].index.tolist()
        
    #     distances_man, indices_man = self.nn_manhattan.kneighbors(new_mf_scaled, n_neighbors=k)
    #     similar_datasets_man = self.metafeatures_df.iloc[indices_man[0]].index.tolist()
        
    #     all_similar = list(set(similar_datasets_euc + similar_datasets_man))
        
    #     # Weight by distance
    #     weighted_performances = {}
    #     for pipeline in self.performance_matrix.index:
    #         weighted_sum = 0
    #         weight_sum = 0
            
    #         for i, dataset in enumerate(similar_datasets_euc):
    #             if dataset in self.performance_matrix.columns:
    #                 perf = self.performance_matrix.loc[pipeline, dataset]
    #                 if not np.isnan(perf):
    #                     weight = 1.0 / (1.0 + distances_euc[0][i])
    #                     weighted_sum += perf * weight
    #                     weight_sum += weight
            
    #         for i, dataset in enumerate(similar_datasets_man):
    #             if dataset in self.performance_matrix.columns:
    #                 perf = self.performance_matrix.loc[pipeline, dataset]
    #                 if not np.isnan(perf):
    #                     weight = 1.0 / (1.0 + distances_man[0][i])
    #                     weighted_sum += perf * weight
    #                     weight_sum += weight
            
    #         if weight_sum > 0:
    #             weighted_performances[pipeline] = weighted_sum / weight_sum
        
    #     if not weighted_performances:
    #         return None
        
    #     # Get top pipeline
    #     top_pipeline_name = max(weighted_performances.keys(), key=lambda x: weighted_performances[x])
    #     top_pipeline_performance = weighted_performances[top_pipeline_name]
    #     top_pipeline_config = next(config for config in pipeline_configs if config['name'] == top_pipeline_name)
        
    #     # Determine confidence
    #     avg_distance = (np.mean(distances_euc[0]) + np.mean(distances_man[0])) / 2
    #     confidence = 'high' if avg_distance < 1.0 else 'medium' if avg_distance < 2.0 else 'low'
        
    #     # Rank all pipelines
    #     pipeline_ranking = sorted(weighted_performances.keys(), key=lambda x: weighted_performances[x], reverse=True)
        
    #     return {
    #         'pipeline_config': top_pipeline_config,
    #         'expected_performance': top_pipeline_performance,
    #         'similar_datasets': all_similar[:k],
    #         'pipeline_ranking': pipeline_ranking,
    #         'confidence': confidence,
    #         'similarity_scores': {
    #             'euclidean_distance': np.mean(distances_euc[0]),
    #             'manhattan_distance': np.mean(distances_man[0])
    #         }
    #     }

    def train_ranker(self):
        """Train an XGBRanker on historical dataset–pipeline performances"""
        X_train = []
        y_train = []
        group = []
        
        # Debug information
        print(f"Training data: {self.performance_matrix.shape} performance matrix")
        print(f"Metafeatures: {self.metafeatures_df.shape}")
        
        # Build training data
        for dataset_id in self.performance_matrix.columns:
            if dataset_id not in self.metafeatures_df.index:
                print(f"Skipping {dataset_id} - not found in metafeatures")
                continue
                
            mf = self.metafeatures_df.loc[dataset_id].values
            pipeline_scores = self.performance_matrix[dataset_id]
    
            valid_scores = pipeline_scores.dropna()
            if len(valid_scores) == 0:
                continue
    
            # Instead of hashing, use normalized index
            pipeline_names = self.performance_matrix.index.tolist()
            for pipeline_name, score in valid_scores.items():
                pipeline_idx = pipeline_names.index(pipeline_name) / len(pipeline_names)
                X_train.append(np.concatenate([mf, [pipeline_idx]]))
                y_train.append(score)
    
            group.append(len(valid_scores))
    
        if len(X_train) == 0:
            print("⚠️ No training data available for ranker.")
            return None
    
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training ranker with {len(X_train)} examples, {len(group)} groups")
    
        # Train the ranker with more conservative parameters
        self.ranker = XGBRanker(
            objective='rank:pairwise',
            n_estimators=100,     # Reduced from 200
            learning_rate=0.05,   # Reduced from 0.1
            max_depth=4,          # Reduced from 6
            gamma=1,              # Regularization parameter
            random_state=42
        )
    
        self.ranker.fit(
            X_train,
            y_train,
            group=group,
            verbose=True
        )
    
        # Validate on training data
        preds = self.ranker.predict(X_train)
        print("Training predictions range:", min(preds), "-", max(preds))
        
        print("✅ Ranker training completed.")
        return self.ranker
        
    def recommend(self, new_dataset, k=10):
        """Recommend preprocessing pipelines for a new dataset using trained XGBRanker"""
        if self.ranker is None:
            # Fallback to simple recommendation
            print("Ranker not trained, using fallback recommendation")
            return self._fallback_recommendation()
        
        # Extract meta-features for new dataset
        new_metafeatures = extract_enhanced_metafeatures(new_dataset)
        if not new_metafeatures:
            print("Could not extract metafeatures, using fallback recommendation")
            return self._fallback_recommendation()
        
        # Align feature space with training
        new_mf_df = pd.DataFrame([new_metafeatures])
        new_mf_df = new_mf_df.reindex(columns=self.metafeatures_df.columns, fill_value=0)
        
        # Apply preprocessing (imputer + scaler)
        new_mf_imputed = self.imputer.transform(new_mf_df)
        new_mf_scaled = self.scaler.transform(new_mf_imputed).flatten()
        
        # Build feature matrix for all pipelines
        pipeline_names = self.performance_matrix.index.tolist()
        X_new = []
        
        # Use normalized index instead of hash
        for pipeline_name in pipeline_names:
            pipeline_idx = pipeline_names.index(pipeline_name) / len(pipeline_names)
            features_with_id = np.concatenate([new_mf_scaled, [pipeline_idx]])
            X_new.append(features_with_id)
        
        X_new = np.array(X_new)
        
        # Predict performance scores and normalize to [0,1] range
        preds = self.ranker.predict(X_new)
        preds = np.clip(preds, 0, 1)  # Clip to valid range
        
        weighted_performances = dict(zip(pipeline_names, preds))
        
        print(f"Debug - predicted scores (clipped to [0,1]):")
        for pipeline, score in sorted(weighted_performances.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {pipeline}: {score:.4f}")
    
    # Rest of the recommendation code...
    
        if not weighted_performances:
            return None
    
        # --- 4. Rank pipelines ---
        pipeline_ranking = sorted(weighted_performances.keys(),
                                  key=lambda x: weighted_performances[x],
                                  reverse=True)
    
        top_pipeline_name = pipeline_ranking[0]
        top_pipeline_score = weighted_performances[top_pipeline_name]
        top_pipeline_config = next(cfg for cfg in pipeline_configs if cfg['name'] == top_pipeline_name)
    
        # --- 5. Confidence estimation (based on score spread) ---
        score_values = np.array(list(weighted_performances.values()))
        spread = score_values.max() - score_values.min()
        confidence = 'high' if spread > 0.2 else 'medium' if spread > 0.1 else 'low'


        return {
            'pipeline_config': top_pipeline_config,
            'expected_performance': float(top_pipeline_score),
            'similar_datasets': [],  # XGBRanker is global, no neighbors
            'pipeline_ranking': pipeline_ranking[:k],  # top-k pipelines
            'confidence': confidence,
            'similarity_scores': {
                'model_type': 'XGBRanker'
            }
        }


class BayesianSurrogateRecommender:
    """
    Bayesian surrogate recommender that predicts pipeline performance for a new dataset.
    Uses a RandomForestRegressor to model the relationship between meta-features, pipeline identity,
    and performance. Handles high-dimensional meta-features better than KNN approaches.
    """
    def __init__(self, performance_matrix, metafeatures_df, random_state=42):
        self.performance_matrix = performance_matrix.astype(float)
        self.metafeatures_df = metafeatures_df
        self.random_state = random_state
        
        # Preprocessing for meta-features
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        # Encode pipeline identity
        self.pipeline_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Surrogate model - RandomForestRegressor is effective for high-dimensional data
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        )
        
    def fit(self):
        """
        Train surrogate model to predict pipeline performance based on meta-features.
        """
        # Use original or reduced meta-features
        mf_df = self.metafeatures_df
        
        # Skip if insufficient data
        if mf_df is None or mf_df.empty or self.performance_matrix.empty:
            print("Insufficient data for training surrogate model")
            return False
            
        # Ensure all datasets in performance matrix are also in metafeatures
        common_datasets = set(mf_df.index).intersection(set(self.performance_matrix.columns))
        if len(common_datasets) < 3:
            print(f"Too few common datasets between metafeatures and performance matrix: {len(common_datasets)}")
            return False
        
        # Preprocess meta-features
        X_mf = self.imputer.fit_transform(mf_df)
        X_mf = self.scaler.fit_transform(X_mf)
        
        # Build training data: [meta-features, pipeline_onehot] → performance
        X_train = []
        y_train = []
        pipeline_names = []
        
        for dataset_name in self.performance_matrix.columns:
            if dataset_name not in mf_df.index:
                continue
                
            # Get meta-features for this dataset
            dataset_mf = X_mf[mf_df.index.get_loc(dataset_name)]
            
            # For each pipeline with valid performance on this dataset
            for pipeline_name, performance in self.performance_matrix[dataset_name].items():
                if not np.isnan(performance):
                    X_train.append(np.hstack([dataset_mf, [pipeline_name]]))  # Temporarily add pipeline name as a feature
                    y_train.append(performance)
                    pipeline_names.append(pipeline_name)
        
        if len(X_train) == 0:
            print("No valid training examples for surrogate model")
            return False
            
        # Extract pipeline names for encoding
        pipeline_names = np.array(pipeline_names).reshape(-1, 1)
        
        # One-hot encode pipeline names
        pipeline_encoded = self.pipeline_encoder.fit_transform(pipeline_names)
        
        # Create final training matrix: [meta-features, pipeline_onehot]
        X_final = []
        for i, x in enumerate(X_train):
            # Remove temporary pipeline name string
            meta_features = x[:-1]
            # Add one-hot encoded pipeline
            X_final.append(np.hstack([meta_features, pipeline_encoded[i]]))
        
        # Train surrogate model
        print(f"Training surrogate model with {len(X_final)} examples")
        self.model.fit(np.array(X_final), np.array(y_train))
        print("Surrogate model training complete")
        
        # Calculate feature importance
        feature_importance = self.model.feature_importances_
        print("\nTop 10 most important features:")
        n_mf_features = X_mf.shape[1]
        
        # Meta-feature importance
        if n_mf_features > 0:
            mf_importance = feature_importance[:n_mf_features]
            top_mf_idx = np.argsort(mf_importance)[::-1][:10]
            for i, idx in enumerate(top_mf_idx):
                if idx < len(mf_df.columns):
                    print(f"  {i+1}. Meta-feature '{mf_df.columns[idx]}': {mf_importance[idx]:.4f}")
                    
        return True
    
    def recommend(self, new_dataset_metafeatures, k=5):
        """
        Recommend top-k pipelines for a new dataset based on predicted performance.
        
        Args:
            new_dataset_metafeatures: Meta-features for the new dataset
            k: Number of top pipelines to recommend
            
        Returns:
            Dictionary with recommendation results
        """
        if not hasattr(self, 'model') or self.model is None:
            print("Surrogate model not trained")
            return None
            
        # Convert meta-features to proper format
        if isinstance(new_dataset_metafeatures, dict):
            # Align with training meta-features
            mf_values = []
            for col in self.metafeatures_df.columns:
                mf_values.append(new_dataset_metafeatures.get(col, 0))
            new_mf = np.array(mf_values).reshape(1, -1)
        else:
            new_mf = np.array(new_dataset_metafeatures).reshape(1, -1)
            
        # Preprocess using fitted transformers
        new_mf = self.imputer.transform(new_mf)
        new_mf = self.scaler.transform(new_mf)
        
        # Get all pipeline names
        pipeline_names = self.performance_matrix.index.tolist()
        
        # Create test samples for each pipeline
        X_test = []
        for pipeline in pipeline_names:
            # Encode pipeline
            pipeline_encoded = self.pipeline_encoder.transform([[pipeline]])
            # Combine with meta-features
            X_test.append(np.hstack([new_mf[0], pipeline_encoded[0]]))
            
        # Predict performance for each pipeline
        X_test = np.array(X_test)
        predictions = self.model.predict(X_test)
        
        # Rank pipelines by predicted performance
        pipeline_scores = {pipeline: score for pipeline, score in zip(pipeline_names, predictions)}
        ranked_pipelines = sorted(pipeline_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top pipeline and its configuration
        top_pipeline = ranked_pipelines[0][0]
        top_pipeline_config = next(cfg for cfg in pipeline_configs if cfg['name'] == top_pipeline)
        
        # Calculate prediction variance/confidence
        pipeline_preds = []
        for i in range(10):  # Sample 10 trees
            tree_idx = np.random.randint(0, self.model.n_estimators)
            tree_preds = self.model.estimators_[tree_idx].predict(X_test)
            pipeline_preds.append(tree_preds)
            
        pipeline_vars = np.var(pipeline_preds, axis=0)
        confidence_level = 'high' if np.mean(pipeline_vars) < 0.01 else 'medium' if np.mean(pipeline_vars) < 0.05 else 'low'
        
        return {
            'pipeline_config': top_pipeline_config,
            'pipeline_name': top_pipeline,
            'expected_performance': float(ranked_pipelines[0][1]),
            'pipeline_ranking': [p for p, _ in ranked_pipelines[:k]],
            'confidence': confidence_level,
            'model_type': 'BayesianSurrogate'
        }


def run_comprehensive_evaluation():
    """Run comprehensive evaluation with train/test split"""
    print("="*80)
    print("ENHANCED PREPROCESSING RECOMMENDER WITH AUTOGLUON")
    print("="*80)
    
    # Load training datasets
    print("\nLoading training datasets...")
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
        return
    
    # Build training performance matrix
    print("\nBuilding training performance matrix with AutoGluon...")
    train_performance_matrix = build_performance_matrix(train_datasets, pipeline_configs, use_autogluon=True)
    print("\nTraining Performance Matrix:")
    print(train_performance_matrix.round(4))
    
    # Save training results
    train_performance_matrix.to_csv('training_performance_matrix_autogluon.csv')
    print("\nTraining performance matrix")
    # Build training metafeatures matrix
    print("\nBuilding training metafeatures matrix...")
    train_metafeatures_df = build_metafeatures_matrix(train_datasets)
    print("\nTraining Meta-features Matrix:")
    print(train_metafeatures_df.round(4))
    
    if not train_metafeatures_df.empty:
        train_metafeatures_df.to_csv('training_metafeatures.csv')
        print("\nTraining meta-features matrix saved")
    
    # Train recommender system
    print("\nTraining recommender system...")
    recommender = EnhancedPreprocessingRecommender(train_performance_matrix, train_metafeatures_df)
    recommender.train_ranker()
    # Load test datasets
    print("\nLoading test datasets...")
    test_datasets = []
    for dataset_id in test_dataset_ids:
        dataset = load_openml_dataset(dataset_id)
        if dataset:
            test_datasets.append(dataset)
    
    print(f"\nLoaded {len(test_datasets)} test datasets")
    
    # Evaluate on each test dataset
    test_results = []
    
    for test_dataset in test_datasets:
        print(f"\n" + "="*60)
        print(f"TESTING ON {test_dataset['name']} (ID: {test_dataset['id']})")
        print("="*60)
        
        # Get recommendation
        recommendation = recommender.recommend(test_dataset)
        
        if recommendation:
            print(f"\nRecommendation (confidence: {recommendation['confidence']}):")
            print("Pipeline configuration:")
            for key, value in recommendation['pipeline_config'].items():
                if key != 'name':
                    print(f"  {key}: {value}")
            print(f"Expected performance: {recommendation['expected_performance']:.4f}")
            print(f"Similar datasets: {recommendation['similar_datasets']}")
            
            # Evaluate all pipelines on test dataset
            print("\nEvaluating all pipelines with AutoGluon...")
            test_performances = {}
            
            for config in pipeline_configs:
                print(f"  Testing {config['name']}...")
                perf = evaluate_pipeline_with_autogluon(test_dataset, config)
                test_performances[config['name']] = perf
                status = "⭐ RECOMMENDED" if config['name'] == recommendation['pipeline_config']['name'] else ""
                print(f"    {config['name']:20s}: {perf:.4f} {status}")
            
            # Analysis
            valid_performances = [(name, perf) for name, perf in test_performances.items() if not np.isnan(perf)]
            
            if valid_performances:
                sorted_pipelines = sorted(valid_performances, key=lambda x: x[1], reverse=True)
                best_pipeline, best_performance = sorted_pipelines[0]
                
                rec_name = recommendation['pipeline_config']['name']
                rec_performance = test_performances[rec_name]
                
                print(f"\nResults Analysis:")
                print(f"  Best actual pipeline: {best_pipeline} ({best_performance:.4f})")
                print(f"  Recommended pipeline: {rec_name} ({rec_performance:.4f})")
                
                # Calculate metrics
                pipeline_names = [name for name, _ in sorted_pipelines]
                rec_rank = pipeline_names.index(rec_name) + 1 if rec_name in pipeline_names else len(pipeline_names)
                perf_diff = rec_performance - best_performance if not np.isnan(rec_performance) else float('-inf')
                
                print(f"  Recommendation rank: {rec_rank}/{len(pipeline_names)}")
                print(f"  Performance gap: {perf_diff:.4f}")
                
                # Success metrics
                top3 = rec_rank <= 3
                top_half = rec_rank <= len(pipeline_names) // 2
                better_than_baseline = (not np.isnan(rec_performance) and 
                                      not np.isnan(test_performances.get('baseline', np.nan)) and
                                      rec_performance > test_performances['baseline'])
                
                print(f"  Success metrics:")
                print(f"    Top 3: {'✓' if top3 else '✗'}")
                print(f"    Top half: {'✓' if top_half else '✗'}")
                print(f"    Better than baseline: {'✓' if better_than_baseline else '✗'}")
                
                # Store results
                test_results.append({
                    'dataset_id': test_dataset['id'],
                    'dataset_name': test_dataset['name'],
                    'baseline_performance': test_performances['baseline'],
                    'recommended_pipeline': rec_name,
                    'recommended_performance': rec_performance,
                    'best_pipeline': best_pipeline,
                    'best_performance': best_performance,
                    'rank': rec_rank,
                    'total_pipelines': len(pipeline_names),
                    'performance_gap': perf_diff,
                    'confidence': recommendation['confidence'],
                    'top3': top3,
                    'top_half': top_half,
                    'better_than_baseline': better_than_baseline,
                    'all_performances': test_performances
                })
                
                # Confidence assessment
                if recommendation['confidence'] == 'high' and top3:
                    print(f"  🎯 High confidence recommendation was successful!")
                elif recommendation['confidence'] == 'low' and not top_half:
                    print(f"  ⚠️  Low confidence recommendation performed as expected")
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
        else:
            print("  ❌ Could not generate recommendation")
            test_results.append({
                'dataset_id': test_dataset['id'],
                'dataset_name': test_dataset['name'],
                'error': 'No recommendation generated'
            })
    
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
        avg_perf_gap = np.mean([r['performance_gap'] for r in valid_results if not np.isnan(r['performance_gap'])])
        
        print(f"\nAggregate Performance:")
        print(f"  Total test datasets: {total_datasets}")
        print(f"  Top 3 recommendations: {top3_count}/{total_datasets} ({top3_count/total_datasets*100:.1f}%)")
        print(f"  Top half recommendations: {top_half_count}/{total_datasets} ({top_half_count/total_datasets*100:.1f}%)")
        print(f"  Better than baseline: {better_than_baseline_count}/{total_datasets} ({better_than_baseline_count/total_datasets*100:.1f}%)")
        print(f"  Average rank: {avg_rank:.2f}")
        print(f"  Average performance gap: {avg_perf_gap:.4f}")
        
        # Confidence analysis
        confidence_analysis = {}
        for result in valid_results:
            conf = result['confidence']
            if conf not in confidence_analysis:
                confidence_analysis[conf] = {'count': 0, 'top3': 0, 'top_half': 0}
            confidence_analysis[conf]['count'] += 1
            if result['top3']:
                confidence_analysis[conf]['top3'] += 1
            if result['top_half']:
                confidence_analysis[conf]['top_half'] += 1
        
        print(f"\nConfidence Analysis:")
        for conf, stats in confidence_analysis.items():
            top3_rate = stats['top3'] / stats['count'] * 100 if stats['count'] > 0 else 0
            top_half_rate = stats['top_half'] / stats['count'] * 100 if stats['count'] > 0 else 0
            print(f"  {conf.capitalize()} confidence ({stats['count']} datasets):")
            print(f"    Top 3 rate: {top3_rate:.1f}%")
            print(f"    Top half rate: {top_half_rate:.1f}%")
        
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
    
    # Save detailed results
    if test_results:
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'all_performances'} for r in test_results])
        results_df.to_csv('test_evaluation_results.csv', index=False)
        print(f"\nDetailed results saved to 'test_evaluation_results.csv'")
    
    print(f"\n🎉 Comprehensive evaluation completed!")
    print(f"Training datasets: {len(train_datasets)}")
    print(f"Test datasets: {len(test_datasets)}")
    print(f"Pipeline configurations: {len(pipeline_configs)}")
    print(f"Meta-features: {len(train_metafeatures_df.columns) if not train_metafeatures_df.empty else 0}")

def run_enhanced_comprehensive_evaluation(
    train_frac=0.6, val_frac=0.2, test_frac=0.2,
    reduce_metafeatures=False, surrogate_model=True,
    n_components=20, reduction_method='pca'
):
    """
    Enhanced comprehensive evaluation using train/val/test splits per dataset.
    
    Args:
        train_frac: Fraction of dataset to use for training
        val_frac: Fraction of dataset to use for validation (and recommender training)
        test_frac: Fraction of dataset to keep as holdout test set
        reduce_metafeatures: Whether to apply dimensionality reduction to meta-features
        surrogate_model: Whether to use Bayesian surrogate model for recommendation
        n_components: Number of components for dimensionality reduction
        reduction_method: Method for dimensionality reduction ('pca', 'kpca', 'tsne')
    """
    print("="*80)
    print("ENHANCED PREPROCESSING RECOMMENDER WITH TRAIN/VAL/TEST SPLITS")
    print("="*80)

    ag_args_fit = {
        "ag.max_memory_usage_ratio": 0.9,  # Use at most 30% of available memory
        'num_gpus': 0,                     # Use CPU only for stability
        'num_cpus': min(10, os.cpu_count()) # Limit CPU usage
    }
            
    
    # Load training datasets
    print("\nLoading training datasets...")
    np.random.seed(42)
    train_datasets = []
    
    for dataset_id in train_dataset_ids:
        dataset = load_openml_dataset(dataset_id)
        if dataset:
            train_datasets.append(dataset)
        if len(train_datasets) >= 50:  # Limit for computational efficiency
            break
    
    print(f"\nLoaded {len(train_datasets)} training datasets")
    
    if len(train_datasets) < 3:
        print("Need at least 3 training datasets to proceed")
        return
    
    # Build training performance matrix with splits
    print("\nBuilding training performance matrix with train/val splits...")
    train_performance_matrix, dataset_splits = build_performance_matrix_with_splits(
        train_datasets, pipeline_configs, use_autogluon=True,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )
    print("\nTraining Performance Matrix (Validation Scores):")
    print(train_performance_matrix.round(4))
    
    # Save training results
    train_performance_matrix.to_csv('training_performance_matrix_with_splits.csv')
    print("\nTraining performance matrix saved")
    
    # Build training metafeatures matrix
    print("\nBuilding training metafeatures matrix...")
    train_metafeatures_df = build_metafeatures_matrix(train_datasets)
    print("\nTraining Meta-features Matrix:")
    print(train_metafeatures_df.head().round(4))
    
    if not train_metafeatures_df.empty:
        train_metafeatures_df.to_csv('training_metafeatures_full.csv')
        print(f"\nTraining meta-features matrix saved ({train_metafeatures_df.shape[1]} features)")
    
    
    
    # Train recommender systems
    print("\nTraining recommender systems...")
    
    # Original KNN-based recommender
    knn_recommender = EnhancedPreprocessingRecommender(train_performance_matrix, train_metafeatures_df)
    knn_recommender.train_ranker()
    
    # Bayesian surrogate recommender if requested
    surrogate_recommender = None
    if surrogate_model:
        print("\nTraining Bayesian surrogate recommender...")
        surrogate_recommender = BayesianSurrogateRecommender(
            train_performance_matrix, 
            train_metafeatures_df
        )
        surrogate_recommender.fit()
    
    # Load test datasets
    print("\nLoading test datasets...")
    test_datasets = []
    for dataset_id in test_dataset_ids:
        dataset = load_openml_dataset(dataset_id)
        if dataset:
            test_datasets.append(dataset)
    
    print(f"\nLoaded {len(test_datasets)} test datasets")
    
    # Evaluate on each test dataset
    test_results = []
    
    for test_dataset in test_datasets:
        print(f"\n" + "="*60)
        print(f"TESTING ON {test_dataset['name']} (ID: {test_dataset['id']})")
        print("="*60)
        
        # Split test dataset
        test_splits = split_dataset_stratified(
            test_dataset, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
        )
        
        # Extract meta-features
        test_metafeatures = extract_enhanced_metafeatures(test_dataset)
        
        # Get recommendations from both recommenders
        recommendations = {}
        
        # KNN recommender
        knn_rec = knn_recommender.recommend(test_dataset)
        if knn_rec:
            recommendations['knn'] = knn_rec
            print(f"\nKNN Recommendation (confidence: {knn_rec.get('confidence', 'unknown')}):\n  Pipeline: {knn_rec['pipeline_config']['name']}\n  Expected performance: {knn_rec.get('expected_performance', 'unknown'):.4f}")
        
        # Surrogate recommender
        if surrogate_recommender:
            surrogate_rec = surrogate_recommender.recommend(test_metafeatures)
            if surrogate_rec:
                recommendations['surrogate'] = surrogate_rec
                print(f"\nBayesian Surrogate Recommendation (confidence: {surrogate_rec.get('confidence', 'unknown')}):\n  Pipeline: {surrogate_rec['pipeline_name']}\n  Expected performance: {surrogate_rec.get('expected_performance', 'unknown'):.4f}")
        
        # Evaluate all pipelines on the test dataset's holdout test split
        print("\nEvaluating all pipelines on holdout test split...")
        test_performances = {}
        
        for config in pipeline_configs:
            try:
                # Apply preprocessing
                X_train_val = pd.concat([test_splits['X_train'], test_splits['X_val']])
                y_train_val = pd.concat([test_splits['y_train'], test_splits['y_val']])
                
                X_train_val_processed, y_train_val_processed, X_test_processed, y_test_processed = preprocess_consistent_across_splits(
    X_train_val, y_train_val, test_splits['X_test'], test_splits['y_test'], config
)


                
                if X_train_val_processed is None or X_test_processed is None or X_train_val_processed.empty or X_test_processed.empty:
                    test_performances[config['name']] = np.nan
                    continue
                
                # Sample data if too large to avoid memory issues
                if len(X_train_val_processed) > 5000:
                    print(f"  Sampling train data from {len(X_train_val_processed)} to 5000 samples")
                    idx = np.random.choice(len(X_train_val_processed), 5000, replace=False)
                    X_train_val_processed = X_train_val_processed.iloc[idx]
                    y_train_val_processed = y_train_val_processed.iloc[idx] if hasattr(y_train_val_processed, 'iloc') else y_train_val_processed[idx]
                
                # Train on combined train+val, test on holdout test
                train_data = X_train_val_processed.copy()
                train_data['target'] = y_train_val_processed
                
                # Use AutoGluon with fallback options
                temp_dir = tempfile.mkdtemp()
                try:
                    # Check class distribution first to avoid memory errors
                    class_counts = train_data['target'].value_counts()
                    if class_counts.min() < 8 or len(class_counts) > 30:
                        # For problematic datasets, use RandomForest directly
                        print(f"  Using RandomForest instead of AutoGluon (min class count: {class_counts.min()}, num classes: {len(class_counts)})")
                        rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=2, random_state=42)
                        rf.fit(X_train_val_processed, y_train_val_processed)
                        preds = rf.predict(X_test_processed)
                        score = accuracy_score(y_test_processed, preds)
                        test_performances[config['name']] = score
                        print(f"  {config['name']}: {score:.4f}")
                        continue
                    
                    # Use AutoGluon with reduced memory settings
                    # Check class distribution first to avoid memory errors
                    class_counts = train_data['target'].value_counts()
                    if class_counts.min() < 8 or len(class_counts) > 30:
                        # For problematic datasets, use RandomForest directly
                        print(f"  Using RandomForest instead of AutoGluon (min class count: {class_counts.min()}, num classes: {len(class_counts)})")
                        rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=2, random_state=42)
                        rf.fit(X_train_val_processed, y_train_val_processed)
                        preds = rf.predict(X_test_processed)
                        score = accuracy_score(y_test_processed, preds)
                        test_performances[config['name']] = score
                        print(f"  {config['name']}: {score:.4f}")
                        continue
                    
                    # Use AutoGluon with reduced memory settings
                    predictor = TabularPredictor(
                        label='target',
                        path=temp_dir,
                        problem_type=AUTOGLUON_CONFIG['problem_type'],
                        eval_metric=AUTOGLUON_CONFIG['eval_metric'],
                        verbosity=1
                    )
                    
                    
                    predictor.fit(
                        train_data,
                        time_limit=AUTOGLUON_CONFIG['time_limit'],
                        presets=AUTOGLUON_CONFIG['presets'],
                        included_model_types=AUTOGLUON_CONFIG['included_model_types'],
                        hyperparameter_tune_kwargs=AUTOGLUON_CONFIG['hyperparameter_tune_kwargs'],
                        ag_args_fit=ag_args_fit
                    )
                    
                    # Predict on test set
                    preds = predictor.predict(X_test_processed)
                    score = accuracy_score(y_test_processed, preds)
                    test_performances[config['name']] = score
                    print(f"  {config['name']}: {score:.4f}")
                finally:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        
            except Exception as e:
                print(f"  Error evaluating {config['name']}: {e}")
                print("  Falling back to RandomForest classifier...")
                try:
                    # Try fallback to RandomForest with reduced complexity
                    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=2, random_state=42)
                    rf.fit(X_train_val_processed, y_train_val_processed)
                    preds = rf.predict(X_test_processed)
                    score = accuracy_score(y_test_processed, preds)
                    test_performances[config['name']] = score
                    print(f"  {config['name']} (fallback): {score:.4f}")
                except Exception as inner_e:
                    print(f"  Fallback also failed: {inner_e}")
                    test_performances[config['name']] = np.nan
        
        # Analysis
        valid_performances = [(name, perf) for name, perf in test_performances.items() if not np.isnan(perf)]
        
        if valid_performances and recommendations:
            valid_performances.sort(key=lambda x: x[1], reverse=True)
            best_pipeline, best_performance = valid_performances[0]
            
            result = {
                'dataset_id': test_dataset['id'],
                'dataset_name': test_dataset['name'],
                'best_pipeline': best_pipeline,
                'best_performance': best_performance,
                'all_performances': test_performances,
            }
            
            # Analyze KNN recommendation
            if 'knn' in recommendations:
                knn_recommended = recommendations['knn']['pipeline_config']['name']
                knn_expected = recommendations['knn'].get('expected_performance', 0)
                knn_actual = test_performances.get(knn_recommended, np.nan)
                
                result.update({
                    'knn_recommended_pipeline': knn_recommended,
                    'knn_expected_performance': knn_expected,
                    'knn_actual_performance': knn_actual,
                    'knn_rank': [name for name, _ in valid_performances].index(knn_recommended) + 1 if knn_recommended in [name for name, _ in valid_performances] else len(valid_performances) + 1,
                    'knn_performance_gap': best_performance - knn_actual if not np.isnan(knn_actual) else np.nan,
                    'knn_confidence': recommendations['knn'].get('confidence', 'unknown')
                })
                
                print(f"\nKNN Recommendation Analysis:\n  Recommended pipeline: {knn_recommended}\n  Expected performance: {knn_expected:.4f}\n  Actual performance: {knn_actual:.4f}\n  Rank among all pipelines: {result['knn_rank']}/{len(valid_performances)}\n  Performance gap vs best: {result['knn_performance_gap']:.4f}")
            
            # Analyze Surrogate recommendation
            if 'surrogate' in recommendations:
                surrogate_recommended = recommendations['surrogate']['pipeline_name']
                surrogate_expected = recommendations['surrogate'].get('expected_performance', 0)
                surrogate_actual = test_performances.get(surrogate_recommended, np.nan)
                
                result.update({
                    'surrogate_recommended_pipeline': surrogate_recommended,
                    'surrogate_expected_performance': surrogate_expected,
                    'surrogate_actual_performance': surrogate_actual,
                    'surrogate_rank': [name for name, _ in valid_performances].index(surrogate_recommended) + 1 if surrogate_recommended in [name for name, _ in valid_performances] else len(valid_performances) + 1,
                    'surrogate_performance_gap': best_performance - surrogate_actual if not np.isnan(surrogate_actual) else np.nan,
                    'surrogate_confidence': recommendations['surrogate'].get('confidence', 'unknown')
                })
                
                print(f"\nSurrogate Recommendation Analysis:\n  Recommended pipeline: {surrogate_recommended}\n  Expected performance: {surrogate_expected:.4f}\n  Actual performance: {surrogate_actual:.4f}\n  Rank among all pipelines: {result['surrogate_rank']}/{len(valid_performances)}\n  Performance gap vs best: {result['surrogate_performance_gap']:.4f}")
            
            test_results.append(result)
            
        else:
            print("  ❌ Could not generate or evaluate recommendations")
            test_results.append({
                'dataset_id': test_dataset['id'],
                'dataset_name': test_dataset['name'],
                'error': 'No recommendation generated or evaluated'
            })
    
    # Overall summary
    print(f"\n" + "="*80)
    print("OVERALL ENHANCED EVALUATION SUMMARY")
    print("="*80)
    
    # Filter valid results
    valid_results = [r for r in test_results if 'error' not in r]
    
    if valid_results:
        # Calculate KNN metrics
        knn_results = [r for r in valid_results if 'knn_recommended_pipeline' in r]
        
        if knn_results:
            knn_top1 = sum(1 for r in knn_results if r['knn_rank'] == 1)
            knn_top3 = sum(1 for r in knn_results if r['knn_rank'] <= 3)
            knn_avg_rank = np.mean([r['knn_rank'] for r in knn_results])
            knn_avg_gap = np.mean([r['knn_performance_gap'] for r in knn_results if not np.isnan(r['knn_performance_gap'])])
            
            print(f"\nKNN Recommender Performance:\n  Total test datasets: {len(knn_results)}\n  Top 1 recommendations: {knn_top1}/{len(knn_results)} ({knn_top1/len(knn_results)*100:.1f}%)\n  Top 3 recommendations: {knn_top3}/{len(knn_results)} ({knn_top3/len(knn_results)*100:.1f}%)\n  Average rank: {knn_avg_rank:.2f}\n  Average performance gap: {knn_avg_gap:.4f}")
        
        # Calculate Surrogate metrics
        surrogate_results = [r for r in valid_results if 'surrogate_recommended_pipeline' in r]
        
        if surrogate_results:
            surrogate_top1 = sum(1 for r in surrogate_results if r['surrogate_rank'] == 1)
            surrogate_top3 = sum(1 for r in surrogate_results if r['surrogate_rank'] <= 3)
            surrogate_avg_rank = np.mean([r['surrogate_rank'] for r in surrogate_results])
            surrogate_avg_gap = np.mean([r['surrogate_performance_gap'] for r in surrogate_results if not np.isnan(r['surrogate_performance_gap'])])
            
            print(f"\nBayesian Surrogate Recommender Performance:\n  Total test datasets: {len(surrogate_results)}\n  Top 1 recommendations: {surrogate_top1}/{len(surrogate_results)} ({surrogate_top1/len(surrogate_results)*100:.1f}%)\n  Top 3 recommendations: {surrogate_top3}/{len(surrogate_results)} ({surrogate_top3/len(surrogate_results)*100:.1f}%)\n  Average rank: {surrogate_avg_rank:.2f}\n  Average performance gap: {surrogate_avg_gap:.4f}")
        
        # Compare the two approaches if both available
        if knn_results and surrogate_results:
            print("\nComparison (KNN vs Surrogate):\n  Top 1 rate: {knn_top1/len(knn_results)*100:.1f}% vs {surrogate_top1/len(surrogate_results)*100:.1f}%\n  Top 3 rate: {knn_top3/len(knn_results)*100:.1f}% vs {surrogate_top3/len(surrogate_results)*100:.1f}%\n  Average rank: {knn_avg_rank:.2f} vs {surrogate_avg_rank:.2f}\n  Average gap: {knn_avg_gap:.4f} vs {surrogate_avg_gap:.4f}")
    
    # Save detailed results
    if test_results:
        results_df = pd.DataFrame(test_results)
        results_df.to_csv('enhanced_evaluation_results.csv', index=False)
        print(f"\nDetailed results saved to 'enhanced_evaluation_results.csv'")
    
    print(f"\n🎉 Enhanced comprehensive evaluation completed!")
    print(f"Training datasets: {len(train_datasets)}")
    print(f"Test datasets: {len(test_datasets)}")
    print(f"Pipeline configurations: {len(pipeline_configs)}")
    print(f"Meta-features: {len(train_metafeatures_df.columns) if not train_metafeatures_df.empty else 0}")
    
    return test_results

# Main execution
if __name__ == "__main__":
    # Uncomment the approach you want to run
    
    # Original evaluation
    # run_comprehensive_evaluation()

    
    # Enhanced evaluation with train/val/test splits
    run_enhanced_comprehensive_evaluation(
        train_frac=0.6,       # 60% for training
        val_frac=0.2,         # 20% for validation
        test_frac=0.2,        # 20% for holdout test
        surrogate_model=True,      # Use Bayesian surrogate model
    )