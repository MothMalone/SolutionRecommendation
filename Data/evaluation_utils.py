import pandas as pd
import numpy as np
import os
import warnings
import tempfile
import shutil
import uuid
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from autogluon.tabular import TabularPredictor
from scipy.stats import zscore


# Configuration constants - Updated to match new AutoGluon config
AUTOGLUON_CONFIG = {
    "eval_metric": "accuracy",
    "time_limit": 600,  # 10 minutes per dataset (5 minutes in original comment but 600 seconds)
    "presets": "medium_quality",
    "verbosity": 4,
    "hyperparameter_tune_kwargs": None,
    "ag_args_fit": {
        "ag.max_memory_usage_ratio": 0.9,
    },
    "seed": 42
}

# Legacy constants (kept for backwards compatibility)
AG_ARGS_FIT = AUTOGLUON_CONFIG["ag_args_fit"]

STABLE_MODELS = [
    "GBM", "CAT", "XGB", "RF", "XT", "KNN", "LR", "NN_TORCH", "FASTAI",
    "NN_MXNET", "TABPFN", "DUMMY", "NB"
]


class Preprocessor:
    """A stateful class to handle fitting and transforming data to prevent data leakage."""
    def __init__(self, config):
        self.config = config
        self.column_transformer = None
        self.selection_model = None
        self.reduction_model = None
        self.outlier_models = {}
        self.fitted_columns = None
        self.fitted = False

    def fit(self, X, y):
        try:
            X_processed = X.copy()
            y_processed = y.copy()
            
            #  Fit ColumnTransformer for imputation, scaling, encoding 
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            
            imputers = {
                'mean': SimpleImputer(strategy='mean'), 
                'median': SimpleImputer(strategy='median'), 
                'knn': KNNImputer(n_neighbors=min(5, len(X)-1)), 
                'most_frequent': SimpleImputer(strategy='most_frequent'), 
                'constant': SimpleImputer(strategy='constant', fill_value=0)
            }
            scalers = {
                'standard': StandardScaler(), 
                'minmax': MinMaxScaler(), 
                'robust': RobustScaler(), 
                'maxabs': MaxAbsScaler()
            }
            
            numeric_steps = []
            if self.config.get('imputation') in imputers: 
                numeric_steps.append(('imputer', imputers[self.config['imputation']]))
            if self.config.get('scaling') in scalers: 
                numeric_steps.append(('scaler', scalers[self.config['scaling']]))
            numeric_pipeline = Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough'

            categorical_pipeline = 'passthrough'
            if self.config.get('encoding') == 'onehot' and categorical_features:
                categorical_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), 
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
                ])

            transformers = []
            if numeric_features:
                transformers.append(('num', numeric_pipeline, numeric_features))
            if categorical_features:
                transformers.append(('cat', categorical_pipeline, categorical_features))
            
            if not transformers:
                raise ValueError("No features to transform")
                
            self.column_transformer = ColumnTransformer(transformers, remainder='drop')
            self.column_transformer.fit(X_processed)
            X_processed = pd.DataFrame(
                self.column_transformer.transform(X_processed), 
                columns=self.column_transformer.get_feature_names_out(), 
                index=X.index
            )
            
            if X_processed.shape[1] == 0:
                raise ValueError("No features remained after initial transformation")
            
            # Fit Feature Selection Models
            if self.config['feature_selection'] in ['k_best', 'mutual_info']:
                k = min(20, X_processed.shape[1])
                if k > 0 and len(X_processed) > k:  # Need enough samples
                    selector_func = f_classif if self.config['feature_selection'] == 'k_best' else mutual_info_classif
                    self.selection_model = SelectKBest(selector_func, k=k)
                    self.selection_model.fit(X_processed, y_processed)
                    X_processed = pd.DataFrame(
                        self.selection_model.transform(X_processed), 
                        columns=self.selection_model.get_feature_names_out(), 
                        index=X_processed.index
                    )
            elif self.config['feature_selection'] == 'variance_threshold':
                if X_processed.shape[1] > 1:
                    selector = VarianceThreshold(threshold=0.0)
                    selector.fit(X_processed)
                    if selector.transform(X_processed).shape[1] > 0:
                        self.selection_model = selector
                        X_processed = pd.DataFrame(
                            selector.transform(X_processed),
                            index=X_processed.index
                        )
            
            # Fit Dimensionality Reduction Models 
            if self.config['dimensionality_reduction'] in ['pca', 'svd']:
                n_components = min(10, X_processed.shape[1], len(X_processed) - 1)
                if n_components > 0:
                    reducer = PCA(n_components=n_components) if self.config['dimensionality_reduction'] == 'pca' else TruncatedSVD(n_components=n_components)
                    self.reduction_model = reducer.fit(X_processed)

            # Fit Outlier Removal Models
            if self.config['outlier_removal'] in ['isolation_forest', 'lof'] and len(X_processed) > 10:
                model = IsolationForest(random_state=42, contamination=0.05) if self.config['outlier_removal'] == 'isolation_forest' else LocalOutlierFactor()
                self.outlier_models['model'] = model.fit(X_processed)

            self.fitted = True
            self.fitted_columns = X_processed.columns
            
        except Exception as e:
            print(f"      Warning: Preprocessor fit failed: {e}")
            self.fitted = False

    def transform(self, X, y):
        if not self.fitted: 
            print("      Warning: Preprocessor not fitted, returning original data")
            return X.reset_index(drop=True), y.reset_index(drop=True)
        
        try:
            X_processed = X.copy()
            y_processed = y.copy()

            # Impute, scale, encode
            X_processed = pd.DataFrame(
                self.column_transformer.transform(X_processed), 
                columns=self.column_transformer.get_feature_names_out(), 
                index=X_processed.index
            )
            
            # Feature selection
            if self.selection_model:
                X_processed = pd.DataFrame(
                    self.selection_model.transform(X_processed), 
                    columns=self.selection_model.get_feature_names_out() if hasattr(self.selection_model, 'get_feature_names_out') else None,
                    index=X_processed.index
                )
            
            # Dimensionality reduction
            if self.reduction_model:
                X_processed = pd.DataFrame(
                    self.reduction_model.transform(X_processed), 
                    index=X_processed.index
                )
            
            if self.config['outlier_removal'] != 'none' and len(X_processed) > 20:
                original_size = len(X_processed)
                
                if self.config['outlier_removal'] == 'iqr':
                    mask = pd.Series(True, index=X_processed.index)
                    for col in X_processed.columns:
                        q1, q3 = X_processed[col].quantile(0.25), X_processed[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:  # Avoid division by zero
                            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                            mask = mask & (X_processed[col] >= lower) & (X_processed[col] <= upper)
                    X_processed = X_processed[mask]
                    
                elif self.config['outlier_removal'] == 'zscore':
                    try:
                        z_scores = np.abs(zscore(X_processed, nan_policy='omit'))
                        mask = (z_scores < 3).all(axis=1)
                        X_processed = X_processed[mask]
                    except:
                        pass  # Keep original data if zscore fails
                        
                elif self.config['outlier_removal'] in ['isolation_forest', 'lof'] and 'model' in self.outlier_models:
                    try:
                        predictions = self.outlier_models['model'].fit_predict(X_processed)
                        X_processed = X_processed[predictions == 1]
                    except:
                        pass  # Keep original data if outlier detection fails
                
                if len(X_processed) < original_size * 0.1:  # Keep at least 10% of data
                    print(f"      Warning: Outlier removal would remove too much data, skipping")
                    X_processed = X.copy()
                    X_processed = pd.DataFrame(
                        self.column_transformer.transform(X_processed), 
                        columns=self.column_transformer.get_feature_names_out(), 
                        index=X_processed.index
                    )
                    if self.selection_model:
                        X_processed = pd.DataFrame(
                            self.selection_model.transform(X_processed), 
                            index=X_processed.index
                        )
                    if self.reduction_model:
                        X_processed = pd.DataFrame(
                            self.reduction_model.transform(X_processed), 
                            index=X_processed.index
                        )
                else:
                    y_processed = y_processed.loc[X_processed.index]

            return X_processed.reset_index(drop=True), y_processed.reset_index(drop=True)
            
        except Exception as e:
            print(f"      Warning: Transform failed: {e}, returning original data")
            return X.reset_index(drop=True), y.reset_index(drop=True)


def can_stratify(y, test_size=0.2, min_samples_per_class=2):
    """Check if stratification is possible given class distribution."""
    if y.nunique() <= 1:
        return False
    
    class_counts = y.value_counts()
    min_count = class_counts.min()
    
    # Need at least min_samples_per_class in each split
    min_test_samples = max(1, int(test_size * len(y) / y.nunique()))
    min_train_samples = min_samples_per_class
    
    return min_count >= (min_test_samples + min_train_samples)


def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """Perform train-test split with optional stratification."""
    try:
        if can_stratify(y, test_size):
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception as e:
        print(f"    Warning: Stratification failed ({e}), using random split")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_openml_dataset(dataset_id):
    """Enhanced dataset loading with better error tracking"""
    try:
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X, y = dataset.data, dataset.target
        
        # Handle categorical columns properly
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(['category']).columns
            if len(categorical_cols) > 0:
                X = X.copy()  
                X.loc[:, categorical_cols] = X.loc[:, categorical_cols].astype(object)
        
        # Handle target encoding
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = pd.Series(LabelEncoder().fit_transform(y), name=y.name)
        
        # Remove rows with missing targets
        valid_indices = y.dropna().index
        X = X.loc[valid_indices].reset_index(drop=True)
        y = y.loc[valid_indices].reset_index(drop=True)
        
        # Subsample if too large
        if len(X) > 5000:
            X, y = shuffle(X, y, n_samples=5000, random_state=42)
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        
        if len(X) < 20:
            print(f"ERROR: Dataset {dataset_id} too small: {len(X)} samples")
            return None
            
        print(f"SUCCESS: Loaded dataset {dataset_id}: Shape={X.shape}, Classes={y.nunique()}")
        return {'id': dataset_id, 'name': f"D_{dataset_id}", 'X': X, 'y': y.astype(int)}
        
    except Exception as e:
        print(f"ERROR: Failed to load dataset {dataset_id}: {e}")
        with open('failed_datasets.log', 'a') as f:
            f.write(f"Dataset {dataset_id}: {str(e)}\n")
        return None


def get_metafeatures(dataset_id, meta_features_df):
    try:
        return meta_features_df.loc[dataset_id].to_dict()
    except KeyError:
        print(f"    Warning: No metafeatures found for dataset {dataset_id}")
        return None


def get_temp_dir():
    """Create a new temporary directory with random name to avoid conflicts"""
    temp_dir = os.path.join(tempfile.gettempdir(), f"ag_temp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def run_autogluon_evaluation(X_train, y_train, X_test, y_test):
    """
    Evaluate using AutoGluon with the specified configuration.
    Uses IdentityFeatureGenerator to prevent AutoGluon from doing its own preprocessing.
    """
    if len(X_train) == 0 or len(X_test) == 0:
        print("      Warning: Empty dataset provided to AutoGluon")
        return np.nan
        
    temp_dir = get_temp_dir()
    
    # Suppress "path already exists" warning
    warnings.filterwarnings("ignore", message="path already exists! This predictor may overwrite")
    
    try:
        from autogluon.features.generators import IdentityFeatureGenerator
        
        # Reset column names to avoid mismatch issues
        X_train_ag = X_train.copy()
        X_train_ag.columns = [f"col_{i}" for i in range(X_train_ag.shape[1])]
        
        X_test_ag = X_test.copy()
        X_test_ag.columns = [f"col_{i}" for i in range(X_test_ag.shape[1])]
        
        # Prepare training data
        train_data = X_train_ag.copy()
        train_data['target'] = y_train.values
        
        # Detect problem type based on target
        unique_classes = y_train.nunique()
        if unique_classes == 2:
            problem_type = 'binary'
        elif unique_classes > 20 and np.issubdtype(y_train.dtype, np.number):
            problem_type = 'regression'
        else:
            problem_type = 'multiclass'
        
        # Determine eval metric
        eval_metric = 'r2' if problem_type == 'regression' else AUTOGLUON_CONFIG['eval_metric']
        
        # Create predictor with new config
        predictor = TabularPredictor(
            label='target', 
            path=temp_dir, 
            problem_type=problem_type, 
            eval_metric=eval_metric,
            verbosity=AUTOGLUON_CONFIG['verbosity']
        )
        
        # Fit with new config - using IdentityFeatureGenerator to prevent extra preprocessing
        predictor.fit(
            train_data=train_data,
            time_limit=AUTOGLUON_CONFIG['time_limit'],
            presets=AUTOGLUON_CONFIG['presets'],
            hyperparameter_tune_kwargs=AUTOGLUON_CONFIG['hyperparameter_tune_kwargs'],
            ag_args_fit=AUTOGLUON_CONFIG['ag_args_fit'],
            feature_generator=IdentityFeatureGenerator(),  # Prevent AutoGluon from preprocessing
            raise_on_no_models_fitted=False
        )
        
        # Make predictions
        preds = predictor.predict(X_test_ag)
        
        # Calculate score based on problem type
        if problem_type == 'regression':
            from sklearn.metrics import r2_score
            return r2_score(y_test, preds)
        else:
            return accuracy_score(y_test, preds)
        
    except Exception as e:
        print(f"      Warning: AutoGluon evaluation failed: {e}")
        print("      Fallback: using RandomForestClassifier/Regressor")
        
        # Fallback to a simple RandomForest based on problem type
        try:
            # Determine problem type for fallback
            unique_classes = y_train.nunique()
            if unique_classes > 20 and np.issubdtype(y_train.dtype, np.number):
                problem_type = 'regression'
            else:
                problem_type = 'classification'
            
            if problem_type == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import r2_score
                model = RandomForestRegressor(
                    n_estimators=50, 
                    random_state=AUTOGLUON_CONFIG['seed'], 
                    max_depth=10
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return r2_score(y_test, y_pred)
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=AUTOGLUON_CONFIG['seed'], 
                    max_depth=10
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return accuracy_score(y_test, y_pred)
                
        except Exception as e2:
            print(f"      Warning: Fallback model also failed: {e2}")
            return np.nan
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def run_experiment_for_dataset(dataset, meta_features_df, global_performance_matrix=None, recommender_type='baseline', ground_truth_perf_matrix=None, use_influence=False, influence_method='performance_variance', evaluate_only_baseline=False):
    """
    Run experiment for a dataset. Can use pre-computed ground truth performance or evaluate pipelines.
    
    Args:
        dataset: Dictionary with dataset information
        meta_features_df: DataFrame with metafeatures
        global_performance_matrix: Performance matrix for training recommender (90 training datasets)
        recommender_type: Type of recommender to use
        ground_truth_perf_matrix: Optional DataFrame with pre-computed performance for TEST datasets only
        use_influence: Whether to use DPO-style influence weighting (for PMM/BalancedPMM)
        influence_method: Method for calculating influence scores
        evaluate_only_baseline: If True, only evaluate 'baseline' pipeline (skip all others)
    """
    print(f"\n{'='*30} EXPERIMENT FOR {dataset['name']} ({dataset['id']}) {'='*30}")
    
    # Important: ground_truth_perf_matrix should ONLY contain TEST datasets, NOT training datasets
    # This ensures we're evaluating on truly unseen data
    
    # Check if this is a TEST dataset with pre-computed ground truth
    use_ground_truth = (
        ground_truth_perf_matrix is not None and 
        isinstance(ground_truth_perf_matrix, pd.DataFrame) and
        dataset['name'] in ground_truth_perf_matrix.columns
    )
    
    if use_ground_truth:
        print(f"  ✅ Using pre-computed TEST ground truth performance (skipping evaluation)...")
        
        # Extract performance for this dataset
        local_perf_matrix = ground_truth_perf_matrix[[dataset['name']]].copy()
        valid_local_perf = local_perf_matrix.dropna()
        
        if valid_local_perf.empty:
            print("  WARNING: No valid ground truth performance results, skipping...")
            return local_perf_matrix, None, None
        
        print(f"  Found ground truth performance for {len(valid_local_perf)} pipelines")
        
        # Show top 3 pipelines from ground truth
        top_3_ground_truth = valid_local_perf.sort_values(by=dataset['name'], ascending=False).head(3)
        print(f"  Top-3 pipelines from ground truth:")
        for idx, (pipeline, row) in enumerate(top_3_ground_truth.iterrows(), 1):
            score = row[dataset['name']]
            print(f"    {idx}. {pipeline}: {score:.4f}")
    
    else:
        # This dataset is NOT in the test ground truth, so we need to evaluate it
        # This is expected for datasets not in the pre-computed test set
        if ground_truth_perf_matrix is not None:
            available_test_datasets = list(ground_truth_perf_matrix.columns)
            print(f"  ⚠️ Dataset {dataset['name']} not in pre-computed test ground truth.")
            print(f"     Available test datasets: {available_test_datasets}")
            print(f"     Will evaluate all pipelines (this may take time)...")
        else:
            print(f"  ⚠️ No ground truth file provided, evaluating all pipelines...")
        
        # Use the full dataset
        X_full = dataset['X']
        y_full = dataset['y']
        
        # Split into train/test
        X_train, X_test, y_train, y_test = safe_train_test_split(
            X_full, y_full, test_size=0.3, random_state=42
        )
        
        print(f"  Dataset size: {len(X_full)} samples, split into train: {len(X_train)}, test: {len(X_test)}")
        
        # Check if dataset is too small
        if len(X_train) < 10 or len(X_test) < 5:
            print("  WARNING: Dataset too small for reliable evaluation, skipping...")
            return None, None, None
        
        from recommender_trainer import pipeline_configs
        
        # Filter pipelines based on evaluate_only_baseline flag
        if evaluate_only_baseline:
            print("  Evaluating ONLY baseline pipeline...")
            configs_to_evaluate = [c for c in pipeline_configs if c['name'] == 'baseline']
        else:
            print("  Evaluating all pipelines...")
            configs_to_evaluate = pipeline_configs
        
        local_perf = {}
        
        for config in configs_to_evaluate:
            print(f"    Evaluating '{config['name']}'...")
            try:
                if config['name'] == 'baseline':
                    score = run_autogluon_evaluation(X_train, y_train, X_test, y_test)
                else:
                    preprocessor = Preprocessor(config)
                    preprocessor.fit(X_train, y_train)
                    X_train_p, y_train_p = preprocessor.transform(X_train, y_train)
                    X_test_p, y_test_p = preprocessor.transform(X_test, y_test)
                    
                    if len(X_train_p) > 0 and len(X_test_p) > 0:
                        score = run_autogluon_evaluation(X_train_p, y_train_p, X_test_p, y_test_p)
                    else:
                        print(f"      Warning: Preprocessing resulted in empty dataset")
                        score = np.nan
                        
                local_perf[config['name']] = score
                
            except Exception as e:
                print(f"      ERROR: Could not evaluate pipeline {config['name']}. Reason: {e}")
                local_perf[config['name']] = np.nan
        
        local_perf_matrix = pd.DataFrame.from_dict(local_perf, orient='index', columns=[dataset['name']])
        
        # Check if we have any valid results
        valid_local_perf = local_perf_matrix.dropna()
        if valid_local_perf.empty:
            print("  WARNING: No valid local performance results, skipping recommender...")
            return local_perf_matrix, None, None
    
    # Determine the recommended pipeline based on recommender_type
    print(f"  Using {recommender_type} recommender for recommendation...")
    
    if recommender_type == 'baseline':
        # Just use the local best performance
        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
        print(f"  Using baseline recommender (best local performance)")
    
    elif recommender_type == 'autogluon' and global_performance_matrix is not None:
        # Use AutoGluon recommender
        from recommender_trainer import AutoGluonPipelineRecommender
        
        try:
            print("  Training AutoGluon recommender...")
            recommender = AutoGluonPipelineRecommender(
                global_performance_matrix, 
                meta_features_df
            )
            
            if not recommender.fit():
                print("  WARNING: AutoGluon recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                if isinstance(recommendation_result, str):
                    recommended_pipeline = recommendation_result
                else:
                    # Handle tuple or other return type
                    recommended_pipeline = recommendation_result[0] if isinstance(recommendation_result, tuple) else recommendation_result
                print(f"  AutoGluon recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: AutoGluon recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            
    elif recommender_type == 'hybrid' and global_performance_matrix is not None:
        # Use Hybrid recommender
        from recommender_trainer import HybridMetaRecommender
        
        try:
            print("  Training Hybrid recommender...")
            # Use influence parameters from function arguments
            if use_influence:
                print(f"  🎯 Using influence weighting (method: {influence_method})")
            
            recommender = HybridMetaRecommender(
                global_performance_matrix, 
                meta_features_df,
                use_influence_weighting=use_influence,
                influence_method=influence_method
            )
            
            if not recommender.fit():
                print("  WARNING: Hybrid recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                result = recommender.recommend(dataset['id'])
                # Handle if recommender returns a tuple (pipeline_name, scores_dict)
                if isinstance(result, tuple):
                    recommended_pipeline = result[0]  # Get just the pipeline name
                    print(f"  Hybrid recommender suggests: {recommended_pipeline} (with scores)")
                else:
                    recommended_pipeline = result
                    print(f"  Hybrid recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Hybrid recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'surrogate' and global_performance_matrix is not None:
        # Use Bayesian Surrogate recommender
        from recommender_trainer import BayesianSurrogateRecommender
        
        try:
            print("  Training Bayesian Surrogate recommender...")
            recommender = BayesianSurrogateRecommender(
                global_performance_matrix, 
                meta_features_df
            )
            
            if not recommender.fit():
                print("  WARNING: Surrogate recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                if isinstance(recommendation_result, tuple) and len(recommendation_result) > 0:
                    recommended_pipeline = recommendation_result[0]
                else:
                    recommended_pipeline = recommendation_result
                print(f"  Surrogate recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Surrogate recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    # Handle new recommender types
    elif recommender_type == 'random' and global_performance_matrix is not None:
        # Use Random recommender
        from recommender_trainer import RandomRecommender
        
        try:
            print("  Using Random recommender...")
            recommender = RandomRecommender(global_performance_matrix, meta_features_df)
            recommender.fit()
            recommended_pipeline, predictions = recommender.recommend(dataset['id'])
            print(f"  Random recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Random recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'avgrank' and global_performance_matrix is not None:
        # Use Average Rank recommender
        from recommender_trainer import AverageRankRecommender
        
        try:
            print("  Using Average Rank recommender...")
            recommender = AverageRankRecommender(global_performance_matrix, meta_features_df)
            recommender.fit()
            recommended_pipeline, predictions = recommender.recommend(dataset['id'])
            print(f"  Average Rank recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Average Rank recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'l1' and global_performance_matrix is not None:
        # Use L1 Distance recommender
        from recommender_trainer import L1Recommender
        
        try:
            print("  Using L1 Distance recommender...")
            recommender = L1Recommender(global_performance_matrix, meta_features_df)
            recommender.fit()
            recommended_pipeline, predictions = recommender.recommend(dataset['id'])
            print(f"  L1 Distance recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: L1 Distance recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'basic' and global_performance_matrix is not None:
        # Use Basic recommender
        from recommender_trainer import BasicRecommender
        
        try:
            print("  Using Basic recommender...")
            recommender = BasicRecommender(global_performance_matrix, meta_features_df)
            recommender.fit()
            recommended_pipeline, predictions = recommender.recommend(dataset['id'])
            print(f"  Basic recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Basic recommender failed: {e}")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'knn' and global_performance_matrix is not None:
        # Use KNN recommender
        from recommender_trainer import KnnRecommender
        
        try:
            print("  Training KNN recommender...")
            recommender = KnnRecommender(global_performance_matrix, meta_features_df)
            if not recommender.fit():
                print("  WARNING: KNN recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                # Handle tuple return: (pipeline_name, predictions_dict)
                if isinstance(recommendation_result, tuple):
                    recommended_pipeline, predictions = recommendation_result
                else:
                    recommended_pipeline = recommendation_result
                print(f"  ✅ KNN recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: KNN recommender failed: {e}")
            import traceback
            traceback.print_exc()
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'rf' and global_performance_matrix is not None:
        # Use Random Forest recommender
        from recommender_trainer import RFRecommender
        
        try:
            print("  Training Random Forest recommender...")
            recommender = RFRecommender(global_performance_matrix, meta_features_df)
            if not recommender.fit():
                print("  WARNING: RF recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                # Handle tuple return: (pipeline_name, predictions_dict)
                if isinstance(recommendation_result, tuple):
                    recommended_pipeline, predictions = recommendation_result
                else:
                    recommended_pipeline = recommendation_result
                print(f"  ✅ Random Forest recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Random Forest recommender failed: {e}")
            import traceback
            traceback.print_exc()
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'nn' and global_performance_matrix is not None:
        # Use Neural Network recommender
        from recommender_trainer import NNRecommender
        
        try:
            print("  Training Neural Network recommender...")
            recommender = NNRecommender(global_performance_matrix, meta_features_df)
            if not recommender.fit():
                print("  WARNING: NN recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                # Handle tuple return: (pipeline_name, predictions_dict)
                if isinstance(recommendation_result, tuple):
                    recommended_pipeline, predictions = recommendation_result
                else:
                    recommended_pipeline = recommendation_result
                print(f"  ✅ Neural Network recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Neural Network recommender failed: {e}")
            import traceback
            traceback.print_exc()
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'regressor' and global_performance_matrix is not None:
        # Use Regressor recommender
        from recommender_trainer import RegressorRecommender
        
        try:
            print("  Training Regressor recommender...")
            recommender = RegressorRecommender(global_performance_matrix, meta_features_df)
            if not recommender.fit():
                print("  WARNING: Regressor recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                # Handle tuple return: (pipeline_name, predictions_dict)
                if isinstance(recommendation_result, tuple):
                    recommended_pipeline, predictions = recommendation_result
                else:
                    recommended_pipeline = recommendation_result
                print(f"  ✅ Regressor recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: Regressor recommender failed: {e}")
            import traceback
            traceback.print_exc()
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'adaboost' and global_performance_matrix is not None:
        # Use AdaBoost Regressor recommender
        from recommender_trainer import AdaBoostRegressorRecommender
        
        try:
            print("  Training AdaBoost Regressor recommender...")
            recommender = AdaBoostRegressorRecommender(global_performance_matrix, meta_features_df)
            if not recommender.fit():
                print("  WARNING: AdaBoost recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommendation_result = recommender.recommend(dataset['id'])
                # Handle tuple return: (pipeline_name, predictions_dict)
                if isinstance(recommendation_result, tuple):
                    recommended_pipeline, predictions = recommendation_result
                else:
                    recommended_pipeline = recommendation_result
                print(f"  ✅ AdaBoost Regressor recommender suggests: {recommended_pipeline}")
        except Exception as e:
            print(f"  ERROR: AdaBoost Regressor recommender failed: {e}")
            import traceback
            traceback.print_exc()
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'pmm' and global_performance_matrix is not None:
        # Use PMM recommender
        from recommender_trainer import PmmRecommender
        
        try:
            influence_status = "WITH" if use_influence else "WITHOUT"
            print(f"  Training PMM recommender {influence_status} influence weighting...")
            if use_influence:
                print(f"    Influence method: {influence_method}")
            recommender = PmmRecommender(
                num_epochs=20, 
                batch_size=64,
                use_influence_weighting=use_influence,
                influence_method=influence_method
            )
            training_result = recommender.fit(global_performance_matrix, meta_features_df)
            
            if not training_result:
                print("  WARNING: PMM recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                # Check if model learned meaningful representations
                if len(recommender.dataset_embeddings) == 0:
                    print("  WARNING: PMM recommender has no embeddings, using local best pipeline...")
                    recommended_pipeline = valid_local_perf.idxmax().iloc[0]
                else:
                    # Convert dataset ID to the correct format if needed
                    dataset_id = dataset['id']
                    print(f"  Getting PMM recommendation for dataset {dataset_id}...")
                    recommendation_result = recommender.recommend(dataset_id, global_performance_matrix)
                    
                    if recommendation_result and 'pipeline' in recommendation_result and recommendation_result['pipeline']:
                        recommended_pipeline = recommendation_result['pipeline']
                        print(f"  PMM recommender suggests: {recommended_pipeline}")
                        
                        # Show additional information about similar datasets
                        if 'similar_datasets' in recommendation_result:
                            similar_datasets = recommendation_result['similar_datasets'][:3]
                            similarity_scores = recommendation_result.get('similarity_scores', {})
                            
                            # Check if similarities are all zero (model didn't learn)
                            all_zero = all(abs(similarity_scores.get(ds, 0.0)) < 1e-6 for ds in similar_datasets)
                            if all_zero:
                                print("  ⚠️ WARNING: All similarities are zero. Model didn't learn properly.")
                                print("  Falling back to local best pipeline...")
                                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
                            else:
                                print(f"  Based on similar datasets: {similar_datasets}")
                                
                                # Show influence scores if available
                                influence_scores = recommendation_result.get('influence_scores', {})
                                influence_weighted = recommendation_result.get('influence_weighted', False)
                                
                                if influence_weighted and influence_scores:
                                    print(f"  🎯 Using influence weighting (method: {recommender.influence_method})")
                                
                                for ds in similar_datasets:
                                    sim_score = similarity_scores.get(ds, 0.0)
                                    if influence_weighted and ds in influence_scores:
                                        inf_score = influence_scores[ds]
                                        print(f"    - Dataset {ds} (similarity: {sim_score:.4f}, influence: {inf_score:.3f})")
                                    else:
                                        print(f"    - Dataset {ds} (similarity: {sim_score:.4f})")
                    else:
                        print(f"  Warning: PMM recommender returned invalid result: {recommendation_result}")
                        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
        except Exception as e:
            import traceback
            print(f"  ERROR: PMM recommender failed: {e}")
            print("  Traceback:")
            traceback.print_exc()
            print("  Falling back to local best pipeline...")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'balancedpmm' and global_performance_matrix is not None:
        # Use Balanced PMM recommender
        from recommender_trainer import BalancedPmmRecommender
        
        try:
            influence_status = "WITH" if use_influence else "WITHOUT"
            print(f"  Training Balanced PMM recommender {influence_status} influence weighting...")
            if use_influence:
                print(f"    Influence method: {influence_method}")
            recommender = BalancedPmmRecommender(
                num_epochs=20, 
                batch_size=64,
                use_influence_weighting=use_influence,
                influence_method=influence_method
            )
            training_result = recommender.fit(global_performance_matrix, meta_features_df)
            
            if not training_result:
                print("  WARNING: Balanced PMM recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                # Check if model learned meaningful representations
                if len(recommender.dataset_embeddings) == 0:
                    print("  WARNING: Balanced PMM recommender has no embeddings, using local best pipeline...")
                    recommended_pipeline = valid_local_perf.idxmax().iloc[0]
                else:
                    # Convert dataset ID to the correct format if needed
                    dataset_id = dataset['id']
                    print(f"  Getting Balanced PMM recommendation for dataset {dataset_id}...")
                    recommendation_result = recommender.recommend(dataset_id, global_performance_matrix)
                    
                    if recommendation_result and 'pipeline' in recommendation_result and recommendation_result['pipeline']:
                        recommended_pipeline = recommendation_result['pipeline']
                        print(f"  Balanced PMM recommender suggests: {recommended_pipeline}")
                        
                        # Show additional information about similar datasets
                        if 'similar_datasets' in recommendation_result:
                            similar_datasets = recommendation_result['similar_datasets'][:3]
                            similarity_scores = recommendation_result.get('similarity_scores', {})
                            
                            # Check if similarities are all zero (model didn't learn)
                            all_zero = all(abs(similarity_scores.get(ds, 0.0)) < 1e-6 for ds in similar_datasets)
                            if all_zero:
                                print("  ⚠️ WARNING: All similarities are zero. Model didn't learn properly.")
                                print("  Falling back to local best pipeline...")
                                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
                            else:
                                print(f"  Based on similar datasets: {similar_datasets}")
                                for ds in similar_datasets:
                                    score = similarity_scores.get(ds, 0.0)
                                    print(f"    - Dataset {ds} (similarity: {score:.4f})")
                    else:
                        print(f"  Warning: Balanced PMM recommender returned invalid result: {recommendation_result}")
                        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
        except Exception as e:
            import traceback
            print(f"  ERROR: Balanced PMM recommender failed: {e}")
            print("  Traceback:")
            traceback.print_exc()
            print("  Falling back to local best pipeline...")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    elif recommender_type == 'paper_pmm' and global_performance_matrix is not None:
        # Use Paper-Style PMM recommender (dataset+pipeline pairs)
        from pmm_paper_style import PaperPmmRecommender
        
        try:
            influence_status = "WITH" if use_influence else "WITHOUT"
            print(f"  Training Paper-Style PMM recommender {influence_status} influence weighting...")
            print(f"    NOTE: This uses (dataset, pipeline) feature pairs like the paper")
            if use_influence:
                print(f"    Influence method: {influence_method}")
            
            recommender = PaperPmmRecommender(
                hidden_dim=128,
                embedding_dim=64,
                margin=0.8,
                batch_size=64,
                num_epochs=50,
                learning_rate=0.001,
                use_influence_weighting=use_influence,
                influence_method=influence_method
            )
            
            training_result = recommender.fit(global_performance_matrix, meta_features_df, verbose=True)
            
            if not training_result:
                print("  WARNING: Paper-Style PMM recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                # Get dataset metafeatures
                dataset_id = dataset['id']
                print(f"  Getting Paper-Style PMM recommendation for dataset {dataset_id}...")
                
                # Get metafeatures for this dataset
                if dataset_id in meta_features_df.index:
                    dataset_metafeats = meta_features_df.loc[dataset_id].values
                else:
                    # Try with string conversion
                    dataset_id_str = str(dataset_id)
                    if dataset_id_str in meta_features_df.index:
                        dataset_metafeats = meta_features_df.loc[dataset_id_str].values
                    else:
                        print(f"  WARNING: Dataset {dataset_id} not in metafeatures, using local best pipeline...")
                        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
                        dataset_metafeats = None
                
                if dataset_metafeats is not None:
                    # Get recommendations (returns list of pipeline names)
                    recommendations = recommender.recommend(dataset_metafeats, top_k=3)
                    
                    if recommendations and len(recommendations) > 0:
                        recommended_pipeline = recommendations[0]
                        print(f"  Paper-Style PMM recommender suggests: {recommended_pipeline}")
                        print(f"  Top-3 recommendations: {recommendations}")
                        
                        # Show influence information if available
                        if use_influence and hasattr(recommender, 'dataset_influence_scores'):
                            print(f"  🎯 Using influence weighting (method: {influence_method})")
                            # Show top 3 most influential datasets
                            top_influential = sorted(
                                recommender.dataset_influence_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:3]
                            print(f"  Most influential training datasets:")
                            for ds, score in top_influential:
                                print(f"    - Dataset {ds}: influence score {score:.3f}")
                    else:
                        print(f"  Warning: Paper-Style PMM recommender returned empty recommendations")
                        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
        except Exception as e:
            import traceback
            print(f"  ERROR: Paper-Style PMM recommender failed: {e}")
            print("  Traceback:")
            traceback.print_exc()
            print("  Falling back to local best pipeline...")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            
    else:
        # Default to local best
        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
        print(f"  Using default recommender (best local performance)")
    
    # Get top 3 pipelines
    top_3 = valid_local_perf.sort_values(by=dataset['name'], ascending=False).head(3).index.tolist()
    print(f"  Top-3 pipelines: {top_3}")
    
    scores_dict = valid_local_perf[dataset['name']].to_dict()
    recommended_score = scores_dict[recommended_pipeline]
    print(f"  Best pipeline: {recommended_pipeline} (score: {recommended_score:.4f})")
    
    # Since we've already evaluated all pipelines directly on the test set above,
    # we use the same results for analysis
    
    test_results_df = local_perf_matrix
    valid_test_results = valid_local_perf
    
    print("  Creating rankings with proper handling of ties...")
    sorted_scores = sorted(valid_test_results[dataset['name']].items(), 
                        key=lambda x: x[1], reverse=True)

    current_rank = 1
    last_score = None
    rank_mapping = {}
    skip = 0

    for i, (pipeline, score) in enumerate(sorted_scores):
        if last_score is not None and score != last_score:
            current_rank += skip
            skip = 0
        
        rank_mapping[pipeline] = current_rank
        skip += 1
        last_score = score

    baseline_rank = rank_mapping.get('baseline', float('inf'))
    recommended_rank = rank_mapping.get(recommended_pipeline, float('inf'))

    if recommended_pipeline in rank_mapping and 'baseline' in rank_mapping:
        recommended_score = valid_test_results.loc[recommended_pipeline, dataset['name']]
        baseline_score = valid_test_results.loc['baseline', dataset['name']]
        
        if np.isclose(recommended_score, baseline_score, rtol=1e-5, atol=1e-8):
            better_than_baseline = "equal"
        elif recommended_score > baseline_score:
            better_than_baseline = "yes"
        else:
            better_than_baseline = "no"
    else:
        better_than_baseline = "unknown"

    print(f"  BEST PIPELINE on Test Split: '{sorted_scores[0][0]}'")
    print(f"  RANK of Recommended Pipeline: {recommended_rank}/{len(rank_mapping)}")
    print(f"  RANK of Baseline Pipeline: {baseline_rank}/{len(rank_mapping)}")
    print(f"  COMPARISON to Baseline: {better_than_baseline}")

    score_gap = np.nan
    if recommended_pipeline in valid_test_results.index:
        best_score = sorted_scores[0][1]
        recommended_score = valid_test_results.loc[recommended_pipeline, dataset['name']]
        score_gap = best_score - recommended_score
        print(f"  PERFORMANCE GAP: {score_gap:.4f} ({recommended_score:.4f} vs {best_score:.4f})")
    
    summary = {
        'dataset': dataset['name'], 
        'recommendation': recommended_pipeline, 
        'ground_truth_best': sorted_scores[0][0],
        'rank': recommended_rank,
        'baseline_rank': baseline_rank,
        'better_than_baseline': better_than_baseline,
        'recommended_score': valid_test_results.loc[recommended_pipeline, dataset['name']] if recommended_pipeline in valid_test_results.index else np.nan,
        'baseline_score': valid_test_results.loc['baseline', dataset['name']] if 'baseline' in valid_test_results.index else np.nan,  
        'best_score': sorted_scores[0][1],
        'score_gap': score_gap if recommended_pipeline in valid_test_results.index else np.nan,
        'num_valid_pipelines': len(valid_test_results)
    }
    
    return local_perf_matrix, test_results_df, summary


def analyze_recommendations(test_summary_df, meta_features_df=None):
    """Analyze the performance of recommendations in detail."""
    print("\n" + "="*80)
    print("DETAILED RECOMMENDATION ANALYSIS")
    print("="*80)
    
    # Improvement over baseline
    better_than_baseline = (test_summary_df['better_than_baseline'] == 'yes').mean() * 100
    equal_to_baseline = (test_summary_df['better_than_baseline'] == 'equal').mean() * 100
    worse_than_baseline = (test_summary_df['better_than_baseline'] == 'no').mean() * 100
    
    print(f"Comparison to baseline:")
    print(f"- Better than baseline: {better_than_baseline:.1f}%")
    print(f"- Equal to baseline: {equal_to_baseline:.1f}%")
    print(f"- Worse than baseline: {worse_than_baseline:.1f}%")
    
    # Average performance by pipeline
    recommendations = test_summary_df['recommendation'].value_counts()
    ground_truths = test_summary_df['ground_truth_best'].value_counts()
    
    print("\nMost frequently recommended pipelines:")
    for pipeline, count in recommendations.items():
        print(f"- {pipeline}: {count} times ({count/len(test_summary_df)*100:.1f}%)")
    
    print("\nMost frequently optimal pipelines:")
    for pipeline, count in ground_truths.items():
        print(f"- {pipeline}: {count} times ({count/len(test_summary_df)*100:.1f}%)")
    
    # Average gap by dataset characteristics
    if meta_features_df is not None and 'NumberOfFeatures' in meta_features_df.columns:
        try:
            # Convert dataset column to string
            test_summary_with_str_index = test_summary_df.copy()
            test_summary_with_str_index.index = test_summary_with_str_index.index.astype(str)
            
            # Convert metafeatures index to string 
            meta_features_str_index = meta_features_df.copy()
            meta_features_str_index.index = meta_features_str_index.index.astype(str)
            
            # Extract numeric dataset IDs from test dataset names (e.g., 'D_1503' -> '1503')
            dataset_id_mapping = {}
            for dataset_name in test_summary_with_str_index.index:
                if dataset_name.startswith('D_'):
                    numeric_id = dataset_name.split('_')[1]
                    dataset_id_mapping[dataset_name] = numeric_id
            
            # Create a subset of metafeatures with matching IDs
            matching_metafeatures = []
            matching_test_data = []
            
            for dataset_name, numeric_id in dataset_id_mapping.items():
                if numeric_id in meta_features_str_index.index:
                    matching_test_data.append(test_summary_with_str_index.loc[dataset_name])
                    matching_metafeatures.append(meta_features_str_index.loc[numeric_id])
            
            if matching_test_data and matching_metafeatures:
                # Create DataFrames with consistent indices
                matched_test_df = pd.DataFrame(matching_test_data)
                matched_test_df.index = [f"dataset_{i}" for i in range(len(matched_test_df))]
                
                matched_meta_df = pd.DataFrame(matching_metafeatures)
                matched_meta_df.index = [f"dataset_{i}" for i in range(len(matched_meta_df))]
                
                test_with_features = matched_test_df.join(
                    matched_meta_df[['NumberOfFeatures', 'NumberOfInstances']]
                )
                
                # Group by dataset size
                small = test_with_features[test_with_features['NumberOfInstances'] < 1000]
                medium = test_with_features[(test_with_features['NumberOfInstances'] >= 1000) & 
                                            (test_with_features['NumberOfInstances'] < 5000)]
                large = test_with_features[test_with_features['NumberOfInstances'] >= 5000]
                
                print("\nPerformance by dataset size:")
                if len(small) > 0:
                    print(f"- Small datasets (<1000 samples): Avg rank {small['rank'].mean():.2f}")
                if len(medium) > 0:
                    print(f"- Medium datasets (1000-5000 samples): Avg rank {medium['rank'].mean():.2f}")
                if len(large) > 0:
                    print(f"- Large datasets (>5000 samples): Avg rank {large['rank'].mean():.2f}")
            else:
                print("\nCould not match test datasets with metafeatures for size analysis")
                
        except Exception as e:
            print(f"\nError in dataset size analysis: {e}")
    else:
        print("\nNo metafeatures available for dataset size analysis")