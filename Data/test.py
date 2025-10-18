import pandas as pd
import numpy as np
import os
import warnings
import tempfile
import shutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from autogluon.tabular import TabularPredictor
from scipy.stats import zscore
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: USER-DEFINED CONFIGURATIONS
# =============================================================================

AG_ARGS_FIT = {
    "ag.max_memory_usage_ratio": 0.3,
    'num_gpus': 0,
    'num_cpus': min(10, os.cpu_count() if os.cpu_count() else 4)
}

STABLE_MODELS = [
    "GBM", "CAT", "XGB", "RF", "XT", "KNN", "LR", "NN_TORCH", "FASTAI",
    "NN_MXNET", "TABPFN", "DUMMY", "NB"
]

train_dataset_ids = [
    # 22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
    # 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 53, 54, 55,
    # 56, 57, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
    # 187, 188, 275, 276,
    # 277, 278, 285, 300, 301, 307, 308,
    # 310, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336,
    # 337, 338, 339, 340, 342, 343, 346, 372, 375,
    # 378, 443, 444, 446, 448, 449, 450, 451, 452, 453, 454, 455, 457, 458, 459, 461,
    # 462, 463, 464, 465, 467, 468, 469, 471, 2009, 2804, 2309, 1907
    [2, 7, 16, 20, 
    51, 197, 224, 381, 
    382, 405, 416, 422, 488, 503, 507, 
    513, 567, 580, 584, 703, 725, 737, 
    750, 757, 773, 796, 802, 805, 816, 818, 831, 838,
    852, 870, 871, 918, 925, 930, 934, 940, 947, 950, 
    954, 967, 980, 983, 984, 986, 991, 993, 995, 996, 
    999, 1000, 1002, 1012, 1017, 1018, 1021, 1022, 1024, 
    1026, 1047, 1056, 1067, 1068, 1069, 1100, 1116, 1442, 
    1450, 1452, 1453, 1455, 1466, 1467, 1472, 1475, 1482, 1484,
    1493, 1508, 1511, 1512, 1519, 1566, 4552, 23380, 40474, 40497, 
    40536, 40588, 40595, 40649, 40670, 40686, 40693, 40706, 40966, 41470,
    41476, 41482, 41487, 41488, 41489, 41491
    ]
]
test_dataset_ids = [
    1503, 
    # 23517, 1551, 1552, 
    # 183, 255, 545, 
    # 546, 475, 481, 516, 3, 6, 8, 10, 12, 14, 9, 11 ,5
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

# =============================================================================
# SECTION 2: DATA & PREPROCESSING UTILITIES
# =============================================================================


import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class HybridMetaRecommender:
    """
    A hybrid recommender that combines dataset similarity (KNN) with 
    gradient boosting for performance prediction.
    """
    
    def __init__(self, performance_matrix, metafeatures_df):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        
        self.knn_model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')  
        
        # For performance prediction
        self.xgb_model = None
        self.pipeline_names = None
        self.trained = False
        
    def fit(self):
        """Train both KNN and XGBoost models on available performance data."""
        try:
            numeric_columns = {}
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        num_id = int(col.split('_')[1])
                        numeric_columns[col] = num_id
                except:
                    pass
            
            common_datasets = []
            for col, num_id in numeric_columns.items():
                if num_id in self.metafeatures_df.index:
                    common_datasets.append((col, num_id))
            
            if len(common_datasets) < 5:  # Need minimum datasets for KNN
                print("    Warning: Not enough common datasets for training")
                return False
            
            print(f"    Found {len(common_datasets)} common datasets for training")
            
            X_knn = []
            dataset_ids = []
            
            for col, num_id in common_datasets:
                metafeatures = self.metafeatures_df.loc[num_id].values
                X_knn.append(metafeatures)
                dataset_ids.append(num_id)
                
            # Handle NaN values in metafeatures
            X_knn = np.array(X_knn)
            
            # Check if there are any NaN values
            if np.isnan(X_knn).any():
                print(f"    Warning: Found NaN values in metafeatures, imputing with median...")
                X_knn = self.imputer.fit_transform(X_knn)
            else:
                self.imputer.fit(X_knn)
            
            # Scale features for KNN
            X_knn_scaled = self.scaler.fit_transform(X_knn)
            
            k = min(5, max(3, int(np.sqrt(len(common_datasets)))))
            self.knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
            self.knn_model.fit(X_knn_scaled)
            
            X_xgb = []
            y_xgb = []
            self.pipeline_names = self.performance_matrix.index.tolist()
            
            for col, num_id in common_datasets:
                metafeatures = self.metafeatures_df.loc[num_id].values
                
                # Handle NaN in individual metafeatures
                if np.isnan(metafeatures).any():
                    metafeatures = self.imputer.transform([metafeatures])[0]
                
                # Get performance scores for all pipelines on this dataset
                for pipeline in self.pipeline_names:
                    score = self.performance_matrix.loc[pipeline, col]
                    if not np.isnan(score):
                        # Enhanced feature representation: metafeatures + pipeline characteristics
                        pipeline_features = self._get_pipeline_features(pipeline)
                        features = np.concatenate([metafeatures, pipeline_features])
                        X_xgb.append(features)
                        y_xgb.append(score)
            
            if len(X_xgb) < 20:  # Need minimum samples for XGBoost
                print(f"    Warning: Not enough training samples ({len(X_xgb)})")
                return False
                
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            self.xgb_model.fit(X_xgb, y_xgb)
            
            self.trained = True
            print(f"    ✅ Hybrid recommender trained on {len(X_xgb)} examples")
            
            self.dataset_mapping = {i: (col, num_id) for i, (col, num_id) in enumerate(common_datasets)}
            self.X_knn_scaled = X_knn_scaled
            
            return True
            
        except Exception as e:
            print(f"    Error training hybrid recommender: {e}")
            return False
    
    def _get_pipeline_features(self, pipeline_name):
        """Convert pipeline configuration to feature vector."""
        pipeline_config = next((cfg for cfg in pipeline_configs if cfg['name'] == pipeline_name), None)
        if not pipeline_config:
            # Return zeros if pipeline not found
            return np.zeros(21)  # 21 = total features for all pipeline components
        
        # Create feature vector from pipeline components
        features = []
        
        # Imputation (one-hot, 6 options)
        imputation_methods = ['none', 'mean', 'median', 'knn', 'most_frequent', 'constant']
        features.extend([1 if pipeline_config.get('imputation') == method else 0 for method in imputation_methods])
        
        # Scaling (one-hot, 5 options)
        scaling_methods = ['none', 'standard', 'minmax', 'robust', 'maxabs']
        features.extend([1 if pipeline_config.get('scaling') == method else 0 for method in scaling_methods])
        
        # Feature selection (one-hot, 4 options)
        selection_methods = ['none', 'k_best', 'mutual_info', 'variance_threshold']
        features.extend([1 if pipeline_config.get('feature_selection') == method else 0 for method in selection_methods])
        
        # Outlier removal (one-hot, 5 options)
        outlier_methods = ['none', 'iqr', 'zscore', 'isolation_forest', 'lof']
        features.extend([1 if pipeline_config.get('outlier_removal') == method else 0 for method in outlier_methods])
        
        # Dimensionality reduction (one-hot, 3 options)
        dim_red_methods = ['none', 'pca', 'svd']
        features.extend([1 if pipeline_config.get('dimensionality_reduction') == method else 0 for method in dim_red_methods])
        
        return np.array(features)
        
    def recommend(self, dataset_id):
        """Recommend pipeline using a hybrid of KNN similarity and XGBoost prediction."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None
                
            metafeatures = self.metafeatures_df.loc[dataset_id].values.reshape(1, -1)
            
            # Handle NaN values in target metafeatures
            if np.isnan(metafeatures).any():
                print(f"    Warning: NaN values found in metafeatures for dataset {dataset_id}, imputing...")
                metafeatures = self.imputer.transform(metafeatures)
            
            metafeatures_scaled = self.scaler.transform(metafeatures)
            distances, indices = self.knn_model.kneighbors(metafeatures_scaled)
            
            # Get similar dataset weights (inverse distance)
            weights = 1.0 / (distances.flatten() + 1e-8)  # Avoid division by zero
            weights = weights / weights.sum()  # Normalize
            
            similar_dataset_performances = {}
            
            for i, idx in enumerate(indices.flatten()):
                col, _ = self.dataset_mapping[idx]
                weight = weights[i]
                
                for pipeline in self.pipeline_names:
                    score = self.performance_matrix.loc[pipeline, col]
                    if not np.isnan(score):
                        if pipeline not in similar_dataset_performances:
                            similar_dataset_performances[pipeline] = 0
                        similar_dataset_performances[pipeline] += score * weight
            
            xgb_predictions = {}
            
            for pipeline in self.pipeline_names:
                pipeline_features = self._get_pipeline_features(pipeline)
                features = np.concatenate([metafeatures.flatten(), pipeline_features])
                pred = self.xgb_model.predict([features])[0]
                xgb_predictions[pipeline] = pred
            
            final_predictions = {}
            knn_weight = 0.4  # Weight for KNN similarity-based scores
            xgb_weight = 0.6  # Weight for XGBoost predictions
            
            for pipeline in self.pipeline_names:
                knn_score = similar_dataset_performances.get(pipeline, 0)
                xgb_score = xgb_predictions[pipeline]
                
                # Weighted combination
                final_score = (knn_weight * knn_score) + (xgb_weight * xgb_score)
                final_predictions[pipeline] = final_score
            
            # Find the best pipeline
            best_pipeline = max(final_predictions.items(), key=lambda x: x[1])[0]
            print(f"    🔮 Hybrid recommender suggests: {best_pipeline} (predicted score: {final_predictions[best_pipeline]:.4f})")
            
            # Return top-3 recommendations for reference
            top_pipelines = sorted(final_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top-3 pipelines: {[p[0] for p in top_pipelines]}")
            
            return best_pipeline
            
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            return None



class BayesianSurrogateRecommender:
    """Recommends preprocessing pipelines based on dataset meta-features using a RandomForest surrogate model."""
    
    def __init__(self, performance_matrix, metafeatures_df):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pipeline_names = None
        self.trained = False
    
    def fit(self):
        """Train the surrogate model on available performance data."""
        try:
            # Convert string column names (like 'D_516') to numeric IDs for matching
            numeric_columns = {}
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        num_id = int(col.split('_')[1])
                        numeric_columns[col] = num_id
                except:
                    pass
            
            # Find common datasets between performance matrix and metafeatures
            common_datasets = []
            for col, num_id in numeric_columns.items():
                if num_id in self.metafeatures_df.index:
                    common_datasets.append((col, num_id))
            
            if len(common_datasets) < 1:
                print("    Warning: No common datasets between metafeatures and performance matrix")
                return False
            
            print(f"    Found {len(common_datasets)} common datasets for training")
            
            X_train = []
            y_train = []
            self.pipeline_names = self.performance_matrix.index.tolist()
            
            for col, num_id in common_datasets:
                metafeatures = self.metafeatures_df.loc[num_id].values
                
                # Get performance scores for all pipelines on this dataset
                for pipeline in self.pipeline_names:
                    score = self.performance_matrix.loc[pipeline, col]
                    if not np.isnan(score):
                        # For each pipeline-dataset pair, create a feature vector:
                        # [metafeatures + one-hot encoded pipeline]
                        pipeline_idx = self.pipeline_names.index(pipeline)
                        pipeline_onehot = np.zeros(len(self.pipeline_names))
                        pipeline_onehot[pipeline_idx] = 1
                        
                        features = np.concatenate([metafeatures, pipeline_onehot])
                        X_train.append(features)
                        y_train.append(score)
            
            if len(X_train) < 10:  # Need minimum samples for training
                print(f"    Warning: Not enough training samples ({len(X_train)})")
                return False
                

            self.model.fit(X_train, y_train)
            self.trained = True
            print(f"    ✅ Surrogate model trained on {len(X_train)} examples")
            return True
            
        except Exception as e:
            print(f"    Error training surrogate model: {e}")
            return False
    
    def recommend(self, dataset_id):
        """Recommend the best pipeline for a dataset based on its meta-features."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None
            
        try:
            # Get metafeatures for the target dataset
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None
                
            metafeatures = self.metafeatures_df.loc[dataset_id].values
            
            # Predict performance for each pipeline
            predictions = {}
            for pipeline in self.pipeline_names:
                pipeline_idx = self.pipeline_names.index(pipeline)
                pipeline_onehot = np.zeros(len(self.pipeline_names))
                pipeline_onehot[pipeline_idx] = 1
                
                features = np.concatenate([metafeatures, pipeline_onehot])
                pred = self.model.predict([features])[0]
                predictions[pipeline] = pred
            
            # Find the best pipeline
            best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
            print(f"    🔮 Surrogate model recommends: {best_pipeline} (predicted score: {predictions[best_pipeline]:.4f})")
            return best_pipeline
            
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            return None

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
            print(f"❌ Dataset {dataset_id} too small: {len(X)} samples")
            return None
            
        print(f"✅ Loaded dataset {dataset_id}: Shape={X.shape}, Classes={y.nunique()}")
        return {'id': dataset_id, 'name': f"D_{dataset_id}", 'X': X, 'y': y.astype(int)}
        
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_id}: {e}")
        with open('failed_datasets.log', 'a') as f:
            f.write(f"Dataset {dataset_id}: {str(e)}\n")
        return None
        
def get_metafeatures(dataset_id, meta_features_df):
    try:
        return meta_features_df.loc[dataset_id].to_dict()
    except KeyError:
        print(f"    Warning: No metafeatures found for dataset {dataset_id}")
        return None

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
            
            # 2. Fit Feature Selection Models
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

# =============================================================================
# SECTION 4: EVALUATION & RECOMMENDER
# =============================================================================
def get_temp_dir():
    """Create a new temporary directory with random name to avoid conflicts"""
    import uuid
    temp_dir = os.path.join(tempfile.gettempdir(), f"ag_temp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def run_autogluon_evaluation(X_train, y_train, X_test, y_test):
    if len(X_train) == 0 or len(X_test) == 0:
        print("      Warning: Empty dataset provided to AutoGluon")
        return np.nan
        
    temp_dir = get_temp_dir()
    try:
        # Reset column names to avoid mismatch issues
        X_train_ag = X_train.copy()
        X_train_ag.columns = [f"col_{i}" for i in range(X_train_ag.shape[1])]
        
        X_test_ag = X_test.copy()
        X_test_ag.columns = [f"col_{i}" for i in range(X_test_ag.shape[1])]
        
        train_data = X_train_ag.copy()
        train_data['target'] = y_train.values
        
        problem_type = 'binary' if y_train.nunique() <= 2 else 'multiclass'
        predictor = TabularPredictor(
            label='target', 
            path=temp_dir, 
            problem_type=problem_type, 
            eval_metric='accuracy', 
            verbosity=0
        )
        
        predictor.fit(
            train_data, 
            time_limit=600, 
            presets='medium_quality',
            included_model_types=STABLE_MODELS, 
            hyperparameter_tune_kwargs=None,
            feature_generator=None, 
            ag_args_fit=AG_ARGS_FIT, 
            raise_on_no_models_fitted=False
        )
        
        preds = predictor.predict(X_test_ag)
        return accuracy_score(y_test, preds)
        
    except Exception as e:
        print(f"      Warning: AutoGluon evaluation failed: {e}")
        
        # Fallback to a simple RandomForest
        try:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            return accuracy_score(y_test, preds)
        except Exception as e2:
            print(f"      Warning: Fallback classifier also failed: {e2}")
            return np.nan
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

# Modify the main experiment function
def run_experiment_for_dataset(dataset, meta_features_df, global_performance_matrix=None):
    print(f"\n{'='*30} EXPERIMENT FOR {dataset['name']} ({dataset['id']}) {'='*30}")
    
    # Use safe splitting function
    X_train_val, X_test, y_train_val, y_test = safe_train_test_split(
        dataset['X'], dataset['y'], test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = safe_train_test_split(
        X_train_val, y_train_val, test_size=0.5, random_state=42
    )
    
    print(f"  Splits created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Check if splits are too small
    if len(X_train) < 10 or len(X_val) < 5 or len(X_test) < 5:
        print("  ⚠️ Dataset too small for reliable evaluation, skipping...")
        return None, None, None
    
    print("  Building local knowledge base (Train -> Val)...")
    local_perf = {}
    
    for config in pipeline_configs:
        print(f"    Evaluating '{config['name']}'...")
        try:
            if config['name'] == 'baseline':
                score = run_autogluon_evaluation(X_train, y_train, X_val, y_val)
            else:
                preprocessor = Preprocessor(config)
                preprocessor.fit(X_train, y_train)
                X_train_p, y_train_p = preprocessor.transform(X_train, y_train)
                X_val_p, y_val_p = preprocessor.transform(X_val, y_val)
                
                if len(X_train_p) > 0 and len(X_val_p) > 0:
                    score = run_autogluon_evaluation(X_train_p, y_train_p, X_val_p, y_val_p)
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
        print("  ⚠️ No valid local performance results, skipping recommender...")
        return local_perf_matrix, None, None
    
    # Determine the recommended pipeline
    if global_performance_matrix is not None and len(global_performance_matrix.columns) > 0:
        # Using global knowledge
        print("  Using global knowledge for recommendation...")
        meta_features = get_metafeatures(dataset['id'], meta_features_df)
        if not meta_features:
            print("  ⚠️ No metafeatures available, using local best pipeline...")
            recommended_pipeline = valid_local_perf.idxmax().iloc[0]
        else:
            # recommender = BayesianSurrogateRecommender(
            #     global_performance_matrix, 
            #     meta_features_df
            # )
            recommender = HybridMetaRecommender(
                global_performance_matrix, 
                meta_features_df
            )
            if not recommender.fit():
                print("  ⚠️ Recommender training failed, using local best pipeline...")
                recommended_pipeline = valid_local_perf.idxmax().iloc[0]
            else:
                recommended_pipeline = recommender.recommend(dataset['id'])

                pipeline_names = [config['name'] for config in pipeline_configs]



    else:
        # No global knowledge, just use local best
        print("  No global knowledge available, using local best pipeline...")
        recommended_pipeline = valid_local_perf.idxmax().iloc[0]
    
    print(f"\n  🎯 Recommender's Choice (from Val set): '{recommended_pipeline}'")

    # print("  Evaluating ground truth on the TEST split...")
    # ground_truth_perf = {}
    # MODIFIED SECTION: Evaluate all pipelines (including recommended) on the SAME test split
    print("  Evaluating all pipelines on the SAME test split for fair comparison...")
    
    # Dictionary to store the preprocessed test datasets for each pipeline
    preprocessed_test_data = {}
    
    for config in pipeline_configs:
        pipeline_name = config['name']
        print(f"    Preparing test data for '{pipeline_name}'...")


        # print(f"    Evaluating '{config['name']}' on test set...")
        try:
            # if config['name'] == 'baseline':
            #     score = run_autogluon_evaluation(X_train_val, y_train_val, X_test, y_test)
            if pipeline_name == 'baseline':
                # For baseline, just use the original test data
                preprocessed_test_data[pipeline_name] = {
                    'X_train': X_train_val.copy(),
                    'y_train': y_train_val.copy(),
                    'X_test': X_test.copy(),
                    'y_test': y_test.copy()
                }
            else:
                preprocessor = Preprocessor(config)
                preprocessor.fit(X_train_val, y_train_val)
                X_train_val_p, y_train_val_p = preprocessor.transform(X_train_val, y_train_val)
                X_test_p, y_test_p = preprocessor.transform(X_test, y_test)
                
                # if len(X_train_val_p) > 0 and len(X_test_p) > 0:
                #     score = run_autogluon_evaluation(X_train_val_p, y_train_val_p, X_test_p, y_test_p)
                # else:
                #     print(f"      Warning: Preprocessing resulted in empty dataset")
                #     score = np.nan
                if len(X_train_val_p) > 0 and len(X_test_p) > 0:
                    preprocessed_test_data[pipeline_name] = {
                        'X_train': X_train_val_p,
                        'y_train': y_train_val_p,
                        'X_test': X_test_p,
                        'y_test': y_test_p
                    }
                else:
                    print(f"      Warning: Preprocessing for '{pipeline_name}' resulted in empty dataset")
                    test_results[pipeline_name] = np.nan
                    
            # ground_truth_perf[config['name']] = score
            
        except Exception as e:
            print(f"      ERROR: Could not prepare test data for '{pipeline_name}'. Reason: {e}")
        # except Exception as e:
        #      print(f"      ERROR: Could not evaluate ground truth for {config['name']}. Reason: {e}")
        #      ground_truth_perf[config['name']] = np.nan

    # gt_df = pd.DataFrame.from_dict(ground_truth_perf, orient='index', columns=[dataset['name']])
    # valid_gt = gt_df.dropna().sort_values(by=dataset['name'], ascending=False)
    
    # if valid_gt.empty: 
    #     print("  ⚠️ No valid ground truth results")
    #     return local_perf_matrix, gt_df, None

    # gt_ranking = valid_gt.index.tolist()
    # rank = gt_ranking.index(recommended_pipeline) + 1 if recommended_pipeline in gt_ranking else -1
    
    # print(f"  🏆 Ground Truth Best (on Test set): '{gt_ranking[0]}'")
    # print(f"  📊 Final Rank of Recommendation: {rank}/{len(gt_ranking)}")

    # Step 2: Evaluate all pipelines on their preprocessed test data
    print("  Evaluating pipelines on test split...")
    test_results = {}
    
    for pipeline_name, data in preprocessed_test_data.items():
        print(f"    Evaluating '{pipeline_name}' on test split...")
        try:
            score = run_autogluon_evaluation(
                data['X_train'], data['y_train'], 
                data['X_test'], data['y_test']
            )
            test_results[pipeline_name] = score
        except Exception as e:
            print(f"      ERROR: Test evaluation failed for '{pipeline_name}'. Reason: {e}")
            test_results[pipeline_name] = np.nan
    
    test_results_df = pd.DataFrame.from_dict(test_results, orient='index', columns=[dataset['name']])
    valid_test_results = test_results_df.dropna().sort_values(by=dataset['name'], ascending=False)
    
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

    print(f"  🏆 Best Pipeline on Test Split: '{sorted_scores[0][0]}'")
    print(f"  📊 Rank of Recommended Pipeline: {recommended_rank}/{len(rank_mapping)}")
    print(f"  📊 Rank of Baseline Pipeline: {baseline_rank}/{len(rank_mapping)}")
    print(f"  📊 Better than Baseline: {better_than_baseline}")

    score_gap = np.nan
    if recommended_pipeline in valid_test_results.index:
        best_score = sorted_scores[0][1]
        recommended_score = valid_test_results.loc[recommended_pipeline, dataset['name']]
        score_gap = best_score - recommended_score
        print(f"  📉 Performance Gap: {score_gap:.4f} ({recommended_score:.4f} vs {best_score:.4f})")
    
    
    summary = {
        'dataset': dataset['name'], 
        'recommendation': recommended_pipeline, 
        # 'ground_truth_best': gt_ranking[0], 
        'ground_truth_best': sorted_scores[0][0],
        # 'rank': rank,
        'rank': recommended_rank,
        'baseline_rank': baseline_rank,
        'better_than_baseline': better_than_baseline,
        # 'recommended_score': gt_df.loc[recommended_pipeline, dataset['name']] if recommended_pipeline in gt_df.index else np.nan,
        'recommended_score': valid_test_results.loc[recommended_pipeline, dataset['name']] if recommended_pipeline in valid_test_results.index else np.nan,
        'baseline_score': valid_test_results.loc['baseline', dataset['name']] if 'baseline' in valid_test_results.index else np.nan,  
        # 'best_score': valid_gt.iloc[0, 0],
        'best_score': sorted_scores[0][1],
        'score_gap': score_gap if recommended_pipeline in valid_test_results.index else np.nan,
        # 'num_valid_pipelines': len(valid_gt)
        'num_valid_pipelines': len(valid_test_results)

    }
    # return local_perf_matrix, gt_df, summary
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

def main():
    print("🚀 STARTING RECOMMENDER SYSTEM EXPERIMENT 🚀")
    
    meta_features_df = pd.read_csv("dataset_feats.csv", index_col=0)
    print(f"✅ Loaded metafeatures for {len(meta_features_df)} datasets")
   
    print("\n" + "="*80)
    print("PHASE 1: BUILDING GLOBAL KNOWLEDGE (TRAINING DATASETS)")
    print("="*80)
    
    training_local_perfs, training_gt_perfs, training_summaries = [], [], []
    successful_training = 0
    failed_training = 0
    
    for i, ds_id in enumerate(train_dataset_ids):
        print(f"\n[{i+1}/{len(train_dataset_ids)}] Processing TRAINING dataset {ds_id}...")
        
        dataset = load_openml_dataset(ds_id)
        if dataset is None:
            failed_training += 1
            continue
            
        try:
            local_perf, gt_perf, summary = run_experiment_for_dataset(dataset, meta_features_df)
            
            if local_perf is not None: 
                training_local_perfs.append(local_perf)
            if gt_perf is not None: 
                training_gt_perfs.append(gt_perf)
            if summary is not None: 
                training_summaries.append(summary)
                successful_training += 1
            else:
                failed_training += 1
                
        except Exception as e:
            print(f"  ❌ Experiment failed for training dataset {ds_id}: {e}")
            failed_training += 1
            continue
    
    if training_local_perfs:
        global_performance_matrix = pd.concat(training_local_perfs, axis=1)
        global_performance_matrix.to_csv('training_performance_matrix.csv')
        print(f"✅ Saved global performance matrix to 'training_performance_matrix.csv' ({global_performance_matrix.shape})")
    else:
        print("⚠️ No training performance data collected")
        global_performance_matrix = pd.DataFrame()
    
    print("\n" + "="*80)
    print("PHASE 2: EVALUATING ON TEST DATASETS")
    print("="*80)
    
    test_local_perfs, test_gt_perfs, test_summaries = [], [], []
    successful_test = 0
    failed_test = 0
    
    for i, ds_id in enumerate(test_dataset_ids):
        print(f"\n[{i+1}/{len(test_dataset_ids)}] Processing TEST dataset {ds_id}...")
        
        dataset = load_openml_dataset(ds_id)
        if dataset is None:
            failed_test += 1
            continue
            
        try:
            local_perf, gt_perf, summary = run_experiment_for_dataset(
                dataset, 
                meta_features_df,
                global_performance_matrix
            )
            
            if local_perf is not None: 
                test_local_perfs.append(local_perf)
            if gt_perf is not None: 
                test_gt_perfs.append(gt_perf)
            if summary is not None: 
                test_summaries.append(summary)
                successful_test += 1
            else:
                failed_test += 1
                
        except Exception as e:
            print(f"  ❌ Experiment failed for test dataset {ds_id}: {e}")
            failed_test += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETION SUMMARY")
    print(f"{'='*80}")
    print(f"Training datasets: {successful_training} successful, {failed_training} failed")
    print(f"Test datasets: {successful_test} successful, {failed_test} failed")
    
    if test_local_perfs:
        test_local_df = pd.concat(test_local_perfs, axis=1)
        test_local_df.to_csv('test_local_performance.csv')
        print(f"\n✅ Saved test local performances to 'test_local_performance.csv' ({test_local_df.shape})")
        
    if test_gt_perfs:
        test_gt_df = pd.concat(test_gt_perfs, axis=1)
        test_gt_df.to_csv('test_ground_truth_performance.csv')
        print(f"✅ Saved test ground truth performances to 'test_ground_truth_performance.csv' ({test_gt_df.shape})")
        
    if test_summaries:
        test_summary_df = pd.DataFrame(test_summaries).set_index('dataset')
        test_summary_df.to_csv('test_evaluation_summary.csv')
        print(f"✅ Saved test summary to 'test_evaluation_summary.csv' ({test_summary_df.shape})")
        
        print(f"\n{'='*80}")
        print("🏁 TEST RESULTS SUMMARY 🏁")
        print(f"{'='*80}")
        print(test_summary_df[['recommendation', 'ground_truth_best', 'rank', 'baseline_rank', 'better_than_baseline', 'recommended_score',  'baseline_score', 'best_score']])
        
        valid_ranks = test_summary_df[test_summary_df['rank'] > 0]['rank']
        if not valid_ranks.empty:
            avg_rank = valid_ranks.mean()
            top1_accuracy = (valid_ranks == 1).mean() * 100
            top3_accuracy = (valid_ranks <= 3).mean() * 100
            
            print(f"\n📊 TEST PERFORMANCE METRICS:")
            print(f"- Average Rank: {avg_rank:.2f}")
            print(f"- Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"- Top-3 Accuracy: {top3_accuracy:.2f}%")
            print(f"- Valid Recommendations: {len(valid_ranks)}/{len(test_summary_df)}")
            
            # Show score differences
            valid_summaries = test_summary_df[test_summary_df['rank'] > 0]
            if not valid_summaries.empty:
                score_diff = valid_summaries['best_score'] - valid_summaries['recommended_score']
                avg_score_diff = score_diff.mean()
                print(f"- Average Score Gap: {avg_score_diff:.4f}")
                print(f"- Score Gap Std: {score_diff.std():.4f}")
        else:
            print("⚠️ No valid recommendations were made on test datasets")
    else:
        print("⚠️ No successful test experiments completed")

    analyze_recommendations(test_summary_df, meta_features_df)

def quick_test():
    """Run a quick test with just a few datasets"""
    print("🚀 STARTING QUICK TEST OF RECOMMENDER SYSTEM 🚀")
    
    meta_features_df = pd.read_csv("dataset_feats.csv", index_col=0)
    print(f"✅ Loaded metafeatures for {len(meta_features_df)} datasets")
   
    # Use a small subset of training datasets
    quick_train_ids = train_dataset_ids[:5] 
    quick_test_ids = test_dataset_ids[:2]    
    
    print("\n" + "="*80)
    print("PHASE 1: BUILDING GLOBAL KNOWLEDGE (TRAINING DATASETS)")
    print("="*80)
    
    training_local_perfs = []
    
    for i, ds_id in enumerate(quick_train_ids):
        print(f"\n[{i+1}/{len(quick_train_ids)}] Processing TRAINING dataset {ds_id}...")
        
        dataset = load_openml_dataset(ds_id)
        if dataset is None:
            continue
            
        try:
            local_perf, _, _ = run_experiment_for_dataset(dataset, meta_features_df)
            
            if local_perf is not None: 
                training_local_perfs.append(local_perf)
                
        except Exception as e:
            print(f"  ❌ Experiment failed for training dataset {ds_id}: {e}")
            continue
    
    if training_local_perfs:
        global_performance_matrix = pd.concat(training_local_perfs, axis=1)
    else:
        print("⚠️ No training performance data collected")
        global_performance_matrix = pd.DataFrame()
    
    print("\n" + "="*80)
    print("PHASE 2: EVALUATING ON TEST DATASETS")
    print("="*80)
    
    test_summaries = []
    
    for i, ds_id in enumerate(quick_test_ids):
        print(f"\n[{i+1}/{len(quick_test_ids)}] Processing TEST dataset {ds_id}...")
        
        dataset = load_openml_dataset(ds_id)
        if dataset is None:
            continue
            
        try:
            _, _, summary = run_experiment_for_dataset(
                dataset, 
                meta_features_df,
                global_performance_matrix
            )
            
            if summary is not None: 
                test_summaries.append(summary)
                
        except Exception as e:
            print(f"  ❌ Experiment failed for test dataset {ds_id}: {e}")
            continue
    
    if test_summaries:
        test_summary_df = pd.DataFrame(test_summaries).set_index('dataset')
        print(f"\n{'='*80}")
        print("🏁 QUICK TEST RESULTS SUMMARY 🏁")
        print(f"{'='*80}")
        print(test_summary_df[['recommendation', 'ground_truth_best', 'rank', 'baseline_rank', 'better_than_baseline']])
        
        if len(test_summary_df) > 0:
            analyze_recommendations(test_summary_df, meta_features_df)

    else:
        print("⚠️ No successful test experiments completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SoluRec pipeline recommender experiment')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with fewer datasets')
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        main()