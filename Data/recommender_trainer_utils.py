import pandas as pd
import numpy as np
import os
import warnings
import tempfile
import shutil
import uuid
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from scipy.stats import zscore
from autogluon.tabular import TabularPredictor

# Utility functions for evaluation
def get_temp_dir():
    """Create a new temporary directory with random name to avoid conflicts"""
    temp_dir = os.path.join(tempfile.gettempdir(), f"ag_temp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

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
    """Get metafeatures for a dataset if available"""
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

def run_autogluon_evaluation(X_train, y_train, X_test, y_test):
    """Run AutoGluon evaluation on the given data."""
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
            included_model_types=['GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'LR', 'NN_TORCH', 'FASTAI',
                                 'NN_MXNET', 'TABPFN', 'DUMMY', 'NB'], 
            hyperparameter_tune_kwargs=None,
            feature_generator=None, 
            ag_args_fit={"ag.max_memory_usage_ratio": 0.3,
                        'num_gpus': 0,
                        'num_cpus': min(10, os.cpu_count() if os.cpu_count() else 4)}, 
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