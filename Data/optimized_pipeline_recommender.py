"""
Optimized Pipeline Recommender
================================
Uses Bayesian Optimization (SMAC3) to search for the best preprocessing pipeline
configuration for each test dataset based on:
  - Historical performance (preprocessed_performance.csv)
  - Dataset metafeatures (dataset_feats.csv)

The optimization searches over the space of:
  - imputation: none, mean, median, knn, most_frequent, constant
  - scaling: none, standard, minmax, robust, maxabs
  - encoding: none, onehot
  - feature_selection: none, k_best, mutual_info, variance_threshold
  - outlier_removal: none, iqr, zscore, isolation_forest, lof
  - dimensionality_reduction: none, pca, svd
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ConfigSpace import (
    ConfigurationSpace,
    Categorical,
    Configuration
)
from smac import HyperparameterOptimizationFacade, Scenario
import tempfile
import shutil

warnings.filterwarnings('ignore')


class OptimizedPipelineRecommender:
    """
    Uses SMAC3 Bayesian Optimization to find the best pipeline configuration
    for a given dataset based on historical performance and metafeatures.
    """
    
    def __init__(self, performance_matrix, metafeatures_df, 
                 n_trials=50, use_similarity_weighting=True):
        """
        Args:
            performance_matrix: DataFrame (12 pipelines × N datasets)
            metafeatures_df: DataFrame (N datasets × M features)
            n_trials: Number of SMAC optimization trials
            use_similarity_weighting: Weight similar datasets more in prediction
        """
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.n_trials = n_trials
        self.use_similarity_weighting = use_similarity_weighting
        
        # Pipeline configuration mapping
        self.pipeline_configs = {
            'baseline': {'imputation': 'none', 'scaling': 'none', 'encoding': 'none', 
                        'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
            'simple_preprocess': {'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
                                 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
            'robust_preprocess': {'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 
                                 'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'},
            'feature_selection': {'imputation': 'median', 'scaling': 'standard', 'encoding': 'onehot', 
                                 'feature_selection': 'k_best', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
            'dimension_reduction': {'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
                                   'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
            'conservative': {'imputation': 'median', 'scaling': 'minmax', 'encoding': 'onehot', 
                           'feature_selection': 'variance_threshold', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
            'aggressive': {'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot', 
                          'feature_selection': 'k_best', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'pca'},
            'knn_impute_pca': {'imputation': 'knn', 'scaling': 'standard', 'encoding': 'onehot', 
                              'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'},
            'mutual_info_zscore': {'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot', 
                                  'feature_selection': 'mutual_info', 'outlier_removal': 'zscore', 'dimensionality_reduction': 'none'},
            'constant_maxabs_iforest': {'imputation': 'constant', 'scaling': 'maxabs', 'encoding': 'onehot', 
                                       'feature_selection': 'variance_threshold', 'outlier_removal': 'isolation_forest', 'dimensionality_reduction': 'none'},
            'mean_minmax_lof_svd': {'imputation': 'mean', 'scaling': 'minmax', 'encoding': 'onehot', 
                                   'feature_selection': 'k_best', 'outlier_removal': 'lof', 'dimensionality_reduction': 'svd'},
            'mostfreq_standard_iqr': {'imputation': 'most_frequent', 'scaling': 'standard', 'encoding': 'onehot', 
                                     'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'}
        }
        
        # Reverse mapping: config -> pipeline name (for lookup)
        self.config_to_pipeline = {}
        for name, config in self.pipeline_configs.items():
            config_tuple = tuple(sorted(config.items()))
            self.config_to_pipeline[config_tuple] = name
        
        # Build surrogate model
        self.surrogate_model = None
        self.scaler = StandardScaler()
        self.knn_model = None
        self.trained = False
        
    def fit(self):
        """Train the surrogate model on historical data."""
        print(f"\n{'='*80}")
        print("TRAINING OPTIMIZED PIPELINE RECOMMENDER")
        print(f"{'='*80}")
        
        try:
            # Prepare training data: (dataset_metafeatures + pipeline_config) -> performance
            X_train = []
            y_train = []
            
            for col in self.performance_matrix.columns:
                dataset_id = int(col.split('_')[1])
                
                if dataset_id not in self.metafeatures_df.index:
                    continue
                
                dataset_mf = self.metafeatures_df.loc[dataset_id].values
                
                for pipeline_name in self.performance_matrix.index:
                    performance = self.performance_matrix.loc[pipeline_name, col]
                    
                    if np.isnan(performance):
                        continue
                    
                    # Encode pipeline config as features
                    config = self.pipeline_configs[pipeline_name]
                    config_features = self._encode_pipeline_config(config)
                    
                    # Concatenate: [dataset_metafeatures, pipeline_config_features]
                    combined_features = np.concatenate([dataset_mf, config_features])
                    
                    X_train.append(combined_features)
                    y_train.append(performance)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Handle NaN in metafeatures
            X_train = np.nan_to_num(X_train, nan=0.0)
            
            print(f"✅ Training data: {len(X_train)} examples")
            print(f"   ({len(self.performance_matrix.columns)} datasets × {len(self.pipeline_configs)} pipelines)")
            
            # Scale features
            X_train = self.scaler.fit_transform(X_train)
            
            # Train surrogate model (Random Forest)
            self.surrogate_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.surrogate_model.fit(X_train, y_train)
            
            print(f"✅ Surrogate model trained (R² = {self.surrogate_model.score(X_train, y_train):.4f})")
            
            # Build KNN model for similarity weighting
            if self.use_similarity_weighting:
                mf_matrix = []
                for col in self.performance_matrix.columns:
                    dataset_id = int(col.split('_')[1])
                    if dataset_id in self.metafeatures_df.index:
                        mf_matrix.append(self.metafeatures_df.loc[dataset_id].values)
                
                mf_matrix = np.array(mf_matrix)
                mf_matrix = np.nan_to_num(mf_matrix, nan=0.0)
                
                self.knn_model = NearestNeighbors(n_neighbors=min(10, len(mf_matrix)))
                self.knn_model.fit(mf_matrix)
                
                print(f"✅ KNN similarity model built")
            
            self.trained = True
            return True
            
        except Exception as e:
            print(f"❌ Error training optimized recommender: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _encode_pipeline_config(self, config):
        """
        Encode pipeline configuration as numerical features.
        
        Returns:
            np.array: One-hot encoded features for each component
        """
        # Define all possible values for each component
        component_values = {
            'imputation': ['none', 'mean', 'median', 'knn', 'most_frequent', 'constant'],
            'scaling': ['none', 'standard', 'minmax', 'robust', 'maxabs'],
            'encoding': ['none', 'onehot'],
            'feature_selection': ['none', 'k_best', 'mutual_info', 'variance_threshold'],
            'outlier_removal': ['none', 'iqr', 'zscore', 'isolation_forest', 'lof'],
            'dimensionality_reduction': ['none', 'pca', 'svd']
        }
        
        features = []
        for component, possible_values in component_values.items():
            # One-hot encode
            one_hot = [1 if config[component] == val else 0 for val in possible_values]
            features.extend(one_hot)
        
        return np.array(features, dtype=float)
    
    def _decode_smac_config(self, smac_config):
        """Convert SMAC Configuration to pipeline config dict."""
        return {
            'imputation': smac_config['imputation'],
            'scaling': smac_config['scaling'],
            'encoding': smac_config['encoding'],
            'feature_selection': smac_config['feature_selection'],
            'outlier_removal': smac_config['outlier_removal'],
            'dimensionality_reduction': smac_config['dimensionality_reduction']
        }
    
    def _predict_performance(self, dataset_metafeatures, pipeline_config):
        """
        Predict performance of a pipeline config on a dataset.
        
        Args:
            dataset_metafeatures: np.array of dataset metafeatures
            pipeline_config: dict of pipeline configuration
            
        Returns:
            float: Predicted performance (higher is better)
        """
        # Encode pipeline config
        config_features = self._encode_pipeline_config(pipeline_config)
        
        # Concatenate features
        combined_features = np.concatenate([dataset_metafeatures, config_features])
        combined_features = np.nan_to_num(combined_features, nan=0.0)
        
        # Scale
        combined_features = self.scaler.transform(combined_features.reshape(1, -1))
        
        # Predict
        prediction = self.surrogate_model.predict(combined_features)[0]
        
        return prediction
    
    def recommend(self, dataset_id):
        """
        Recommend the best pipeline configuration for a dataset using SMAC optimization.
        
        Args:
            dataset_id: int, dataset ID
            
        Returns:
            tuple: (best_pipeline_name_or_config, optimization_details)
        """
        if not self.trained:
            print("❌ Model not trained!")
            return None, {}
        
        try:
            # Get dataset metafeatures
            if dataset_id not in self.metafeatures_df.index:
                print(f"❌ No metafeatures for dataset {dataset_id}")
                return None, {}
            
            dataset_mf = self.metafeatures_df.loc[dataset_id].values
            dataset_mf = np.nan_to_num(dataset_mf, nan=0.0)
            
            print(f"\n🔍 Optimizing pipeline for dataset {dataset_id}...")
            
            # Define configuration space
            cs = ConfigurationSpace()
            cs.add_hyperparameters([
                Categorical('imputation', ['none', 'mean', 'median', 'knn', 'most_frequent', 'constant']),
                Categorical('scaling', ['none', 'standard', 'minmax', 'robust', 'maxabs']),
                Categorical('encoding', ['none', 'onehot']),
                Categorical('feature_selection', ['none', 'k_best', 'mutual_info', 'variance_threshold']),
                Categorical('outlier_removal', ['none', 'iqr', 'zscore', 'isolation_forest', 'lof']),
                Categorical('dimensionality_reduction', ['none', 'pca', 'svd'])
            ])
            
            # Define objective function
            def objective(config: Configuration, seed: int = 0):
                """Objective: Maximize predicted performance (SMAC minimizes, so negate)"""
                pipeline_config = self._decode_smac_config(config)
                predicted_perf = self._predict_performance(dataset_mf, pipeline_config)
                return -predicted_perf  # Negate because SMAC minimizes
            
            # Run SMAC optimization
            scenario = Scenario(
                configspace=cs,
                n_trials=self.n_trials,
                deterministic=True,
                n_workers=1
            )
            
            smac = HyperparameterOptimizationFacade(
                scenario=scenario,
                target_function=objective,
                overwrite=True
            )
            
            incumbent = smac.optimize()
            
            # Get best configuration
            best_config = self._decode_smac_config(incumbent)
            best_predicted_perf = -objective(incumbent)  # Negate back
            
            print(f"✅ Optimization complete!")
            print(f"   Predicted performance: {best_predicted_perf:.4f}")
            print(f"   Best config: {best_config}")
            
            # Check if this matches any existing pipeline
            config_tuple = tuple(sorted(best_config.items()))
            if config_tuple in self.config_to_pipeline:
                best_pipeline_name = self.config_to_pipeline[config_tuple]
                print(f"   → Matches existing pipeline: {best_pipeline_name}")
            else:
                best_pipeline_name = f"optimized_{dataset_id}"
                print(f"   → New custom pipeline: {best_pipeline_name}")
            
            # Evaluate all 12 standard pipelines for comparison
            pipeline_predictions = {}
            for name, config in self.pipeline_configs.items():
                pred = self._predict_performance(dataset_mf, config)
                pipeline_predictions[name] = pred
            
            best_standard_pipeline = max(pipeline_predictions.items(), key=lambda x: x[1])
            
            details = {
                'optimized_config': best_config,
                'optimized_pipeline_name': best_pipeline_name,
                'predicted_performance': best_predicted_perf,
                'best_standard_pipeline': best_standard_pipeline[0],
                'best_standard_predicted_perf': best_standard_pipeline[1],
                'improvement': best_predicted_perf - best_standard_pipeline[1],
                'all_predictions': pipeline_predictions
            }
            
            print(f"   Best standard pipeline: {best_standard_pipeline[0]} ({best_standard_pipeline[1]:.4f})")
            print(f"   Improvement: {details['improvement']:.4f} ({details['improvement']/best_standard_pipeline[1]*100:.2f}%)")
            
            return best_config, details
            
        except Exception as e:
            print(f"❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
