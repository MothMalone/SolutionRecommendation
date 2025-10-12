import pandas as pd
import numpy as np
import os
import warnings
import tempfile
import shutil
import uuid
import argparse
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import zscore
import scipy.stats as st
import xgboost as xgb
from autogluon.tabular import TabularPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import evaluation utilities
from evaluation_utils import (
    Preprocessor, can_stratify, safe_train_test_split, load_openml_dataset,
    get_metafeatures, run_autogluon_evaluation, run_experiment_for_dataset,
    analyze_recommendations
)

# Import optimization-based recommender
from optimized_pipeline_recommender import OptimizedPipelineRecommender



os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')

# =============================================================================
# USER-DEFINED CONFIGURATIONS - Kept the same as in original script
# =============================================================================

AG_ARGS_FIT = {
    "ag.max_memory_usage_ratio": 0.3,
    'num_gpus': 1,
    'num_cpus': min(10, os.cpu_count() if os.cpu_count() else 4)
}

STABLE_MODELS = [
    "GBM", "CAT", "XGB", "RF", "XT", "KNN", "LR", "NN_TORCH", "FASTAI",
    "NN_MXNET", "TABPFN", "DUMMY", "NB"
]

train_dataset_ids = [
    22, 23, 24, 26, 28, 29, 30, 31, 32, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 53, 54, 55,
    56, 57, 59, 60, 61, 62, 163, 164, 171, 181, 182, 185, 186,
    187, 188, 275, 276, 277, 278, 285, 300, 301, 307, 308,
    310, 311, 312, 313, 316, 327, 328, 329, 333, 334, 335, 336,
    337, 338, 339, 340, 342, 343, 346, 372, 375
]

test_dataset_ids = [
    1503, 23517, 1551, 1552, 183, 255, 545, 546, 475, 481, 
    516, 3, 6, 8, 10, 12, 14, 9, 11, 5
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
# RECOMMENDER CLASSES - Modified to include all recommender types
# =============================================================================

class PmmRecommender:
    """
    Pipeline Meta-Model (PMM) Recommender using Siamese Network for dataset similarity.
    This recommender uses a Siamese neural network to learn a similarity metric 
    between datasets based on their meta-features.
    
    Enhanced with sample-level influence weighting to prioritize more informative datasets.
    """
    
    def __init__(self, hidden_dim=64, embedding_dim=32, margin=1.0, 
                 batch_size=32, num_epochs=20, learning_rate=0.001, 
                 similarity_threshold=0.8, use_influence_weighting=True,
                 influence_method='performance_variance'):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        self.use_influence_weighting = use_influence_weighting
        self.influence_method = influence_method  # 'performance_variance', 'gradient_norm', 'data_diversity'
        self.model = None
        self.dataset_embeddings = {}
        self.dataset_influence_scores = {}  # NEW: Store influence scores for each dataset
        self.is_trained = False
        self.scaler = StandardScaler()
        self.pipeline_names = []
        self.dataset_names = []
        self.d_format_to_dataset = {}  # Mapping from D_X format to dataset ID
    
    def _calculate_dataset_influence_scores(self, performance_matrix, metafeatures_df, dataset_id_to_column):
        """
        Calculate influence scores for each training dataset.
        Higher scores indicate more influential/informative datasets.
        
        Methods:
        1. 'performance_variance': Datasets that show high variance in pipeline performance
           are more informative (can better distinguish between good/bad pipelines)
        2. 'data_diversity': Datasets with unique metafeature profiles are more valuable
        3. 'discriminative_power': Datasets where the best pipeline significantly outperforms others
        
        Args:
            performance_matrix: Performance data
            metafeatures_df: Metafeatures data
            dataset_id_to_column: Mapping from dataset IDs to performance matrix columns
            
        Returns:
            Dictionary mapping dataset_id to influence score
        """
        influence_scores = {}
        
        print(f"    Calculating influence scores using method: {self.influence_method}")
        
        if self.influence_method == 'performance_variance':
            # Datasets with higher variance in pipeline performance are more informative
            for dataset_id, col in dataset_id_to_column.items():
                if col in performance_matrix.columns:
                    perfs = performance_matrix[col].dropna()
                    if len(perfs) > 0:
                        # Higher variance = more discriminative = more influential
                        variance = perfs.var()
                        influence_scores[dataset_id] = variance
        
        elif self.influence_method == 'discriminative_power':
            # Datasets where the best pipeline significantly outperforms the worst
            for dataset_id, col in dataset_id_to_column.items():
                if col in performance_matrix.columns:
                    perfs = performance_matrix[col].dropna()
                    if len(perfs) >= 2:
                        # Gap between best and worst pipeline
                        gap = perfs.max() - perfs.min()
                        # Also consider how clear the ranking is (std of top 3 vs bottom 3)
                        sorted_perfs = perfs.sort_values(ascending=False)
                        if len(sorted_perfs) >= 6:
                            top_3_std = sorted_perfs.iloc[:3].std()
                            bottom_3_std = sorted_perfs.iloc[-3:].std()
                            # Clear separation between top and bottom = more informative
                            clarity = gap / (top_3_std + bottom_3_std + 1e-6)
                        else:
                            clarity = gap
                        influence_scores[dataset_id] = clarity
        
        elif self.influence_method == 'data_diversity':
            # Datasets with unique metafeature profiles are more valuable
            all_metafeatures = []
            dataset_ids_ordered = []
            
            for dataset_id in dataset_id_to_column.keys():
                if dataset_id in metafeatures_df.index:
                    mf = metafeatures_df.loc[dataset_id].values
                    if not np.isnan(mf).any():
                        all_metafeatures.append(mf)
                        dataset_ids_ordered.append(dataset_id)
            
            if len(all_metafeatures) > 1:
                all_metafeatures = np.array(all_metafeatures)
                # Calculate distance to nearest neighbor (uniqueness)
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(all_metafeatures)
                # Set diagonal to infinity to ignore self-distance
                np.fill_diagonal(distances, np.inf)
                # Distance to nearest neighbor = uniqueness score
                min_distances = distances.min(axis=1)
                
                for i, dataset_id in enumerate(dataset_ids_ordered):
                    influence_scores[dataset_id] = min_distances[i]
        
        else:  # Default: equal weighting
            for dataset_id in dataset_id_to_column.keys():
                influence_scores[dataset_id] = 1.0
        
        # Normalize scores with stronger differentiation to make influence more impactful
        if influence_scores:
            scores = np.array(list(influence_scores.values()))
            dataset_ids_list = list(influence_scores.keys())
            
            # Avoid division by zero
            if scores.std() > 1e-8:
                # Z-score normalization
                scores = (scores - scores.mean()) / scores.std()
                # Apply STRONGER exponential to emphasize differences (exp(2*z) instead of exp(z))
                # This creates much more separation between high and low influence
                scores = np.exp(2.0 * scores)
                # Normalize to [0, 1]
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                # Scale to [0.1, 2.0] for 20x range (instead of 0.5-1.5 for 3x range)
                # This means influential datasets get 20x more weight than uninfluential ones
                scores = 0.1 + (scores * 1.9)
            else:
                scores = np.ones_like(scores)
            
            # Update influence scores with normalized values
            for i, dataset_id in enumerate(dataset_ids_list):
                influence_scores[dataset_id] = scores[i]
            
            # Print detailed statistics
            print(f"    ✅ Calculated influence scores for {len(influence_scores)} datasets")
            print(f"       Range: [{scores.min():.3f}, {scores.max():.3f}] (ratio: {scores.max()/scores.min():.1f}x)")
            print(f"       Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
            
            # Show top 5 most influential datasets
            top_influential = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"       Top 5 most influential datasets:")
            for ds_id, score in top_influential:
                print(f"         Dataset {ds_id}: {score:.3f}")
        
        return influence_scores
    
    def _map_dataset_to_column(self, dataset_id, performance_matrix):
        """
        Map a dataset ID to its corresponding column in the performance matrix
        
        Args:
            dataset_id: Dataset ID to map
            performance_matrix: Performance matrix DataFrame
            
        Returns:
            Column name if found, None otherwise
        """
        # Case 1: Direct match
        if dataset_id in performance_matrix.columns:
            return dataset_id
            
        # Case 2: D_X format
        formatted_id = f"D_{dataset_id}"
        if formatted_id in performance_matrix.columns:
            return formatted_id
            
        # Case 3: Strip D_ prefix if present in dataset_id
        if isinstance(dataset_id, str) and dataset_id.startswith("D_"):
            numeric_id = dataset_id[2:]
            if numeric_id in performance_matrix.columns:
                return numeric_id
            try:
                # Try as integer
                int_id = int(numeric_id)
                if int_id in performance_matrix.columns:
                    return int_id
            except:
                pass
                
        # Case 4: Try numeric conversion for string IDs
        if isinstance(dataset_id, str) and dataset_id.isdigit():
            # Try as integer
            int_id = int(dataset_id)
            if int_id in performance_matrix.columns:
                return int_id
            # Try with D_ prefix
            formatted_int_id = f"D_{int_id}"
            if formatted_int_id in performance_matrix.columns:
                return formatted_int_id
        
        # Not found
        return None
        
    def fit(self, performance_matrix, metafeatures_df):
        """
        Train Siamese network to learn dataset similarity
        
        Args:
            performance_matrix: DataFrame with pipelines as rows, datasets as columns
            metafeatures_df: DataFrame with datasets as rows, meta-features as columns
        """
        try:
            # Store dataset and pipeline info
            self.pipeline_names = performance_matrix.index.tolist()
            self.dataset_names = performance_matrix.columns.tolist()
            
            # Check for NaN values in metafeatures
            nan_count = metafeatures_df.isna().sum().sum()
            if nan_count > 0:
                print(f"    ⚠️ Found {nan_count} NaN values in metafeatures. Imputing missing values.")
                # Impute missing values with median
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                imputed_values = imputer.fit_transform(metafeatures_df.values)
                metafeatures_df = pd.DataFrame(
                    imputed_values, index=metafeatures_df.index, columns=metafeatures_df.columns
                )
                print(f"    ✅ Successfully imputed missing values in metafeatures.")
            
            # Scale meta-features
            self.metafeatures_df = metafeatures_df.copy()
            metafeatures_values = self.scaler.fit_transform(metafeatures_df.values)
            scaled_metafeatures_df = pd.DataFrame(
                metafeatures_values, index=metafeatures_df.index, columns=metafeatures_df.columns
            )
            
            # Create dataset
            siamese_dataset = self._create_siamese_dataset(scaled_metafeatures_df, performance_matrix)
            
            if len(siamese_dataset) < 10:  # Need minimum dataset pairs for training
                print("    Warning: Not enough dataset pairs for training")
                return False
                
            # Split into train and validation sets
            train_size = int(0.8 * len(siamese_dataset))
            val_size = len(siamese_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                siamese_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            
            # Initialize model
            input_dim = metafeatures_df.shape[1]
            self.model = self._create_siamese_model(input_dim)
            
            # Loss function and optimizer
            criterion = self._contrastive_loss
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10  # Early stopping patience
            
            for epoch in range(self.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                
                for data1, data2, label in train_loader:
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output1, output2 = self.model(data1, data2)
                    
                    # Calculate loss
                    loss = criterion(output1, output2, label)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * data1.size(0)
                
                train_loss /= len(train_loader.dataset)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for data1, data2, label in val_loader:
                        output1, output2 = self.model(data1, data2)
                        loss = criterion(output1, output2, label)
                        val_loss += loss.item() * data1.size(0)
                
                val_loss /= len(val_loader.dataset)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            
            # Compute and store embeddings ONLY for datasets in the performance matrix
            self.model.eval()
            self.dataset_embeddings = {}
            self.d_format_to_dataset = {}  # New mapping to help with D_X format lookups
            
            # First create mapping between dataset IDs in metafeatures and performance matrix columns
            dataset_id_to_column = {}
            for col in self.dataset_names:  # These are performance matrix columns
                # Try to extract dataset ID from column name (e.g., D_123 -> 123)
                if col.startswith('D_') and col[2:].isdigit():
                    dataset_id = int(col[2:])
                    if dataset_id in metafeatures_df.index:
                        dataset_id_to_column[dataset_id] = col
                        # Also store the reverse mapping
                        self.d_format_to_dataset[col] = dataset_id
                elif col in metafeatures_df.index:
                    dataset_id_to_column[col] = col
            
            print(f"    Found {len(dataset_id_to_column)} dataset mappings between metafeatures and performance matrix")
            print(f"    ⚠️ IMPORTANT: Only storing embeddings for {len(dataset_id_to_column)} datasets that are in the performance matrix")
            
            # Calculate influence scores for each training dataset
            if self.use_influence_weighting:
                self.dataset_influence_scores = self._calculate_dataset_influence_scores(
                    performance_matrix, metafeatures_df, dataset_id_to_column
                )
            else:
                # Equal weighting for all datasets
                self.dataset_influence_scores = {ds_id: 1.0 for ds_id in dataset_id_to_column.keys()}
            
            with torch.no_grad():
                # Store embeddings ONLY for dataset IDs that are in the performance matrix
                valid_embeddings_count = 0
                for dataset_id in dataset_id_to_column.keys():  # CHANGED: Only iterate over datasets in performance matrix
                    # Get the scaled metafeatures for this dataset
                    try:
                        mf_values = scaled_metafeatures_df.loc[dataset_id].values
                        
                        # Check for NaN or infinite values in metafeatures
                        if np.isnan(mf_values).any() or np.isinf(mf_values).any():
                            print(f"    ⚠️ WARNING: NaN/Inf values in metafeatures for dataset {dataset_id}")
                            mf_values = np.nan_to_num(mf_values, nan=0.0, posinf=1.0, neginf=-1.0)
                            
                        # Get embedding from model
                        metafeatures = torch.FloatTensor(mf_values)
                        embedding = self.model.get_embedding(metafeatures.unsqueeze(0)).squeeze().cpu().numpy()
                        
                        # Check for NaN values in the embedding
                        if np.isnan(embedding).any() or np.isinf(embedding).any():
                            print(f"    ⚠️ WARNING: NaN/Inf values in embedding for dataset {dataset_id}, replacing with zeros")
                            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # Store the fixed embedding
                        self.dataset_embeddings[dataset_id] = embedding
                        valid_embeddings_count += 1
                        
                        # Add D_X format mapping for numeric dataset IDs
                        if isinstance(dataset_id, (int, float)) or (isinstance(dataset_id, str) and dataset_id.isdigit()):
                            d_format_id = f"D_{dataset_id}"
                            self.d_format_to_dataset[d_format_id] = dataset_id
                            
                    except Exception as e:
                        print(f"    ❌ Error generating embedding for dataset {dataset_id}: {e}")
                
                print(f"    Successfully created {valid_embeddings_count} valid embeddings out of {len(dataset_id_to_column)} training datasets")
            
            print(f"    Created {len(self.d_format_to_dataset)} D_X format mappings for datasets")
            print(f"    ✅ PMM model ready with embeddings for {len(self.dataset_embeddings)} datasets (from performance matrix)")
            
            # Calculate influence scores for datasets if enabled
            if self.use_influence_weighting and self.influence_method != 'none':
                print(f"\n    🎯 Calculating influence scores using method: {self.influence_method}")
                self.dataset_influence_scores = self._calculate_dataset_influence_scores(
                    performance_matrix, metafeatures_df, dataset_id_to_column
                )
            else:
                # All datasets have equal weight
                self.dataset_influence_scores = {ds_id: 1.0 for ds_id in self.dataset_embeddings.keys()}
                print(f"    Using equal weighting for all datasets (influence disabled)")
            
            self.is_trained = True
            print("    PMM Siamese network training completed!")
            return True
            
        except Exception as e:
            print(f"    Error training PMM Siamese network: {e}")
            self.is_trained = False
            return False
    
    def _create_siamese_dataset(self, metafeatures_df, performance_matrix):
        """Create dataset for Siamese network training"""
        class SiameseDataset(torch.utils.data.Dataset):
            def __init__(self, metafeatures_df, performance_matrix, similarity_threshold):
                self.metafeatures = metafeatures_df.values
                self.dataset_names = metafeatures_df.index.tolist()
                self.n_datasets = len(self.dataset_names)  # Add this line to define n_datasets
                
                # Map dataset IDs to indices - handle both int and string indices
                self.dataset_idx_map = {}
                for idx, name in enumerate(self.dataset_names):
                    self.dataset_idx_map[name] = idx  # Original format
                    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
                        # Also add D_X format mapping for numeric IDs
                        formatted_name = f"D_{name}"
                        self.dataset_idx_map[formatted_name] = idx
                        
                print(f"    Dataset idx map sample: {list(self.dataset_idx_map.items())[:5]}")
                
                # Create a mapping between performance matrix column names and metafeatures dataset IDs
                self.column_to_dataset = {}
                self.dataset_to_column = {}
                
                print(f"    Performance matrix columns: {list(performance_matrix.columns)[:5]} (showing first 5)")
                print(f"    Metafeatures dataset IDs: {self.dataset_names[:5]} (showing first 5)")
                
                # Comprehensive mapping approach (silent to reduce verbosity)
                for col in performance_matrix.columns:
                    # Case 1: Direct match (column name is exactly a dataset ID)
                    if col in self.dataset_names:
                        self.column_to_dataset[col] = col
                        self.dataset_to_column[col] = col
                        
                    # Case 2: D_X format mapping to numeric ID
                    elif col.startswith('D_') and col[2:].isdigit():
                        # Try as integer ID
                        numeric_id = int(col[2:])
                        if numeric_id in self.dataset_names:
                            self.column_to_dataset[col] = numeric_id
                            self.dataset_to_column[numeric_id] = col
                            
                        # Try as string ID (for dataset IDs stored as strings)
                        str_id = col[2:]
                        if str_id in self.dataset_names:
                            self.column_to_dataset[col] = str_id
                            self.dataset_to_column[str_id] = col
                
                # Print mapping statistics
                print(f"    Found {len(self.column_to_dataset)} mappable datasets between metafeatures and performance matrix")
                if len(self.column_to_dataset) > 0:
                    print(f"    Sample mappings: {list(self.column_to_dataset.items())[:3]}")
                else:
                    print(f"    ⚠️ WARNING: No mappings found! This will prevent the model from learning properly.")
                    # Additional mapping attempt for datasets with different formats
                    for ds_id in self.dataset_names:
                        for col in performance_matrix.columns:
                            # Try mapping with D_ prefix for numeric IDs
                            if isinstance(ds_id, (int, float)) or (isinstance(ds_id, str) and ds_id.isdigit()):
                                formatted_id = f"D_{ds_id}"
                                if formatted_id == col:
                                    self.column_to_dataset[col] = ds_id
                                    self.dataset_to_column[ds_id] = col
                                    print(f"    Alternative mapping: {col} -> {ds_id}")
                            
                            # Try removing D_ prefix from column
                            if col.startswith('D_'):
                                col_id = col[2:]
                                try:
                                    # Try as integer
                                    col_num = int(col_id)
                                    if col_num == ds_id or str(col_num) == str(ds_id):
                                        self.column_to_dataset[col] = ds_id
                                        self.dataset_to_column[ds_id] = col
                                        print(f"    Numeric mapping: {col} -> {ds_id}")
                                except:
                                    pass
                
                # Initialize similarity matrix
                self.similarity_matrix = np.zeros((self.n_datasets, self.n_datasets))
                
                # Calculate pairwise correlations between datasets
                valid_dataset_pairs = 0
                for i in range(self.n_datasets):
                    for j in range(i+1, self.n_datasets):
                        ds_i = self.dataset_names[i]
                        ds_j = self.dataset_names[j]
                        
                        # Find corresponding columns in performance matrix
                        col_i = self.dataset_to_column.get(ds_i)
                        col_j = self.dataset_to_column.get(ds_j)
                        
                        # Skip if either dataset is not in performance matrix
                        if col_i is None or col_j is None:
                            continue
                            
                        # Get performance vectors
                        perf_i = performance_matrix[col_i].values
                        perf_j = performance_matrix[col_j].values
                        
                        # Use only pipelines where both datasets have values
                        valid_idx = ~np.isnan(perf_i) & ~np.isnan(perf_j)
                        valid_count = np.sum(valid_idx)
                        
                        if valid_count > 3:  # Need at least 3 common pipelines
                            try:
                                corr = np.corrcoef(perf_i[valid_idx], perf_j[valid_idx])[0, 1]
                                if not np.isnan(corr):
                                    self.similarity_matrix[i, j] = corr
                                    self.similarity_matrix[j, i] = corr
                                    valid_dataset_pairs += 1
                            except Exception as e:
                                print(f"    Error calculating correlation for datasets {ds_i} and {ds_j}: {e}")
                
                print(f"    Successfully calculated similarities for {valid_dataset_pairs} dataset pairs")
                
                # Create dataset pairs with labels
                self.pairs = []
                self.labels = []
                
                # Note: n_datasets is now defined at the beginning of __init__
                
                for i in range(self.n_datasets):
                    for j in range(i+1, self.n_datasets):
                        similarity = self.similarity_matrix[i, j]
                        if not np.isnan(similarity) and similarity != 0.0:
                            # Label 1 for similar, 0 for dissimilar
                            label = 1.0 if similarity >= similarity_threshold else 0.0
                            self.pairs.append((i, j))
                            self.labels.append(label)
                
                # Convert to numpy arrays
                self.pairs = np.array(self.pairs)
                self.labels = np.array(self.labels)
                
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                pair = self.pairs[idx]
                i, j = pair
                
                data1 = torch.FloatTensor(self.metafeatures[i])
                data2 = torch.FloatTensor(self.metafeatures[j])
                label = torch.FloatTensor([self.labels[idx]])
                
                return data1, data2, label
        
        return SiameseDataset(metafeatures_df, performance_matrix, self.similarity_threshold)
    
    def _create_siamese_model(self, input_dim):
        """Create Siamese network model"""
        class SiameseNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, embedding_dim):
                super(SiameseNetwork, self).__init__()
                
                # Feature extraction network
                self.feature_net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, embedding_dim)
                )
                
            def forward_one(self, x):
                """Process one input through the network"""
                return self.feature_net(x)
            
            def forward(self, x1, x2):
                """Process a pair of inputs"""
                output1 = self.forward_one(x1)
                output2 = self.forward_one(x2)
                return output1, output2
            
            def get_embedding(self, x):
                """Get embedding for a single input"""
                return self.forward_one(x)
        
        return SiameseNetwork(input_dim, self.hidden_dim, self.embedding_dim)
    
    def _contrastive_loss(self, output1, output2, label):
        """Contrastive loss for Siamese network"""
        margin = self.margin
        
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # If label=1 (similar), minimize distance
        # If label=0 (dissimilar), maximize distance up to margin
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss
    
    def recommend(self, new_dataset_metafeatures, performance_matrix=None, k=5):
        """
        Recommend pipelines for a new dataset using learned similarity
        
        Args:
            new_dataset_metafeatures: Meta-features for new dataset or dataset_id
            performance_matrix: DataFrame with pipeline performances (optional)
            k: Number of similar datasets to consider
            
        Returns:
            Dictionary with pipeline recommendations and similarity information
        """
        if not self.is_trained or self.model is None:
            print("    PMM model not trained. Returning None.")
            return None
            
        # If performance_matrix is not provided, try to use the one from training
        if performance_matrix is None:
            print("    Warning: No performance matrix provided, trying to use training data.")
            try:
                # Try to load the performance matrix from a file
                performance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocessed_performance.csv")
                if os.path.exists(performance_path):
                    performance_matrix = pd.read_csv(performance_path, index_col=0)
                    print(f"    Successfully loaded performance matrix from file with shape {performance_matrix.shape}.")
                    
                    # Examine the format of columns for debugging
                    column_formats = {}
                    for col in performance_matrix.columns[:10]:  # Look at first 10 columns
                        if col.startswith('D_'):
                            column_formats['D_X format'] = column_formats.get('D_X format', 0) + 1
                        elif col.isdigit():
                            column_formats['numeric'] = column_formats.get('numeric', 0) + 1
                        else:
                            column_formats['other'] = column_formats.get('other', 0) + 1
                    
                    print(f"    Performance matrix column formats: {column_formats}")
                    
                    # If we have D_X format columns, pre-create mappings for all our dataset IDs
                    if 'D_X format' in column_formats and column_formats['D_X format'] > 0:
                        print("    Creating mappings for D_X format columns...")
                        # For each dataset in our embeddings, create a mapping to the D_X format
                        for ds_id in self.dataset_embeddings.keys():
                            if isinstance(ds_id, (int, float)) or (isinstance(ds_id, str) and ds_id.isdigit()):
                                d_format = f"D_{ds_id}"
                                if d_format in performance_matrix.columns:
                                    self.d_format_to_dataset[d_format] = ds_id
                                    print(f"    Mapped {d_format} -> {ds_id}")
                        
                        print(f"    Created {len(self.d_format_to_dataset)} D_X format mappings")
                else:
                    print(f"    Performance matrix file not found at {performance_path}")
                    raise FileNotFoundError(f"Performance matrix not found at {performance_path}")
            except Exception as e:
                print(f"    Error loading performance matrix: {e}")
                # Create an empty performance matrix with the pipelines from training
                from recommender_trainer import pipeline_configs
                pipelines = [config['name'] for config in pipeline_configs]
                performance_matrix = pd.DataFrame(index=pipelines, columns=self.dataset_names)
        
        # Create dataset ID to column mapping
        dataset_to_column = {}
        
        print(f"    Performance matrix columns (first 5): {list(performance_matrix.columns)[:5]}")
        print(f"    Dataset embeddings keys (first 5): {list(self.dataset_embeddings.keys())[:5]}")
        
        # Comprehensive mapping between dataset IDs and performance matrix columns
        mapping_count = 0
        for ds_id in self.dataset_embeddings.keys():
            mapped = False
            
            # Case 1: Direct match - column name exactly matches dataset ID
            if ds_id in performance_matrix.columns:
                dataset_to_column[ds_id] = ds_id
                mapped = True
                mapping_count += 1
                # print(f"    Direct match: {ds_id}")
            
            # Case 2: D_X format - try both integer and string conversions
            if not mapped:
                # Format as D_X
                formatted_id = f"D_{ds_id}"
                if formatted_id in performance_matrix.columns:
                    dataset_to_column[ds_id] = formatted_id
                    mapped = True
                    mapping_count += 1
                    # print(f"    D_X format match: {ds_id} -> {formatted_id}")
            
            # Case 3: Try numeric conversion if dataset_id is a string
            if not mapped and isinstance(ds_id, str):
                try:
                    if ds_id.isdigit():
                        numeric_id = int(ds_id)
                        formatted_id = f"D_{numeric_id}"
                        if formatted_id in performance_matrix.columns:
                            dataset_to_column[ds_id] = formatted_id
                            mapped = True
                            mapping_count += 1
                            # print(f"    Numeric conversion match: {ds_id} -> {formatted_id}")
                except:
                    pass
            
            # Case 4: Reverse mapping - look for D_X in performance matrix that might match this dataset
            if not mapped:
                for col in performance_matrix.columns:
                    if col.startswith('D_') and col[2:].isdigit():
                        col_numeric = int(col[2:])
                        # Check if numeric parts match
                        try:
                            if isinstance(ds_id, int) and ds_id == col_numeric:
                                dataset_to_column[ds_id] = col
                                mapped = True
                                mapping_count += 1
                                # print(f"    Reverse match: {ds_id} -> {col}")
                            elif isinstance(ds_id, str) and ds_id.isdigit() and int(ds_id) == col_numeric:
                                dataset_to_column[ds_id] = col
                                mapped = True
                                mapping_count += 1
                                # print(f"    Reverse string match: {ds_id} -> {col}")
                        except:
                            pass
        
        # Print some diagnostics
        print(f"    ✅ Created mapping for {mapping_count}/{len(self.dataset_embeddings)} datasets to performance matrix columns")
        if len(dataset_to_column) == 0:
            print(f"    ⚠️ WARNING: No mappings found between dataset IDs and performance matrix columns!")
            print(f"    Dataset IDs (first 5): {list(self.dataset_embeddings.keys())[:5]}")
            print(f"    Performance matrix columns (first 5): {list(performance_matrix.columns)[:5]}")
                
        # If new_dataset_metafeatures is a dataset ID, get its metafeatures
        if isinstance(new_dataset_metafeatures, (int, str)):
            dataset_id = new_dataset_metafeatures
            original_dataset_id = dataset_id  # Keep the original for reporting
            found = False
            
            # Debug information
            print(f"    Processing dataset ID: {dataset_id}")
            print(f"    Available dataset IDs in metafeatures: {self.metafeatures_df.index.tolist()[:10]}... (showing first 10)")
            
            # Case 1: Direct lookup in metafeatures index
            if dataset_id in self.metafeatures_df.index:
                print(f"    Found dataset ID {dataset_id} in metafeatures (direct match).")
                new_dataset_metafeatures = self.metafeatures_df.loc[dataset_id]
                found = True
            
            # Case 2: Try removing D_ prefix if present
            if not found and isinstance(dataset_id, str) and dataset_id.startswith("D_"):
                numeric_id = dataset_id[2:]
                # Try as numeric ID
                try:
                    numeric_id = int(numeric_id)
                    if numeric_id in self.metafeatures_df.index:
                        print(f"    Found dataset ID {numeric_id} in metafeatures after removing D_ prefix.")
                        new_dataset_metafeatures = self.metafeatures_df.loc[numeric_id]
                        dataset_id = numeric_id
                        found = True
                except:
                    # Try as string ID
                    if numeric_id in self.metafeatures_df.index:
                        print(f"    Found dataset ID {numeric_id} in metafeatures after removing D_ prefix (as string).")
                        new_dataset_metafeatures = self.metafeatures_df.loc[numeric_id]
                        dataset_id = numeric_id
                        found = True
            
            # Case 3: Try adding D_ prefix for metafeatures lookup
            if not found:
                formatted_id = f"D_{dataset_id}"
                if formatted_id in self.metafeatures_df.index:
                    print(f"    Found dataset ID {formatted_id} in metafeatures by adding D_ prefix.")
                    new_dataset_metafeatures = self.metafeatures_df.loc[formatted_id]
                    dataset_id = formatted_id
                    found = True
            
            # Case 4: Try numeric conversion if string
            if not found and isinstance(dataset_id, str):
                try:
                    if dataset_id.isdigit():
                        numeric_id = int(dataset_id)
                        if numeric_id in self.metafeatures_df.index:
                            print(f"    Found dataset ID {numeric_id} in metafeatures after numeric conversion.")
                            new_dataset_metafeatures = self.metafeatures_df.loc[numeric_id]
                            dataset_id = numeric_id
                            found = True
                except:
                    pass
            
            # Case 5: Try all available formats in metafeatures
            if not found:
                print(f"    Trying all format conversions for dataset ID {dataset_id}...")
                for index_id in self.metafeatures_df.index:
                    # Compare as strings to catch numeric equivalence
                    if str(index_id) == str(dataset_id):
                        print(f"    Found dataset ID {index_id} in metafeatures by string comparison.")
                        new_dataset_metafeatures = self.metafeatures_df.loc[index_id]
                        dataset_id = index_id
                        found = True
                        break
                    # Try with D_ prefix/suffix variations
                    elif isinstance(index_id, str) and index_id.startswith("D_") and index_id[2:] == str(dataset_id):
                        print(f"    Found dataset ID {index_id} in metafeatures with D_ prefix.")
                        new_dataset_metafeatures = self.metafeatures_df.loc[index_id]
                        dataset_id = index_id
                        found = True
                        break
                    elif isinstance(dataset_id, str) and dataset_id.startswith("D_") and dataset_id[2:] == str(index_id):
                        print(f"    Found dataset ID {index_id} in metafeatures by removing D_ prefix from input.")
                        new_dataset_metafeatures = self.metafeatures_df.loc[index_id]
                        dataset_id = index_id
                        found = True
                        break
            
            # Fallback if dataset not found
            if not found:
                print(f"    ❌ Dataset ID {original_dataset_id} not found in metafeatures after all attempts.")
                # Default to first dataset as fallback
                if len(self.metafeatures_df) > 0:
                    fallback_id = self.metafeatures_df.index[0]
                    print(f"    Using fallback dataset: {fallback_id}")
                    new_dataset_metafeatures = self.metafeatures_df.loc[fallback_id]
                    dataset_id = fallback_id
                else:
                    return None
        
        try:
            # Convert meta-features to DataFrame format if needed
            if isinstance(new_dataset_metafeatures, dict):
                new_mf_df = pd.DataFrame([new_dataset_metafeatures])
            elif isinstance(new_dataset_metafeatures, pd.Series):
                new_mf_df = pd.DataFrame([new_dataset_metafeatures.values], 
                                        columns=new_dataset_metafeatures.index)
            
            # Align columns with training data
            new_mf_df = new_mf_df.reindex(columns=self.metafeatures_df.columns, fill_value=0)
            
            # Check for NaN values in the dataset metafeatures
            if new_mf_df.isna().any().any():
                print("    ⚠️ NaN values detected in new dataset metafeatures. Imputing with median values.")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                new_mf_values = imputer.fit_transform(new_mf_df.values)
                new_mf_df = pd.DataFrame(new_mf_values, columns=new_mf_df.columns)
                print("    ✅ Successfully imputed missing values.")
            
            # Scale using the same scaler
            new_mf_scaled = self.scaler.transform(new_mf_df.values)
            
            # Get embedding for new dataset
            self.model.eval()
            with torch.no_grad():
                new_embedding = self.model.get_embedding(
                    torch.FloatTensor(new_mf_scaled)
                ).squeeze().numpy()
                
                # Verify embedding quality
                if np.isnan(new_embedding).any():
                    print("    ⚠️ NaN values detected in embedding after model forward pass. Fixing...")
                    new_embedding = np.nan_to_num(new_embedding)
                    print(f"    Fixed embedding shape: {new_embedding.shape}")
                
                # Verify embedding has non-zero values (model is actually learning)
                if np.all(np.abs(new_embedding) < 1e-6):
                    print("    ⚠️ WARNING: Embedding has extremely small values. Model may not be learning properly.")
            
            # Calculate similarity to all training datasets
            similarities = {}
            
            # Print debug information about embeddings
            print(f"    Dataset embeddings available for {len(self.dataset_embeddings)} datasets")
            print(f"    New dataset embedding shape: {new_embedding.shape}")
            if len(self.dataset_embeddings) > 0:
                sample_key = list(self.dataset_embeddings.keys())[0]
                print(f"    Sample embedding shape: {self.dataset_embeddings[sample_key].shape}")
                print(f"    Sample embedding values: {self.dataset_embeddings[sample_key][:5]} (first 5)")
                print(f"    New embedding values: {new_embedding[:5]} (first 5)")
            
            # Check for NaN or Inf values in the new embedding
            if np.isnan(new_embedding).any() or np.isinf(new_embedding).any():
                print(f"    ⚠️ WARNING: NaN/Inf values detected in the new dataset embedding!")
                # Replace problematic values with zeros/bounds to allow calculations to continue
                new_embedding = np.nan_to_num(new_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
                print(f"    Fixed embedding values: {new_embedding[:5]} (first 5)")
                
            # Calculate cosine similarity with all stored dataset embeddings
            similarity_count = 0
            invalid_count = 0
            
            for dataset_id, embedding in self.dataset_embeddings.items():
                # Cosine similarity calculation with robust error handling
                try:
                    # Clean the stored embedding if needed
                    if np.isnan(embedding).any() or np.isinf(embedding).any():
                        print(f"    ⚠️ WARNING: NaN/Inf values in embedding for dataset {dataset_id}")
                        embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Calculate vector norms
                    norm_new = np.linalg.norm(new_embedding)
                    norm_emb = np.linalg.norm(embedding)
                    
                    # Check for zero/near-zero norms to avoid division issues
                    if norm_new < 1e-8 or norm_emb < 1e-8:
                        print(f"    Warning: Near-zero norm detected for dataset {dataset_id}")
                        similarities[dataset_id] = 0.0
                        invalid_count += 1
                    else:
                        # Calculate cosine similarity
                        dot_product = np.dot(new_embedding, embedding)
                        similarity = dot_product / (norm_new * norm_emb)
                        
                        # Validate similarity value and bound it to [-1, 1] range
                        if np.isnan(similarity) or np.isinf(similarity):
                            print(f"    ⚠️ Invalid similarity value for dataset {dataset_id}, using 0.0 instead")
                            similarities[dataset_id] = 0.0
                            invalid_count += 1
                        else:
                            # Bound to valid cosine similarity range
                            similarity = max(-1.0, min(1.0, similarity))
                            similarities[dataset_id] = similarity
                            similarity_count += 1
                            
                except Exception as e:
                    print(f"    Error calculating similarity for dataset {dataset_id}: {e}")
                    similarities[dataset_id] = 0.0
                    invalid_count += 1
            
            print(f"    Calculated {similarity_count} valid similarities and encountered {invalid_count} calculation issues")
            
            # Print debug information
            print(f"    Calculated similarities for {len(similarities)} datasets")
            if similarities:
                print(f"    Similarity range: {min(similarities.values()):.4f} to {max(similarities.values()):.4f}")
                # Print top 5 similarities
                top_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"    Top similarities:")
                for ds_id, sim in top_similarities:
                    column_name = dataset_to_column.get(ds_id, "Not in performance matrix")
                    print(f"      Dataset {ds_id}: {sim:.4f} (Column: {column_name})")
            
            # Get k most similar datasets
            most_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
            similar_datasets = [d[0] for d in most_similar]
            
            print(f"    Most similar datasets: {similar_datasets}")
            
            # Check if we have performance data for these datasets
            found_in_performance = []
            not_found_in_performance = []
            
            # For each similar dataset, check if we can find it in the performance matrix
            for ds in similar_datasets:
                # Try direct match first
                if ds in performance_matrix.columns:
                    found_in_performance.append(ds)
                # Try D_X format
                elif f"D_{ds}" in performance_matrix.columns:
                    found_in_performance.append(ds)
                else:
                    not_found_in_performance.append(ds)
            
            if not_found_in_performance:
                print(f"    ⚠️ WARNING: {len(not_found_in_performance)} similar datasets not found in performance matrix")
                print(f"    Missing datasets: {not_found_in_performance}")
            
            if found_in_performance:
                print(f"    ✅ Found {len(found_in_performance)} similar datasets in performance matrix")
                print(f"    Datasets with performance data: {found_in_performance}")
            
            # Weight pipelines by similarity and performance
            weighted_performances = {}
            
            # Create a robust dataset_to_column mapping for similar datasets using our mapping function
            complete_dataset_to_column = {}
            
            # Copy existing mappings
            for ds_id, col in dataset_to_column.items():
                complete_dataset_to_column[ds_id] = col
            
            # For each similar dataset that doesn't have a mapping yet
            for dataset in [d for d in most_similar if d[0] not in complete_dataset_to_column]:
                ds_id = dataset[0]
                
                # Use our dedicated mapping function
                column = self._map_dataset_to_column(ds_id, performance_matrix)
                if column:
                    complete_dataset_to_column[ds_id] = column
                    print(f"    Found column match for dataset {ds_id} -> {column}")
                
                # If not found with the function, try a more exhaustive search
                elif isinstance(ds_id, int) or (isinstance(ds_id, str) and ds_id.isdigit()):
                    # Try D_X format with variations
                    try:
                        numeric_val = int(ds_id) if isinstance(ds_id, str) else ds_id
                        for col in performance_matrix.columns:
                            if col.startswith('D_') and col[2:].isdigit():
                                col_val = int(col[2:])
                                if col_val == numeric_val:
                                    complete_dataset_to_column[ds_id] = col
                                    print(f"    Found column through exhaustive search: {ds_id} -> {col}")
                                    break
                    except:
                        pass
            
            # Print mapping information for the most similar datasets
            print(f"    Mappings for the most similar datasets:")
            for ds_id in [d[0] for d in most_similar]:
                if ds_id in complete_dataset_to_column:
                    print(f"      Dataset {ds_id} maps to column {complete_dataset_to_column[ds_id]}")
                else:
                    print(f"      ❌ Dataset {ds_id} has NO mapping to any performance matrix column")
            
            # Now use the complete mapping for weighted performance calculation
            for pipeline in performance_matrix.index:
                weighted_sum = 0
                weight_sum = 0
                used_datasets = []
                
                for dataset, similarity in most_similar:
                    col_to_use = None
                    
                    # Check multiple ways to find the right column
                    # 1. Check if we have a mapping for this dataset
                    if dataset in complete_dataset_to_column:
                        col_to_use = complete_dataset_to_column[dataset]
                    # 2. Try direct D_X format
                    elif isinstance(dataset, (int, float)) or (isinstance(dataset, str) and dataset.isdigit()):
                        d_format = f"D_{dataset}"
                        if d_format in performance_matrix.columns:
                            col_to_use = d_format
                    # 3. Check our d_format_to_dataset mapping (reverse lookup)
                    else:
                        for d_format, ds_id in self.d_format_to_dataset.items():
                            if ds_id == dataset and d_format in performance_matrix.columns:
                                col_to_use = d_format
                                break
                    
                    if col_to_use:
                        try:
                            perf = performance_matrix.loc[pipeline, col_to_use]
                            if not pd.isna(perf):
                                # Combine similarity score with influence score
                                influence_score = self.dataset_influence_scores.get(dataset, 1.0)
                                # Weight = similarity × influence score
                                # This gives more weight to both similar AND influential datasets
                                weight = max(0.01, similarity * influence_score)
                                
                                weighted_sum += perf * weight
                                weight_sum += weight
                                used_datasets.append((dataset, influence_score))
                                
                                # Print for first pipeline to show influence weighting in action
                                if pipeline == performance_matrix.index[0] and self.use_influence_weighting:
                                    print(f"        Dataset {dataset}: sim={similarity:.4f}, influence={influence_score:.3f}, weight={weight:.4f}")
                        except Exception as e:
                            print(f"    Error accessing performance for {dataset}/{col_to_use}: {e}")
                    else:
                        # Only print once per dataset
                        if pipeline == performance_matrix.index[0]:
                            print(f"    No column mapping found for dataset {dataset}")
                
                if weight_sum > 0:
                    weighted_performances[pipeline] = weighted_sum / weight_sum
                    # Only print for first few pipelines to reduce verbosity
                    # print(f"    Pipeline {pipeline} weighted by datasets: {used_datasets}, score={weighted_performances[pipeline]:.4f}")
                else:
                    # If no similar datasets found in performance matrix, use average performance
                    valid_perfs = performance_matrix.loc[pipeline].dropna()
                    if not valid_perfs.empty:
                        avg_perf = valid_perfs.mean()
                        weighted_performances[pipeline] = avg_perf
                        print(f"    No similar datasets with performance data found for pipeline {pipeline}. Using average: {avg_perf:.4f}")
                    else:
                        weighted_performances[pipeline] = 0
                        print(f"    No performance data available for pipeline {pipeline}")
            
            # Rank pipelines by weighted performance
            ranked_pipelines = sorted(
                weighted_performances.items(), key=lambda x: x[1], reverse=True
            )
            
            # Show top pipeline performances
            print("    Top recommended pipelines:")
            for p, score in ranked_pipelines[:3]:
                print(f"    - {p}: {score:.4f}")
            
            result = {
                'pipeline': ranked_pipelines[0][0] if ranked_pipelines else None,
                'pipeline_ranking': [p[0] for p in ranked_pipelines],
                'performance_scores': {p[0]: p[1] for p in ranked_pipelines},
                'similar_datasets': similar_datasets,
                'similarity_scores': {d: s for d, s in most_similar},
                'influence_scores': {d: self.dataset_influence_scores.get(d, 1.0) for d in similar_datasets},
                'dataset_used': dataset_id if isinstance(new_dataset_metafeatures, (int, str)) else None,
                'influence_weighted': self.use_influence_weighting
            }
            
            return result
            
        except Exception as e:
            import traceback
            print(f"    Error in PMM recommendation: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            return None


class BalancedPmmRecommender(PmmRecommender):
    """
    Balanced Pipeline Meta-Model (PMM) Recommender using Siamese Network for dataset similarity.
    This variant ensures balanced training of similar and dissimilar dataset pairs.
    """
    
    def _create_siamese_dataset(self, metafeatures_df, performance_matrix):
        """Create balanced dataset for Siamese network training"""
        class BalancedSiameseDataset(torch.utils.data.Dataset):
            def __init__(self, metafeatures_df, performance_matrix, similarity_threshold):
                self.metafeatures = metafeatures_df.values
                self.dataset_names = metafeatures_df.index.tolist()
                
                # Map dataset IDs to indices - handle both int and string indices
                self.dataset_idx_map = {}
                for idx, name in enumerate(self.dataset_names):
                    self.dataset_idx_map[name] = idx  # Original format
                    
                    # Also add string version for easier lookups
                    if not isinstance(name, str):
                        self.dataset_idx_map[str(name)] = idx
                    
                    # Add D_X format mapping for numeric IDs
                    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
                        formatted_name = f"D_{name}"
                        self.dataset_idx_map[formatted_name] = idx
                        
                print(f"    Dataset idx map sample: {list(self.dataset_idx_map.items())[:5]}")
                
                # Create a mapping between performance matrix column names and metafeatures dataset IDs
                self.column_to_dataset = {}
                self.dataset_to_column = {}
                
                print(f"    Performance matrix columns: {list(performance_matrix.columns)[:5]} (showing first 5)")
                print(f"    Metafeatures dataset IDs: {self.dataset_names[:5]} (showing first 5)")
                
                # Comprehensive mapping approach
                for col in performance_matrix.columns:
                    # Case 1: Direct match (column name is exactly a dataset ID)
                    if col in self.dataset_names:
                        self.column_to_dataset[col] = col
                        self.dataset_to_column[col] = col
                        print(f"    Direct match: {col}")
                        
                    # Case 2: D_X format mapping to numeric ID
                    elif col.startswith('D_') and col[2:].isdigit():
                        # Try as integer ID
                        numeric_id = int(col[2:])
                        if numeric_id in self.dataset_names:
                            self.column_to_dataset[col] = numeric_id
                            self.dataset_to_column[numeric_id] = col
                            # print(f"    D_X format match: {col} -> {numeric_id}")
                            
                        # Try as string ID (for dataset IDs stored as strings)
                        str_id = col[2:]
                        if str_id in self.dataset_names:
                            self.column_to_dataset[col] = str_id
                            self.dataset_to_column[str_id] = col
                            # print(f"    D_X format match (string): {col} -> {str_id}")
                
                # If no mappings found, try comprehensive mapping
                if len(self.column_to_dataset) == 0:
                    print(f"    ⚠️ WARNING: No mappings found! Trying comprehensive mapping approach.")
                    for ds_id in self.dataset_names:
                        for col in performance_matrix.columns:
                            # Try mapping with D_ prefix for numeric IDs
                            if isinstance(ds_id, (int, float)) or (isinstance(ds_id, str) and ds_id.isdigit()):
                                formatted_id = f"D_{ds_id}"
                                if formatted_id == col:
                                    self.column_to_dataset[col] = ds_id
                                    self.dataset_to_column[ds_id] = col
                                    print(f"    Alternative mapping: {col} -> {ds_id}")
                            
                            # Try removing D_ prefix from column
                            if col.startswith('D_'):
                                col_id = col[2:]
                                try:
                                    # Try as integer
                                    col_num = int(col_id)
                                    if col_num == ds_id or str(col_num) == str(ds_id):
                                        self.column_to_dataset[col] = ds_id
                                        self.dataset_to_column[ds_id] = col
                                        print(f"    Numeric mapping: {col} -> {ds_id}")
                                except:
                                    pass
                
                print(f"    Found {len(self.column_to_dataset)} mappable datasets between metafeatures and performance matrix")
                if len(self.column_to_dataset) > 0:
                    print(f"    Sample mappings: {list(self.column_to_dataset.items())[:3]}")
                
                # Store number of datasets for later use
                self.n_datasets = len(self.dataset_names)
                
                # Initialize similarity matrix
                self.similarity_matrix = np.zeros((self.n_datasets, self.n_datasets))
                
                # Calculate pairwise correlations between datasets
                for i in range(self.n_datasets):
                    for j in range(i+1, self.n_datasets):
                        ds_i = self.dataset_names[i]
                        ds_j = self.dataset_names[j]
                        
                        # Find corresponding columns in performance matrix
                        col_i = self.dataset_to_column.get(ds_i)
                        col_j = self.dataset_to_column.get(ds_j)
                        
                        # Skip if either dataset is not in performance matrix
                        if col_i is None or col_j is None:
                            continue
                            
                        # Get performance vectors
                        perf_i = performance_matrix[col_i].values
                        perf_j = performance_matrix[col_j].values
                        
                        # Use only pipelines where both datasets have valid values
                        valid_idx = ~np.isnan(perf_i) & ~np.isnan(perf_j) & ~np.isinf(perf_i) & ~np.isinf(perf_j)
                        valid_count = np.sum(valid_idx)
                        
                        if valid_count > 3:  # Need at least 3 common pipelines
                            try:
                                # Calculate correlation coefficients
                                valid_perf_i = perf_i[valid_idx]
                                valid_perf_j = perf_j[valid_idx]
                                
                                # Double check that we don't have constant values (would cause NaN correlation)
                                if np.std(valid_perf_i) > 1e-8 and np.std(valid_perf_j) > 1e-8:
                                    corr = np.corrcoef(valid_perf_i, valid_perf_j)[0, 1]
                                    
                                    # Ensure correlation is valid
                                    if not np.isnan(corr) and not np.isinf(corr):
                                        # Bound to valid correlation range
                                        corr = max(-1.0, min(1.0, corr))
                                        self.similarity_matrix[i, j] = corr
                                        self.similarity_matrix[j, i] = corr
                                    else:
                                        print(f"    ⚠️ Invalid correlation between datasets {ds_i} and {ds_j}")
                                else:
                                    print(f"    ⚠️ Constant performance values for datasets {ds_i} and {ds_j}")
                            except Exception as e:
                                print(f"    Error calculating correlation between {ds_i} and {ds_j}: {e}")
                
                # Create dataset pairs with labels
                similar_pairs = []
                dissimilar_pairs = []
                
                for i in range(self.n_datasets):
                    for j in range(i+1, self.n_datasets):
                        similarity = self.similarity_matrix[i, j]
                        if not np.isnan(similarity):
                            if similarity >= similarity_threshold:
                                similar_pairs.append((i, j, 1.0))
                            else:
                                dissimilar_pairs.append((i, j, 0.0))
                
                # Balance similar and dissimilar pairs
                min_samples = min(len(similar_pairs), len(dissimilar_pairs))
                if min_samples > 0:
                    # Randomly sample to balance classes
                    if len(similar_pairs) > min_samples:
                        similar_pairs = np.random.choice(similar_pairs, min_samples, replace=False)
                    if len(dissimilar_pairs) > min_samples:
                        dissimilar_pairs = np.random.choice(dissimilar_pairs, min_samples, replace=False)
                
                # Combine and shuffle
                all_pairs = similar_pairs + dissimilar_pairs
                np.random.shuffle(all_pairs)
                
                # Separate into pairs and labels
                self.pairs = [(p[0], p[1]) for p in all_pairs]
                self.labels = [p[2] for p in all_pairs]
                
                # Convert to numpy arrays
                self.pairs = np.array(self.pairs)
                self.labels = np.array(self.labels)
                
            def __len__(self):
                return len(self.pairs)
            
            def __getitem__(self, idx):
                pair = self.pairs[idx]
                i, j = pair
                
                data1 = torch.FloatTensor(self.metafeatures[i])
                data2 = torch.FloatTensor(self.metafeatures[j])
                label = torch.FloatTensor([self.labels[idx]])
                
                return data1, data2, label
        
        return BalancedSiameseDataset(metafeatures_df, performance_matrix, self.similarity_threshold)


class HybridMetaRecommender:
    """
    A hybrid recommender that combines dataset similarity (KNN) with 
    gradient boosting for performance prediction.
    """
    
    def __init__(self, performance_matrix, metafeatures_df, use_influence_weighting=False, influence_method='performance_variance'):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        
        self.knn_model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')  
        
        # For performance prediction
        self.xgb_model = None
        self.pipeline_names = None
        self.trained = False
        
        # Influence weighting parameters
        self.use_influence_weighting = use_influence_weighting
        self.influence_method = influence_method
        self.dataset_influence_scores = {}  # Store influence scores for each dataset
        
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
            
            # Calculate influence scores if enabled
            if self.use_influence_weighting:
                print(f"    Calculating influence scores using method: {self.influence_method}")
                self._calculate_dataset_influence_scores()
                print(f"    ✅ Calculated influence scores for {len(self.dataset_influence_scores)} datasets")
            
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
    
    def _calculate_dataset_influence_scores(self):
        """
        Calculate DPO-inspired influence scores for all training datasets.
        
        Key Principles (from DPO - Direct Preference Optimization):
        1. Discriminative Power: Datasets that clearly separate good/bad pipelines
        2. Information Gain: Datasets providing unique information
        3. Reliability: Consistent, non-noisy performance patterns
        
        This makes the recommender learn more from "informative" datasets.
        """
        if not self.dataset_mapping:
            return
        
        # Extract dataset IDs from mapping
        dataset_ids = [num_id for _, num_id in self.dataset_mapping.values()]
        
        # Calculate all metafeatures matrix for diversity calculation
        all_metafeatures = []
        valid_dataset_ids = []
        for dataset_id in dataset_ids:
            if dataset_id in self.metafeatures_df.index:
                mf = self.metafeatures_df.loc[dataset_id].values
                if not np.isnan(mf).all():
                    all_metafeatures.append(mf)
                    valid_dataset_ids.append(dataset_id)
        
        if len(all_metafeatures) > 0:
            all_metafeatures = np.array(all_metafeatures)
            # Impute NaNs for diversity calculation
            if np.isnan(all_metafeatures).any():
                temp_imputer = SimpleImputer(strategy='median')
                all_metafeatures = temp_imputer.fit_transform(all_metafeatures)
        
        for dataset_id in dataset_ids:
            # Find the column name for this dataset
            col = None
            for c, num_id in self.dataset_mapping.values():
                if num_id == dataset_id:
                    col = c
                    break
            
            if col is None or col not in self.performance_matrix.columns:
                self.dataset_influence_scores[dataset_id] = 0.5  # Neutral score
                continue
            
            # Get performance values for this dataset
            performances = self.performance_matrix[col].dropna()
            
            if len(performances) < 3:  # Need at least 3 pipelines
                self.dataset_influence_scores[dataset_id] = 0.5
                continue
            
            perf_values = performances.values
            
            # ==========================================
            # COMPONENT 1: Discriminative Power (DPO Core)
            # ==========================================
            # Measures how well this dataset separates good from bad pipelines
            # High discriminative power = clear winner/loser pipelines
            
            sorted_perfs = np.sort(perf_values)
            
            # Gap between top-3 and bottom-3
            if len(sorted_perfs) >= 6:
                top_3_mean = sorted_perfs[-3:].mean()
                bottom_3_mean = sorted_perfs[:3].mean()
                gap = top_3_mean - bottom_3_mean
                
                # Also consider the spread within top-3 (low spread = clear winner)
                top_3_std = sorted_perfs[-3:].std()
                bottom_3_std = sorted_perfs[:3].std()
                
                # Discriminative power = large gap + low variance in top/bottom groups
                discriminative_power = gap / (1.0 + top_3_std + bottom_3_std)
            else:
                # For fewer pipelines, use range / std
                discriminative_power = (sorted_perfs.max() - sorted_perfs.min()) / (perf_values.std() + 1e-6)
            
            # ==========================================
            # COMPONENT 2: Information Gain
            # ==========================================
            # Measures how unique/diverse this dataset is
            # High information gain = learns something new
            
            if dataset_id in valid_dataset_ids and len(all_metafeatures) > 1:
                dataset_mf = self.metafeatures_df.loc[dataset_id].values
                if not np.isnan(dataset_mf).all():
                    # Impute NaNs in this dataset's features
                    if np.isnan(dataset_mf).any():
                        dataset_mf = self.imputer.transform([dataset_mf])[0]
                    
                    # Calculate average distance to all other datasets
                    distances = []
                    for other_mf in all_metafeatures:
                        # Use imputed features
                        dist = np.linalg.norm(dataset_mf - other_mf)
                        distances.append(dist)
                    
                    # Information gain = how far from others (more unique = more informative)
                    # Use median distance to be robust to outliers
                    information_gain = np.median(distances) if len(distances) > 0 else 1.0
                else:
                    information_gain = 1.0
            else:
                information_gain = 1.0
            
            # ==========================================
            # COMPONENT 3: Reliability Score
            # ==========================================
            # Measures consistency of performance patterns
            # High reliability = trustworthy signal (not too noisy)
            
            # Variance-to-range ratio (low = consistent patterns)
            perf_range = perf_values.max() - perf_values.min()
            perf_var = perf_values.var()
            
            if perf_range > 0:
                # Low variance relative to range = reliable
                reliability = 1.0 - min(1.0, perf_var / (perf_range + 1e-6))
            else:
                reliability = 0.5  # Neutral if no variance
            
            # Also check coefficient of variation (CV)
            cv = perf_values.std() / (perf_values.mean() + 1e-6)
            reliability = (reliability + (1.0 / (1.0 + cv))) / 2.0
            
            # ==========================================
            # AGGREGATE INFLUENCE SCORE (DPO-Style)
            # ==========================================
            
            if self.influence_method == 'performance_variance':
                # Original: Just variance (simple but effective)
                influence = perf_var
                
            elif self.influence_method == 'discriminative_power':
                # DPO-inspired: Focus on datasets with clear winners/losers
                influence = discriminative_power
                
            elif self.influence_method == 'data_diversity':
                # Focus on unique datasets
                influence = information_gain
                
            elif self.influence_method == 'combined':
                # **DPO FULL VERSION**: Combine all three components
                # Normalize each component to [0, 1] range
                norm_discrim = discriminative_power
                norm_info = information_gain
                norm_reliability = reliability
                
                # Weighted combination (DPO emphasizes discriminative power)
                influence = (
                    0.5 * norm_discrim +      # 50%: Can we learn clear preferences?
                    0.3 * norm_info +          # 30%: Is this dataset unique?
                    0.2 * norm_reliability     # 20%: Is the signal reliable?
                )
            else:
                influence = 1.0
            
            self.dataset_influence_scores[dataset_id] = max(0.0, influence)  # Ensure non-negative
        
        # ==========================================
        # NORMALIZE WITH DPO-STYLE SCALING
        # ==========================================
        if self.dataset_influence_scores:
            scores = np.array(list(self.dataset_influence_scores.values()))
            dataset_ids_list = list(self.dataset_influence_scores.keys())
            
            if scores.std() > 1e-8:
                # Step 1: Normalize to [0, 1]
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                
                # Step 2: Apply power transformation to emphasize differences
                # DPO uses exponential weighting: exp(β * score) where β controls sharpness
                beta = 2.0  # Higher beta = more aggressive weighting
                scores = np.exp(beta * scores)
                
                # Step 3: Re-normalize to [0, 1]
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                
                # Step 4: Scale to [0.1, 3.0] for 30x range (DPO uses high dynamic range)
                scores = 0.1 + (scores * 2.9)
            else:
                scores = np.ones_like(scores)
                
            for i, dataset_id in enumerate(dataset_ids_list):
                self.dataset_influence_scores[dataset_id] = scores[i]
                
            print(f"    ✅ DPO-style influence scores calculated for {len(self.dataset_influence_scores)} datasets")
            print(f"       Range: [{scores.min():.3f}, {scores.max():.3f}] (ratio: {scores.max()/scores.min():.1f}x)")
            print(f"       Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
            
            # Show top-5 most influential datasets
            top_5_influential = sorted(self.dataset_influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"       Top-5 most influential datasets:")
            for ds_id, score in top_5_influential:
                print(f"         Dataset {ds_id}: influence={score:.3f}")
        
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
            
            # Apply influence weighting if enabled
            if self.use_influence_weighting and self.dataset_influence_scores:
                print(f"    🎯 Applying influence weighting (method: {self.influence_method})")
                influence_weighted_weights = []
                for i, idx in enumerate(indices.flatten()):
                    _, num_id = self.dataset_mapping[idx]
                    influence_score = self.dataset_influence_scores.get(num_id, 1.0)
                    influence_weighted_weights.append(weights[i] * influence_score)
                
                # Renormalize after influence weighting
                influence_weighted_weights = np.array(influence_weighted_weights)
                if influence_weighted_weights.sum() > 0:
                    weights = influence_weighted_weights / influence_weighted_weights.sum()
                    print(f"    Influence-weighted similar datasets:")
                    for i, idx in enumerate(indices.flatten()):
                        _, num_id = self.dataset_mapping[idx]
                        inf_score = self.dataset_influence_scores.get(num_id, 1.0)
                        print(f"      Dataset {num_id}: distance={distances.flatten()[i]:.4f}, weight={weights[i]:.4f}, influence={inf_score:.3f}")
            
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
            
            return best_pipeline, final_predictions
            
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            return None, {}


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
            return None, {}
            
        try:
            # Get metafeatures for the target dataset
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
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
            return best_pipeline, predictions
            
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            return None, {}


# =============================================================================
# NEW CLASS: AutoGluon-based Recommender
# =============================================================================

class AutoGluonPipelineRecommender:
    """
    A recommender that treats pipeline selection as a classification problem using AutoGluon.
    It concatenates metafeatures with one-hot encoded pipeline characteristics and trains 
    an AutoGluon model to predict the performance score.
    """
    
    def __init__(self, performance_matrix, metafeatures_df):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.predictor = None
        self.pipeline_names = None
        self.imputer = SimpleImputer(strategy='median')
        self.trained = False
        self.feature_names = None
        self.temp_dir = None
        
    def _prepare_training_data(self):
        """Prepare training data for AutoGluon."""
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
        
        if len(common_datasets) < 5:  # Need minimum datasets
            print("    Warning: Not enough common datasets for training")
            return None, None, []
        
        print(f"    Found {len(common_datasets)} common datasets for training")
        
        X_train_rows = []
        y_train = []
        self.pipeline_names = self.performance_matrix.index.tolist()
        
        # Generate feature names for the metafeatures
        metafeature_columns = self.metafeatures_df.columns.tolist()
        
        # Define pipeline one-hot features
        pipeline_feature_names = []
        for config in pipeline_configs:
            pipeline_feature_names.append(f"pipeline_{config['name']}")
        
        self.feature_names = metafeature_columns + pipeline_feature_names
        
        # Collect training data
        for col, num_id in common_datasets:
            metafeatures = self.metafeatures_df.loc[num_id].values
            
            # Handle NaN values
            if np.isnan(metafeatures).any():
                metafeatures = self.imputer.fit_transform([metafeatures])[0]
            
            # Get performances for all pipelines on this dataset
            for pipeline in self.pipeline_names:
                score = self.performance_matrix.loc[pipeline, col]
                if not np.isnan(score):
                    # Create one-hot encoding for pipeline
                    pipeline_onehot = np.zeros(len(pipeline_configs))
                    for i, config in enumerate(pipeline_configs):
                        if config['name'] == pipeline:
                            pipeline_onehot[i] = 1
                            break
                    
                    # Create feature row: metafeatures + pipeline one-hot
                    feature_row = np.concatenate([metafeatures, pipeline_onehot])
                    feature_dict = {name: value for name, value in zip(self.feature_names, feature_row)}
                    X_train_rows.append(feature_dict)
                    y_train.append(score)
        
        if not X_train_rows:
            return None, None, []
            
        X_train_df = pd.DataFrame(X_train_rows)
        return X_train_df, y_train, common_datasets
        
    def fit(self):
        """Train the AutoGluon model on available performance data."""
        try:
            X_train_df, y_train, common_datasets = self._prepare_training_data()
            
            if X_train_df is None or len(X_train_df) < 20:
                print(f"    Warning: Not enough training samples")
                return False
                
            # Create temporary directory for AutoGluon model
            self.temp_dir = os.path.join(tempfile.gettempdir(), f"ag_recommender_{np.random.randint(10000)}")
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Prepare training data for AutoGluon
            train_data = X_train_df.copy()
            train_data['target'] = y_train
            
            print(f"    Training AutoGluon model on {len(train_data)} examples...")
            self.predictor = TabularPredictor(
                label='target',
                path=self.temp_dir,
                problem_type='regression',
                eval_metric='root_mean_squared_error',
                verbosity=2
            )
            
            self.predictor.fit(
                train_data,
                time_limit=3000,  # 8 hours
                presets='best_quality',
                included_model_types=STABLE_MODELS,

                
                hyperparameter_tune_kwargs=None,
                feature_generator=None,
                ag_args_fit=AG_ARGS_FIT
            )
            
            self.trained = True
            print(f"    ✅ AutoGluon recommender trained on {len(train_data)} examples")
            
            # Save model info
            self.common_datasets = common_datasets
            
            # Save model summary to file for reference
            with open('autogluon_recommender_summary.txt', 'w') as f:
                f.write(f"AutoGluon Pipeline Recommender Summary\n")
                f.write(f"=====================================\n")
                f.write(f"Trained on {len(train_data)} examples from {len(common_datasets)} datasets\n")
                f.write(f"Feature names: {self.feature_names}\n")
                f.write(f"Leaderboard:\n")
                f.write(str(self.predictor.leaderboard()))
            
            return True
            
        except Exception as e:
            print(f"    Error training AutoGluon recommender: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def recommend(self, dataset_id):
        """Recommend the best pipeline for a dataset based on its meta-features."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            metafeatures = self.metafeatures_df.loc[dataset_id].values
            
            # Handle NaN values
            if np.isnan(metafeatures).any():
                metafeatures = self.imputer.transform([metafeatures])[0]
            
            # Predict performance for each pipeline
            predictions = {}
            test_rows = []
            
            for pipeline in self.pipeline_names:
                # Create one-hot encoding for pipeline
                pipeline_onehot = np.zeros(len(pipeline_configs))
                for i, config in enumerate(pipeline_configs):
                    if config['name'] == pipeline:
                        pipeline_onehot[i] = 1
                        break
                
                # Create feature row: metafeatures + pipeline one-hot
                feature_row = np.concatenate([metafeatures, pipeline_onehot])
                feature_dict = {name: value for name, value in zip(self.feature_names, feature_row)}
                test_rows.append(feature_dict)
            
            # Create test dataframe
            test_df = pd.DataFrame(test_rows)
            
            # Ensure all required features are present in test data
            # AutoGluon requires exact same columns as were present in training data
            trained_features = self.predictor._learner.feature_generator.features_in
            for feature in trained_features:
                if feature not in test_df.columns:
                    # Add missing columns with default value 0
                    test_df[feature] = 0
            
            # Predict scores for all pipelines at once
            preds = self.predictor.predict(test_df)
            
            # Create dictionary of predictions
            for i, pipeline in enumerate(self.pipeline_names):
                predictions[pipeline] = preds.iloc[i]
            
            # Find the best pipeline
            best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
            print(f"    🔮 AutoGluon recommender suggests: {best_pipeline} (predicted score: {predictions[best_pipeline]:.4f})")
            
            # Return top-3 recommendations for reference
            top_pipelines = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"    Top-3 pipelines: {[p[0] for p in top_pipelines]}")
            
            return best_pipeline, predictions
            
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    def __del__(self):
        """Clean up temporary directory when done."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except:
                pass


# =============================================================================
# NEW RECOMMENDER CLASSES - Simple recommenders
# =============================================================================

class RandomRecommender:
    """Recommends pipelines randomly."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
    
    def fit(self):
        """Train the model on available performance data."""
        try:
            self.pipeline_names = self.performance_matrix.index.tolist()
            self.model = np.arange(len(self.pipeline_names))
            np.random.shuffle(self.model)
            self.trained = True
            print(f"    ✅ Random recommender ready")
            return True
        except Exception as e:
            print(f"    Error initializing random recommender: {e}")
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipelines randomly."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            # Just randomly recommend any pipeline
            best_pipeline = self.pipeline_names[self.model[0]]
            
            # Generate random scores for all pipelines
            predictions = {}
            for pipeline in self.pipeline_names:
                predictions[pipeline] = np.random.random()
                
            print(f"    🎲 Random recommender suggests: {best_pipeline} (random score)")
            return best_pipeline, predictions
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            return None, {}


class AverageRankRecommender:
    """Recommends pipelines based on their average rank across all datasets."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
    
    def fit(self):
        """Compute average ranks for pipelines."""
        try:
            if self.performance_matrix is None:
                print("    Error: No performance matrix provided")
                return False
                
            self.pipeline_names = self.performance_matrix.index.tolist()
            
            # Calculate average performance for each pipeline
            avg_performance = self.performance_matrix.mean(axis=1, skipna=True)
            
            # Sort pipelines by average performance (descending)
            self.model = avg_performance.sort_values(ascending=False).index.tolist()
            self.trained = True
            print(f"    ✅ Average rank recommender trained")
            return True
        except Exception as e:
            print(f"    Error training average rank recommender: {e}")
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline based on average rank."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            # Recommend the pipeline with highest average performance
            best_pipeline = self.model[0]
            
            # Create score dictionary from the average performances
            predictions = {}
            for i, pipeline in enumerate(self.model):
                predictions[pipeline] = 1.0 - (i / len(self.model))  # Higher for better ranks
                
            print(f"    📊 Average rank recommender suggests: {best_pipeline}")
            return best_pipeline, predictions
        except Exception as e:
            print(f"    Error making recommendation: {e}")
            return None, {}


class L1Recommender:
    """Recommends pipelines based on L1 distance between datasets."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.Ytrain = None
        self.Ftrain = None
        self.trained = False
        self.pipeline_names = None
        self.name = 'L1'
    
    def fit(self):
        """Train the L1 recommender on available performance data."""
        try:
            if self.performance_matrix is None or self.metafeatures_df is None:
                print("    Error: Performance matrix and metafeatures are required")
                return False
            
            # Get dataset IDs from performance matrix columns
            dataset_ids = []
            features = []
            
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        ds_id = int(col.split('_')[1])
                        if ds_id in self.metafeatures_df.index:
                            dataset_ids.append(ds_id)
                            features.append(self.metafeatures_df.loc[ds_id].values)
                except:
                    pass
            
            if len(dataset_ids) < 5:
                print(f"    Warning: Not enough common datasets for training ({len(dataset_ids)})")
                return False
                
            # Convert to matrix format
            self.Ytrain = self.performance_matrix.values  # (n_pipelines, n_datasets)
            self.Ftrain = np.array(features)  # (n_datasets, n_features)
            
            # Handle NaN values in the feature matrix
            self.Ftrain = np.nan_to_num(self.Ftrain, nan=0.0)
            
            self.pipeline_names = self.performance_matrix.index.tolist()
            self.trained = True
            print(f"    ✅ L1 recommender trained on {len(dataset_ids)} datasets")
            return True
        except Exception as e:
            print(f"    Error training L1 recommender: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recommend(self, dataset_id, n_init=5):
        """Recommend pipeline based on L1 distance to similar datasets."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            # Get features for the target dataset
            ftest = self.metafeatures_df.loc[dataset_id].values
            
            # Handle NaN values
            ftest = np.nan_to_num(ftest, nan=0.0)
            
            # Calculate L1 distances to all training datasets
            distances = np.abs(self.Ftrain - ftest).sum(axis=1)
            
            # Find closest datasets
            ix_closest = np.argsort(distances)[:n_init]
            
            # Find non-NaN pipelines for closest datasets
            ix_nonnan_pipelines = np.where(np.invert(np.isnan(self.Ytrain[:, ix_closest].sum(axis=1))))[0]
            
            # Calculate ranks for these pipelines on closest datasets
            ranks = np.apply_along_axis(st.rankdata, 0, self.Ytrain[ix_nonnan_pipelines[:, None], ix_closest])
            
            # Get average pipeline ranks
            ave_pipeline_ranks = ranks.mean(axis=1)
            
            # Sort pipelines by rank (lowest average rank = best)
            ix_init = ix_nonnan_pipelines[np.argsort(ave_pipeline_ranks)]
            
            # Create predictions dictionary
            predictions = {}
            for i, pipe_idx in enumerate(ix_init):
                if i < len(self.pipeline_names):
                    pipeline = self.pipeline_names[pipe_idx]
                    predictions[pipeline] = 1.0 - (i / len(ix_init))  # Higher score for better rank
            
            best_pipeline = self.pipeline_names[ix_init[0]] if len(ix_init) > 0 else None
            
            if best_pipeline is None:
                print("    ⚠️ L1 recommender couldn't find a suitable pipeline")
                return None, {}
                
            print(f"    📏 L1 recommender suggests: {best_pipeline}")
            return best_pipeline, predictions
        except Exception as e:
            print(f"    Error making L1 recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


class BasicRecommender:
    """Simple recommender that uses average pipeline performance."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
    
    def fit(self):
        """Train the basic recommender (just compute average performance)."""
        try:
            if self.performance_matrix is None:
                print("    Error: No performance matrix provided")
                return False
                
            # Calculate average performance for each pipeline across all datasets
            avg_performance = self.performance_matrix.mean(axis=1, skipna=True)
            
            # Sort pipelines by average performance (descending)
            self.model = avg_performance.sort_values(ascending=False).index.tolist()
            self.pipeline_names = self.performance_matrix.index.tolist()
            self.trained = True
            print(f"    ✅ Basic recommender trained")
            return True
        except Exception as e:
            print(f"    Error training basic recommender: {e}")
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline based on average performance."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            # Simply recommend the pipeline with highest average performance
            best_pipeline = self.model[0] if len(self.model) > 0 else None
            
            # Create predictions dictionary based on average performance rank
            predictions = {}
            for i, pipeline in enumerate(self.model):
                predictions[pipeline] = 1.0 - (i / len(self.model))
                
            print(f"    📈 Basic recommender suggests: {best_pipeline}")
            return best_pipeline, predictions
        except Exception as e:
            print(f"    Error making basic recommendation: {e}")
            return None, {}


# =============================================================================
# ML-BASED RECOMMENDER CLASSES
# =============================================================================

class KnnRecommender:
    """Recommends pipelines using k-nearest neighbors classification."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
        self.name = 'CLF-kNN'
    
    def fit(self):
        """Train a KNN model to recommend pipelines."""
        try:
            if self.performance_matrix is None or self.metafeatures_df is None:
                print("    Error: Performance matrix and metafeatures are required")
                return False
            
            # Get dataset IDs and features from the metafeatures DataFrame
            feature_data = {}
            for idx in self.metafeatures_df.index:
                feature_data[idx] = self.metafeatures_df.loc[idx].values
            
            # Prepare training data
            X = []
            y = []
            
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        ds_id = int(col.split('_')[1])
                        if ds_id in feature_data:
                            # Get the best pipeline for this dataset (index of max value)
                            col_values = self.performance_matrix[col].values
                            best_idx = np.nanargmax(col_values)
                            
                            # Add to training data if valid
                            if not np.isnan(col_values[best_idx]):
                                X.append(feature_data[ds_id])
                                y.append(best_idx)
                except:
                    continue
            
            if len(X) < 5:
                print(f"    Warning: Not enough training examples for KNN ({len(X)})")
                return False
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train KNN model
            self.model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)))
            self.model.fit(X_train, y_train)
            
            # Validate model
            train_acc = self.model.score(X_train, y_train) * 100
            val_acc = self.model.score(X_val, y_val) * 100
            print(f"    KNN model training accuracy: {train_acc:.2f}%, validation: {val_acc:.2f}%")
            
            # Store pipeline names
            self.pipeline_names = self.performance_matrix.index.tolist()
            self.trained = True
            print(f"    ✅ KNN recommender trained on {len(X)} examples")
            return True
        except Exception as e:
            print(f"    Error training KNN recommender: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline using KNN classification."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            # Get features for the target dataset
            X = self.metafeatures_df.loc[dataset_id].values.reshape(1, -1)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Predict probabilities
            probs = self.model.predict_proba(X)[0]
            
            # Create predictions dictionary
            predictions = {}
            classes = self.model.classes_
            
            for i, pipeline_idx in enumerate(classes):
                if pipeline_idx < len(self.pipeline_names):
                    pipeline = self.pipeline_names[pipeline_idx]
                    predictions[pipeline] = float(probs[i])
            
            # Find best pipeline
            if len(predictions) > 0:
                best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
                print(f"    🧠 KNN recommender suggests: {best_pipeline} (confidence: {predictions[best_pipeline]:.4f})")
                return best_pipeline, predictions
            else:
                print("    ⚠️ KNN recommender couldn't make a prediction")
                return None, {}
        except Exception as e:
            print(f"    Error making KNN recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


class RFRecommender:
    """Recommends pipelines using Random Forest classification."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
        self.name = 'CLF-RF'
    
    def fit(self):
        """Train a Random Forest model to recommend pipelines."""
        try:
            if self.performance_matrix is None or self.metafeatures_df is None:
                print("    Error: Performance matrix and metafeatures are required")
                return False
            
            # Get dataset IDs and features from the metafeatures DataFrame
            feature_data = {}
            for idx in self.metafeatures_df.index:
                feature_data[idx] = self.metafeatures_df.loc[idx].values
            
            # Prepare training data
            X = []
            y = []
            
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        ds_id = int(col.split('_')[1])
                        if ds_id in feature_data:
                            # Get the best pipeline for this dataset (index of max value)
                            col_values = self.performance_matrix[col].values
                            best_idx = np.nanargmax(col_values)
                            
                            # Add to training data if valid
                            if not np.isnan(col_values[best_idx]):
                                X.append(feature_data[ds_id])
                                y.append(best_idx)
                except:
                    continue
            
            if len(X) < 5:
                print(f"    Warning: Not enough training examples for Random Forest ({len(X)})")
                return False
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train Random Forest model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Validate model
            train_acc = self.model.score(X_train, y_train) * 100
            val_acc = self.model.score(X_val, y_val) * 100
            print(f"    Random Forest model training accuracy: {train_acc:.2f}%, validation: {val_acc:.2f}%")
            
            # Store pipeline names
            self.pipeline_names = self.performance_matrix.index.tolist()
            self.trained = True
            print(f"    ✅ Random Forest recommender trained on {len(X)} examples")
            return True
        except Exception as e:
            print(f"    Error training Random Forest recommender: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline using Random Forest classification."""
        if not self.trained:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            # Get features for the target dataset
            X = self.metafeatures_df.loc[dataset_id].values.reshape(1, -1)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Predict probabilities
            probs = self.model.predict_proba(X)[0]
            
            # Create predictions dictionary
            predictions = {}
            classes = self.model.classes_
            
            for i, pipeline_idx in enumerate(classes):
                if pipeline_idx < len(self.pipeline_names):
                    pipeline = self.pipeline_names[pipeline_idx]
                    predictions[pipeline] = float(probs[i])
            
            # Find best pipeline
            if len(predictions) > 0:
                best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
                print(f"    🌳 Random Forest recommender suggests: {best_pipeline} (confidence: {predictions[best_pipeline]:.4f})")
                return best_pipeline, predictions
            else:
                print("    ⚠️ Random Forest recommender couldn't make a prediction")
                return None, {}
        except Exception as e:
            print(f"    Error making Random Forest recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


# Neural Network Classifier implementation
class NNClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10, output_dim=3):
        super(NNClassifier, self).__init__()
        # Improved architecture with batch norm and dropout
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.nn1 = nn.Linear(input_dim, 2*hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(2*hidden_dim)
        self.nn2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.pred = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.nn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.bn2(x)
        x = self.nn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return self.pred(x)


class NNRecommender:
    """Recommends pipelines using Neural Network classification."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
        self.name = 'CLF-NN'
        self.encoder = None
        self.save_path = './model/nn_recommender.pkl'
    
    def fit(self):
        """Train a Neural Network model to recommend pipelines."""
        try:
            if self.performance_matrix is None or self.metafeatures_df is None:
                print("    Error: Performance matrix and metafeatures are required")
                return False
            
            # Get dataset IDs and features from the metafeatures DataFrame
            feature_data = {}
            for idx in self.metafeatures_df.index:
                feature_data[idx] = self.metafeatures_df.loc[idx].values
            
            # Prepare training data
            X = []
            y = []
            
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        ds_id = int(col.split('_')[1])
                        if ds_id in feature_data:
                            # Get the best pipeline for this dataset (index of max value)
                            col_values = self.performance_matrix[col].values
                            best_idx = np.nanargmax(col_values)
                            
                            # Add to training data if valid
                            if not np.isnan(col_values[best_idx]):
                                X.append(feature_data[ds_id])
                                y.append(best_idx)
                except:
                    continue
            
            if len(X) < 10:
                print(f"    Warning: Not enough training examples for Neural Network ({len(X)})")
                return False
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Create encoder to map class indices to continuous values
            from sklearn import preprocessing
            self.encoder = preprocessing.LabelEncoder()
            y = self.encoder.fit_transform(y)
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Calculate class weights to handle imbalance
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weights_dict = {i: w for i, w in zip(classes, class_weights)}
            weights_tensor = torch.tensor([class_weights_dict.get(i, 1.0) for i in range(len(self.encoder.classes_))]).float()
            
            # Create neural network model with better architecture
            input_dim = X.shape[1]
            hidden_dim = min(128, input_dim * 2)  # Larger hidden layer
            output_dim = len(self.encoder.classes_)
            
            model = NNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)  # Add class weights
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # Adam optimizer
            
            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train).float()
            y_train_tensor = torch.tensor(y_train).long()
            X_val_tensor = torch.tensor(X_val).float()
            y_val_tensor = torch.tensor(y_val).long()
            
            # Training loop with more epochs
            n_epochs = 300  # Increased from 100
            batch_size = min(32, len(X_train))
            best_val_acc = 0
            
            for epoch in range(n_epochs):
                # Training
                model.train()
                indices = torch.randperm(len(X_train_tensor))
                
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_x = X_train_tensor[batch_indices]
                    batch_y = y_train_tensor[batch_indices]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    _, predicted = torch.max(val_outputs, 1)
                    val_acc = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model, self.save_path)
            
            # Load the best model (PyTorch 2.6+ requires weights_only=False for custom classes)
            self.model = torch.load(self.save_path, weights_only=False)
            
            # Evaluate on train and validation sets
            self.model.eval()
            with torch.no_grad():
                train_outputs = self.model(X_train_tensor)
                _, train_predicted = torch.max(train_outputs, 1)
                train_acc = (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor) * 100
                
                val_outputs = self.model(X_val_tensor)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == y_val_tensor).sum().item() / len(y_val_tensor) * 100
            
            print(f"    Neural Network model training accuracy: {train_acc:.2f}%, validation: {val_acc:.2f}%")
            
            # Store pipeline names
            self.pipeline_names = self.performance_matrix.index.tolist()
            self.trained = True
            print(f"    ✅ Neural Network recommender trained on {len(X)} examples")
            return True
        except Exception as e:
            print(f"    Error training Neural Network recommender: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline using Neural Network classification."""
        if not self.trained or self.model is None:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            # Get features for the target dataset
            features = self.metafeatures_df.loc[dataset_id].values
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)
            
            # Convert to PyTorch tensor
            X = torch.tensor(features).float().unsqueeze(0)  # Add batch dimension
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                probs = F.softmax(outputs, dim=1)[0]
            
            # Convert class indices back to pipeline indices
            predictions = {}
            for i, prob in enumerate(probs):
                pipeline_idx = self.encoder.inverse_transform([i])[0]
                if pipeline_idx < len(self.pipeline_names):
                    pipeline = self.pipeline_names[pipeline_idx]
                    predictions[pipeline] = float(prob)
            
            # Find best pipeline
            if len(predictions) > 0:
                best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
                print(f"    🧠 Neural Network recommender suggests: {best_pipeline} (confidence: {predictions[best_pipeline]:.4f})")
                return best_pipeline, predictions
            else:
                print("    ⚠️ Neural Network recommender couldn't make a prediction")
                return None, {}
        except Exception as e:
            print(f"    Error making Neural Network recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


# =============================================================================
# REGRESSION-BASED RECOMMENDER CLASSES
# =============================================================================

# Neural Network Regressor implementation
class NNRegressor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10, output_dim=1):
        super(NNRegressor, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.nn1 = nn.Linear(input_dim, 2*hidden_dim)
        self.nn2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.pred = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.nn1(x)
        x = F.relu(x)
        x = self.nn2(x)
        x = F.relu(x)
        return self.pred(x)


class RegressorRecommender:
    """Recommends pipelines using neural network regression to predict performance."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
        self.pipeline_features = None
        self.name = 'Regressor'
        self.save_path = './model/nn_regressor.pkl'
    
    def fit(self):
        """Train a neural network regressor to predict pipeline performance."""
        try:
            if self.performance_matrix is None or self.metafeatures_df is None:
                print("    Error: Performance matrix and metafeatures are required")
                return False
            
            # Get dataset IDs and features from the metafeatures DataFrame
            feature_data = {}
            for idx in self.metafeatures_df.index:
                feature_data[idx] = self.metafeatures_df.loc[idx].values
            
            # Get pipeline features - simple one-hot encoding for now
            self.pipeline_names = self.performance_matrix.index.tolist()
            pipeline_count = len(self.pipeline_names)
            self.pipeline_features = np.eye(pipeline_count)
            
            # Prepare training data
            X_data = []
            y_data = []
            
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        ds_id = int(col.split('_')[1])
                        if ds_id in feature_data:
                            dataset_features = feature_data[ds_id]
                            
                            for i, pipeline in enumerate(self.pipeline_names):
                                score = self.performance_matrix.loc[pipeline, col]
                                if not np.isnan(score):
                                    # Combine dataset features and pipeline features
                                    pipeline_feats = self.pipeline_features[i]
                                    combined_features = np.concatenate([dataset_features, pipeline_feats])
                                    X_data.append(combined_features)
                                    y_data.append(score)
                except:
                    continue
            
            if len(X_data) < 20:
                print(f"    Warning: Not enough training examples for regressor ({len(X_data)})")
                return False
                
            # Convert to numpy arrays
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create neural network regressor
            input_dim = X.shape[1]
            hidden_dim = min(64, input_dim * 2)
            
            model = NNRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
            
            # Convert data to PyTorch tensors
            X_train_tensor = torch.tensor(X_train).float()
            y_train_tensor = torch.tensor(y_train).float().view(-1, 1)
            X_val_tensor = torch.tensor(X_val).float()
            y_val_tensor = torch.tensor(y_val).float().view(-1, 1)
            
            # Training loop
            n_epochs = 100
            batch_size = min(32, len(X_train))
            best_val_loss = float('inf')
            
            for epoch in range(n_epochs):
                # Training
                model.train()
                indices = torch.randperm(len(X_train_tensor))
                
                for i in range(0, len(X_train_tensor), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_x = X_train_tensor[batch_indices]
                    batch_y = y_train_tensor[batch_indices]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model, self.save_path)
            
            # Load the best model (PyTorch 2.6+ requires weights_only=False for custom classes)
            self.model = torch.load(self.save_path, weights_only=False)
            
            # Evaluate on train and validation sets
            self.model.eval()
            with torch.no_grad():
                train_outputs = self.model(X_train_tensor)
                train_loss = criterion(train_outputs, y_train_tensor).item()
                
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            print(f"    Neural Network regressor training loss: {train_loss:.4f}, validation: {val_loss:.4f}")
            
            self.trained = True
            print(f"    ✅ Neural Network regressor trained on {len(X_data)} examples")
            return True
        except Exception as e:
            print(f"    Error training Neural Network regressor: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline by predicting performance using regression."""
        if not self.trained or self.model is None:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            # Get features for the target dataset
            dataset_features = self.metafeatures_df.loc[dataset_id].values
            
            # Handle NaN values
            dataset_features = np.nan_to_num(dataset_features, nan=0.0)
            
            # Predict performance for each pipeline
            predictions = {}
            
            self.model.eval()
            for i, pipeline in enumerate(self.pipeline_names):
                # Combine dataset features with pipeline features
                pipeline_feats = self.pipeline_features[i]
                combined_features = np.concatenate([dataset_features, pipeline_feats])
                
                # Convert to PyTorch tensor
                X = torch.tensor(combined_features).float().unsqueeze(0)  # Add batch dimension
                
                # Predict performance
                with torch.no_grad():
                    output = self.model(X)
                    predicted_score = float(output.item())
                
                predictions[pipeline] = predicted_score
            
            # Find best pipeline
            if len(predictions) > 0:
                best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
                print(f"    📈 Regressor recommender suggests: {best_pipeline} (predicted score: {predictions[best_pipeline]:.4f})")
                return best_pipeline, predictions
            else:
                print("    ⚠️ Regressor recommender couldn't make a prediction")
                return None, {}
        except Exception as e:
            print(f"    Error making regression recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


class AdaBoostRegressorRecommender:
    """Recommends pipelines using AdaBoost regression to predict performance."""
    
    def __init__(self, performance_matrix=None, metafeatures_df=None):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.model = None
        self.trained = False
        self.pipeline_names = None
        self.pipeline_features = None
        self.name = 'AdaBoostRegressor'
    
    def fit(self):
        """Train an AdaBoost regressor to predict pipeline performance."""
        try:
            if self.performance_matrix is None or self.metafeatures_df is None:
                print("    Error: Performance matrix and metafeatures are required")
                return False
            
            # Get dataset IDs and features from the metafeatures DataFrame
            feature_data = {}
            for idx in self.metafeatures_df.index:
                feature_data[idx] = self.metafeatures_df.loc[idx].values
            
            # Get pipeline features - simple one-hot encoding for now
            self.pipeline_names = self.performance_matrix.index.tolist()
            pipeline_count = len(self.pipeline_names)
            self.pipeline_features = np.eye(pipeline_count)
            
            # Prepare training data
            X_data = []
            y_data = []
            
            for col in self.performance_matrix.columns:
                try:
                    if '_' in col:
                        ds_id = int(col.split('_')[1])
                        if ds_id in feature_data:
                            dataset_features = feature_data[ds_id]
                            
                            for i, pipeline in enumerate(self.pipeline_names):
                                score = self.performance_matrix.loc[pipeline, col]
                                if not np.isnan(score):
                                    # Combine dataset features and pipeline features
                                    pipeline_feats = self.pipeline_features[i]
                                    combined_features = np.concatenate([dataset_features, pipeline_feats])
                                    X_data.append(combined_features)
                                    y_data.append(score)
                except:
                    continue
            
            if len(X_data) < 20:
                print(f"    Warning: Not enough training examples for AdaBoost ({len(X_data)})")
                return False
                
            # Convert to numpy arrays
            X = np.array(X_data)
            y = np.array(y_data)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0)
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train AdaBoost regressor with better hyperparameters
            self.model = AdaBoostRegressor(
                n_estimators=200,  # Increased from 50
                learning_rate=0.05,  # Lower learning rate for better convergence
                loss='exponential',  # Better loss function
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Validate model
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            print(f"    AdaBoost regressor training R²: {train_score:.4f}, validation: {val_score:.4f}")
            
            self.trained = True
            print(f"    ✅ AdaBoost regressor trained on {len(X_data)} examples")
            return True
        except Exception as e:
            print(f"    Error training AdaBoost regressor: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def recommend(self, dataset_id):
        """Recommend pipeline by predicting performance using AdaBoost regression."""
        if not self.trained or self.model is None:
            print("    Warning: Model not trained, cannot make recommendations")
            return None, {}
            
        try:
            if dataset_id not in self.metafeatures_df.index:
                print(f"    Warning: No metafeatures for dataset {dataset_id}")
                return None, {}
                
            # Get features for the target dataset
            dataset_features = self.metafeatures_df.loc[dataset_id].values
            
            # Handle NaN values
            dataset_features = np.nan_to_num(dataset_features, nan=0.0)
            
            # Predict performance for each pipeline
            predictions = {}
            
            for i, pipeline in enumerate(self.pipeline_names):
                # Combine dataset features with pipeline features
                pipeline_feats = self.pipeline_features[i]
                combined_features = np.concatenate([dataset_features, pipeline_feats])
                
                # Reshape for prediction
                X = combined_features.reshape(1, -1)
                
                # Predict performance
                predicted_score = float(self.model.predict(X)[0])
                predictions[pipeline] = predicted_score
            
            # Find best pipeline
            if len(predictions) > 0:
                best_pipeline = max(predictions.items(), key=lambda x: x[1])[0]
                print(f"    📊 AdaBoost recommender suggests: {best_pipeline} (predicted score: {predictions[best_pipeline]:.4f})")
                return best_pipeline, predictions
            else:
                print("    ⚠️ AdaBoost recommender couldn't make a prediction")
                return None, {}
        except Exception as e:
            print(f"    Error making AdaBoost recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None, {}


# =============================================================================
# EVALUATION FUNCTION - From original script
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


# =============================================================================
# MAIN FUNCTION - For training and evaluating recommenders
# =============================================================================

def main(meta_features_df=None, performance_matrix=None, recommender_type='autogluon'):
    print("STARTING RECOMMENDER TRAINING & EVALUATION")
    
    # If parameters are not provided, try to load them from files
    if meta_features_df is None or performance_matrix is None:
        print("\nLoading existing data...")
        try:
            if performance_matrix is None:
                performance_matrix = pd.read_csv("training_performance_matrix.csv", index_col=0)
            if meta_features_df is None:
                meta_features_df = pd.read_csv("dataset_feats.csv", index_col=0)
            print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
            print(f"SUCCESS: Loaded metafeatures for {len(meta_features_df)} datasets")
        except FileNotFoundError as e:
            print(f"ERROR: Error loading required files: {e}")
            print("Please make sure 'training_performance_matrix.csv' and 'dataset_feats.csv' exist.")
            return
    
    print("\n" + "="*80)
    print(f"TRAINING {recommender_type.upper()} RECOMMENDER")
    print("="*80)
    
    recommender = None
    recommender_success = False
    
    # Create the appropriate recommender based on the type
    if recommender_type == 'hybrid':
        print("\nTraining Hybrid Meta Recommender...")
        recommender = HybridMetaRecommender(
            performance_matrix, 
            meta_features_df,
            use_influence_weighting=args.use_influence,
            influence_method=args.influence_method
        )
        recommender_success = recommender.fit()
    
    elif recommender_type == 'surrogate':
        print("\nTraining Bayesian Surrogate Recommender...")
        recommender = BayesianSurrogateRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
    
    elif recommender_type == 'autogluon':
        print("\nTraining AutoGluon Pipeline Recommender...")
        recommender = AutoGluonPipelineRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
    
    elif recommender_type == 'random':
        print("\nInitializing Random Recommender...")
        recommender = RandomRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'average-rank':
        print("\nTraining Average Rank Recommender...")
        recommender = AverageRankRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'l1':
        print("\nTraining L1 Recommender...")
        recommender = L1Recommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'basic':
        print("\nTraining Basic Recommender...")
        recommender = BasicRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'knn':
        print("\nTraining KNN Recommender...")
        recommender = KnnRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'rf':
        print("\nTraining Random Forest Recommender...")
        recommender = RFRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'nn':
        print("\nTraining Neural Network Recommender...")
        recommender = NNRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'regressor':
        print("\nTraining Regressor Recommender...")
        recommender = RegressorRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'adaboost':
        print("\nTraining AdaBoost Recommender...")
        recommender = AdaBoostRegressorRecommender(performance_matrix, meta_features_df)
        recommender_success = recommender.fit()
        
    elif recommender_type == 'pmm':
        print("\nTraining PMM Recommender...")
        recommender = PmmRecommender()
        recommender_success = recommender.fit(performance_matrix, meta_features_df)
        
    elif recommender_type == 'balancedpmm':
        print("\nTraining Balanced PMM Recommender...")
        recommender = BalancedPmmRecommender()
        recommender_success = recommender.fit(performance_matrix, meta_features_df)
    
    elif recommender_type == 'paper_pmm':
        print("\nTraining Paper-Style PMM Recommender (Dataset+Pipeline pairs)...")
        from pmm_paper_style import PaperPmmRecommender
        recommender = PaperPmmRecommender(
            hidden_dim=128,
            embedding_dim=64,
            margin=0.8,
            batch_size=64,
            num_epochs=500,
            learning_rate=0.001,
            use_influence_weighting=args.use_influence,
            influence_method=args.influence_method if args.use_influence else 'performance_variance'
        )
        if args.use_influence:
            print(f"  Using influence weighting with method: {args.influence_method}")
        recommender_success = recommender.fit(performance_matrix, meta_features_df, verbose=True)
    
    elif recommender_type == 'optimizer':
        print("\nTraining Optimization-Based Pipeline Recommender (SMAC3)...")
        print("  This recommender searches the configuration space to construct custom pipelines")
        recommender = OptimizedPipelineRecommender(
            n_trials=50,  # Number of SMAC evaluations per dataset
            random_state=42
        )
        recommender_success = recommender.fit(performance_matrix, meta_features_df)
    
    else:  # Default to baseline (no actual recommender needed)
        print("\nUsing baseline recommender (best local performance)")
        recommender_success = True
    
    print("\n" + "="*80)
    print("EVALUATION ON TEST DATASET")
    print("="*80)
    
    # NEW: For paper_pmm only, evaluate on ALL test datasets using ground truth
    if recommender_type == 'paper_pmm' and recommender_success and recommender is not None:
        try:
            ground_truth_perf_matrix = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
            print(f"✅ Loaded test ground truth: {ground_truth_perf_matrix.shape}")
            print(f"   Test datasets: {list(ground_truth_perf_matrix.columns)}")
            
            test_summaries = []
            
            for ds_col in ground_truth_perf_matrix.columns:
                # Extract dataset ID
                if ds_col.startswith('D_'):
                    ds_id = int(ds_col[2:])
                else:
                    ds_id = ds_col
                
                # Check if dataset has metafeatures
                if ds_id not in meta_features_df.index:
                    print(f"  ⚠️  Dataset {ds_id} not in metafeatures, skipping")
                    continue
                
                # Get metafeatures
                dataset_metafeats = meta_features_df.loc[ds_id].values
                
                # Get ground truth performances
                ground_truth_perf = ground_truth_perf_matrix[ds_col].dropna()
                
                if len(ground_truth_perf) == 0:
                    continue
                
                print(f"\n  Evaluating on dataset {ds_id}...")
                
                # Get recommendation from paper_pmm
                top_k_pipelines = recommender.recommend(dataset_metafeats, top_k=5)
                recommended_pipeline = top_k_pipelines[0]
                
                # Evaluate recommendation
                best_pipeline = ground_truth_perf.idxmax()
                best_score = ground_truth_perf.max()
                recommended_score = ground_truth_perf[recommended_pipeline]
                baseline_score = ground_truth_perf.get('baseline', np.nan)
                
                # Calculate rank
                sorted_pipelines = ground_truth_perf.sort_values(ascending=False)
                rank = list(sorted_pipelines.index).index(recommended_pipeline) + 1
                
                print(f"    Recommended: {recommended_pipeline} (rank {rank}/{len(ground_truth_perf)})")
                print(f"    Score: {recommended_score:.4f} (best: {best_score:.4f}, gap: {best_score-recommended_score:.4f})")
                
                test_summaries.append({
                    'dataset': ds_col,
                    'recommended': recommended_pipeline,
                    'best': best_pipeline,
                    'rank': rank,
                    'recommended_score': recommended_score,
                    'best_score': best_score,
                    'baseline_score': baseline_score,
                    'score_gap': best_score - recommended_score,
                    'better_than_baseline': 'yes' if recommended_score > baseline_score else ('equal' if np.isclose(recommended_score, baseline_score) else 'no')
                })
            
            # Show aggregate results
            if test_summaries:
                summary_df = pd.DataFrame(test_summaries)
                
                print("\n" + "="*80)
                print("AGGREGATE TEST RESULTS")
                print("="*80)
                print(f"Total test datasets evaluated: {len(summary_df)}")
                print(f"Average rank: {summary_df['rank'].mean():.2f}")
                print(f"Top-1 accuracy: {(summary_df['rank'] == 1).mean()*100:.1f}%")
                print(f"Top-3 accuracy: {(summary_df['rank'] <= 3).mean()*100:.1f}%")
                print(f"Average score gap: {summary_df['score_gap'].mean():.4f}")
                print(f"Better than baseline: {(summary_df['better_than_baseline'] == 'yes').mean()*100:.1f}%")
                
                # Save results
                summary_df.to_csv('paper_pmm_test_results.csv', index=False)
                print(f"\n✅ Saved detailed results to 'paper_pmm_test_results.csv'")
                
                # Show some example recommendations
                print("\n" + "="*80)
                print("SAMPLE RECOMMENDATIONS")
                print("="*80)
                for i, row in summary_df.head(5).iterrows():
                    print(f"{row['dataset']}: Recommended={row['recommended']} (rank {row['rank']}), Best={row['best']}, Score={row['recommended_score']:.3f}")
                
                # Done - skip the old single-dataset evaluation
                print("\n[SUCCESS] Paper-PMM evaluation completed on all test datasets!")
                print("\n[FINISH] RECOMMENDER TRAINING & EVALUATION COMPLETED [FINISH]")
                return
                
        except Exception as e:
            print(f"⚠️  Could not load test ground truth: {e}")
            print(f"   Falling back to single-dataset evaluation...")
    
    # OLD: Single dataset evaluation (for baseline or if ground truth not available)
    
    # Try to load a test dataset for evaluation
    test_dataset_ids = [1503, 23517, 1551, 1552, 255, 546, 475, 481, 516, 6, 8, 10, 12, 14, 9, 11, 5]
    
    # Let's take the first test dataset ID that exists in the metafeatures
    test_dataset_id = None
    
    # For PMM recommenders, we need the test dataset to be in both metafeatures and performance_matrix
    if recommender_type in ['pmm', 'balancedpmm']:
        for dataset_id in meta_features_df.index:
            # Try to find this dataset ID in the performance matrix (looking for D_X format)
            formatted_id = f"D_{dataset_id}"
            if formatted_id in performance_matrix.columns:
                test_dataset_id = dataset_id
                print(f"Found dataset {dataset_id} in both metafeatures and performance matrix (as {formatted_id})")
                break
    
    # Fallback to the traditional approach
    if test_dataset_id is None:
        for dataset_id in test_dataset_ids:
            if dataset_id in meta_features_df.index:
                test_dataset_id = dataset_id
                break
    
    if test_dataset_id is None:
        print("❌ No suitable test dataset found in metafeatures.")
        return
    
    print(f"\nEvaluating recommender on dataset ID {test_dataset_id}...")
    
    # Get recommendations from the recommender
    recommendations = {}
    scores = {}
    similarity_info = {}
    
    if recommender_success and recommender_type != 'baseline':
        try:
            # Handle different return formats based on recommender type
            if recommender_type in ['pmm', 'balancedpmm']:
                # PMM recommenders return a dictionary
                print(f"\nGetting recommendation from {recommender_type.upper()} recommender for dataset {test_dataset_id}...")
                recommendation_result = recommender.recommend(test_dataset_id, performance_matrix)
                
                if recommendation_result and recommendation_result.get('pipeline'):
                    recommendation = recommendation_result['pipeline']
                    score_dict = recommendation_result.get('performance_scores', {})
                    similar_datasets = recommendation_result.get('similar_datasets', [])
                    similarity_scores = recommendation_result.get('similarity_scores', {})
                    
                    recommendations[recommender_type] = recommendation
                    scores[recommender_type] = score_dict
                    similarity_info[recommender_type] = {
                        'similar_datasets': similar_datasets,
                        'similarity_scores': similarity_scores
                    }
                    
                    # Print similar datasets information
                    print(f"\nMost similar datasets to {test_dataset_id}:")
                    for i, dataset in enumerate(similar_datasets[:5]):
                        sim_score = similarity_scores.get(dataset, 0.0)
                        print(f"  {i+1}. Dataset {dataset} (similarity: {sim_score:.4f})")
                    
                    # Print detailed scores for top 5 pipelines
                    print("\nTop 5 pipeline scores:")
                    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                    for pipeline, score in sorted_scores:
                        print(f"  {pipeline}: {score:.4f}")
                else:
                    print(f"\n❌ PMM recommender failed for dataset {test_dataset_id}")
                    print(f"Result: {recommendation_result}")
            
            elif recommender_type == 'paper_pmm':
                # Paper-style PMM takes dataset metafeatures and returns list of pipelines
                print(f"\nGetting recommendation from PAPER-PMM recommender for dataset {test_dataset_id}...")
                
                # Get dataset metafeatures
                if test_dataset_id in meta_features_df.index:
                    dataset_metafeats = meta_features_df.loc[test_dataset_id].values
                else:
                    # Try with string conversion
                    test_dataset_id_str = str(test_dataset_id)
                    if test_dataset_id_str in meta_features_df.index:
                        dataset_metafeats = meta_features_df.loc[test_dataset_id_str].values
                    else:
                        print(f"❌ Dataset {test_dataset_id} not found in metafeatures")
                        dataset_metafeats = None
                
                if dataset_metafeats is not None:
                    # Get recommendations (returns list of pipeline names)
                    top_k_pipelines = recommender.recommend(dataset_metafeats, top_k=5)
                    
                    if top_k_pipelines and len(top_k_pipelines) > 0:
                        recommendation = top_k_pipelines[0]
                        recommendations[recommender_type] = recommendation
                        
                        print(f"  ✅ Paper-PMM recommended: {recommendation}")
                        print(f"  Top-5 recommendations: {top_k_pipelines}")
                        
                        # Create score dict (paper-pmm doesn't return explicit scores, use rank)
                        score_dict = {pipeline: 1.0 / (i+1) for i, pipeline in enumerate(top_k_pipelines)}
                        scores[recommender_type] = score_dict
                        
                        # Show influence information if available
                        if args.use_influence and hasattr(recommender, 'dataset_influence_scores'):
                            print(f"\n  🎯 Used influence weighting (method: {args.influence_method})")
                            # Show top 5 most influential datasets
                            top_influential = sorted(
                                recommender.dataset_influence_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:5]
                            print(f"  Most influential training datasets:")
                            for ds, score in top_influential:
                                print(f"    - Dataset {ds}: influence score {score:.3f}")
                    else:
                        print(f"❌ Paper-PMM returned empty recommendations")
                else:
                    print(f"❌ Could not get metafeatures for dataset {test_dataset_id}")
            
            else:
                # Original format for other recommenders
                recommendation, score_dict = recommender.recommend(test_dataset_id)
                if recommendation:
                    recommendations[recommender_type] = recommendation
                    scores[recommender_type] = score_dict
        except Exception as e:
            import traceback
            print(f"\n❌ Error in recommendation: {e}")
            traceback.print_exc()
            print("Using baseline recommender as fallback")
    
    # Print comparison of recommendations
    print("\n" + "="*80)
    print("RECOMMENDER COMPARISON")
    print("="*80)
    
    for recommender_name, pipeline in recommendations.items():
        print(f"{recommender_name.capitalize()} Recommender: {pipeline}")
        
        # Show similarity info for PMM recommenders
        if recommender_name in ['pmm', 'balancedpmm'] and recommender_name in similarity_info:
            similar_datasets = similarity_info[recommender_name].get('similar_datasets', [])
            similarity_scores = similarity_info[recommender_name].get('similarity_scores', {})
            
            if similar_datasets:
                print("\nTop similar datasets used for recommendation:")
                for i, dataset in enumerate(similar_datasets[:3]):
                    sim_score = similarity_scores.get(dataset, 0.0)
                    col_name = None
                    
                    # Try to find the corresponding column in performance matrix
                    if dataset in performance_matrix.columns:
                        col_name = dataset
                    elif f"D_{dataset}" in performance_matrix.columns:
                        col_name = f"D_{dataset}"
                        
                    if col_name:
                        best_pipeline = performance_matrix[col_name].idxmax()
                        best_score = performance_matrix[col_name].max()
                        print(f"  {i+1}. Dataset {dataset} (similarity: {sim_score:.4f}, best pipeline: {best_pipeline} with score {best_score:.4f})")
                    else:
                        print(f"  {i+1}. Dataset {dataset} (similarity: {sim_score:.4f}, not found in performance matrix)")
        
        # Show top 5 pipelines if we have score information
        if recommender_name in scores and scores[recommender_name]:
            print("\nTop 5 pipelines:")
            sorted_scores = sorted(scores[recommender_name].items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (pipe, score) in enumerate(sorted_scores):
                print(f"  {i+1}. {pipe} (score: {score:.4f})")
                
            # Print score distribution statistics
            score_values = list(scores[recommender_name].values())
            if score_values:
                print(f"\nScore statistics:")
                print(f"  Min: {min(score_values):.4f}")
                print(f"  Max: {max(score_values):.4f}")
                print(f"  Mean: {sum(score_values) / len(score_values):.4f}")
                print(f"  Std Dev: {np.std(score_values):.4f}")
                
                # Calculate number of pipelines with non-zero scores
                non_zero_scores = sum(1 for s in score_values if abs(s) > 1e-6)
                print(f"  Non-zero scores: {non_zero_scores}/{len(score_values)} ({non_zero_scores/len(score_values)*100:.1f}%)")
    
    # Find any column in performance matrix that corresponds to this dataset ID
    test_column = None
    for col in performance_matrix.columns:
        if col == str(test_dataset_id) or col == f"D_{test_dataset_id}":
            test_column = col
            break
        elif col.startswith('D_') and col[2:] == str(test_dataset_id):
            test_column = col
            break
    
    if test_column:
        print("\nActual performances in training data:")
        actual_scores = performance_matrix[test_column].sort_values(ascending=False)
        for i, (pipeline, score) in enumerate(actual_scores.items()):
            if not np.isnan(score) and i < 10:  # Show top 10
                print(f"  {pipeline}: {score:.4f}")
    
    print("\n" + "="*80)
    print("SAVING MODELS AND RESULTS")
    print("="*80)
    
    # Save recommender comparison to file
    with open('recommender_comparison.txt', 'w') as f:
        f.write("RECOMMENDER COMPARISON\n")
        f.write("=" * 40 + "\n\n")
        
        for recommender_name, pipeline in recommendations.items():
            f.write(f"{recommender_name.capitalize()} Recommender: {pipeline}\n")
            
            # Write top 5 recommendations
            if recommender_name in scores and scores[recommender_name]:
                f.write("\nTop 5 pipelines:\n")
                sorted_scores = sorted(scores[recommender_name].items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (pipe, score) in enumerate(sorted_scores):
                    f.write(f"  {i+1}. {pipe} (score: {score:.4f})\n")
                    
                # Write score distribution statistics
                score_values = list(scores[recommender_name].values())
                if score_values:
                    f.write(f"\nScore statistics:\n")
                    f.write(f"  Min: {min(score_values):.4f}\n")
                    f.write(f"  Max: {max(score_values):.4f}\n")
                    f.write(f"  Mean: {sum(score_values) / len(score_values):.4f}\n")
                    f.write(f"  Std Dev: {np.std(score_values):.4f}\n")
            f.write("\n")
        
        if test_column:
            f.write("\nActual performances in training data:\n")
            actual_scores = performance_matrix[test_column].sort_values(ascending=False)
            for pipeline, score in actual_scores.items():
                if not np.isnan(score):
                    f.write(f"  {pipeline}: {score:.4f}\n")
    
    print("[SUCCESS] Saved recommender comparison to 'recommender_comparison.txt'")
    
    print("\n[FINISH] RECOMMENDER TRAINING & EVALUATION COMPLETED [FINISH]")


def run_evaluation(meta_features_df, performance_matrix=None, is_quick_test=False, recommender_type='baseline'):
    """Run a comprehensive evaluation of pipeline recommendations on test datasets."""
    print(f"STARTING PIPELINE RECOMMENDER EVALUATION WITH {recommender_type.upper()} RECOMMENDER")
    
    # Try to load ground truth performance data to skip evaluation
    ground_truth_perf_matrix = None
    try:
        ground_truth_perf_matrix = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
        print(f"✅ SUCCESS: Loaded ground truth performance matrix with shape {ground_truth_perf_matrix.shape}")
        print(f"   This will skip the expensive pipeline evaluation step!")
    except Exception as e:
        print(f"INFO: No ground truth performance file found ({e})")
        print(f"   Will evaluate pipelines from scratch (this will be slow)...")
    
    print("\n" + "="*80)
    print("EVALUATING ALL PIPELINES ON TEST DATASETS")
    print("="*80)
    
    test_local_perfs = []
    test_summaries = []
    successful_test = 0
    failed_test = 0
    
    # Use fewer datasets for quick test
    test_ids = test_dataset_ids[:3] if is_quick_test else test_dataset_ids
    
    for i, ds_id in enumerate(test_ids):
        print(f"\n[{i+1}/{len(test_ids)}] Processing TEST dataset {ds_id}...")
        
        dataset = load_openml_dataset(ds_id)
        if dataset is None:
            failed_test += 1
            continue
            
        try:
            local_perf, _, summary = run_experiment_for_dataset(
                dataset, 
                meta_features_df,
                global_performance_matrix=performance_matrix,
                recommender_type=recommender_type,
                ground_truth_perf_matrix=ground_truth_perf_matrix,
                use_influence=args.use_influence,
                influence_method=args.influence_method
            )
            
            if local_perf is not None: 
                test_local_perfs.append(local_perf)
            if summary is not None: 
                test_summaries.append(summary)
                successful_test += 1
            else:
                failed_test += 1
                
        except Exception as e:
            print(f"  ERROR: Experiment failed for test dataset {ds_id}: {e}")
            failed_test += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETION SUMMARY")
    print(f"{'='*80}")
    print(f"Test datasets: {successful_test} successful, {failed_test} failed")
    
    if test_local_perfs:
        test_local_df = pd.concat(test_local_perfs, axis=1)
        test_local_df.to_csv('test_local_performance.csv')
        print(f"\nSUCCESS: Saved test local performances to 'test_local_performance.csv' ({test_local_df.shape})")
        
        # We're using the same test_local_df as the ground truth performances
        test_local_df.to_csv('test_ground_truth_performance.csv')
        print(f"SUCCESS: Saved test ground truth performances to 'test_ground_truth_performance.csv' ({test_local_df.shape})")
        
    if test_summaries:
        test_summary_df = pd.DataFrame(test_summaries).set_index('dataset')
        test_summary_df.to_csv('test_evaluation_summary.csv')
        print(f"SUCCESS: Saved test summary to 'test_evaluation_summary.csv' ({test_summary_df.shape})")
        
        print(f"\n{'='*80}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        print(test_summary_df[['recommendation', 'ground_truth_best', 'rank', 'baseline_rank', 'better_than_baseline', 'recommended_score',  'baseline_score', 'best_score']])
        
        valid_ranks = test_summary_df[test_summary_df['rank'] > 0]['rank']
        if not valid_ranks.empty:
            avg_rank = valid_ranks.mean()
            top1_accuracy = (valid_ranks == 1).mean() * 100
            top3_accuracy = (valid_ranks <= 3).mean() * 100
            
            print(f"\nTEST PERFORMANCE METRICS:")
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
            print("WARNING: No valid recommendations were made on test datasets")
    else:
        print("WARNING: No successful test experiments completed")

    analyze_recommendations(test_summary_df, meta_features_df)
    return test_summary_df


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SoluRec - Pipeline recommender system')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on test datasets')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with fewer datasets')
    parser.add_argument('--debug', action='store_true', help='Run in deep debugging mode with detailed analysis')
    parser.add_argument('--debug-datasets', type=int, default=5, help='Number of datasets to analyze in debug mode')
    parser.add_argument('--recommender', type=str, default='baseline', 
                       choices=['baseline', 'autogluon', 'hybrid', 'surrogate', 'random', 
                               'avgrank', 'l1', 'basic', 'knn', 'rf', 'nn', 
                               'regressor', 'adaboost', 'pmm', 'balancedpmm', 'paper_pmm', 'optimizer'], 
                       help='Type of recommender to use')
    parser.add_argument('--use-influence', action='store_true', 
                       help='Enable DPO-style influence weighting for PMM/BalancedPMM recommenders')
    parser.add_argument('--influence-method', type=str, default='performance_variance',
                       choices=['performance_variance', 'dataset_diversity', 'prediction_confidence', 'combined'],
                       help='Method for calculating influence scores')
    args = parser.parse_args()
    
    # Load saved meta-features and preprocessed performance matrix if they exist
    try:
        meta_features_df = pd.read_csv('dataset_feats.csv', index_col=0)
        print(f"SUCCESS: Loaded {len(meta_features_df)} dataset metafeatures")
        
        if args.debug:
            # Deep debugging mode
            print("\n🔬 ENTERING DEBUG MODE")
            print("="*80)
            
            # Load performance matrix
            try:
                performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
            except Exception as e:
                print(f"ERROR: Performance matrix required for debug mode: {e}")
                exit(1)
            
            # Load ground truth if available
            try:
                ground_truth_matrix = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded test ground truth with shape {ground_truth_matrix.shape}")
            except Exception as e:
                print(f"INFO: No test ground truth found: {e}")
                ground_truth_matrix = None
            
            # Determine test datasets
            if ground_truth_matrix is not None:
                # Use datasets from ground truth file
                test_dataset_ids = [int(col.split('_')[1]) for col in ground_truth_matrix.columns if col.startswith('D_')]
            else:
                # Use a default set of test datasets
                test_dataset_ids = [1503, 1551, 255, 183, 1552]
            
            print(f"\nTest datasets: {test_dataset_ids[:args.debug_datasets]}")
            
            # Train recommenders to debug
            recommender_configs = []
            
            if args.recommender == 'pmm':
                influence_status = "WITH" if args.use_influence else "WITHOUT"
                print(f"\n Training PMM recommender {influence_status} influence weighting...")
                if args.use_influence:
                    print(f"  Influence method: {args.influence_method}")
                pmm_rec = PmmRecommender(
                    num_epochs=20, 
                    batch_size=64,
                    use_influence_weighting=args.use_influence,
                    influence_method=args.influence_method
                )
                if pmm_rec.fit(performance_matrix, meta_features_df):
                    recommender_configs.append((pmm_rec, 'pmm'))
                else:
                    print("ERROR: PMM recommender training failed")
                    exit(1)
                    
            elif args.recommender == 'balancedpmm':
                influence_status = "WITH" if args.use_influence else "WITHOUT"
                print(f"\nTraining Balanced PMM recommender {influence_status} influence weighting...")
                if args.use_influence:
                    print(f"  Influence method: {args.influence_method}")
                bal_pmm_rec = BalancedPmmRecommender(
                    num_epochs=20, 
                    batch_size=64,
                    use_influence_weighting=args.use_influence,
                    influence_method=args.influence_method
                )
                if bal_pmm_rec.fit(performance_matrix, meta_features_df):
                    recommender_configs.append((bal_pmm_rec, 'balancedpmm'))
                else:
                    print("ERROR: Balanced PMM recommender training failed")
                    exit(1)
                    
            elif args.recommender == 'hybrid':
                print("\nTraining Hybrid recommender...")
                hybrid_rec = HybridMetaRecommender(
                    performance_matrix, 
                    meta_features_df,
                    use_influence_weighting=args.use_influence,
                    influence_method=args.influence_method
                )
                if hybrid_rec.fit():
                    recommender_configs.append((hybrid_rec, 'hybrid'))
                else:
                    print("ERROR: Hybrid recommender training failed")
                    exit(1)
            else:
                print(f"ERROR: Debug mode currently only supports pmm, balancedpmm, and hybrid recommenders")
                exit(1)
            
            # Run deep debugging analysis
            from debug_analysis import run_debug_mode
            output_dir = run_debug_mode(
                recommender_configs=recommender_configs,
                test_dataset_ids=test_dataset_ids,
                metafeatures_df=meta_features_df,
                performance_matrix=performance_matrix,
                ground_truth_matrix=ground_truth_matrix,
                max_datasets=args.debug_datasets
            )
            
            print(f"\n🎉 Debug analysis complete! Check the '{output_dir}' directory for detailed results.")
            
        elif args.evaluate:
            # If performance matrix exists, use it; otherwise, build it
            try:
                performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
                run_evaluation(meta_features_df, performance_matrix, args.quick, args.recommender)
            except Exception as e:
                print(f"INFO: No existing performance matrix found: {e}")
                print("Building performance matrix from scratch...")
                run_evaluation(meta_features_df, None, args.quick, args.recommender)
        else:
            # Regular recommender training flow
            try:
                performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
                main(meta_features_df, performance_matrix, args.recommender)
            except Exception as e:
                print(f"ERROR: Error loading performance matrix: {e}")
                print("Please run preprocess_metafeatures.py first to generate the required files.")
    except Exception as e:
        print(f"ERROR: Error loading saved data: {e}")
        print("Please run preprocess_metafeatures.py first to generate the required files.")