"""
Paper-Style PMM Recommender - Correct Implementation
Learns performance ranking directly using (dataset + pipeline) metafeature combinations.
This is the architecture described in the paper you provided.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')


# =============================================================================
# NEURAL NETWORK COMPONENTS (from paper)
# =============================================================================

def lam_distance(x1, x2, function='sigmoid'):
    """
    Compute distance between two performance predictions.
    Paper uses sigmoid to transform differences to [0,1] range.
    """
    if function == 'none':
        return (x1 - x2).reshape(-1)
    elif function == 'sigmoid':
        return torch.sigmoid(x1 - x2).reshape(-1)
    else:
        return torch.sigmoid(x1 - x2).reshape(-1)


class LamNet(nn.Module):
    """
    Embedding Network (from paper): Maps (dataset + pipeline) features to performance score.
    Input: [dataset_metafeatures, pipeline_metafeatures]
    Output: Predicted performance score [0, 100]
    """
    
    def __init__(self, n_input, n_hidden, n_output=1):
        super(LamNet, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(n_input)
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.pred = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.hidden1(x)
        x = F.relu(x)
        
        x = self.hidden2(x)
        x = F.relu(x)
        
        out = self.pred(x)
        return torch.sigmoid(out) * 100  # Scale to 0-100


class SiameseNet(nn.Module):
    """
    Siamese Network (from paper): Twin architecture for comparing performance predictions.
    Both branches use the SAME LamNet to ensure consistent scoring.
    """
    
    def __init__(self, lamNet):
        super(SiameseNet, self).__init__()
        self.lam = lamNet  # Shared embedding network
    
    def forward(self, x1, x2):
        """Process two (dataset, pipeline) pairs for training"""
        out1 = self.lam(x1)
        out2 = self.lam(x2)
        return out1, out2
    
    def get_lam(self, x):
        """Get prediction for a single (dataset, pipeline) pair"""
        return self.lam(x)
    
    def predict_proba(self, dataset_feats, pipeline_feats_matrix):
        """
        Predict performance for one dataset with all pipelines.
        
        Args:
            dataset_feats: Single dataset metafeatures [n_dataset_features]
            pipeline_feats_matrix: All pipeline metafeatures [n_pipelines, n_pipeline_features]
        
        Returns:
            Performance scores for all pipelines [n_pipelines]
        """
        self.eval()
        # Repeat dataset features for each pipeline
        n_pipelines = pipeline_feats_matrix.shape[0]
        dataset_repeated = np.repeat(dataset_feats.reshape(1, -1), n_pipelines, axis=0)
        
        # Concatenate: [dataset_feats, pipeline_feats]
        combined = np.concatenate((dataset_repeated, pipeline_feats_matrix), axis=1)
        combined_tensor = torch.tensor(combined).float()
        
        with torch.no_grad():
            scores = self.get_lam(combined_tensor)
        
        return scores.reshape(-1).numpy()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (from paper): Teaches network to rank pairs correctly.
    - Similar pairs (y=1): Minimize distance
    - Dissimilar pairs (y=0): Maximize distance beyond margin
    """
    
    def __init__(self, margin=0.8, function='sigmoid', distance_function='l1'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.function = function
        self.distance_function = distance_function
        self.eps = 1e-9
    
    def forward(self, out, target, size_average=True):
        out1, out2 = out
        distance = lam_distance(out1, out2, self.function)
        
        if self.distance_function == 'l2':
            losses = 0.5 * ((1 - target).float() * torch.pow(distance, 2) + 
                           target.float() * torch.pow(F.relu(self.margin - (distance + self.eps)), 2))
        else:
            losses = 0.5 * ((1 - target).float() * distance + 
                           target.float() * F.relu(self.margin - (distance + self.eps)))
        
        return losses.mean() if size_average else losses.sum()


class LamDataset(Dataset):
    """PyTorch Dataset for pairwise training"""
    
    def __init__(self, pairs1, pairs2, labels=None):
        self.pairs1 = pairs1
        self.pairs2 = pairs2
        self.labels = labels
    
    def __getitem__(self, index):
        if self.labels is None:
            return self.pairs1[index], self.pairs2[index]
        else:
            return self.pairs1[index], self.pairs2[index], self.labels[index]
    
    def __len__(self):
        return len(self.pairs1)


# =============================================================================
# PAPER-STYLE PMM RECOMMENDER
# =============================================================================

class PaperPmmRecommender:
    """
    PMM Recommender following the paper's architecture.
    
    Key differences from dataset-similarity PMM:
    1. Uses (dataset + pipeline) metafeature pairs as input
    2. Predicts performance scores directly
    3. Learns rankings through contrastive loss on pairs
    4. Supports influence weighting for more informative datasets
    """
    
    def __init__(self, hidden_dim=64, embedding_dim=32, margin=0.8,
                 batch_size=32, num_epochs=50, learning_rate=0.001,
                 use_influence_weighting=True, influence_method='performance_variance'):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.use_influence_weighting = use_influence_weighting
        self.influence_method = influence_method
        
        self.model = None
        self.pipeline_features = None
        self.pipeline_names = []
        self.is_trained = False
        self.influence_scores = {}
        self.dataset_scaler = StandardScaler()  # For normalizing dataset features
        self.imputer = None  # For handling NaN values in dataset features
        
    def _extract_pipeline_features(self):
        """
        Extract pipeline features from pipeline_configs.
        Creates one-hot encoding of pipeline components.
        """
        from recommender_trainer import pipeline_configs
        
        pipeline_features = []
        self.pipeline_names = []
        
        for config in pipeline_configs:
            self.pipeline_names.append(config['name'])
            
            features = []
            
            # Imputation (one-hot, 6 options)
            imputation_methods = ['none', 'mean', 'median', 'knn', 'most_frequent', 'constant']
            features.extend([1 if config.get('imputation') == method else 0 for method in imputation_methods])
            
            # Scaling (one-hot, 5 options)
            scaling_methods = ['none', 'standard', 'minmax', 'robust', 'maxabs']
            features.extend([1 if config.get('scaling') == method else 0 for method in scaling_methods])
            
            # Feature selection (one-hot, 4 options)
            selection_methods = ['none', 'k_best', 'mutual_info', 'variance_threshold']
            features.extend([1 if config.get('feature_selection') == method else 0 for method in selection_methods])
            
            # Outlier removal (one-hot, 5 options)
            outlier_methods = ['none', 'iqr', 'zscore', 'isolation_forest', 'lof']
            features.extend([1 if config.get('outlier_removal') == method else 0 for method in outlier_methods])
            
            # Dimensionality reduction (one-hot, 3 options)
            dim_red_methods = ['none', 'pca', 'svd']
            features.extend([1 if config.get('dimensionality_reduction') == method else 0 for method in dim_red_methods])
            
            pipeline_features.append(features)
        
        return np.array(pipeline_features, dtype=np.float32)
    
    def _calculate_influence_scores(self, performance_matrix, metafeatures_df):
        """Calculate influence scores for training datasets (same as before)"""
        influence_scores = {}
        
        print(f"    Calculating influence scores using method: {self.influence_method}")
        
        if self.influence_method == 'performance_variance':
            for col in performance_matrix.columns:
                perfs = performance_matrix[col].dropna()
                if len(perfs) > 0:
                    variance = perfs.var()
                    influence_scores[col] = variance
        
        elif self.influence_method == 'discriminative_power':
            for col in performance_matrix.columns:
                perfs = performance_matrix[col].dropna()
                if len(perfs) >= 2:
                    gap = perfs.max() - perfs.min()
                    sorted_perfs = perfs.sort_values(ascending=False)
                    if len(sorted_perfs) >= 6:
                        top_3_std = sorted_perfs.iloc[:3].std()
                        bottom_3_std = sorted_perfs.iloc[-3:].std()
                        clarity = gap / (top_3_std + bottom_3_std + 1e-6)
                    else:
                        clarity = gap
                    influence_scores[col] = clarity
        
        elif self.influence_method == 'data_diversity':
            all_metafeatures = []
            dataset_cols_ordered = []
            
            for col in performance_matrix.columns:
                # Map column to dataset ID
                if col.startswith('D_'):
                    ds_id = int(col[2:])
                else:
                    ds_id = col
                
                if ds_id in metafeatures_df.index:
                    mf = metafeatures_df.loc[ds_id].values
                    if not np.isnan(mf).any():
                        all_metafeatures.append(mf)
                        dataset_cols_ordered.append(col)
            
            if len(all_metafeatures) > 1:
                all_metafeatures = np.array(all_metafeatures)
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(all_metafeatures)
                np.fill_diagonal(distances, np.inf)
                min_distances = distances.min(axis=1)
                
                for i, col in enumerate(dataset_cols_ordered):
                    influence_scores[col] = min_distances[i]
        else:
            for col in performance_matrix.columns:
                influence_scores[col] = 1.0
        
        # Normalize
        if influence_scores:
            scores_array = np.array(list(influence_scores.values()))
            if scores_array.max() > 0:
                scores_array = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min() + 1e-9)
                scores_array = np.power(scores_array, 2)  # Stronger differentiation
                for i, col in enumerate(influence_scores.keys()):
                    influence_scores[col] = scores_array[i]
        
        return influence_scores
    
    def _create_training_pairs(self, performance_matrix, metafeatures_df, pipeline_features, total_pairs=10000):
        """
        Create training pairs of (dataset, pipeline) combinations.
        
        Paper's approach:
        1. Sample datasets (optionally with influence weighting)
        2. Sample pipelines
        3. Get their performances
        4. Create label: y=1 if perf1 >= perf2, else y=0
        """
        n_pipelines = len(self.pipeline_names)
        n_datasets = len(performance_matrix.columns)
        
        # Get dataset metafeatures and normalize them
        dataset_cols_list = []
        dataset_feats_list = []
        col_to_ds_id = {}
        
        for col in performance_matrix.columns:
            if col.startswith('D_'):
                ds_id = int(col[2:])
            else:
                ds_id = col
            
            if ds_id in metafeatures_df.index:
                dataset_cols_list.append(col)
                dataset_feats_list.append(metafeatures_df.loc[ds_id].values)
                col_to_ds_id[col] = ds_id
        
        # NEW: Fit scaler on all dataset features and normalize
        dataset_feats_array = np.array(dataset_feats_list)
        
        # CRITICAL: Handle NaN values before normalization
        # Replace NaN with median of each feature column
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        dataset_feats_imputed = imputer.fit_transform(dataset_feats_array)
        
        # Now normalize
        dataset_feats_normalized = self.dataset_scaler.fit_transform(dataset_feats_imputed)
        
        # Store imputer for use in recommend()
        self.imputer = imputer
        
        # Create dictionary with normalized features
        dataset_feats_dict = {}
        for i, col in enumerate(dataset_cols_list):
            dataset_feats_dict[col] = dataset_feats_normalized[i]
        
        # Calculate sampling probabilities
        if self.use_influence_weighting and self.influence_scores:
            sampling_weights = np.array([self.influence_scores.get(col, 1.0) for col in performance_matrix.columns])
            sampling_weights = sampling_weights / sampling_weights.sum()
        else:
            sampling_weights = np.ones(n_datasets) / n_datasets
        
        pairs1 = []
        pairs2 = []
        labels = []
        
        dataset_cols = list(performance_matrix.columns)
        
        pair_count = 0
        max_attempts = total_pairs * 10
        attempts = 0
        
        print(f"    Generating {total_pairs} training pairs...")
        
        while pair_count < total_pairs and attempts < max_attempts:
            attempts += 1
            
            # Sample datasets with influence-based probability
            ds_idx1, ds_idx2 = np.random.choice(n_datasets, size=2, replace=True, p=sampling_weights)
            ds_col1 = dataset_cols[ds_idx1]
            ds_col2 = dataset_cols[ds_idx2]
            
            # Sample pipelines uniformly
            pipe_idx1, pipe_idx2 = np.random.randint(0, n_pipelines, size=2)
            pipe_name1 = self.pipeline_names[pipe_idx1]
            pipe_name2 = self.pipeline_names[pipe_idx2]
            
            # Get performances
            perf1 = performance_matrix.loc[pipe_name1, ds_col1]
            perf2 = performance_matrix.loc[pipe_name2, ds_col2]
            
            # Skip if missing data
            if np.isnan(perf1) or np.isnan(perf2):
                continue
            
            # Skip if no metafeatures
            if ds_col1 not in dataset_feats_dict or ds_col2 not in dataset_feats_dict:
                continue
            
            # Create feature vectors: [dataset_features, pipeline_features]
            f1 = np.concatenate([dataset_feats_dict[ds_col1], pipeline_features[pipe_idx1]])
            f2 = np.concatenate([dataset_feats_dict[ds_col2], pipeline_features[pipe_idx2]])
            
            pairs1.append(torch.tensor(f1).float())
            pairs2.append(torch.tensor(f2).float())
            
            # Label: 1 if perf1 >= perf2, else 0
            label = 1 if perf1 >= perf2 else 0
            labels.append(label)
            
            pair_count += 1
        
        print(f"    Created {len(labels)} pairs (similar: {sum(labels)}, dissimilar: {len(labels) - sum(labels)})")
        
        return pairs1, pairs2, labels
    
    def fit(self, performance_matrix, metafeatures_df, verbose=False):
        """
        Train the paper-style PMM recommender.
        
        Args:
            performance_matrix: [n_pipelines, n_datasets] performance scores
            metafeatures_df: [n_datasets, n_features] dataset metafeatures
            verbose: Print training progress
        """
        try:
            print("  Training Paper-Style PMM Recommender...")
            
            # Extract pipeline features
            self.pipeline_features = self._extract_pipeline_features()
            print(f"    Pipeline features shape: {self.pipeline_features.shape}")
            
            # Calculate influence scores
            if self.use_influence_weighting:
                self.influence_scores = self._calculate_influence_scores(performance_matrix, metafeatures_df)
                print(f"    Influence scores - Min: {min(self.influence_scores.values()):.3f}, "
                      f"Max: {max(self.influence_scores.values()):.3f}")
            
            # Create training pairs
            pairs1, pairs2, labels = self._create_training_pairs(
                performance_matrix, metafeatures_df, self.pipeline_features, total_pairs=10000
            )
            
            if len(labels) < 100:
                print(f"    ERROR: Not enough training pairs ({len(labels)})")
                return False
            
            # Split into train/val/test
            n_pairs = len(labels)
            n_train = int(n_pairs * 0.7)
            n_val = int(n_pairs * 0.15)
            
            train_dataset = LamDataset(pairs1[:n_train], pairs2[:n_train], labels[:n_train])
            val_dataset = LamDataset(pairs1[n_train:n_train+n_val], pairs2[n_train:n_train+n_val], 
                                    labels[n_train:n_train+n_val])
            test_dataset = LamDataset(pairs1[n_train+n_val:], pairs2[n_train+n_val:], 
                                     labels[n_train+n_val:])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            n_dataset_feats = metafeatures_df.shape[1]
            n_pipeline_feats = self.pipeline_features.shape[1]
            input_dim = n_dataset_feats + n_pipeline_feats
            
            print(f"    Input dimension: {input_dim} (dataset: {n_dataset_feats}, pipeline: {n_pipeline_feats})")
            
            lamNet = LamNet(n_input=input_dim, n_hidden=self.hidden_dim, n_output=1)
            self.model = SiameseNet(lamNet)
            
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, 
                                       momentum=0.9, weight_decay=0.0001)
            criterion = ContrastiveLoss(margin=self.margin, function='sigmoid', distance_function='l1')
            
            # Training loop
            best_val_acc = 0
            best_epoch = 0
            
            print(f"    Training for {self.num_epochs} epochs...")
            
            for epoch in range(self.num_epochs):
                # Train
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for x1, x2, y in train_loader:
                    optimizer.zero_grad()
                    out = self.model(x1, x2)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * len(y)
                    train_total += len(y)
                    
                    # Calculate accuracy
                    out1, out2 = out
                    distance = lam_distance(out1, out2, 'sigmoid')
                    predictions = (distance >= 0.8).float()
                    train_correct += (predictions == y.squeeze()).sum().item()
                
                train_loss /= train_total
                train_acc = train_correct / train_total * 100
                
                # Validate
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for x1, x2, y in val_loader:
                        out = self.model(x1, x2)
                        loss = criterion(out, y)
                        
                        val_loss += loss.item() * len(y)
                        val_total += len(y)
                        
                        out1, out2 = out
                        distance = lam_distance(out1, out2, 'sigmoid')
                        predictions = (distance >= 0.8).float()
                        val_correct += (predictions == y.squeeze()).sum().item()
                
                val_loss /= val_total
                val_acc = val_correct / val_total * 100
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                
                if verbose and epoch % 10 == 0:
                    print(f"    Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}%")
            
            print(f"    Training completed! Best validation accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"    ERROR: Paper-Style PMM training failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
            return False
    
    def recommend(self, dataset_metafeatures, top_k=5):
        """
        Recommend top pipelines for a new dataset.
        
        Args:
            dataset_metafeatures: Metafeatures array for the dataset (can be array or dataset_id will be ignored)
            top_k: Number of top pipelines to return
        
        Returns:
            List of top pipeline names
        """
        if not self.is_trained:
            print("    ERROR: Model not trained")
            return ['baseline']
        
        try:
            # Handle if dataset_metafeatures is passed as array directly
            if isinstance(dataset_metafeatures, np.ndarray):
                dataset_feats = dataset_metafeatures
            else:
                # Assume it's a pandas Series or similar
                dataset_feats = np.array(dataset_metafeatures)
            
            # CRITICAL: Impute NaN values first, then normalize
            dataset_feats_imputed = self.imputer.transform(dataset_feats.reshape(1, -1))
            dataset_feats_normalized = self.dataset_scaler.transform(dataset_feats_imputed).reshape(-1)
            
            # Predict performance for all pipelines using normalized features
            scores = self.model.predict_proba(dataset_feats_normalized, self.pipeline_features)
            
            # Create scores dictionary
            scores_dict = {self.pipeline_names[i]: scores[i] for i in range(len(scores))}
            
            # Get top N pipelines
            sorted_pipelines = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            top_n = [p[0] for p in sorted_pipelines[:top_k]]
            
            print(f"    Paper-Style PMM recommends: {top_n[0]} (score: {scores_dict[top_n[0]]:.2f})")
            print(f"    Top-{top_k}: {top_n}")
            
            return top_n
            
        except Exception as e:
            print(f"    ERROR: Paper-Style PMM recommendation failed: {e}")
            import traceback
            traceback.print_exc()
            return ['baseline']


if __name__ == "__main__":
    print("Paper-Style PMM Recommender Module")
    print("This implements the PMM architecture from the paper:")
    print("- Uses (dataset + pipeline) metafeature pairs")
    print("- Predicts performance scores via Siamese network")
    print("- Learns rankings through contrastive loss")
    print("- Supports influence weighting for informative datasets")
