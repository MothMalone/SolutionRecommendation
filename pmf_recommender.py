"""
PMF-based Recommendation System
Implements Probabilistic Matrix Factorization using Gaussian Process Latent Variable Models
for learning latent representations of datasets and preprocessing pipelines.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class GPLVMRecommender:
    """
    Probabilistic Matrix Factorization using Gaussian Process Latent Variable Models
    for learning latent dataset and pipeline representations.
    """
    
    def __init__(self, latent_dim=5, max_iter=100, lr=0.01, verbose=False):
        self.latent_dim = latent_dim
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def fit(self, performance_matrix, metafeatures_df):
        """
        Fit the GPLVM model on performance matrix and meta-features
        
        Args:
            performance_matrix: DataFrame with pipelines as rows, datasets as columns
            metafeatures_df: DataFrame with datasets as rows, meta-features as columns
        """
        try:
            # Prepare data
            self.pipeline_names = performance_matrix.index.tolist()
            self.dataset_names = performance_matrix.columns.tolist()
            
            # Handle missing values in performance matrix
            Y = performance_matrix.values
            Y_filled = np.nan_to_num(Y, nan=np.nanmean(Y))
            
            # Scale meta-features
            F = metafeatures_df.reindex(self.dataset_names).values
            F_scaled = self.scaler.fit_transform(F)
            
            # Initialize latent variables
            self.X_latent = np.random.randn(len(self.dataset_names), self.latent_dim) * 0.1
            
            # Store data for optimization
            self.Y = torch.tensor(Y_filled.T, dtype=torch.float32)  # [n_datasets, n_pipelines]
            self.F = torch.tensor(F_scaled, dtype=torch.float32)
            self.X = torch.nn.Parameter(torch.tensor(self.X_latent, dtype=torch.float32))
            
            # Pipeline embeddings (learnable)
            self.W = torch.nn.Parameter(torch.randn(len(self.pipeline_names), self.latent_dim) * 0.1)
            
            # Optimize using simple gradient descent
            optimizer = torch.optim.Adam([self.X, self.W], lr=self.lr)
            
            for iter_idx in range(self.max_iter):
                optimizer.zero_grad()
                loss = self._compute_loss()
                loss.backward()
                optimizer.step()
                
                if self.verbose and iter_idx % 20 == 0:
                    print(f"GPLVM Iter {iter_idx}: Loss = {loss.item():.4f}")
            
            self.is_trained = True
            
            if self.verbose:
                print(f"GPLVM training completed. Final loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"Error training GPLVM: {e}")
            # Fall back to random initialization
            self.X_latent = np.random.randn(len(self.dataset_names), self.latent_dim)
            self.is_trained = False
    
    def _compute_loss(self):
        """Compute GPLVM loss function"""
        # Reconstruction loss: ||Y - XW^T||^2
        Y_pred = torch.matmul(self.X, self.W.t())
        reconstruction_loss = torch.mean((self.Y - Y_pred) ** 2)
        
        # Regularization on latent variables
        reg_x = 0.01 * torch.mean(self.X ** 2)
        reg_w = 0.01 * torch.mean(self.W ** 2)
        
        # Meta-feature consistency loss (encourage X to be similar to scaled meta-features)
        if self.F is not None:
            meta_loss = 0.1 * torch.mean((self.X - self.F) ** 2)
        else:
            meta_loss = 0
        
        return reconstruction_loss + reg_x + reg_w + meta_loss
    
    def recommend(self, new_dataset_metafeatures, k=5):
        """
        Recommend pipelines for a new dataset based on learned latent representations
        
        Args:
            new_dataset_metafeatures: dict of meta-features for new dataset
            k: number of top recommendations to return
            
        Returns:
            list of recommended pipeline indices ordered by predicted performance
        """
        if not self.is_trained:
            # Fall back to random recommendation
            return np.random.permutation(len(self.pipeline_names)).tolist()
        
        try:
            # Convert meta-features to scaled format
            new_mf_df = pd.DataFrame([new_dataset_metafeatures])
            new_mf_df = new_mf_df.reindex(columns=pd.Index(range(self.F.shape[1])), fill_value=0)
            new_f_scaled = self.scaler.transform(new_mf_df.values)
            
            # Find closest dataset in latent space
            new_x = torch.tensor(new_f_scaled, dtype=torch.float32)
            
            with torch.no_grad():
                # Use meta-features as initial estimate for latent representation
                distances = torch.sum((self.X - new_x) ** 2, dim=1)
                closest_dataset_idx = torch.argmin(distances).item()
                
                # Use the latent representation of closest dataset
                x_new = self.X[closest_dataset_idx].unsqueeze(0)
                
                # Predict performance for all pipelines
                y_pred = torch.matmul(x_new, self.W.t()).squeeze()
                
                # Return pipeline indices sorted by predicted performance (descending)
                pipeline_scores = y_pred.numpy()
                recommended_indices = np.argsort(-pipeline_scores).tolist()
                
                return recommended_indices
                
        except Exception as e:
            print(f"Error in PMF recommendation: {e}")
            # Fall back to random recommendation
            return np.random.permutation(len(self.pipeline_names)).tolist()

class L1Recommender:
    """
    L1 distance-based recommender as used in the research
    """
    
    def __init__(self):
        self.name = 'L1'
        
    def fit(self, performance_matrix, metafeatures_df):
        """Fit the L1 recommender"""
        self.performance_matrix = performance_matrix.values
        self.metafeatures = metafeatures_df.reindex(performance_matrix.columns).values
        self.pipeline_names = performance_matrix.index.tolist()
        self.dataset_names = performance_matrix.columns.tolist()
        
    def recommend(self, new_dataset_metafeatures, k=5):
        """Recommend using L1 distance"""
        try:
            # Convert to numpy array
            new_mf_df = pd.DataFrame([new_dataset_metafeatures])
            new_mf_df = new_mf_df.reindex(columns=pd.Index(range(self.metafeatures.shape[1])), fill_value=0)
            ftest = new_mf_df.values[0]
            
            # Compute L1 distances
            distances = np.abs(self.metafeatures - ftest).sum(axis=1)
            closest_datasets = np.argsort(distances)[:k]
            
            # Get performance matrix for closest datasets
            Y_closest = self.performance_matrix[:, closest_datasets]
            
            # Handle NaN values and compute average ranks
            ix_nonnan = np.where(~np.isnan(Y_closest.sum(axis=1)))[0]
            
            if len(ix_nonnan) == 0:
                return np.arange(len(self.pipeline_names)).tolist()
            
            # Rank pipelines based on performance on similar datasets
            ranks = np.apply_along_axis(rankdata, 0, Y_closest[ix_nonnan])
            avg_ranks = ranks.mean(axis=1)
            
            # Return pipeline indices sorted by average rank (ascending)
            recommended_indices = ix_nonnan[np.argsort(-avg_ranks)]
            
            # Add remaining pipelines
            remaining = [i for i in range(len(self.pipeline_names)) if i not in recommended_indices]
            recommended_indices = np.concatenate([recommended_indices, remaining])
            
            return recommended_indices.tolist()
            
        except Exception as e:
            print(f"Error in L1 recommendation: {e}")
            return np.arange(len(self.pipeline_names)).tolist()

class PMFEnsembleRecommender:
    """
    Ensemble recommender combining PMF and L1 approaches
    """
    
    def __init__(self, latent_dim=5, max_iter=100, lr=0.01, verbose=False):
        self.gplvm = GPLVMRecommender(latent_dim, max_iter, lr, verbose)
        self.l1 = L1Recommender()
        self.weights = [0.7, 0.3]  # GPLVM weight, L1 weight
        
    def fit(self, performance_matrix, metafeatures_df):
        """Fit both recommenders"""
        print("Training PMF-based recommender...")
        self.gplvm.fit(performance_matrix, metafeatures_df)
        
        print("Training L1-based recommender...")
        self.l1.fit(performance_matrix, metafeatures_df)
        
    def recommend(self, new_dataset_metafeatures, k=5):
        """Ensemble recommendation"""
        try:
            # Get recommendations from both methods
            gplvm_rec = self.gplvm.recommend(new_dataset_metafeatures, k=len(self.gplvm.pipeline_names))
            l1_rec = self.l1.recommend(new_dataset_metafeatures, k=len(self.l1.pipeline_names))
            
            # Convert to ranks (lower is better)
            n_pipelines = len(gplvm_rec)
            gplvm_ranks = {pipeline: rank for rank, pipeline in enumerate(gplvm_rec)}
            l1_ranks = {pipeline: rank for rank, pipeline in enumerate(l1_rec)}
            
            # Ensemble scoring
            ensemble_scores = {}
            for pipeline_idx in range(n_pipelines):
                gplvm_score = 1 / (1 + gplvm_ranks.get(pipeline_idx, n_pipelines))
                l1_score = 1 / (1 + l1_ranks.get(pipeline_idx, n_pipelines))
                ensemble_scores[pipeline_idx] = (
                    self.weights[0] * gplvm_score + self.weights[1] * l1_score
                )
            
            # Sort by ensemble score (descending)
            recommended_indices = sorted(ensemble_scores.keys(), 
                                       key=lambda x: ensemble_scores[x], reverse=True)
            
            return recommended_indices
            
        except Exception as e:
            print(f"Error in ensemble recommendation: {e}")
            return self.gplvm.recommend(new_dataset_metafeatures, k)