"""
Siamese Network for Learning Dataset Similarity
Implements contrastive learning to create a better similarity metric between datasets
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
warnings.filterwarnings('ignore')

class SiameseNetwork(nn.Module):
    """
    Siamese network architecture for learning similarity between datasets
    """
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=32):
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

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese network
    Pushes similar pairs closer and dissimilar pairs further apart
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        Args:
            output1, output2: embeddings from the siamese network
            label: 1 for similar pairs, 0 for dissimilar pairs
        """
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # If label=1 (similar), minimize distance
        # If label=0 (dissimilar), maximize distance up to margin
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss

class SiameseDataset(Dataset):
    """Dataset for Siamese network training"""
    def __init__(self, metafeatures_df, performance_matrix, similarity_threshold=0.8):
        """
        Args:
            metafeatures_df: DataFrame with meta-features
            performance_matrix: DataFrame with pipeline performances
            similarity_threshold: Datasets with performance correlation above this are considered similar
        """
        self.metafeatures = metafeatures_df.values
        self.dataset_names = metafeatures_df.index.tolist()
        
        # Calculate performance similarity matrix
        perf_matrix = performance_matrix.values.T  # (n_datasets, n_pipelines)
        self.n_datasets = perf_matrix.shape[0]
        
        # Handle NaN values for correlation calculation
        masked_perf = np.ma.masked_invalid(perf_matrix)
        
        # Calculate pairwise correlations between datasets
        self.similarity_matrix = np.zeros((self.n_datasets, self.n_datasets))
        for i in range(self.n_datasets):
            for j in range(i+1, self.n_datasets):
                # Use only pipelines where both datasets have values
                mask = ~np.ma.getmask(masked_perf[i]) & ~np.ma.getmask(masked_perf[j])
                if np.sum(mask) > 3:  # Need at least 3 common pipelines
                    corr = np.corrcoef(masked_perf[i][mask], masked_perf[j][mask])[0, 1]
                    self.similarity_matrix[i, j] = corr
                    self.similarity_matrix[j, i] = corr
        
        # Create dataset pairs with labels
        self.pairs = []
        self.labels = []
        
        for i in range(self.n_datasets):
            for j in range(i+1, self.n_datasets):
                similarity = self.similarity_matrix[i, j]
                if not np.isnan(similarity):
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

class SiameseSimilarityRecommender:
    """
    Recommender system using Siamese network to learn dataset similarity
    """
    def __init__(self, hidden_dim=64, embedding_dim=32, margin=1.0, 
                 batch_size=32, num_epochs=100, learning_rate=0.001, 
                 similarity_threshold=0.8, verbose=True):
        """
        Args:
            hidden_dim: Dimension of hidden layer
            embedding_dim: Dimension of embedding layer
            margin: Margin for contrastive loss
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            similarity_threshold: Threshold for considering datasets similar
            verbose: Whether to print training progress
        """
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose
        self.scaler = StandardScaler()
        
        # Will be initialized during training
        self.model = None
        self.dataset_embeddings = None
        self.is_trained = False
        
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
            
            # Scale meta-features
            self.metafeatures_df = metafeatures_df.copy()
            metafeatures_values = self.scaler.fit_transform(metafeatures_df.values)
            scaled_metafeatures_df = pd.DataFrame(
                metafeatures_values, index=metafeatures_df.index, columns=metafeatures_df.columns
            )
            
            # Create dataset
            siamese_dataset = SiameseDataset(
                scaled_metafeatures_df, performance_matrix, 
                similarity_threshold=self.similarity_threshold
            )
            
            # Split into train and validation sets
            train_size = int(0.8 * len(siamese_dataset))
            val_size = len(siamese_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                siamese_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            
            # Initialize model
            input_dim = metafeatures_df.shape[1]
            self.model = SiameseNetwork(
                input_dim=input_dim, 
                hidden_dim=self.hidden_dim, 
                embedding_dim=self.embedding_dim
            )
            
            # Loss function and optimizer
            criterion = ContrastiveLoss(margin=self.margin)
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
                
                # Print progress
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Compute and store embeddings for all datasets
            self.model.eval()
            self.dataset_embeddings = {}
            
            with torch.no_grad():
                for idx, dataset_name in enumerate(self.dataset_names):
                    if dataset_name in metafeatures_df.index:
                        metafeatures = torch.FloatTensor(scaled_metafeatures_df.loc[dataset_name].values)
                        embedding = self.model.get_embedding(metafeatures.unsqueeze(0)).squeeze().numpy()
                        self.dataset_embeddings[dataset_name] = embedding
            
            self.is_trained = True
            
            if self.verbose:
                print("Siamese network training completed!")
                
        except Exception as e:
            print(f"Error training Siamese network: {e}")
            self.is_trained = False
    
    def recommend(self, new_dataset_metafeatures, performance_matrix, k=5):
        """
        Recommend pipelines for a new dataset using learned similarity
        
        Args:
            new_dataset_metafeatures: Dict of meta-features for new dataset
            performance_matrix: DataFrame with pipeline performances
            k: Number of similar datasets to consider
            
        Returns:
            List of recommended pipeline indices in order of predicted performance
        """
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Convert meta-features to DataFrame format
            new_mf_df = pd.DataFrame([new_dataset_metafeatures])
            
            # Align columns with training data
            new_mf_df = new_mf_df.reindex(columns=self.metafeatures_df.columns, fill_value=0)
            
            # Scale using the same scaler
            new_mf_scaled = self.scaler.transform(new_mf_df.values)
            
            # Get embedding for new dataset
            self.model.eval()
            with torch.no_grad():
                new_embedding = self.model.get_embedding(
                    torch.FloatTensor(new_mf_scaled)
                ).squeeze().numpy()
            
            # Calculate similarity to all training datasets
            similarities = {}
            for dataset_name, embedding in self.dataset_embeddings.items():
                # Cosine similarity
                similarity = np.dot(new_embedding, embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(embedding)
                )
                similarities[dataset_name] = similarity
            
            # Get k most similar datasets
            most_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
            similar_datasets = [d[0] for d in most_similar]
            
            # Weight pipelines by similarity and performance
            weighted_performances = {}
            for pipeline in performance_matrix.index:
                weighted_sum = 0
                weight_sum = 0
                
                for dataset, similarity in most_similar:
                    if dataset in performance_matrix.columns:
                        perf = performance_matrix.loc[pipeline, dataset]
                        if not pd.isna(perf):
                            weight = max(0, similarity)  # Ensure non-negative weight
                            weighted_sum += perf * weight
                            weight_sum += weight
                
                if weight_sum > 0:
                    weighted_performances[pipeline] = weighted_sum / weight_sum
                else:
                    weighted_performances[pipeline] = 0
            
            # Rank pipelines by weighted performance
            ranked_pipelines = sorted(
                weighted_performances.items(), key=lambda x: x[1], reverse=True
            )
            
            return {
                'pipeline': ranked_pipelines[0][0] if ranked_pipelines else None,
                'pipeline_ranking': [p[0] for p in ranked_pipelines],
                'performance_scores': {p[0]: p[1] for p in ranked_pipelines},
                'similar_datasets': similar_datasets,
                'similarity_scores': {d: s for d, s in most_similar}
            }
            
        except Exception as e:
            print(f"Error in Siamese recommendation: {e}")
            return None