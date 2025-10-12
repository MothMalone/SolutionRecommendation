import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch is not available. Siamese network recommender will be disabled.")

# Import enhanced metafeatures extractor
try:
    from enhanced_metafeatures import EnhancedMetaFeatureExtractor
    def extract_enhanced_metafeatures(dataset):
        extractor = EnhancedMetaFeatureExtractor()
        return extractor.extract_metafeatures(dataset)
except ImportError:
    # Fallback to simple extraction if enhanced version is not available
    def extract_enhanced_metafeatures(dataset):
        # Basic meta-feature extraction
        X = dataset.get('X', None)
        y = dataset.get('y', None)
        
        if X is None or y is None:
            return {}
            
        features = {}
        features['n_samples'] = X.shape[0]
        features['n_features'] = X.shape[1]
        
        try:
            features['n_classes'] = len(np.unique(y))
        except:
            features['n_classes'] = 2  # Default for binary classification
            
        return features

class EnhancedPreprocessingRecommender:
    def __init__(self, performance_matrix, metafeatures_df, pipeline_configs, use_pmf=True, use_siamese=True):
        self.performance_matrix = performance_matrix.astype(float)
        self.metafeatures_df = metafeatures_df
        self.pipeline_configs = pipeline_configs
        self.use_pmf = use_pmf
        self.use_siamese = use_siamese and TORCH_AVAILABLE
        
        if len(metafeatures_df) == 0:
            print("Warning: No meta-features available. Using fallback recommender.")
            self.use_pmf = False
            self.use_siamese = False
            return
        
        # Prepare meta-features for similarity search
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.metafeatures_imputed = self.imputer.fit_transform(metafeatures_df)
        self.metafeatures_scaled = self.scaler.transform(self.metafeatures_imputed)
        
        # Initialize recommenders
        self.pmf_recommender = None
        self.siamese_recommender = None
        
        # Initialize PMF recommender if enabled
        if self.use_pmf:
            try:
                from pmf_recommender import PMFEnsembleRecommender
                print("Initializing PMF-based recommender...")
                self.pmf_recommender = PMFEnsembleRecommender(
                    latent_dim=min(5, min(len(performance_matrix), len(performance_matrix.columns))),
                    max_iter=50,
                    lr=0.01,
                    verbose=True
                )
                self.pmf_recommender.fit(performance_matrix, metafeatures_df)
                print("PMF recommender trained successfully!")
            except Exception as e:
                print(f"Failed to initialize PMF recommender: {e}")
                self.pmf_recommender = None
                self.use_pmf = False
        
        # Initialize Siamese recommender if enabled
        if self.use_siamese:
            try:
                from siamese_recommender import SiameseSimilarityRecommender
                print("Initializing Siamese network recommender...")
                self.siamese_recommender = SiameseSimilarityRecommender(
                    hidden_dim=64,
                    embedding_dim=32,
                    num_epochs=50,  # Reduced for faster training
                    verbose=True
                )
                self.siamese_recommender.fit(performance_matrix, metafeatures_df)
                print("Siamese recommender trained successfully!")
            except Exception as e:
                print(f"Failed to initialize Siamese recommender: {e}")
                self.siamese_recommender = None
                self.use_siamese = False
        
        # Fit nearest neighbors model for traditional approach
        k_neighbors = min(len(metafeatures_df), 5)
        if k_neighbors > 0:
            self.nn_euclidean = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
            self.nn_euclidean.fit(self.metafeatures_scaled)
            
            self.nn_manhattan = NearestNeighbors(n_neighbors=k_neighbors, metric='manhattan')
            self.nn_manhattan.fit(self.metafeatures_scaled)
        else:
            self.nn_euclidean = None
            self.nn_manhattan = None
    
    def recommend(self, new_dataset, k=3, method='ensemble'):
        """
        Recommend preprocessing pipeline for a new dataset
        
        Args:
            new_dataset: dataset dict with 'X', 'y', etc.
            k: number of similar datasets to consider for k-NN
            method: 'ensemble', 'pmf', 'siamese', 'knn' - which recommendation method to use
        
        Returns:
            dict with detailed recommendation information
        """
        # Extract meta-features for new dataset
        new_metafeatures = extract_enhanced_metafeatures(new_dataset)
        
        if not new_metafeatures:
            return self._get_default_recommendation()
        
        # Choose recommendation method
        if method == 'pmf' and self.use_pmf and self.pmf_recommender is not None:
            return self._recommend_pmf(new_metafeatures)
        elif method == 'siamese' and self.use_siamese and self.siamese_recommender is not None:
            return self._recommend_siamese(new_metafeatures)
        elif method == 'knn' or (not self.use_pmf and not self.use_siamese):
            return self._recommend_knn(new_dataset, k)
        else:
            # Ensemble: combine all available recommenders
            return self._recommend_ensemble(new_dataset, new_metafeatures, k)
    
    def _recommend_pmf(self, new_metafeatures):
        """PMF-based recommendation"""
        try:
            # Get PMF recommendation
            recommended_indices = self.pmf_recommender.recommend(new_metafeatures)
            
            # Return top recommendation
            if recommended_indices:
                top_pipeline_idx = recommended_indices[0]
                if top_pipeline_idx < len(self.pipeline_configs):
                    pipeline_config = self.pipeline_configs[top_pipeline_idx]
                    return {
                        'pipeline_config': pipeline_config,
                        'pipeline_name': pipeline_config['name'],
                        'expected_performance': 0.8,  # Placeholder
                        'confidence': 'high',
                        'method': 'PMF',
                        'pipeline_ranking': [self.pipeline_configs[i]['name'] for i in recommended_indices[:5] 
                                            if i < len(self.pipeline_configs)],
                        'similar_datasets': []
                    }
            
            return self._get_default_recommendation()
            
        except Exception as e:
            print(f"Error in PMF recommendation: {e}")
            return self._get_default_recommendation()
    
    def _recommend_siamese(self, new_metafeatures):
        """Siamese network based recommendation"""
        try:
            # Get Siamese recommendation
            result = self.siamese_recommender.recommend(
                new_metafeatures, self.performance_matrix, k=5
            )
            
            if result and result['pipeline']:
                pipeline_name = result['pipeline']
                pipeline_config = next(
                    (config for config in self.pipeline_configs if config['name'] == pipeline_name),
                    self.pipeline_configs[0]
                )
                
                return {
                    'pipeline_config': pipeline_config,
                    'pipeline_name': pipeline_name,
                    'expected_performance': result['performance_scores'].get(pipeline_name, 0.7),
                    'confidence': 'high',
                    'method': 'Siamese',
                    'pipeline_ranking': result['pipeline_ranking'][:5],
                    'similar_datasets': result['similar_datasets'],
                    'similarity_scores': result['similarity_scores']
                }
            
            return self._get_default_recommendation()
            
        except Exception as e:
            print(f"Error in Siamese recommendation: {e}")
            return self._get_default_recommendation()
    
    def _recommend_knn(self, new_dataset, k=3):
        """Traditional k-NN based recommendation"""
        if self.nn_euclidean is None or len(self.metafeatures_df) == 0:
            return self._get_default_recommendation()
        
        # Extract meta-features for new dataset
        new_metafeatures = extract_enhanced_metafeatures(new_dataset)
        
        if not new_metafeatures:
            return self._get_default_recommendation()
        
        # Convert to DataFrame and align columns
        new_mf_df = pd.DataFrame([new_metafeatures])
        new_mf_df = new_mf_df.reindex(columns=self.metafeatures_df.columns, fill_value=0)
        
        # Impute and scale
        new_mf_imputed = self.imputer.transform(new_mf_df)
        new_mf_scaled = self.scaler.transform(new_mf_imputed)
        
        # Find nearest neighbors
        k = min(k, len(self.metafeatures_df))
        
        distances_euc, indices_euc = self.nn_euclidean.kneighbors(new_mf_scaled, n_neighbors=k)
        similar_datasets_euc = self.metafeatures_df.iloc[indices_euc[0]].index.tolist()
        
        distances_man, indices_man = self.nn_manhattan.kneighbors(new_mf_scaled, n_neighbors=k)
        similar_datasets_man = self.metafeatures_df.iloc[indices_man[0]].index.tolist()
        
        all_similar = list(set(similar_datasets_euc + similar_datasets_man))
        
        # Weight by distance
        weighted_performances = {}
        for pipeline in self.performance_matrix.index:
            total_weighted_perf = 0
            total_weight = 0
            
            # Euclidean weights
            for i, dataset in enumerate(similar_datasets_euc):
                if dataset in self.performance_matrix.columns:
                    perf = self.performance_matrix.loc[pipeline, dataset]
                    if not pd.isna(perf):
                        weight = 1 / (1 + distances_euc[0][i])
                        total_weighted_perf += perf * weight
                        total_weight += weight
            
            # Manhattan weights  
            for i, dataset in enumerate(similar_datasets_man):
                if dataset in self.performance_matrix.columns:
                    perf = self.performance_matrix.loc[pipeline, dataset]
                    if not pd.isna(perf):
                        weight = 1 / (1 + distances_man[0][i])
                        total_weighted_perf += perf * weight * 0.5  # Lower weight for Manhattan
                        total_weight += weight * 0.5
            
            if total_weight > 0:
                weighted_performances[pipeline] = total_weighted_perf / total_weight
            else:
                weighted_performances[pipeline] = 0
        
        if not weighted_performances:
            return self._get_default_recommendation()
        
        # Get top pipeline
        best_pipeline_name = max(weighted_performances.items(), key=lambda x: x[1])[0]
        best_pipeline_config = next((config for config in self.pipeline_configs 
                                   if config['name'] == best_pipeline_name), 
                                  self.pipeline_configs[0])
        best_performance = weighted_performances[best_pipeline_name]
        
        # Get pipeline ranking
        pipeline_ranking = sorted(weighted_performances.keys(), key=lambda x: weighted_performances[x], reverse=True)
        
        # Determine confidence based on distance
        avg_distance = (np.mean(distances_euc[0]) + np.mean(distances_man[0])) / 2
        confidence = 'high' if avg_distance < 0.5 else 'medium' if avg_distance < 1.5 else 'low'
        
        return {
            'pipeline_config': best_pipeline_config,
            'pipeline_name': best_pipeline_name,
            'expected_performance': best_performance,
            'confidence': confidence,
            'method': 'k-NN',
            'pipeline_ranking': pipeline_ranking[:5],
            'similar_datasets': all_similar[:k],
            'similarity_scores': {
                'euclidean_distance': np.mean(distances_euc[0]),
                'manhattan_distance': np.mean(distances_man[0])
            }
        }
    
    def _recommend_ensemble(self, new_dataset, new_metafeatures, k=3):
        """Ensemble recommendation combining all available recommenders"""
        try:
            recommendations = []
            weights = {}
            
            # Get recommendations from each available method
            if self.use_pmf and self.pmf_recommender is not None:
                pmf_result = self._recommend_pmf(new_metafeatures)
                if pmf_result:
                    recommendations.append(pmf_result)
                    weights['PMF'] = 0.35
            
            if self.use_siamese and self.siamese_recommender is not None:
                siamese_result = self._recommend_siamese(new_metafeatures)
                if siamese_result:
                    recommendations.append(siamese_result)
                    weights['Siamese'] = 0.35
            
            knn_result = self._recommend_knn(new_dataset, k)
            if knn_result:
                recommendations.append(knn_result)
                weights['k-NN'] = 0.3
            
            if not recommendations:
                return self._get_default_recommendation()
            
            # Vote for pipelines
            pipeline_votes = {}
            for rec in recommendations:
                method = rec['method']
                ranking = rec['pipeline_ranking']
                weight = weights.get(method, 1.0 / len(recommendations))
                
                for i, pipeline in enumerate(ranking):
                    # Weigh by position and method weight
                    position_weight = 1.0 / (i + 1)
                    vote = weight * position_weight
                    
                    if pipeline in pipeline_votes:
                        pipeline_votes[pipeline] += vote
                    else:
                        pipeline_votes[pipeline] = vote
            
            # Get top pipeline
            top_pipeline = max(pipeline_votes.items(), key=lambda x: x[1])[0]
            top_pipeline_config = next((config for config in self.pipeline_configs 
                                      if config['name'] == top_pipeline),
                                     self.pipeline_configs[0])
            
            # Get weighted confidence
            confidence_map = {'high': 2, 'medium': 1, 'low': 0}
            confidence_score = 0
            total_weight = 0
            
            for rec in recommendations:
                if rec['pipeline_name'] == top_pipeline:
                    method = rec['method']
                    weight = weights.get(method, 1.0 / len(recommendations))
                    confidence_score += confidence_map[rec['confidence']] * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_confidence_score = confidence_score / total_weight
                confidence = 'high' if avg_confidence_score > 1.5 else 'medium' if avg_confidence_score > 0.75 else 'low'
            else:
                confidence = 'medium'
            
            # Get expected performance
            performance_votes = [rec['expected_performance'] for rec in recommendations 
                               if rec['pipeline_name'] == top_pipeline]
            expected_performance = np.mean(performance_votes) if performance_votes else 0.7
            
            # Gather similar datasets from all methods
            similar_datasets = set()
            for rec in recommendations:
                if 'similar_datasets' in rec:
                    similar_datasets.update(rec['similar_datasets'])
            
            # Get pipeline ranking
            pipeline_ranking = sorted(pipeline_votes.keys(), key=lambda x: pipeline_votes[x], reverse=True)
            
            return {
                'pipeline_config': top_pipeline_config,
                'pipeline_name': top_pipeline,
                'expected_performance': expected_performance,
                'confidence': confidence,
                'method': 'Ensemble',
                'pipeline_ranking': pipeline_ranking[:5],
                'similar_datasets': list(similar_datasets)[:5],
                'components': [rec['method'] for rec in recommendations]
            }
                
        except Exception as e:
            print(f"Error in ensemble recommendation: {e}")
            # Fall back to k-NN as it's the most reliable
            return self._recommend_knn(new_dataset, k)
    
    def _get_default_recommendation(self):
        """Return default recommendation when others fail"""
        # Find the best performing pipeline on average
        if not self.performance_matrix.empty:
            avg_performance = self.performance_matrix.mean(axis=1, skipna=True).dropna()
            if len(avg_performance) > 0:
                best_pipeline_name = avg_performance.idxmax()
                best_pipeline_config = next((config for config in self.pipeline_configs 
                                          if config['name'] == best_pipeline_name),
                                         self.pipeline_configs[0])
                
                return {
                    'pipeline_config': best_pipeline_config,
                    'pipeline_name': best_pipeline_name,
                    'expected_performance': avg_performance.max(),
                    'confidence': 'low',
                    'method': 'Default (average performance)',
                    'pipeline_ranking': avg_performance.sort_values(ascending=False).index.tolist()[:5],
                    'similar_datasets': []
                }
        
        # Last resort fallback
        return {
            'pipeline_config': self.pipeline_configs[0],
            'pipeline_name': self.pipeline_configs[0]['name'],
            'expected_performance': 0.7,
            'confidence': 'low',
            'method': 'Fallback',
            'pipeline_ranking': [config['name'] for config in self.pipeline_configs[:5]],
            'similar_datasets': []
        }