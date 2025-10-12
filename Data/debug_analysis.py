"""
Deep Debugging and Analysis System for Pipeline Recommenders

This module provides comprehensive debugging, profiling, and visualization
tools to understand recommender behavior in depth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DatasetProfiler:
    """Profiles a dataset to understand its characteristics"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset['X']
        self.y = dataset['y']
        self.profile = {}
        
    def analyze(self):
        """Perform comprehensive dataset analysis"""
        print(f"\n{'='*80}")
        print(f"DEEP PROFILE: Dataset {self.dataset['name']} (ID: {self.dataset['id']})")
        print(f"{'='*80}")
        
        # Basic statistics
        self.profile['n_samples'] = len(self.X)
        self.profile['n_features'] = self.X.shape[1]
        self.profile['n_classes'] = self.y.nunique()
        
        # Feature types
        numeric_features = self.X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X.select_dtypes(exclude=np.number).columns.tolist()
        self.profile['n_numeric'] = len(numeric_features)
        self.profile['n_categorical'] = len(categorical_features)
        
        # Class distribution
        class_counts = self.y.value_counts()
        self.profile['class_balance'] = class_counts.std() / class_counts.mean()
        self.profile['minority_class_ratio'] = class_counts.min() / class_counts.sum()
        
        # Missing values
        self.profile['missing_ratio'] = self.X.isna().sum().sum() / (self.X.shape[0] * self.X.shape[1])
        
        # Dimensionality
        self.profile['dimensionality'] = self.X.shape[1] / self.X.shape[0]
        
        # Feature statistics for numeric features
        if numeric_features:
            numeric_data = self.X[numeric_features]
            self.profile['feature_variance_mean'] = numeric_data.var().mean()
            self.profile['feature_skewness_mean'] = numeric_data.skew().mean()
            self.profile['feature_correlation_mean'] = numeric_data.corr().abs().mean().mean()
        
        # Print profile
        print(f"\n📊 Basic Statistics:")
        print(f"   Samples: {self.profile['n_samples']:,}")
        print(f"   Features: {self.profile['n_features']} ({self.profile['n_numeric']} numeric, {self.profile['n_categorical']} categorical)")
        print(f"   Classes: {self.profile['n_classes']}")
        print(f"   Dimensionality: {self.profile['dimensionality']:.4f} (features/samples)")
        
        print(f"\n⚖️ Class Distribution:")
        print(f"   Balance (std/mean): {self.profile['class_balance']:.4f}")
        print(f"   Minority class ratio: {self.profile['minority_class_ratio']:.4f}")
        for cls, count in class_counts.items():
            print(f"   Class {cls}: {count:,} ({count/len(self.y)*100:.1f}%)")
        
        print(f"\n🔍 Data Quality:")
        print(f"   Missing values: {self.profile['missing_ratio']*100:.2f}%")
        
        if numeric_features:
            print(f"\n📈 Feature Characteristics:")
            print(f"   Mean variance: {self.profile['feature_variance_mean']:.4f}")
            print(f"   Mean skewness: {self.profile['feature_skewness_mean']:.4f}")
            print(f"   Mean correlation: {self.profile['feature_correlation_mean']:.4f}")
        
        # Categorize dataset complexity
        self.profile['complexity'] = self._assess_complexity()
        print(f"\n🎯 Dataset Complexity: {self.profile['complexity']}")
        
        return self.profile
    
    def _assess_complexity(self):
        """Assess overall dataset complexity"""
        score = 0
        
        # High dimensionality increases complexity
        if self.profile['dimensionality'] > 0.1:
            score += 2
        elif self.profile['dimensionality'] > 0.05:
            score += 1
        
        # Class imbalance increases complexity
        if self.profile['class_balance'] > 1.0:
            score += 2
        elif self.profile['class_balance'] > 0.5:
            score += 1
        
        # Many classes increases complexity
        if self.profile['n_classes'] > 10:
            score += 2
        elif self.profile['n_classes'] > 5:
            score += 1
        
        # Missing data increases complexity
        if self.profile['missing_ratio'] > 0.1:
            score += 2
        elif self.profile['missing_ratio'] > 0.05:
            score += 1
        
        if score >= 6:
            return "Very Complex"
        elif score >= 4:
            return "Complex"
        elif score >= 2:
            return "Moderate"
        else:
            return "Simple"


class RecommenderDebugger:
    """Debug and analyze recommender decisions in detail"""
    
    def __init__(self, recommender, recommender_name, performance_matrix, metafeatures_df):
        self.recommender = recommender
        self.recommender_name = recommender_name
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.debug_log = []
        
    def debug_recommendation(self, dataset, k=5):
        """Deep dive into a single recommendation"""
        print(f"\n{'='*80}")
        print(f"🔬 DEBUGGING RECOMMENDATION FOR: {dataset['name']}")
        print(f"{'='*80}")
        
        # Get recommendation
        if self.recommender_name in ['pmm', 'balancedpmm']:
            result = self.recommender.recommend(
                dataset['id'], 
                performance_matrix=self.performance_matrix,
                k=k
            )
        elif self.recommender_name == 'hybrid':
            result = self.recommender.recommend(dataset['id'])
        else:
            result = None
        
        if result is None or (isinstance(result, dict) and not result.get('pipeline')):
            print("❌ Recommendation failed!")
            return None
        
        # Extract recommendation details
        if isinstance(result, dict):
            recommended_pipeline = result['pipeline']
            similar_datasets = result.get('similar_datasets', [])
            similarity_scores = result.get('similarity_scores', {})
            influence_scores = result.get('influence_scores', {})
            performance_scores = result.get('performance_scores', {})
        else:
            recommended_pipeline = result[0] if isinstance(result, tuple) else result
            similar_datasets = []
            similarity_scores = {}
            influence_scores = {}
            performance_scores = {}
        
        print(f"\n✅ Recommended Pipeline: {recommended_pipeline}")
        
        # Analyze similar datasets
        if similar_datasets:
            print(f"\n🎯 Similar Datasets Found: {len(similar_datasets)}")
            print(f"\nTop {min(k, len(similar_datasets))} Most Similar Datasets:")
            
            for i, ds_id in enumerate(similar_datasets[:k], 1):
                sim_score = similarity_scores.get(ds_id, 0.0)
                inf_score = influence_scores.get(ds_id, 0.0)
                
                # Get dataset characteristics
                if ds_id in self.metafeatures_df.index:
                    n_samples = self.metafeatures_df.loc[ds_id, 'NumberOfInstances'] if 'NumberOfInstances' in self.metafeatures_df.columns else '?'
                    n_features = self.metafeatures_df.loc[ds_id, 'NumberOfFeatures'] if 'NumberOfFeatures' in self.metafeatures_df.columns else '?'
                    n_classes = self.metafeatures_df.loc[ds_id, 'NumberOfClasses'] if 'NumberOfClasses' in self.metafeatures_df.columns else '?'
                else:
                    n_samples = n_features = n_classes = '?'
                
                print(f"\n   {i}. Dataset {ds_id}")
                print(f"      Similarity: {sim_score:.4f}")
                if inf_score > 0:
                    print(f"      Influence: {inf_score:.4f} ⭐")
                print(f"      Shape: {n_samples} samples × {n_features} features × {n_classes} classes")
                
                # Show performance on this dataset
                d_format = f"D_{ds_id}"
                if d_format in self.performance_matrix.columns:
                    perfs = self.performance_matrix[d_format].dropna()
                    if not perfs.empty:
                        best_pipeline = perfs.idxmax()
                        best_score = perfs.max()
                        rec_score = perfs.get(recommended_pipeline, 0.0)
                        
                        print(f"      Best pipeline on this dataset: {best_pipeline} ({best_score:.4f})")
                        print(f"      Recommended pipeline score: {rec_score:.4f}")
                        if rec_score == best_score:
                            print(f"      ✅ Recommended pipeline is OPTIMAL here!")
                        else:
                            gap = best_score - rec_score
                            print(f"      ⚠️ Gap from optimal: {gap:.4f}")
        
        # Analyze recommendation quality
        print(f"\n📊 Pipeline Performance Predictions:")
        if performance_scores:
            sorted_pipelines = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (pipeline, score) in enumerate(sorted_pipelines[:5], 1):
                marker = "👑" if pipeline == recommended_pipeline else "  "
                print(f"   {marker} {i}. {pipeline}: {score:.4f}")
        
        # Log this recommendation
        debug_entry = {
            'dataset': dataset['name'],
            'dataset_id': dataset['id'],
            'recommended_pipeline': recommended_pipeline,
            'similar_datasets': similar_datasets[:k],
            'similarity_scores': {k: v for k, v in list(similarity_scores.items())[:k]},
            'influence_scores': {k: v for k, v in list(influence_scores.items())[:k]},
            'performance_predictions': dict(sorted_pipelines[:5]) if performance_scores else {}
        }
        self.debug_log.append(debug_entry)
        
        return debug_entry
    
    def visualize_embeddings(self, test_dataset_ids, output_dir='debug_output'):
        """Visualize dataset embeddings in 2D space"""
        if self.recommender_name not in ['pmm', 'balancedpmm']:
            print("Embedding visualization only available for PMM recommenders")
            return
        
        if not hasattr(self.recommender, 'dataset_embeddings') or len(self.recommender.dataset_embeddings) == 0:
            print("No embeddings available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n📉 Visualizing Dataset Embeddings...")
        
        # Collect embeddings
        dataset_ids = list(self.recommender.dataset_embeddings.keys())
        embeddings = np.array([self.recommender.dataset_embeddings[ds_id] for ds_id in dataset_ids])
        
        # Reduce to 2D using t-SNE
        print("   Running t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(dataset_ids)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Plot training datasets
        train_mask = [ds_id not in test_dataset_ids for ds_id in dataset_ids]
        test_mask = [ds_id in test_dataset_ids for ds_id in dataset_ids]
        
        plt.scatter(embeddings_2d[train_mask, 0], embeddings_2d[train_mask, 1], 
                   c='lightblue', s=100, alpha=0.6, label='Training Datasets', edgecolors='black')
        plt.scatter(embeddings_2d[test_mask, 0], embeddings_2d[test_mask, 1], 
                   c='red', s=200, alpha=0.8, label='Test Datasets', edgecolors='black', marker='*')
        
        # Annotate test datasets
        for i, ds_id in enumerate(dataset_ids):
            if ds_id in test_dataset_ids:
                plt.annotate(f'D_{ds_id}', (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           fontsize=10, fontweight='bold')
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('Dataset Embedding Space (t-SNE Visualization)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'embedding_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved embedding visualization to: {output_path}")
        plt.close()
    
    def compare_recommendations(self, test_results, output_dir='debug_output'):
        """Compare recommendations across multiple test datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Comparing Recommendations Across Datasets...")
        
        # Extract data
        datasets = [entry['dataset'] for entry in self.debug_log]
        recommendations = [entry['recommended_pipeline'] for entry in self.debug_log]
        
        # Get ground truth if available
        if test_results:
            ground_truth = [result.get('ground_truth_best', 'Unknown') for result in test_results]
            ranks = [result.get('rank', 0) for result in test_results]
            scores = [result.get('recommended_score', 0) for result in test_results]
            best_scores = [result.get('best_score', 0) for result in test_results]
            gaps = [result.get('score_gap', 0) for result in test_results]
        else:
            ground_truth = ['Unknown'] * len(datasets)
            ranks = [0] * len(datasets)
            scores = [0] * len(datasets)
            best_scores = [0] * len(datasets)
            gaps = [0] * len(datasets)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Dataset': datasets,
            'Recommended': recommendations,
            'Ground Truth': ground_truth,
            'Rank': ranks,
            'Score': scores,
            'Best Score': best_scores,
            'Gap': gaps,
            'Success': [rec == gt for rec, gt in zip(recommendations, ground_truth)]
        })
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f'{self.recommender_name}_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved comparison to: {csv_path}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Rank distribution
        axes[0, 0].hist(ranks, bins=range(1, max(ranks)+2), edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Rank of Recommended Pipeline', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title(f'Recommendation Rank Distribution\n{self.recommender_name.upper()}', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].axvline(np.mean(ranks), color='red', linestyle='--', label=f'Mean: {np.mean(ranks):.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance gap
        axes[0, 1].bar(range(len(gaps)), gaps, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Dataset Index', fontsize=12)
        axes[0, 1].set_ylabel('Performance Gap', fontsize=12)
        axes[0, 1].set_title('Performance Gap from Optimal\n(Lower is Better)', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].axhline(np.mean(gaps), color='red', linestyle='--', label=f'Mean: {np.mean(gaps):.4f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Pipeline recommendation frequency
        pipeline_counts = pd.Series(recommendations).value_counts()
        axes[1, 0].barh(pipeline_counts.index, pipeline_counts.values, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Frequency', fontsize=12)
        axes[1, 0].set_ylabel('Pipeline', fontsize=12)
        axes[1, 0].set_title('Recommended Pipeline Frequency', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # 4. Success rate
        success_rate = comparison_df['Success'].mean() * 100
        success_counts = comparison_df['Success'].value_counts()
        colors = ['lightgreen' if x else 'lightcoral' for x in [True, False]]
        axes[1, 1].pie(success_counts.values, labels=['Perfect Match', 'Sub-optimal'], 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title(f'Recommendation Accuracy\n{success_rate:.1f}% Perfect Matches', 
                            fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{self.recommender_name}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved analysis plots to: {plot_path}")
        plt.close()
        
        return comparison_df


class SimilarityAnalyzer:
    """Analyze similarity patterns between datasets"""
    
    def __init__(self, metafeatures_df, performance_matrix):
        self.metafeatures_df = metafeatures_df
        self.performance_matrix = performance_matrix
        
    def analyze_similarity_quality(self, test_dataset_id, similar_dataset_ids, output_dir='debug_output'):
        """Analyze whether similar datasets truly have similar performance patterns"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔍 Analyzing Similarity Quality for Dataset {test_dataset_id}...")
        
        # Get metafeatures
        if test_dataset_id not in self.metafeatures_df.index:
            print("   ❌ Test dataset not in metafeatures")
            return
        
        test_mf = self.metafeatures_df.loc[test_dataset_id]
        
        # Compare metafeatures with similar datasets
        print(f"\n   Comparing {len(similar_dataset_ids)} similar datasets:")
        
        for sim_ds_id in similar_dataset_ids[:5]:
            if sim_ds_id in self.metafeatures_df.index:
                sim_mf = self.metafeatures_df.loc[sim_ds_id]
                
                # Calculate metafeature distance
                mf_distance = np.linalg.norm(test_mf.values - sim_mf.values)
                
                print(f"\n   Dataset {sim_ds_id}:")
                print(f"      Metafeature distance: {mf_distance:.4f}")
                
                # Compare key characteristics
                if 'NumberOfInstances' in self.metafeatures_df.columns:
                    test_samples = test_mf['NumberOfInstances']
                    sim_samples = sim_mf['NumberOfInstances']
                    print(f"      Samples: {test_samples:.0f} vs {sim_samples:.0f}")
                
                if 'NumberOfFeatures' in self.metafeatures_df.columns:
                    test_features = test_mf['NumberOfFeatures']
                    sim_features = sim_mf['NumberOfFeatures']
                    print(f"      Features: {test_features:.0f} vs {sim_features:.0f}")
                
                # Check performance correlation if both are in performance matrix
                test_col = f"D_{test_dataset_id}"
                sim_col = f"D_{sim_ds_id}"
                
                if test_col in self.performance_matrix.columns and sim_col in self.performance_matrix.columns:
                    test_perf = self.performance_matrix[test_col].dropna()
                    sim_perf = self.performance_matrix[sim_col].dropna()
                    
                    common_pipelines = test_perf.index.intersection(sim_perf.index)
                    if len(common_pipelines) > 2:
                        correlation = test_perf[common_pipelines].corr(sim_perf[common_pipelines])
                        print(f"      Performance correlation: {correlation:.4f}")
                        
                        if correlation > 0.7:
                            print(f"      ✅ Strong performance similarity!")
                        elif correlation > 0.4:
                            print(f"      ⚠️ Moderate performance similarity")
                        else:
                            print(f"      ❌ Weak performance similarity")


def run_debug_mode(recommender_configs, test_dataset_ids, metafeatures_df, 
                   performance_matrix, ground_truth_matrix=None, max_datasets=5):
    """
    Run comprehensive debugging analysis
    
    Args:
        recommender_configs: List of (recommender, name) tuples
        test_dataset_ids: List of test dataset IDs to analyze
        metafeatures_df: DataFrame with metafeatures
        performance_matrix: Performance matrix for training
        ground_truth_matrix: Optional ground truth for test datasets
        max_datasets: Maximum number of test datasets to analyze in detail
    """
    print("\n" + "="*80)
    print("🔬 DEEP DEBUGGING MODE ACTIVATED")
    print("="*80)
    print(f"\nAnalyzing {min(max_datasets, len(test_dataset_ids))} test datasets in detail...")
    print(f"Recommenders: {[name for _, name in recommender_configs]}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"debug_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Initialize analyzers
    similarity_analyzer = SimilarityAnalyzer(metafeatures_df, performance_matrix)
    
    # Process each test dataset
    all_results = {name: [] for _, name in recommender_configs}
    
    for i, test_id in enumerate(test_dataset_ids[:max_datasets], 1):
        print(f"\n{'#'*80}")
        print(f"ANALYZING TEST DATASET {i}/{min(max_datasets, len(test_dataset_ids))}: {test_id}")
        print(f"{'#'*80}")
        
        # Load dataset
        from evaluation_utils import load_openml_dataset
        dataset = load_openml_dataset(test_id)
        
        if dataset is None:
            print(f"❌ Failed to load dataset {test_id}")
            continue
        
        # Profile the dataset
        profiler = DatasetProfiler(dataset)
        profile = profiler.analyze()
        
        # Analyze each recommender
        for recommender, name in recommender_configs:
            print(f"\n{'─'*80}")
            print(f"Analyzing {name.upper()} Recommender")
            print(f"{'─'*80}")
            
            # Create debugger
            debugger = RecommenderDebugger(recommender, name, performance_matrix, metafeatures_df)
            
            # Debug recommendation
            debug_result = debugger.debug_recommendation(dataset, k=5)
            
            if debug_result and debug_result.get('similar_datasets'):
                # Analyze similarity quality
                similarity_analyzer.analyze_similarity_quality(
                    test_id, 
                    debug_result['similar_datasets'],
                    output_dir
                )
            
            all_results[name].append(debug_result)
    
    # Generate comparative visualizations
    print(f"\n{'='*80}")
    print("GENERATING COMPARATIVE VISUALIZATIONS")
    print(f"{'='*80}")
    
    for recommender, name in recommender_configs:
        debugger = RecommenderDebugger(recommender, name, performance_matrix, metafeatures_df)
        debugger.debug_log = all_results[name]
        
        # Visualize embeddings (for PMM recommenders)
        if name in ['pmm', 'balancedpmm']:
            debugger.visualize_embeddings(test_dataset_ids[:max_datasets], output_dir)
        
        # Compare recommendations
        debugger.compare_recommendations(None, output_dir)
    
    # Save debug log
    debug_log_path = os.path.join(output_dir, 'debug_log.json')
    with open(debug_log_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Debug analysis complete!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"   - embedding_visualization.png (PMM embeddings)")
    print(f"   - *_comparison.csv (recommendation comparisons)")
    print(f"   - *_analysis.png (performance analysis plots)")
    print(f"   - debug_log.json (detailed debug information)")
    
    return output_dir
