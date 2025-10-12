"""
Comprehensive Recommender Comparison Script

This script tests ALL available recommender types on the test datasets
with proper error handling and reporting.

Usage:
    python compare_all_recommenders.py [--quick] [--test-dataset DATASET_ID]
"""

import os
import sys
import pandas as pd
import numpy as np
import traceback
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all necessary classes from recommender_trainer
from recommender_trainer import (
    PmmRecommender, BalancedPmmRecommender, HybridMetaRecommender,
    BayesianSurrogateRecommender, AutoGluonPipelineRecommender,
    RandomRecommender, AverageRankRecommender, L1Recommender,
    BasicRecommender, KnnRecommender, RFRecommender, NNRecommender,
    RegressorRecommender, AdaBoostRegressorRecommender,  # Uncomment if available
    pipeline_configs
)

# Import PaperPmmRecommender
try:
    from pmm_paper_style import PaperPmmRecommender
    PAPER_PMM_AVAILABLE = True
except ImportError:
    print("Warning: PaperPmmRecommender not available")
    PAPER_PMM_AVAILABLE = False

# Define all recommender types to test
ALL_RECOMMENDER_TYPES = [
    # Simple baselines
    ('random', 'Random', RandomRecommender),
    ('average-rank', 'Average Rank', AverageRankRecommender),
    # ('basic', 'Basic', BasicRecommender),
    
    # Classical ML methods
    ('l1', 'L1 Distance', L1Recommender),
    ('knn', 'KNN Classifier', KnnRecommender),
    ('rf', 'Random Forest', RFRecommender),
    ('nn', 'Neural Network', NNRecommender),
    
    # Regression-based methods
    ('regressor', 'NN Regressor', RegressorRecommender),
    ('adaboost', 'AdaBoost Regressor', AdaBoostRegressorRecommender),
    
    # Meta-learning methods
    ('pmm', 'PMM (Original)', PmmRecommender),
    # ('balancedpmm', 'Balanced PMM', BalancedPmmRecommender),  # DISABLED: Outdated implementation with bugs
    ('hybrid', 'Hybrid Meta', HybridMetaRecommender),
    ('surrogate', 'Bayesian Surrogate', BayesianSurrogateRecommender),
    
    # Advanced methods
    ('autogluon', 'AutoGluon', AutoGluonPipelineRecommender),
]

# Add paper_pmm if available
if PAPER_PMM_AVAILABLE:
    ALL_RECOMMENDER_TYPES.append(('paper_pmm', 'Paper PMM', PaperPmmRecommender))

# Test dataset IDs
TEST_DATASET_IDS = [
    1503, 23517, 1551, 1552, 183, 255, 545, 546, 475, 481, 
    516, 3, 6, 8, 10, 12, 14, 9, 11, 5
]


class RecommenderTester:
    """Handles testing and comparison of recommenders."""
    
    def __init__(self, performance_matrix, metafeatures_df, test_ground_truth):
        self.performance_matrix = performance_matrix
        self.metafeatures_df = metafeatures_df
        self.test_ground_truth = test_ground_truth
        self.results = []
        
    def test_recommender(self, recommender_type, recommender_name, recommender_class, 
                        test_dataset_ids=None, use_influence=False):
        """
        Test a single recommender type.
        
        Returns:
            dict: Results including success status, errors, and performance metrics
        """
        print(f"\n{'='*80}")
        print(f"Testing: {recommender_name} ({recommender_type})")
        print(f"{'='*80}")
        
        result = {
            'recommender_type': recommender_type,
            'recommender_name': recommender_name,
            'training_success': False,
            'training_error': None,
            'training_time': 0,
            'recommendation_success': 0,
            'recommendation_failures': 0,
            'total_tests': 0,
            'errors': [],
            'accuracy': 0.0,
            'avg_degradation': 0.0,  # Relative performance degradation (%)
            'recommendations': {}
        }
        
        try:
            # Initialize recommender
            start_time = datetime.now()
            
            if recommender_type == 'paper_pmm':
                # Paper PMM has different initialization (no performance_matrix, no similarity_threshold)
                recommender = recommender_class(
                    hidden_dim=128,
                    margin=0.8,
                    batch_size=64,
                    num_epochs=50,
                    learning_rate=0.001,
                    use_influence_weighting=use_influence,
                    influence_method='combined'
                )
            elif recommender_type in ['pmm', 'balancedpmm']:
                # PMM variants
                recommender = recommender_class(
                    hidden_dim=64,
                    embedding_dim=32,
                    margin=1.0,
                    batch_size=32,
                    num_epochs=20,
                    learning_rate=0.001,
                    similarity_threshold=0.8,
                    use_influence_weighting=use_influence,
                    influence_method='combined'
                )
            elif recommender_type == 'hybrid':
                # Hybrid recommender
                recommender = recommender_class(
                    performance_matrix=self.performance_matrix,
                    metafeatures_df=self.metafeatures_df,
                    use_influence_weighting=use_influence,
                    influence_method='combined'
                )
            else:
                # Standard initialization
                recommender = recommender_class(
                    performance_matrix=self.performance_matrix,
                    metafeatures_df=self.metafeatures_df
                )
            
            # Train the recommender
            print(f"Training {recommender_name}...")
            
            if recommender_type in ['pmm', 'balancedpmm', 'paper_pmm']:
                # PMM variants need both matrices
                train_success = recommender.fit(self.performance_matrix, self.metafeatures_df)
            else:
                # Others use fit() without arguments or with different signature
                train_success = recommender.fit()
            
            training_time = (datetime.now() - start_time).total_seconds()
            result['training_time'] = training_time
            
            if not train_success:
                result['training_error'] = "Training returned False"
                print(f"❌ Training failed: {result['training_error']}")
                return result
            
            result['training_success'] = True
            print(f"✅ Training succeeded in {training_time:.2f} seconds")
            
            # Test recommendations
            if test_dataset_ids is None:
                test_dataset_ids = TEST_DATASET_IDS
            
            correct_predictions = 0
            total_degradation = 0.0  # Relative performance degradation
            valid_tests = 0
            
            for test_dataset_id in test_dataset_ids:
                result['total_tests'] += 1
                
                try:
                    # Convert dataset ID to column name format (e.g., 1503 -> 'D_1503')
                    col_name = f'D_{test_dataset_id}'
                    
                    # Get ground truth best pipeline
                    if col_name not in self.test_ground_truth.columns:
                        result['errors'].append(f"Dataset {test_dataset_id} (col: {col_name}) not in ground truth")
                        result['recommendation_failures'] += 1
                        continue
                    
                    ground_truth_performances = self.test_ground_truth[col_name].dropna()
                    if len(ground_truth_performances) == 0:
                        result['errors'].append(f"No ground truth for dataset {test_dataset_id}")
                        result['recommendation_failures'] += 1
                        continue
                    
                    best_pipeline_gt = ground_truth_performances.idxmax()
                    best_score_gt = ground_truth_performances.max()
                    
                    # Get recommendation - handle different recommender types
                    if recommender_type == 'paper_pmm':
                        # Paper PMM needs metafeatures, not dataset_id
                        if test_dataset_id not in self.metafeatures_df.index:
                            result['errors'].append(f"No metafeatures for dataset {test_dataset_id}")
                            result['recommendation_failures'] += 1
                            continue
                        dataset_mf = self.metafeatures_df.loc[test_dataset_id].values
                        rec_result = recommender.recommend(dataset_mf, top_k=5)
                    else:
                        # Other recommenders use dataset_id
                        try:
                            rec_result = recommender.recommend(test_dataset_id, 
                                                              performance_matrix=self.performance_matrix)
                        except TypeError:
                            # Some recommenders don't accept performance_matrix parameter
                            rec_result = recommender.recommend(test_dataset_id)
                    
                    if rec_result is None:
                        result['errors'].append(f"Recommendation returned None for dataset {test_dataset_id}")
                        result['recommendation_failures'] += 1
                        continue
                    
                    # Extract recommended pipeline
                    if isinstance(rec_result, dict):
                        recommended_pipeline = rec_result.get('pipeline')
                    elif isinstance(rec_result, tuple):
                        recommended_pipeline = rec_result[0]
                    elif isinstance(rec_result, list):
                        # Paper PMM returns list of pipeline names
                        recommended_pipeline = rec_result[0] if len(rec_result) > 0 else None
                    else:
                        recommended_pipeline = rec_result
                    
                    if recommended_pipeline is None:
                        result['errors'].append(f"Could not extract pipeline for dataset {test_dataset_id}")
                        result['recommendation_failures'] += 1
                        continue
                    
                    # Calculate metrics
                    result['recommendation_success'] += 1
                    valid_tests += 1
                    
                    # Check if recommendation is correct
                    if recommended_pipeline == best_pipeline_gt:
                        correct_predictions += 1
                    
                    # Calculate degradation (relative performance loss)
                    if recommended_pipeline in ground_truth_performances.index:
                        recommended_score = ground_truth_performances[recommended_pipeline]
                        
                        # Relative degradation: (best - recommended) / best
                        # e.g., best=0.85, recommended=0.75 → degradation = 0.10/0.85 = 11.76%
                        if best_score_gt > 0:
                            degradation = (best_score_gt - recommended_score) / best_score_gt
                        else:
                            degradation = 0.0
                        
                        total_degradation += degradation
                    else:
                        result['errors'].append(f"Recommended pipeline {recommended_pipeline} not in ground truth for dataset {test_dataset_id}")
                    
                    # Store recommendation
                    result['recommendations'][test_dataset_id] = {
                        'recommended': recommended_pipeline,
                        'ground_truth': best_pipeline_gt,
                        'correct': recommended_pipeline == best_pipeline_gt
                    }
                    
                except Exception as e:
                    error_msg = f"Error recommending for dataset {test_dataset_id}: {str(e)}"
                    result['errors'].append(error_msg)
                    result['recommendation_failures'] += 1
                    print(f"  ⚠️  {error_msg}")
            
            # Calculate final metrics
            if valid_tests > 0:
                result['accuracy'] = correct_predictions / valid_tests
                result['avg_degradation'] = total_degradation / valid_tests
            
            print(f"\n📊 Results for {recommender_name}:")
            print(f"   Training: {'✅ Success' if result['training_success'] else '❌ Failed'} ({result['training_time']:.2f}s)")
            print(f"   Recommendations: {result['recommendation_success']}/{result['total_tests']} successful")
            print(f"   Accuracy: {result['accuracy']*100:.2f}% ({correct_predictions}/{valid_tests} correct)")
            print(f"   Average Degradation: {result['avg_degradation']*100:.2f}% (relative performance loss)")
            if result['errors']:
                print(f"   Errors: {len(result['errors'])} issues encountered")
            
        except Exception as e:
            result['training_error'] = str(e)
            result['errors'].append(f"Fatal error: {str(e)}")
            print(f"❌ Fatal error: {e}")
            print(traceback.format_exc())
        
        return result
    
    def test_all_recommenders(self, test_dataset_ids=None, use_influence=False, 
                             skip_slow=False):
        """
        Test all recommender types.
        
        Args:
            test_dataset_ids: List of dataset IDs to test (None = use all)
            use_influence: Whether to use influence weighting for applicable methods
            skip_slow: Skip slow methods like AutoGluon
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE RECOMMENDER COMPARISON")
        print("="*80)
        print(f"Test datasets: {len(test_dataset_ids) if test_dataset_ids else len(TEST_DATASET_IDS)}")
        print(f"Influence weighting: {use_influence}")
        print(f"Skip slow methods: {skip_slow}")
        print("="*80)
        
        for rec_type, rec_name, rec_class in ALL_RECOMMENDER_TYPES:
            # Skip AutoGluon if requested (very slow)
            if skip_slow and rec_type == 'autogluon':
                print(f"\n⏭️  Skipping {rec_name} (slow method)")
                continue
            
            result = self.test_recommender(
                rec_type, rec_name, rec_class,
                test_dataset_ids=test_dataset_ids,
                use_influence=use_influence
            )
            self.results.append(result)
        
        return self.results
    
    def generate_report(self, output_file='recommender_comparison_report.csv'):
        """Generate a comprehensive CSV report with rankings and baseline comparison."""
        
        if not self.results:
            print("No results to report")
            return
        
        # Get baseline performance for comparison
        baseline_performances = {}
        baseline_pipeline = 'baseline'
        
        for col in self.test_ground_truth.columns:
            if baseline_pipeline in self.test_ground_truth.index:
                baseline_performances[col] = self.test_ground_truth.loc[baseline_pipeline, col]
        
        # Calculate detailed metrics for each recommender
        summary_data = []
        for result in self.results:
            if not result['training_success']:
                # Skip failed training
                summary_data.append({
                    'Recommender': result['recommender_name'],
                    'Type': result['recommender_type'],
                    'Training Success': '❌',
                    'Training Time (s)': f"{result['training_time']:.2f}",
                    'Successful Recs': f"{result['recommendation_success']}/{result['total_tests']}",
                    'Accuracy (%)': 0.0,
                    'Avg Degradation (%)': 999.0,  # Changed from Avg Regret
                    'Avg Rank': 999.0,
                    'Better than Baseline (%)': 0.0,
                    'Equal to Baseline (%)': 0.0,
                    'Worse than Baseline (%)': 0.0,
                    'Errors': len(result['errors'])
                })
                continue
            
            # Calculate average rank across all datasets
            all_ranks = []
            better_than_baseline = 0
            equal_to_baseline = 0
            worse_than_baseline = 0
            valid_comparisons = 0
            
            for test_dataset_id, rec_info in result['recommendations'].items():
                # Get column name
                col_name = f'D_{test_dataset_id}'
                
                if col_name not in self.test_ground_truth.columns:
                    continue
                
                recommended_pipeline = rec_info['recommended']
                
                # Get performance values for all pipelines on this dataset
                dataset_performances = self.test_ground_truth[col_name].dropna()
                
                if recommended_pipeline not in dataset_performances.index:
                    continue
                
                # Calculate rank (1 = best, 12 = worst)
                sorted_pipelines = dataset_performances.sort_values(ascending=False)
                rank = list(sorted_pipelines.index).index(recommended_pipeline) + 1
                all_ranks.append(rank)
                
                # Compare to baseline
                if baseline_pipeline in dataset_performances.index:
                    rec_score = dataset_performances[recommended_pipeline]
                    baseline_score = dataset_performances[baseline_pipeline]
                    
                    valid_comparisons += 1
                    
                    if rec_score > baseline_score:
                        better_than_baseline += 1
                    elif rec_score == baseline_score:
                        equal_to_baseline += 1
                    else:
                        worse_than_baseline += 1
            
            # Calculate percentages
            avg_rank = np.mean(all_ranks) if all_ranks else 999.0
            
            if valid_comparisons > 0:
                pct_better = (better_than_baseline / valid_comparisons) * 100
                pct_equal = (equal_to_baseline / valid_comparisons) * 100
                pct_worse = (worse_than_baseline / valid_comparisons) * 100
            else:
                pct_better = pct_equal = pct_worse = 0.0
            
            summary_data.append({
                'Recommender': result['recommender_name'],
                'Type': result['recommender_type'],
                'Training Success': '✅',
                'Training Time (s)': f"{result['training_time']:.2f}",
                'Successful Recs': f"{result['recommendation_success']}/{result['total_tests']}",
                'Accuracy (%)': result['accuracy'] * 100,
                'Avg Degradation (%)': result['avg_degradation'] * 100,  # Changed from avg_regret
                'Avg Rank': avg_rank,
                'Better than Baseline (%)': pct_better,
                'Equal to Baseline (%)': pct_equal,
                'Worse than Baseline (%)': pct_worse,
                'Errors': len(result['errors'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sort by average rank (ascending = better)
        df_summary = df_summary.sort_values('Avg Rank', ascending=True)
        
        # Save to CSV
        df_summary.to_csv(output_file, index=False)
        print(f"\n📁 Report saved to: {output_file}")
        
        # Print summary table
        print("\n" + "="*100)
        print("COMPREHENSIVE RECOMMENDER RANKING")
        print("="*100)
        print(df_summary.to_string(index=False))
        print("="*100)
        
        # Identify best performers
        trained_results = [r for r in self.results if r['training_success'] and r['recommendation_success'] > 0]
        if trained_results:
            # Find indices in df_summary
            valid_df = df_summary[df_summary['Training Success'] == '✅'].copy()
            
            if len(valid_df) > 0:
                best_rank_idx = valid_df['Avg Rank'].idxmin()
                best_accuracy_idx = valid_df['Accuracy (%)'].idxmax()
                best_degradation_idx = valid_df['Avg Degradation (%)'].idxmin()  # Changed from regret
                best_vs_baseline_idx = valid_df['Better than Baseline (%)'].idxmax()
                
                print(f"\n🏆 BEST PERFORMERS:")
                print(f"   📊 Best Avg Rank: {valid_df.loc[best_rank_idx, 'Recommender']} (Rank: {valid_df.loc[best_rank_idx, 'Avg Rank']:.2f})")
                print(f"   🎯 Best Accuracy: {valid_df.loc[best_accuracy_idx, 'Recommender']} ({valid_df.loc[best_accuracy_idx, 'Accuracy (%)']:.2f}%)")
                print(f"   💎 Lowest Degradation: {valid_df.loc[best_degradation_idx, 'Recommender']} ({valid_df.loc[best_degradation_idx, 'Avg Degradation (%)']:.2f}%)")
                print(f"   🚀 Most Better than Baseline: {valid_df.loc[best_vs_baseline_idx, 'Recommender']} ({valid_df.loc[best_vs_baseline_idx, 'Better than Baseline (%)']:.2f}%)")
        
        # Generate detailed per-dataset analysis
        self._generate_detailed_analysis(output_file)
        
        # Detailed error report
        error_file = output_file.replace('.csv', '_errors.txt')
        with open(error_file, 'w') as f:
            f.write("DETAILED ERROR REPORT\n")
            f.write("="*80 + "\n\n")
            
            for result in self.results:
                if result['errors']:
                    f.write(f"\n{result['recommender_name']} ({result['recommender_type']})\n")
                    f.write("-"*80 + "\n")
                    for error in result['errors']:
                        f.write(f"  • {error}\n")
        
        print(f"📁 Detailed errors saved to: {error_file}")
        
        return df_summary
    
    def _generate_detailed_analysis(self, output_file):
        """Generate detailed per-dataset analysis for debugging."""
        
        analysis_file = output_file.replace('.csv', '_detailed_analysis.csv')
        
        # Collect all recommendations per dataset
        dataset_analysis = []
        
        # Get all test datasets
        test_dataset_ids = set()
        for result in self.results:
            if result['training_success']:
                test_dataset_ids.update(result['recommendations'].keys())
        
        for test_dataset_id in sorted(test_dataset_ids):
            col_name = f'D_{test_dataset_id}'
            
            if col_name not in self.test_ground_truth.columns:
                continue
            
            # Get ground truth performances
            gt_performances = self.test_ground_truth[col_name].dropna()
            
            if len(gt_performances) == 0:
                continue
            
            # Get best pipeline and baseline performance
            best_pipeline = gt_performances.idxmax()
            best_score = gt_performances.max()
            baseline_score = gt_performances.get('baseline', np.nan)
            
            # Rank all pipelines
            sorted_pipelines = gt_performances.sort_values(ascending=False)
            
            # Collect recommendations from each recommender
            for result in self.results:
                if not result['training_success']:
                    continue
                
                if test_dataset_id not in result['recommendations']:
                    continue
                
                rec_info = result['recommendations'][test_dataset_id]
                recommended_pipeline = rec_info['recommended']
                
                # Get performance of recommended pipeline
                rec_score = gt_performances.get(recommended_pipeline, np.nan)
                
                # Calculate rank
                if recommended_pipeline in sorted_pipelines.index:
                    rank = list(sorted_pipelines.index).index(recommended_pipeline) + 1
                else:
                    rank = 999
                
                # Calculate regret
                regret = best_score - rec_score if not np.isnan(rec_score) else 999.0
                
                dataset_analysis.append({
                    'Dataset': test_dataset_id,
                    'Recommender': result['recommender_name'],
                    'Recommended Pipeline': recommended_pipeline,
                    'Pipeline Score': rec_score,
                    'Pipeline Rank': rank,
                    'Regret': regret,
                    'Ground Truth Best': best_pipeline,
                    'Best Score': best_score,
                    'Baseline Score': baseline_score,
                    'Correct': rec_info['correct']
                })
        
        df_analysis = pd.DataFrame(dataset_analysis)
        
        if len(df_analysis) > 0:
            # Sort by dataset and rank
            df_analysis = df_analysis.sort_values(['Dataset', 'Pipeline Rank'])
            df_analysis.to_csv(analysis_file, index=False)
            print(f"📁 Detailed per-dataset analysis saved to: {analysis_file}")
            
            # Generate debugging insights
            self._generate_debugging_insights(df_analysis, output_file)
    
    def _generate_debugging_insights(self, df_analysis, output_file):
        """Generate debugging insights to understand recommender behavior."""
        
        insights_file = output_file.replace('.csv', '_insights.txt')
        
        with open(insights_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("DEBUGGING INSIGHTS: WHY DO RECOMMENDERS PERFORM DIFFERENTLY?\n")
            f.write("="*100 + "\n\n")
            
            # 1. Analyze pipeline diversity
            f.write("1. PIPELINE DIVERSITY ANALYSIS\n")
            f.write("-"*100 + "\n")
            f.write("How diverse are the recommendations from each recommender?\n\n")
            
            for recommender in df_analysis['Recommender'].unique():
                rec_data = df_analysis[df_analysis['Recommender'] == recommender]
                unique_pipelines = rec_data['Recommended Pipeline'].nunique()
                most_common = rec_data['Recommended Pipeline'].value_counts().head(3)
                
                f.write(f"{recommender}:\n")
                f.write(f"  • Unique pipelines recommended: {unique_pipelines}/{len(rec_data)}\n")
                f.write(f"  • Most common recommendations:\n")
                for pipeline, count in most_common.items():
                    f.write(f"      - {pipeline}: {count} times ({count/len(rec_data)*100:.1f}%)\n")
                f.write("\n")
            
            # 2. Analyze ranking patterns
            f.write("\n2. RANKING PATTERN ANALYSIS\n")
            f.write("-"*100 + "\n")
            f.write("What rank do recommenders typically achieve?\n\n")
            
            for recommender in df_analysis['Recommender'].unique():
                rec_data = df_analysis[df_analysis['Recommender'] == recommender]
                rank_dist = rec_data['Pipeline Rank'].value_counts().sort_index()
                
                f.write(f"{recommender}:\n")
                f.write(f"  • Rank distribution:\n")
                for rank, count in rank_dist.items():
                    if count > 0:
                        f.write(f"      Rank {rank}: {count} times ({count/len(rec_data)*100:.1f}%)\n")
                f.write(f"  • Average rank: {rec_data['Pipeline Rank'].mean():.2f}\n")
                f.write(f"  • Median rank: {rec_data['Pipeline Rank'].median():.1f}\n\n")
            
            # 3. Baseline comparison insights
            f.write("\n3. BASELINE COMPARISON INSIGHTS\n")
            f.write("-"*100 + "\n")
            f.write("Why do some recommenders beat baseline more often?\n\n")
            
            # Calculate how often baseline is actually good
            baseline_ranks = []
            for dataset_id in df_analysis['Dataset'].unique():
                dataset_data = df_analysis[df_analysis['Dataset'] == dataset_id]
                baseline_data = dataset_data[dataset_data['Recommended Pipeline'] == 'baseline']
                if len(baseline_data) > 0:
                    baseline_rank = baseline_data.iloc[0]['Pipeline Rank']
                    baseline_ranks.append(baseline_rank)
            
            if baseline_ranks:
                avg_baseline_rank = np.mean(baseline_ranks)
                f.write(f"Baseline pipeline statistics:\n")
                f.write(f"  • Average rank across datasets: {avg_baseline_rank:.2f}\n")
                f.write(f"  • Baseline is rank 1 (best): {sum(1 for r in baseline_ranks if r == 1)} times\n")
                f.write(f"  • Baseline is in top-3: {sum(1 for r in baseline_ranks if r <= 3)} times\n")
                f.write(f"  • Baseline is worst (rank 12): {sum(1 for r in baseline_ranks if r == 12)} times\n\n")
            
            # 4. Dataset difficulty analysis
            f.write("\n4. DATASET DIFFICULTY ANALYSIS\n")
            f.write("-"*100 + "\n")
            f.write("Which datasets are hard for all recommenders?\n\n")
            
            dataset_difficulties = []
            for dataset_id in df_analysis['Dataset'].unique():
                dataset_data = df_analysis[df_analysis['Dataset'] == dataset_id]
                avg_rank = dataset_data['Pipeline Rank'].mean()
                avg_regret = dataset_data['Regret'].mean()
                best_recommender = dataset_data.loc[dataset_data['Pipeline Rank'].idxmin(), 'Recommender']
                
                dataset_difficulties.append({
                    'Dataset': dataset_id,
                    'Avg Rank': avg_rank,
                    'Avg Regret': avg_regret,
                    'Best Recommender': best_recommender
                })
            
            df_difficulty = pd.DataFrame(dataset_difficulties).sort_values('Avg Rank', ascending=False)
            
            f.write("Hardest datasets (high average rank = hard):\n")
            for _, row in df_difficulty.head(5).iterrows():
                f.write(f"  • Dataset {row['Dataset']}: Avg Rank {row['Avg Rank']:.2f}, ")
                f.write(f"Avg Regret {row['Avg Regret']:.4f}, Best: {row['Best Recommender']}\n")
            
            f.write("\nEasiest datasets (low average rank = easy):\n")
            for _, row in df_difficulty.tail(5).iterrows():
                f.write(f"  • Dataset {row['Dataset']}: Avg Rank {row['Avg Rank']:.2f}, ")
                f.write(f"Avg Regret {row['Avg Regret']:.4f}, Best: {row['Best Recommender']}\n")
            
            # 5. Recommender specialization
            f.write("\n\n5. RECOMMENDER SPECIALIZATION\n")
            f.write("-"*100 + "\n")
            f.write("Do certain recommenders excel on specific datasets?\n\n")
            
            for recommender in df_analysis['Recommender'].unique():
                rec_data = df_analysis[df_analysis['Recommender'] == recommender]
                best_datasets = rec_data[rec_data['Pipeline Rank'] == 1]['Dataset'].tolist()
                
                f.write(f"{recommender}:\n")
                f.write(f"  • Achieved rank 1 on {len(best_datasets)} datasets: {best_datasets}\n\n")
        
        print(f"📁 Debugging insights saved to: {insights_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare all recommender types')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test on only 3 datasets')
    parser.add_argument('--test-dataset', type=int, 
                       help='Test on a single specific dataset')
    parser.add_argument('--use-influence', action='store_true',
                       help='Use influence weighting for applicable methods')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow methods like AutoGluon')
    parser.add_argument('--output', type=str, default='recommender_comparison_report.csv',
                       help='Output CSV file name')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    try:
        performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
        metafeatures_df = pd.read_csv('dataset_feats.csv', index_col=0)
        test_ground_truth = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
        
        print(f"✅ Loaded performance matrix: {performance_matrix.shape}")
        print(f"✅ Loaded metafeatures: {metafeatures_df.shape}")
        print(f"✅ Loaded test ground truth: {test_ground_truth.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1
    
    # Determine test datasets
    if args.test_dataset:
        test_datasets = [args.test_dataset]
        print(f"Testing on single dataset: {args.test_dataset}")
    elif args.quick:
        test_datasets = TEST_DATASET_IDS[:3]  # Just first 3
        print(f"Quick test mode: using {len(test_datasets)} datasets")
    else:
        test_datasets = TEST_DATASET_IDS
        print(f"Full test mode: using all {len(test_datasets)} datasets")
    
    # Create tester
    tester = RecommenderTester(performance_matrix, metafeatures_df, test_ground_truth)
    
    # Run tests
    tester.test_all_recommenders(
        test_dataset_ids=test_datasets,
        use_influence=args.use_influence,
        skip_slow=args.skip_slow
    )
    
    # Generate report
    tester.generate_report(output_file=args.output)
    
    print("\n✅ Comparison complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
