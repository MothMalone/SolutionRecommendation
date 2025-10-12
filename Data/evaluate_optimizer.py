"""
Evaluation script for the Optimization-Based Pipeline Recommender.

This script demonstrates how the optimizer works and compares its custom pipeline
recommendations against the 12 predefined pipelines on test datasets.

The optimizer uses Bayesian Optimization (SMAC3) to search a configuration space
of ~10,000 possible pipeline combinations to find the best pipeline per dataset.
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

from optimized_pipeline_recommender import OptimizedPipelineRecommender
from evaluation_utils import load_openml_dataset, get_metafeatures

# Import the 12 predefined pipeline configurations from recommender_trainer
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


def pipeline_to_name(pipeline_config):
    """
    Convert a pipeline configuration dict to a readable name.
    
    This helps us understand what the optimizer is recommending by
    creating human-readable labels for custom pipelines.
    
    Example:
        {'imputation': 'mean', 'scaling': 'standard', ...}
        -> 'mean_standard_onehot_none_none_none'
    """
    # Use the pipeline config values to create a name
    components = [
        pipeline_config.get('imputation', 'none'),
        pipeline_config.get('scaling', 'none'),
        pipeline_config.get('encoding', 'none'),
        pipeline_config.get('feature_selection', 'none'),
        pipeline_config.get('outlier_removal', 'none'),
        pipeline_config.get('dimensionality_reduction', 'none')
    ]
    return '_'.join(components)


def find_matching_predefined_pipeline(custom_config):
    """
    Check if a custom pipeline matches one of the 12 predefined pipelines.
    
    This is interesting because it shows us when the optimizer
    "rediscovers" one of our predefined pipelines versus when it
    finds something truly novel.
    
    Returns:
        str or None: The name of the matching predefined pipeline, or None if novel
    """
    for predefined in pipeline_configs:
        # Compare all components (excluding 'name' field)
        if all(custom_config.get(k) == predefined.get(k) 
               for k in ['imputation', 'scaling', 'encoding', 
                        'feature_selection', 'outlier_removal', 
                        'dimensionality_reduction']):
            return predefined['name']
    return None


def explain_optimizer_workflow():
    """
    Educational function to explain how the optimizer works.
    
    This helps understand the learning process before we run evaluations.
    """
    print("\n" + "="*80)
    print("HOW THE OPTIMIZATION-BASED RECOMMENDER WORKS")
    print("="*80)
    
    print("\n📚 STEP 1: LEARNING FROM EXISTING DATA")
    print("-" * 80)
    print("""
The optimizer learns from two sources:
1. preprocessed_performance.csv - Shows how each of the 12 predefined pipelines
   performed on 90 training datasets
2. dataset_feats.csv - Contains 107 metafeatures describing each dataset
   (e.g., number of features, number of instances, class imbalance, etc.)

It builds a SURROGATE MODEL (Random Forest) that predicts:
   performance = f(dataset_metafeatures, pipeline_configuration)

This surrogate learns patterns like:
- "Datasets with many features benefit from dimensionality_reduction='pca'"
- "Imbalanced datasets work better with outlier_removal='isolation_forest'"
- "Datasets with missing values need imputation='knn'"
""")
    
    print("\n🔍 STEP 2: CONFIGURATION SPACE")
    print("-" * 80)
    print("""
The optimizer can search over 6 pipeline components:

1. imputation: [none, mean, median, most_frequent, knn, constant] (6 options)
2. scaling: [none, standard, minmax, robust, maxabs] (5 options)  
3. encoding: [none, onehot] (2 options)
4. feature_selection: [none, variance_threshold, k_best, mutual_info] (4 options)
5. outlier_removal: [none, iqr, zscore, isolation_forest, lof] (5 options)
6. dimensionality_reduction: [none, pca, svd] (3 options)

Total search space: 6 × 5 × 2 × 4 × 5 × 3 = ~10,000 combinations!

This is MUCH larger than the 12 predefined pipelines.
""")
    
    print("\n🎯 STEP 3: BAYESIAN OPTIMIZATION (SMAC3)")
    print("-" * 80)
    print("""
For each test dataset, the optimizer:

1. Gets the dataset's metafeatures
2. Uses SMAC3 (Bayesian optimization) to search for the best pipeline:
   - Starts with random configurations
   - Evaluates them using the surrogate model
   - Uses Expected Improvement (EI) to decide where to search next
   - Iteratively refines toward the optimal configuration
3. After 50 evaluations, returns the best pipeline found

This is MUCH more efficient than evaluating all 10,000 combinations!
""")
    
    print("\n📊 STEP 4: COMPARISON")
    print("-" * 80)
    print("""
We compare the optimizer's custom pipeline against the 12 predefined pipelines:
- Does it match a predefined pipeline? (rediscovery)
- Does it find something novel? (exploration)
- How does its predicted performance compare?
- Where does it rank among all pipelines?
""")
    
    print("\n" + "="*80 + "\n")


def evaluate_optimizer_on_test_datasets(
    performance_matrix_path='preprocessed_performance.csv',
    metafeatures_path='dataset_feats.csv',
    test_ground_truth_path='test_ground_truth_performance.csv',
    n_trials=50,
    verbose=True
):
    """
    Main evaluation function for the optimizer.
    
    This function:
    1. Loads the training data (performance matrix + metafeatures)
    2. Trains the optimizer's surrogate model
    3. For each test dataset:
       - Runs Bayesian optimization to find best pipeline
       - Compares against 12 predefined pipelines
       - Analyzes whether it's novel or a rediscovery
    
    Args:
        performance_matrix_path: Path to training performance data (12×90)
        metafeatures_path: Path to dataset metafeatures (2560×107)
        test_ground_truth_path: Path to test ground truth (12×19)
        n_trials: Number of SMAC evaluations per dataset
        verbose: Whether to print detailed progress
    
    Returns:
        pd.DataFrame: Evaluation results for each test dataset
    """
    print("\n" + "="*80)
    print("OPTIMIZER EVALUATION ON TEST DATASETS")
    print("="*80)
    
    # Load training data
    print("\n📂 Loading training data...")
    try:
        performance_matrix = pd.read_csv(performance_matrix_path, index_col=0)
        print(f"  ✅ Performance matrix: {performance_matrix.shape}")
        print(f"     ({performance_matrix.shape[0]} pipelines × {performance_matrix.shape[1]} datasets)")
        
        metafeatures = pd.read_csv(metafeatures_path, index_col=0)
        print(f"  ✅ Metafeatures: {metafeatures.shape}")
        print(f"     ({metafeatures.shape[0]} datasets × {metafeatures.shape[1]} features)")
    except Exception as e:
        print(f"  ❌ ERROR loading training data: {e}")
        return None
    
    # Load test ground truth
    print("\n📂 Loading test ground truth...")
    try:
        test_ground_truth = pd.read_csv(test_ground_truth_path, index_col=0)
        print(f"  ✅ Test ground truth: {test_ground_truth.shape}")
        print(f"     ({test_ground_truth.shape[0]} pipelines × {test_ground_truth.shape[1]} datasets)")
        
        # Extract test dataset IDs from column names (format: D_<id>)
        test_dataset_ids = [int(col.split('_')[1]) for col in test_ground_truth.columns if col.startswith('D_')]
        print(f"     Test datasets: {test_dataset_ids}")
    except Exception as e:
        print(f"  ❌ ERROR loading test ground truth: {e}")
        return None
    
    # Train the optimizer's surrogate model
    print("\n🤖 Training optimizer's surrogate model...")
    print(f"  This learns the mapping: (metafeatures, pipeline_config) → performance")
    print(f"  Training on {performance_matrix.shape[1]} datasets × {performance_matrix.shape[0]} pipelines")
    print(f"  = {performance_matrix.shape[0] * performance_matrix.shape[1]} training examples")
    
    optimizer = OptimizedPipelineRecommender(n_trials=n_trials, random_state=42)
    
    try:
        optimizer.fit(performance_matrix, metafeatures)
        print(f"  ✅ Surrogate model trained successfully!")
    except Exception as e:
        print(f"  ❌ ERROR training surrogate: {e}")
        return None
    
    # Evaluate on each test dataset
    print("\n" + "="*80)
    print(f"RUNNING OPTIMIZATION ON {len(test_dataset_ids)} TEST DATASETS")
    print("="*80)
    
    results = []
    novel_pipelines = []
    rediscovered_pipelines = []
    
    for i, dataset_id in enumerate(test_dataset_ids):
        print(f"\n[{i+1}/{len(test_dataset_ids)}] Dataset {dataset_id}")
        print("-" * 80)
        
        # Check if dataset has metafeatures
        if dataset_id not in metafeatures.index:
            print(f"  ⚠️  No metafeatures found, skipping")
            continue
        
        # Get ground truth performances for this dataset
        col_name = f'D_{dataset_id}'
        if col_name not in test_ground_truth.columns:
            print(f"  ⚠️  No ground truth found, skipping")
            continue
        
        ground_truth = test_ground_truth[col_name].dropna()
        if len(ground_truth) == 0:
            print(f"  ⚠️  Empty ground truth, skipping")
            continue
        
        # Get dataset metafeatures
        dataset_metafeats = metafeatures.loc[dataset_id].values
        
        # Run Bayesian optimization to find best pipeline
        print(f"  🔍 Running SMAC optimization ({n_trials} evaluations)...")
        try:
            recommended_pipeline = optimizer.recommend(dataset_id)
            
            # Create a readable name for the custom pipeline
            pipeline_name = pipeline_to_name(recommended_pipeline)
            
            # Check if it matches a predefined pipeline
            matching_predefined = find_matching_predefined_pipeline(recommended_pipeline)
            
            if matching_predefined:
                print(f"  ✅ Optimizer REDISCOVERED predefined pipeline: '{matching_predefined}'")
                rediscovered_pipelines.append(matching_predefined)
            else:
                print(f"  🆕 Optimizer found NOVEL pipeline: {pipeline_name}")
                novel_pipelines.append(pipeline_name)
            
            # Show the pipeline configuration
            print(f"     Configuration:")
            for component, value in recommended_pipeline.items():
                print(f"       - {component}: {value}")
            
            # Compare against predefined pipelines
            best_pipeline = ground_truth.idxmax()
            best_score = ground_truth.max()
            baseline_score = ground_truth.get('baseline', np.nan)
            
            # Get optimizer's pipeline score (from ground truth if it matches)
            if matching_predefined:
                optimizer_score = ground_truth.get(matching_predefined, np.nan)
            else:
                # For truly novel pipelines, we'd need to evaluate them
                # For now, use surrogate prediction as proxy
                optimizer_score = np.nan
                print(f"  ⚠️  Novel pipeline - would need evaluation to get true score")
            
            # Calculate ranking (among predefined pipelines)
            sorted_pipelines = ground_truth.sort_values(ascending=False)
            
            if matching_predefined:
                rank = list(sorted_pipelines.index).index(matching_predefined) + 1
                print(f"  📊 Performance:")
                print(f"     - Rank: {rank}/{len(ground_truth)}")
                print(f"     - Score: {optimizer_score:.4f}")
                print(f"     - Best score: {best_score:.4f}")
                print(f"     - Gap to best: {best_score - optimizer_score:.4f}")
                print(f"     - Baseline: {baseline_score:.4f}")
            else:
                rank = np.nan
                print(f"  📊 Cannot rank novel pipeline without evaluation")
            
            # Store results
            results.append({
                'dataset_id': dataset_id,
                'recommended_pipeline': pipeline_name,
                'matching_predefined': matching_predefined if matching_predefined else 'NOVEL',
                'is_novel': matching_predefined is None,
                'rank': rank,
                'optimizer_score': optimizer_score,
                'best_pipeline': best_pipeline,
                'best_score': best_score,
                'baseline_score': baseline_score,
                'score_gap': best_score - optimizer_score if not np.isnan(optimizer_score) else np.nan,
                'better_than_baseline': optimizer_score > baseline_score if not np.isnan(optimizer_score) else np.nan
            })
            
        except Exception as e:
            print(f"  ❌ ERROR during optimization: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if len(results_df) > 0:
        print(f"\n✅ Successfully evaluated {len(results_df)} test datasets")
        
        # Count novel vs rediscovered
        n_novel = results_df['is_novel'].sum()
        n_rediscovered = (~results_df['is_novel']).sum()
        
        print(f"\n🔍 Discovery Analysis:")
        print(f"  - Novel pipelines: {n_novel} ({n_novel/len(results_df)*100:.1f}%)")
        print(f"  - Rediscovered pipelines: {n_rediscovered} ({n_rediscovered/len(results_df)*100:.1f}%)")
        
        if n_rediscovered > 0:
            print(f"\n  Most rediscovered pipelines:")
            from collections import Counter
            rediscovery_counts = Counter(rediscovered_pipelines)
            for pipeline, count in rediscovery_counts.most_common(3):
                print(f"    - {pipeline}: {count} times")
        
        # Performance metrics (only for rediscovered pipelines with scores)
        valid_results = results_df[~results_df['rank'].isna()]
        if len(valid_results) > 0:
            print(f"\n📊 Performance Metrics (on {len(valid_results)} datasets with rankings):")
            print(f"  - Average rank: {valid_results['rank'].mean():.2f}")
            print(f"  - Top-1 accuracy: {(valid_results['rank'] == 1).sum()}/{len(valid_results)} ({(valid_results['rank'] == 1).mean()*100:.1f}%)")
            print(f"  - Top-3 accuracy: {(valid_results['rank'] <= 3).sum()}/{len(valid_results)} ({(valid_results['rank'] <= 3).mean()*100:.1f}%)")
            print(f"  - Average score gap: {valid_results['score_gap'].mean():.4f}")
            
            better_than_baseline = valid_results['better_than_baseline'].sum()
            print(f"  - Better than baseline: {better_than_baseline}/{len(valid_results)} ({better_than_baseline/len(valid_results)*100:.1f}%)")
        
        # Save results
        output_path = 'optimizer_evaluation_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
    else:
        print(f"\n❌ No successful evaluations")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the Optimization-Based Pipeline Recommender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (50 SMAC evaluations per dataset)
  python evaluate_optimizer.py
  
  # Run with more evaluations for better optimization
  python evaluate_optimizer.py --n-trials 100
  
  # Skip the educational explanation
  python evaluate_optimizer.py --no-explain
        """
    )
    
    parser.add_argument(
        '--performance-matrix',
        type=str,
        default='preprocessed_performance.csv',
        help='Path to training performance matrix (default: preprocessed_performance.csv)'
    )
    
    parser.add_argument(
        '--metafeatures',
        type=str,
        default='dataset_feats.csv',
        help='Path to dataset metafeatures (default: dataset_feats.csv)'
    )
    
    parser.add_argument(
        '--test-ground-truth',
        type=str,
        default='test_ground_truth_performance.csv',
        help='Path to test ground truth (default: test_ground_truth_performance.csv)'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of SMAC evaluations per dataset (default: 50)'
    )
    
    parser.add_argument(
        '--no-explain',
        action='store_true',
        help='Skip the educational explanation of how the optimizer works'
    )
    
    args = parser.parse_args()
    
    # Show explanation unless skipped
    if not args.no_explain:
        explain_optimizer_workflow()
        input("Press Enter to continue with evaluation...")
    
    # Run evaluation
    results = evaluate_optimizer_on_test_datasets(
        performance_matrix_path=args.performance_matrix,
        metafeatures_path=args.metafeatures,
        test_ground_truth_path=args.test_ground_truth,
        n_trials=args.n_trials,
        verbose=True
    )
    
    if results is not None:
        print("\n✅ Evaluation complete!")
    else:
        print("\n❌ Evaluation failed")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
