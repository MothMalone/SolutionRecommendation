"""
Generate Complete Performance Matrix for All 3,600 Pipeline Configurations
Each pipeline is encoded as a vector representing operator positions in the options
"""

import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings('ignore')

from setting_aco import (
    options,
    test_dataset_ids,
    load_openml_dataset,
    apply_preprocessing,
    evaluate_pipeline_with_autogluon
)

# AutoGluon configuration
AUTOGLUON_CONFIG = {
    "eval_metric": "accuracy",
    "time_limit": 100,  # 5 minutes per evaluation
    "presets": "medium_quality",
    "verbosity": 1,
    "hyperparameter_tune_kwargs": None,
    "ag_args_fit": {
        "ag.max_memory_usage_ratio": 0.9,
    },
    "seed": 42
}


def encode_pipeline_to_vector(pipeline_dict, options):
    """
    Encode a pipeline configuration to a vector of operator indices.
    
    Args:
        pipeline_dict: Dict like {'imputation': 'mean', 'scaling': 'robust', ...}
        options: The options dictionary
        
    Returns:
        List of integers representing the index of each selected operator
    """
    vector = []
    for step in ['imputation', 'scaling', 'encoding', 'feature_selection', 
                 'outlier_removal', 'dimensionality_reduction']:
        operator = pipeline_dict.get(step, 'none')
        try:
            idx = options[step].index(operator)
        except ValueError:
            idx = 0  # Default to first option if not found
        vector.append(idx)
    return vector


def vector_to_pipeline_dict(vector, options):
    """
    Convert a vector of indices back to a pipeline dictionary.
    
    Args:
        vector: List of integers [imp_idx, scale_idx, enc_idx, ...]
        options: The options dictionary
        
    Returns:
        Pipeline dictionary
    """
    steps = ['imputation', 'scaling', 'encoding', 'feature_selection', 
             'outlier_removal', 'dimensionality_reduction']
    
    pipeline = {'name': f'pipeline_{"_".join(map(str, vector))}'}
    for i, step in enumerate(steps):
        pipeline[step] = options[step][vector[i]]
    
    return pipeline


def generate_all_pipeline_configurations(options):
    """
    Generate all possible pipeline configurations.
    
    Returns:
        List of (vector, pipeline_dict) tuples
    """
    # Get all combinations
    steps = ['imputation', 'scaling', 'encoding', 'feature_selection', 
             'outlier_removal', 'dimensionality_reduction']
    
    all_combinations = itertools.product(
        range(len(options['imputation'])),
        range(len(options['scaling'])),
        range(len(options['encoding'])),
        range(len(options['feature_selection'])),
        range(len(options['outlier_removal'])),
        range(len(options['dimensionality_reduction']))
    )
    
    configurations = []
    for vector in all_combinations:
        vector_list = list(vector)
        pipeline_dict = vector_to_pipeline_dict(vector_list, options)
        configurations.append((vector_list, pipeline_dict))
    
    return configurations


def evaluate_all_pipelines_on_datasets(
    test_datasets,
    save_path='testing_performance_matrix_autogluon_full.csv',
    checkpoint_interval=50,
    checkpoint_path='checkpoint_performance_matrix.csv'
):
    """
    Evaluate all 3,600 pipelines on all test datasets.
    
    Args:
        test_datasets: List of dataset dictionaries
        save_path: Path to save final matrix
        checkpoint_interval: Save checkpoint every N pipelines
        checkpoint_path: Path for checkpoint saves
    """
    print("="*80)
    print("GENERATING COMPLETE PERFORMANCE MATRIX")
    print("="*80)
    
    # Generate all pipeline configurations
    print("\nGenerating all pipeline configurations...")
    all_configs = generate_all_pipeline_configurations(options)
    n_pipelines = len(all_configs)
    n_datasets = len(test_datasets)
    
    print(f"Total pipelines: {n_pipelines}")
    print(f"Total datasets: {n_datasets}")
    print(f"Total evaluations needed: {n_pipelines * n_datasets}")
    print(f"Estimated time: ~{(n_pipelines * n_datasets * 5) / 60:.1f} minutes (5min per evaluation)")
    
    # Create dataset column names
    dataset_names = [f"D_{ds['id']}" for ds in test_datasets]
    
    # Initialize performance matrix
    # Rows: pipeline vectors (as strings for indexing)
    # Columns: datasets
    performance_data = []
    
    print("\n" + "="*80)
    print("Starting Evaluation")
    print("="*80)
    
    for pipeline_idx, (vector, pipeline_dict) in enumerate(all_configs):
        pipeline_name = "_".join(map(str, vector))
        
        print(f"\nPipeline {pipeline_idx+1}/{n_pipelines}: {pipeline_name}")
        print(f"  Config: {pipeline_dict}")
        
        # Evaluate on all datasets
        row_data = {'pipeline_vector': pipeline_name}
        
        for dataset in test_datasets:
            dataset_name = f"D_{dataset['id']}"
            
            try:
                score = evaluate_pipeline_with_autogluon(dataset, pipeline_dict)
                row_data[dataset_name] = score
                
                if not np.isnan(score):
                    print(f"  {dataset_name}: {score:.4f}")
                else:
                    print(f"  {dataset_name}: NaN (evaluation failed)")
                    
            except Exception as e:
                print(f"  {dataset_name}: Error - {e}")
                row_data[dataset_name] = np.nan
        
        performance_data.append(row_data)
        
        # Save checkpoint
        if (pipeline_idx + 1) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(performance_data)
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"\n✓ Checkpoint saved: {checkpoint_idx+1}/{n_pipelines} pipelines evaluated")
            print(f"  Progress: {((pipeline_idx+1)/n_pipelines)*100:.1f}%")
    
    # Create final DataFrame
    print("\n" + "="*80)
    print("Creating final performance matrix...")
    print("="*80)
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.set_index('pipeline_vector')
    
    # Save to CSV
    performance_df.to_csv(save_path)
    
    print(f"\n✓ Complete performance matrix saved to: {save_path}")
    print(f"  Shape: {performance_df.shape}")
    print(f"  Total cells: {performance_df.size}")
    print(f"  Non-NaN cells: {performance_df.notna().sum().sum()}")
    print(f"  Coverage: {(performance_df.notna().sum().sum() / performance_df.size)*100:.2f}%")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Best pipeline per dataset
    print("\nBest pipeline per dataset:")
    for col in performance_df.columns:
        best_idx = performance_df[col].idxmax()
        best_score = performance_df[col].max()
        if not np.isnan(best_score):
            print(f"  {col}: Pipeline {best_idx} (Score: {best_score:.4f})")
    
    # Overall best pipelines
    print("\nTop 10 pipelines (by average score):")
    avg_scores = performance_df.mean(axis=1).sort_values(ascending=False)
    for i, (pipeline, score) in enumerate(avg_scores.head(10).items()):
        vector = [int(x) for x in pipeline.split('_')]
        pipeline_dict = vector_to_pipeline_dict(vector, options)
        print(f"  {i+1}. {pipeline}: {score:.4f}")
        print(f"      {pipeline_dict}")
    
    return performance_df


def quick_test_subset(n_pipelines=20, n_datasets=3):
    """
    Quick test on a subset of pipelines and datasets.
    """
    print("="*80)
    print("QUICK TEST MODE - Subset Evaluation")
    print("="*80)
    
    # Load subset of datasets
    print(f"\nLoading {n_datasets} test datasets...")
    test_datasets = []
    for dataset_id in test_dataset_ids[:n_datasets]:
        dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
        if dataset:
            test_datasets.append(dataset)
    
    print(f"Loaded {len(test_datasets)} datasets")
    
    # Generate subset of pipeline configurations
    print(f"\nGenerating {n_pipelines} random pipeline configurations...")
    all_configs = generate_all_pipeline_configurations(options)
    
    # Sample random pipelines
    np.random.seed(42)
    selected_indices = np.random.choice(len(all_configs), n_pipelines, replace=False)
    selected_configs = [all_configs[i] for i in selected_indices]
    
    # Create dataset column names
    dataset_names = [f"D_{ds['id']}" for ds in test_datasets]
    
    # Evaluate
    performance_data = []
    
    for pipeline_idx, (vector, pipeline_dict) in enumerate(selected_configs):
        pipeline_name = "_".join(map(str, vector))
        
        print(f"\nPipeline {pipeline_idx+1}/{n_pipelines}: {pipeline_name}")
        
        row_data = {'pipeline_vector': pipeline_name}
        
        for dataset in test_datasets:
            dataset_name = f"D_{dataset['id']}"
            
            try:
                score = evaluate_pipeline_with_autogluon(dataset, pipeline_dict)
                row_data[dataset_name] = score
                print(f"  {dataset_name}: {score:.4f if not np.isnan(score) else 'NaN'}")
            except Exception as e:
                print(f"  {dataset_name}: Error - {e}")
                row_data[dataset_name] = np.nan
        
        performance_data.append(row_data)
    
    # Create DataFrame
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.set_index('pipeline_vector')
    
    # Save
    performance_df.to_csv('test_performance_matrix_subset.csv')
    print(f"\n✓ Test matrix saved to: test_performance_matrix_subset.csv")
    print(f"  Shape: {performance_df.shape}")
    
    return performance_df


if __name__ == "__main__":
    import sys
    
    print("Performance Matrix Generator")
    print("="*80)
    print("\nOptions:")
    print("  1. Quick test (20 pipelines x 3 datasets)")
    print("  2. Full evaluation (3600 pipelines x all test datasets)")
    print("  3. Custom subset")
    print()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        # Quick test
        print("\nRunning quick test...")
        df = quick_test_subset(n_pipelines=20, n_datasets=3)
        
    elif choice == '2':
        # Full evaluation
        print("\nLoading all test datasets...")
        test_datasets = []
        for dataset_id in test_dataset_ids:
            dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
            if dataset:
                test_datasets.append(dataset)
        
        print(f"Loaded {len(test_datasets)} test datasets")
        
        confirm = input(f"\nThis will perform {3600 * len(test_datasets)} evaluations. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            df = evaluate_all_pipelines_on_datasets(
                test_datasets,
                save_path='testing_performance_matrix_autogluon_full.csv',
                checkpoint_interval=50
            )
        else:
            print("Cancelled.")
    
    elif choice == '3':
        # Custom subset
        n_pipelines = int(input("Number of pipelines to test: "))
        n_datasets = int(input("Number of datasets to test: "))
        
        df = quick_test_subset(n_pipelines=n_pipelines, n_datasets=n_datasets)
    
    else:
        print("Invalid choice. Running quick test...")
        df = quick_test_subset()
