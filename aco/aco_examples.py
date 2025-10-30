"""
Complete Example: ACO Pipeline Optimization Tutorial
Demonstrates how to use the ACO system for pipeline optimization
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from setting_aco import (
    options,
    load_openml_dataset,
    test_dataset_ids
)

from aco_main import (
    optimize_dataset_pipeline,
    run_aco_on_test_datasets,
    compare_aco_with_baseline,
    evaluate_pipeline_with_autogluon
)

from aco_visualization import (
    plot_convergence,
    plot_pheromone_heatmap,
    plot_operator_usage,
    plot_multi_dataset_comparison,
    plot_pipeline_comparison,
    create_full_visualization_suite
)

import matplotlib.pyplot as plt


def example_1_single_dataset():
    """
    Example 1: Optimize pipeline for a single dataset
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Dataset Optimization")
    print("="*80)
    
    # Load a single test dataset
    print("\nLoading dataset...")
    dataset = load_openml_dataset(test_dataset_ids[0], test_dataset_ids=test_dataset_ids)
    
    if not dataset:
        print("Failed to load dataset")
        return
    
    print(f"Dataset: {dataset['name']} (ID: {dataset['id']})")
    print(f"Shape: {dataset['X'].shape}")
    print(f"Task: {dataset['task_type']}")
    
    # Run ACO optimization with custom parameters
    print("\nRunning ACO optimization...")
    results = optimize_dataset_pipeline(
        dataset=dataset,
        aco_params={
            'n_ants': 15,           # Number of ants per iteration
            'n_iterations': 30,     # Number of iterations
            'alpha': 1.0,           # Pheromone importance
            'beta': 2.0,            # Heuristic importance
            'rho': 0.1,             # Evaporation rate
            'q0': 0.9,              # Exploitation vs exploration
            'elite_weight': 2.0,    # Weight for best solution
            'verbose': True
        },
        verbose=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Best Pipeline: {results['best_pipeline']}")
    print(f"Total Time: {results['total_time']:.2f} seconds")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_full_visualization_suite(results, dataset['name'], output_dir="./aco_output")
    
    print("\n✓ Example 1 complete!")
    return results


def example_2_batch_optimization():
    """
    Example 2: Optimize pipelines for multiple datasets in batch
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Dataset Optimization")
    print("="*80)
    
    # Load multiple test datasets
    print("\nLoading datasets...")
    test_datasets = []
    for dataset_id in test_dataset_ids[:3]:  # Use first 3 test datasets
        dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
        if dataset:
            test_datasets.append(dataset)
    
    print(f"Loaded {len(test_datasets)} datasets")
    
    # Run batch optimization
    print("\nRunning batch optimization...")
    results = run_aco_on_test_datasets(
        test_datasets=test_datasets,
        aco_params={
            'n_ants': 12,
            'n_iterations': 25,
            'verbose': True
        },
        save_results=True,
        output_prefix="./aco_output/batch_results"
    )
    
    # Plot multi-dataset comparison
    print("\nCreating comparison visualization...")
    plot_multi_dataset_comparison(
        results['summary'],
        save_path="./aco_output/multi_dataset_comparison.png"
    )
    
    print("\n✓ Example 2 complete!")
    return results


def example_3_compare_with_baselines():
    """
    Example 3: Compare ACO results with simple baseline pipelines
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: ACO vs Baseline Pipelines")
    print("="*80)
    
    # Load a test dataset
    print("\nLoading dataset...")
    dataset = load_openml_dataset(test_dataset_ids[0], test_dataset_ids=test_dataset_ids)
    
    if not dataset:
        print("Failed to load dataset")
        return
    
    # Define simple baseline pipelines
    baselines = [
        {'name': 'baseline', 'imputation': 'none', 'scaling': 'none', 'encoding': 'none',
         'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
        {'name': 'simple', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot',
         'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'},
        {'name': 'robust', 'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot',
         'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'},
        {'name': 'aggressive', 'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot',
         'feature_selection': 'k_best', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'pca'},
    ]
    
    print(f"\nComparing ACO with {len(baselines)} baseline pipelines...")
    
    # Run comparison
    comparison_df = compare_aco_with_baseline(
        test_datasets=[dataset],
        baseline_pipelines=baselines,
        aco_params={
            'n_ants': 15,
            'n_iterations': 30,
            'verbose': True
        }
    )
    
    # Display comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(comparison_df[['dataset_name', 'aco_score', 'best_baseline', 
                         'best_baseline_score', 'improvement_pct']].to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('./aco_output/baseline_comparison.csv', index=False)
    print("\n✓ Saved comparison to ./aco_output/baseline_comparison.csv")
    
    print("\n✓ Example 3 complete!")
    return comparison_df


def example_4_custom_aco_parameters():
    """
    Example 4: Experiment with different ACO parameter settings
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: ACO Parameter Sensitivity Analysis")
    print("="*80)
    
    # Load dataset
    dataset = load_openml_dataset(test_dataset_ids[0], test_dataset_ids=test_dataset_ids)
    
    if not dataset:
        print("Failed to load dataset")
        return
    
    # Test different parameter configurations
    param_configs = [
        {
            'name': 'Balanced',
            'params': {'n_ants': 15, 'n_iterations': 25, 'alpha': 1.0, 'beta': 2.0, 'q0': 0.9}
        },
        {
            'name': 'Exploitative',
            'params': {'n_ants': 15, 'n_iterations': 25, 'alpha': 2.0, 'beta': 1.0, 'q0': 0.95}
        },
        {
            'name': 'Explorative',
            'params': {'n_ants': 15, 'n_iterations': 25, 'alpha': 1.0, 'beta': 3.0, 'q0': 0.7}
        },
        {
            'name': 'Many_Ants',
            'params': {'n_ants': 25, 'n_iterations': 15, 'alpha': 1.0, 'beta': 2.0, 'q0': 0.9}
        }
    ]
    
    results_comparison = []
    
    for config in param_configs:
        print(f"\n{'='*80}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'='*80}")
        
        result = optimize_dataset_pipeline(
            dataset=dataset,
            aco_params={**config['params'], 'verbose': False},
            verbose=False
        )
        
        results_comparison.append({
            'config_name': config['name'],
            'best_score': result['best_score'],
            'total_time': result['total_time'],
            'best_pipeline': result['best_pipeline']
        })
        
        print(f"  Best Score: {result['best_score']:.4f}")
        print(f"  Time: {result['total_time']:.2f}s")
    
    # Display comparison
    import pandas as pd
    comparison_df = pd.DataFrame(results_comparison)
    
    print("\n" + "="*80)
    print("PARAMETER CONFIGURATION COMPARISON")
    print("="*80)
    print(comparison_df[['config_name', 'best_score', 'total_time']].to_string(index=False))
    
    # Find best configuration
    best_config = comparison_df.loc[comparison_df['best_score'].idxmax()]
    print(f"\nBest Configuration: {best_config['config_name']}")
    print(f"  Score: {best_config['best_score']:.4f}")
    print(f"  Time: {best_config['total_time']:.2f}s")
    
    print("\n✓ Example 4 complete!")
    return comparison_df


def example_5_analyze_pipeline_components():
    """
    Example 5: Analyze which pipeline components are most important
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Pipeline Component Analysis")
    print("="*80)
    
    # Load dataset
    dataset = load_openml_dataset(test_dataset_ids[0], test_dataset_ids=test_dataset_ids)
    
    if not dataset:
        print("Failed to load dataset")
        return
    
    # Run optimization
    print("\nRunning ACO optimization...")
    results = optimize_dataset_pipeline(
        dataset=dataset,
        aco_params={'n_ants': 15, 'n_iterations': 30, 'verbose': False},
        verbose=False
    )
    
    # Analyze operator statistics
    if 'operator_statistics' in results and not results['operator_statistics'].empty:
        stats = results['operator_statistics']
        
        print("\n" + "="*80)
        print("TOP 10 OPERATORS BY AVERAGE PERFORMANCE")
        print("="*80)
        top_10 = stats.nlargest(10, 'avg_score')
        print(top_10[['step', 'operator', 'avg_score', 'usage_count']].to_string(index=False))
        
        print("\n" + "="*80)
        print("MOST FREQUENTLY USED OPERATORS")
        print("="*80)
        most_used = stats.nlargest(10, 'usage_count')
        print(most_used[['step', 'operator', 'usage_count', 'avg_score']].to_string(index=False))
        
        # Plot operator usage
        plot_operator_usage(results, save_path="./aco_output/operator_analysis.png")
    
    print("\n✓ Example 5 complete!")
    return results


def run_all_examples():
    """
    Run all examples in sequence
    """
    print("\n" + "="*80)
    print("RUNNING ALL ACO PIPELINE OPTIMIZATION EXAMPLES")
    print("="*80)
    
    import os
    os.makedirs("./aco_output", exist_ok=True)
    
    # Example 1
    print("\n\n")
    example_1_single_dataset()
    
    # Example 2
    print("\n\n")
    example_2_batch_optimization()
    
    # Example 3
    print("\n\n")
    example_3_compare_with_baselines()
    
    # Example 4
    print("\n\n")
    example_4_custom_aco_parameters()
    
    # Example 5
    print("\n\n")
    example_5_analyze_pipeline_components()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE!")
    print("="*80)
    print("\nGenerated files in ./aco_output/")


if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs("./aco_output", exist_ok=True)
    
    print("ACO Pipeline Optimization - Examples")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Single dataset optimization")
    print("  2. Batch dataset optimization")
    print("  3. Compare with baseline pipelines")
    print("  4. Parameter sensitivity analysis")
    print("  5. Pipeline component analysis")
    print("  0. Run all examples")
    print()
    
    choice = input("Enter example number (0-5): ").strip()
    
    if choice == '1':
        example_1_single_dataset()
    elif choice == '2':
        example_2_batch_optimization()
    elif choice == '3':
        example_3_compare_with_baselines()
    elif choice == '4':
        example_4_custom_aco_parameters()
    elif choice == '5':
        example_5_analyze_pipeline_components()
    elif choice == '0':
        run_all_examples()
    else:
        print("Invalid choice. Running Example 1...")
        example_1_single_dataset()
