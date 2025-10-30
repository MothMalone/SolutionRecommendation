"""
ACO-based Pipeline Recommendation System
Uses Ant Colony Optimization to find optimal preprocessing pipelines for datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import from clean settings module
from setting_aco import (
    options, 
    apply_preprocessing,
    load_openml_dataset,
    test_dataset_ids,
    AUTOGLUON_CONFIG
)

# Import ACO optimizer
from aco_pipeline_optimizer import ACOPipelineOptimizer, pipeline_dict_to_config

# AutoGluon imports
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import tempfile
import shutil
import uuid
import logging

# Suppress AutoGluon warnings
logging.getLogger("autogluon").setLevel(logging.ERROR)


def evaluate_pipeline_with_autogluon(dataset, pipeline_dict, verbose=False):
    """
    Evaluate a pipeline configuration using AutoGluon.
    
    Args:
        dataset: Dictionary with 'X', 'y', 'name', 'id', 'task_type'
        pipeline_dict: Dictionary mapping preprocessing steps to operators
        verbose: Print evaluation details
        
    Returns:
        Performance score (accuracy for classification, R2 for regression)
    """
    try:
        # Suppress AutoGluon path warnings
        original_warnings = warnings.filters[:]
        warnings.filterwarnings('ignore', message='path already exists')
        
        # Convert pipeline dict to config format
        pipeline_config = pipeline_dict_to_config(pipeline_dict)
        
        X, y = dataset['X'], dataset['y']
        
        # Apply preprocessing
        X_processed, y_processed = apply_preprocessing(X, y, pipeline_config)
        
        if X_processed.empty or len(y_processed) == 0:
            if verbose:
                print(f"  Empty dataset after preprocessing")
            return 0.0
        
        # Detect problem type
        unique_classes = np.unique(y_processed)
        if np.issubdtype(y_processed.dtype, np.number) and len(unique_classes) > 20:
            problem_type = "regression"
        elif len(unique_classes) == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"
        
        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42,
                stratify=y_processed if problem_type != "regression" else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42
            )
        
        # Prepare data for AutoGluon
        train_data = X_train.copy()
        train_data['target'] = y_train
        test_data = X_test.copy()
        
        # Create unique temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Train AutoGluon predictor
            predictor = TabularPredictor(
                label="target",
                path=temp_dir,
                problem_type=problem_type,
                eval_metric=("r2" if problem_type == "regression" else "accuracy"),
                verbosity=AUTOGLUON_CONFIG["verbosity"]
            )
            
            predictor.fit(
                train_data=train_data,
                time_limit=AUTOGLUON_CONFIG["time_limit"],
                presets=AUTOGLUON_CONFIG["presets"],
                hyperparameter_tune_kwargs=AUTOGLUON_CONFIG["hyperparameter_tune_kwargs"],
                ag_args_fit=AUTOGLUON_CONFIG["ag_args_fit"],
                feature_generator=IdentityFeatureGenerator()
            )
            
            # Evaluate
            predictions = predictor.predict(test_data)
            
            if problem_type == "regression":
                score = r2_score(y_test, predictions)
            else:
                score = accuracy_score(y_test, predictions)
            
            return float(score)
            
        except Exception as e:
            if verbose:
                print(f"  AutoGluon error: {e}")
            return 0.0
            
        finally:
            # Cleanup and restore warnings
            shutil.rmtree(temp_dir, ignore_errors=True)
            warnings.filters = original_warnings
    
    except Exception as e:
        if verbose:
            print(f"  Evaluation error: {e}")
        return 0.0


def optimize_dataset_pipeline(dataset, aco_params=None, verbose=True):
    """
    Use ACO to find the optimal pipeline for a single dataset.
    
    Args:
        dataset: Dataset dictionary
        aco_params: Dictionary of ACO parameters (optional)
        verbose: Print progress
        
    Returns:
        Dictionary with optimization results
    """
    # Default ACO parameters
    default_params = {
        'n_ants': 15,
        'n_iterations': 30,
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.1,
        'q0': 0.9,
        'elite_weight': 2.0,
        'verbose': verbose
    }
    
    if aco_params:
        default_params.update(aco_params)
    
    # Create ACO optimizer
    optimizer = ACOPipelineOptimizer(
        options=options,
        **default_params
    )
    
    # Run optimization
    results = optimizer.optimize(
        dataset=dataset,
        evaluate_func=evaluate_pipeline_with_autogluon,
        evaluate_kwargs={'verbose': False}
    )
    
    # Add optimizer statistics
    results['pheromone_summary'] = optimizer.get_pheromone_summary()
    results['operator_statistics'] = optimizer.get_operator_statistics()
    
    return results


def run_aco_on_test_datasets(
    test_datasets,
    aco_params=None,
    save_results=True,
    output_prefix="aco_results"
):
    """
    Run ACO optimization on multiple test datasets.
    
    Args:
        test_datasets: List of dataset dictionaries
        aco_params: ACO parameters (optional)
        save_results: Save results to CSV
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with all results
    """
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"ACO PIPELINE OPTIMIZATION - BATCH MODE")
    print(f"{'='*80}")
    print(f"Total datasets: {len(test_datasets)}")
    print(f"{'='*80}\n")
    
    for idx, dataset in enumerate(test_datasets):
        print(f"\n{'='*80}")
        print(f"Dataset {idx+1}/{len(test_datasets)}: {dataset['name']} (ID: {dataset['id']})")
        print(f"{'='*80}")
        
        # Run optimization
        result = optimize_dataset_pipeline(dataset, aco_params=aco_params, verbose=True)
        
        # Store results
        dataset_result = {
            'dataset_id': dataset['id'],
            'dataset_name': dataset['name'],
            'best_score': result['best_score'],
            'best_pipeline': result['best_pipeline'],
            'total_time': result['total_time'],
            'iterations': len(result['iteration_best_scores']),
            'final_convergence': result['iteration_best_scores'][-1] if result['iteration_best_scores'] else 0
        }
        
        all_results.append(dataset_result)
        
        # Print summary
        print(f"\nDataset {dataset['name']} - Best Score: {result['best_score']:.4f}")
        print(f"Best Pipeline: {result['best_pipeline']}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Average Best Score: {summary_df['best_score'].mean():.4f}")
    print(f"Std Best Score: {summary_df['best_score'].std():.4f}")
    print(f"Min Best Score: {summary_df['best_score'].min():.4f}")
    print(f"Max Best Score: {summary_df['best_score'].max():.4f}")
    print(f"Average Time: {summary_df['total_time'].mean():.2f}s")
    print(f"{'='*80}\n")
    
    # Save results
    if save_results:
        summary_df.to_csv(f'{output_prefix}_summary.csv', index=False)
        print(f"✓ Saved summary to {output_prefix}_summary.csv")
        
        # Save detailed results
        detailed_results = []
        for res in all_results:
            row = {
                'dataset_id': res['dataset_id'],
                'dataset_name': res['dataset_name'],
                'best_score': res['best_score'],
                'total_time': res['total_time']
            }
            # Add pipeline components
            for step, operator in res['best_pipeline'].items():
                row[f'pipeline_{step}'] = operator
            detailed_results.append(row)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(f'{output_prefix}_detailed.csv', index=False)
        print(f"✓ Saved detailed results to {output_prefix}_detailed.csv")
    
    return {
        'summary': summary_df,
        'all_results': all_results
    }


def compare_aco_with_baseline(
    test_datasets,
    baseline_pipelines,
    aco_params=None
):
    """
    Compare ACO-optimized pipelines with baseline pipelines.
    
    Args:
        test_datasets: List of datasets
        baseline_pipelines: List of baseline pipeline configs
        aco_params: ACO parameters
        
    Returns:
        Comparison results
    """
    comparison_results = []
    
    for dataset in test_datasets:
        print(f"\n{'='*80}")
        print(f"Comparing on {dataset['name']} (ID: {dataset['id']})")
        print(f"{'='*80}")
        
        # Run ACO
        print("\nRunning ACO optimization...")
        aco_result = optimize_dataset_pipeline(dataset, aco_params=aco_params, verbose=True)
        
        # Evaluate baselines
        print("\nEvaluating baseline pipelines...")
        baseline_scores = {}
        for pipeline_config in baseline_pipelines:
            score = evaluate_pipeline_with_autogluon(dataset, pipeline_config, verbose=False)
            baseline_scores[pipeline_config['name']] = score
            print(f"  {pipeline_config['name']:20s}: {score:.4f}")
        
        # Find best baseline
        best_baseline_name = max(baseline_scores, key=baseline_scores.get)
        best_baseline_score = baseline_scores[best_baseline_name]
        
        # Calculate improvement
        improvement = aco_result['best_score'] - best_baseline_score
        improvement_pct = (improvement / best_baseline_score * 100) if best_baseline_score > 0 else 0
        
        print(f"\nComparison:")
        print(f"  ACO Best: {aco_result['best_score']:.4f}")
        print(f"  Best Baseline ({best_baseline_name}): {best_baseline_score:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        comparison_results.append({
            'dataset_id': dataset['id'],
            'dataset_name': dataset['name'],
            'aco_score': aco_result['best_score'],
            'best_baseline': best_baseline_name,
            'best_baseline_score': best_baseline_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'aco_pipeline': aco_result['best_pipeline'],
            'baseline_scores': baseline_scores
        })
    
    return pd.DataFrame(comparison_results)


if __name__ == "__main__":
    print("ACO Pipeline Recommendation System")
    print("="*80)
    
    # Example: Load a few test datasets
    print("\nLoading test datasets...")
    test_datasets = []
    for dataset_id in test_dataset_ids[:19]:  # Test on first 3 datasets
        dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
        if dataset:
            test_datasets.append(dataset)
    
    print(f"Loaded {len(test_datasets)} test datasets")
    
    # Run ACO optimization
    print("\nRunning ACO optimization...")
    results = run_aco_on_test_datasets(
        test_datasets,
        aco_params={
            'n_ants': 8,
            'n_iterations': 5,
            'verbose': True
        },
        save_results=True,
        output_prefix="aco_test_run"
    )
    
    print("\n✓ Optimization complete!")
