"""
Parallel Performance Matrix Generator with Progress Tracking
Generates comprehensive performance matrix with fault tolerance and periodic saving
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool, cpu_count, Manager
import itertools
import time
import pickle
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")

from setting_aco import (
    options,
    test_dataset_ids,
    load_openml_dataset,
    apply_preprocessing,
    AUTOGLUON_CONFIG
)

from autogluon.tabular import TabularPredictor
from autogluon.features.generators import IdentityFeatureGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import tempfile
import shutil
import uuid


def encode_pipeline_to_vector(pipeline_dict, options):
    """Encode pipeline as vector of operator indices."""
    vector = []
    for step in ['imputation', 'scaling', 'encoding', 'feature_selection', 
                 'outlier_removal', 'dimensionality_reduction']:
        operator = pipeline_dict.get(step, 'none')
        try:
            idx = options[step].index(operator)
        except ValueError:
            idx = 0
        vector.append(idx)
    return vector


def decode_vector_to_pipeline(vector, options):
    """Decode vector back to pipeline dict."""
    pipeline = {}
    steps = ['imputation', 'scaling', 'encoding', 'feature_selection', 
             'outlier_removal', 'dimensionality_reduction']
    for i, step in enumerate(steps):
        pipeline[step] = options[step][vector[i]]
    return pipeline


def vector_to_string(vector):
    """Convert vector to string for use as index."""
    return '_'.join(map(str, vector))


def string_to_vector(string):
    """Convert string back to vector."""
    return [int(x) for x in string.split('_')]


def generate_all_pipelines(options):
    """Generate all possible pipeline configurations."""
    steps = ['imputation', 'scaling', 'encoding', 'feature_selection', 
             'outlier_removal', 'dimensionality_reduction']
    step_choices = [options[step] for step in steps]
    
    all_pipelines = []
    for combo in itertools.product(*step_choices):
        pipeline = dict(zip(steps, combo))
        vector = encode_pipeline_to_vector(pipeline, options)
        all_pipelines.append((vector, pipeline))
    
    return all_pipelines


def evaluate_single_pipeline(args):
    """
    Evaluate a single pipeline on a single dataset.
    Returns: (pipeline_id, dataset_name, score, error_msg)
    """
    pipeline_dict, dataset, dataset_name, pipeline_id = args
    
    try:
        # Convert to config format
        pipeline_config = {'name': 'eval_pipeline'}
        pipeline_config.update(pipeline_dict)
        
        X, y = dataset['X'], dataset['y']
        
        # Apply preprocessing
        X_processed, y_processed = apply_preprocessing(X, y, pipeline_config)
        
        if X_processed.empty or len(y_processed) == 0:
            return (pipeline_id, dataset_name, 0.0, "Empty after preprocessing")
        
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
        
        # Prepare data
        train_data = X_train.copy()
        train_data['target'] = y_train
        test_data = X_test.copy()
        
        # Create temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"ag_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            predictor = TabularPredictor(
                label="target",
                path=temp_dir,
                problem_type=problem_type,
                eval_metric=("r2" if problem_type == "regression" else "accuracy"),
                verbosity=0
            )
            
            predictor.fit(
                train_data=train_data,
                time_limit=AUTOGLUON_CONFIG["time_limit"],
                presets=AUTOGLUON_CONFIG["presets"],
                hyperparameter_tune_kwargs=AUTOGLUON_CONFIG.get("hyperparameter_tune_kwargs"),
                ag_args_fit=AUTOGLUON_CONFIG.get("ag_args_fit", {}),
                feature_generator=IdentityFeatureGenerator()
            )
            
            predictions = predictor.predict(test_data)
            
            if problem_type == "regression":
                score = r2_score(y_test, predictions)
            else:
                score = accuracy_score(y_test, predictions)
            
            return (pipeline_id, dataset_name, float(score), None)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        return (pipeline_id, dataset_name, 0.0, str(e))


def save_checkpoint(results, checkpoint_file, metadata=None):
    """Save results checkpoint."""
    checkpoint = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_file):
    """Load results from checkpoint."""
    if not os.path.exists(checkpoint_file):
        return {}, {}
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint.get('results', {}), checkpoint.get('metadata', {})


def generate_performance_matrix_parallel(
    datasets,
    all_pipelines,
    options,
    n_workers=None,
    checkpoint_file='performance_matrix_checkpoint.pkl',
    save_every=100,
    output_csv='testing_performance_matrix_autogluon.csv'
):
    """
    Generate performance matrix using parallel processing with progress tracking.
    """
    if n_workers is None:
        n_workers = max(1, min(cpu_count() - 1, 8))  # Cap at 8
    
    print(f"\n{'='*80}")
    print(f"PARALLEL PERFORMANCE MATRIX GENERATION")
    print(f"{'='*80}")
    print(f"Datasets: {len(datasets)}")
    print(f"Pipelines: {len(all_pipelines)}")
    print(f"Total evaluations: {len(datasets) * len(all_pipelines)}")
    print(f"Parallel workers: {n_workers}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Save every: {save_every} evaluations")
    print(f"{'='*80}\n")
    
    # Load checkpoint if exists
    results, metadata = load_checkpoint(checkpoint_file)
    
    if results:
        print(f"✓ Loaded checkpoint with {len(results)} previous results")
        print(f"  Last saved: {metadata.get('timestamp', 'Unknown')}")
    
    # Prepare dataset names
    dataset_names = [d['name'] for d in datasets]
    
    # Create all evaluation tasks
    tasks = []
    for dataset in datasets:
        for vector, pipeline in all_pipelines:
            pipeline_id = vector_to_string(vector)
            
            # Skip if already computed
            if (pipeline_id, dataset['name']) not in results:
                tasks.append((pipeline, dataset, dataset['name'], pipeline_id))
    
    total_tasks = len(tasks)
    already_done = len(results)
    
    print(f"Tasks remaining: {total_tasks}")
    print(f"Already completed: {already_done}")
    print(f"Total: {total_tasks + already_done}\n")
    
    if total_tasks == 0:
        print("✓ All evaluations already completed!")
    else:
        # Estimate time
        avg_time_per_eval = 30  # seconds
        estimated_time = (total_tasks * avg_time_per_eval) / (n_workers * 60)
        print(f"Estimated time: {estimated_time:.1f} minutes\n")
        
        # Run parallel evaluation
        print(f"Starting parallel evaluation...")
        start_time = time.time()
        
        completed_count = 0
        error_count = 0
        
        with Pool(processes=n_workers) as pool:
            # Use imap_unordered for better progress tracking
            if TQDM_AVAILABLE:
                pbar = tqdm(total=total_tasks, desc="Evaluating", unit="eval")
            
            for result in pool.imap_unordered(evaluate_single_pipeline, tasks, chunksize=5):
                pipeline_id, dataset_name, score, error = result
                results[(pipeline_id, dataset_name)] = score
                
                if error:
                    error_count += 1
                
                completed_count += 1
                
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix({
                        'errors': error_count,
                        'avg_score': f'{np.mean([v for v in results.values() if v > 0]):.3f}'
                    })
                else:
                    if completed_count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed
                        remaining = (total_tasks - completed_count) / rate / 60
                        print(f"Progress: {completed_count}/{total_tasks} "
                              f"({completed_count/total_tasks*100:.1f}%) "
                              f"- Rate: {rate:.1f}/s - ETA: {remaining:.1f} min "
                              f"- Errors: {error_count}")
                
                # Save checkpoint periodically
                if completed_count % save_every == 0:
                    metadata = {
                        'completed': already_done + completed_count,
                        'total': already_done + total_tasks,
                        'errors': error_count
                    }
                    save_checkpoint(results, checkpoint_file, metadata)
                    
                    # Also save CSV periodically
                    if completed_count % (save_every * 5) == 0:
                        df = convert_results_to_dataframe(results, all_pipelines, dataset_names)
                        df.to_csv(output_csv)
                        print(f"\n✓ Periodic save: {output_csv} ({df.shape})\n")
            
            if TQDM_AVAILABLE:
                pbar.close()
        
        # Final save
        metadata = {
            'completed': already_done + completed_count,
            'total': already_done + total_tasks,
            'errors': error_count
        }
        save_checkpoint(results, checkpoint_file, metadata)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Parallel evaluation complete!")
        print(f"Completed: {completed_count} evaluations")
        print(f"Errors: {error_count}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Average time per evaluation: {elapsed/completed_count:.2f} seconds")
        print(f"{'='*80}\n")
    
    # Convert results to DataFrame
    print("Converting results to DataFrame...")
    df = convert_results_to_dataframe(results, all_pipelines, dataset_names)
    
    return df


def convert_results_to_dataframe(results, all_pipelines, dataset_names):
    """Convert results dictionary to DataFrame."""
    matrix_data = []
    pipeline_indices = []
    
    for vector, pipeline in all_pipelines:
        pipeline_id = vector_to_string(vector)
        pipeline_indices.append(pipeline_id)
        
        row = []
        for dataset_name in dataset_names:
            score = results.get((pipeline_id, dataset_name), np.nan)
            row.append(score)
        
        matrix_data.append(row)
    
    df = pd.DataFrame(matrix_data, index=pipeline_indices, columns=dataset_names)
    return df


def test_parallel_system(n_test_pipelines=10, n_test_datasets=2):
    """
    Test the parallel system with a small subset.
    """
    print(f"\n{'='*80}")
    print(f"TESTING PARALLEL SYSTEM")
    print(f"{'='*80}\n")
    
    # Load test datasets
    print(f"Loading {n_test_datasets} test datasets...")
    datasets = []
    for dataset_id in test_dataset_ids[:n_test_datasets]:
        dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
        if dataset:
            datasets.append(dataset)
            print(f"  ✓ {dataset['name']}")
    
    # Generate subset of pipelines
    print(f"\nGenerating {n_test_pipelines} random pipelines...")
    all_pipelines = generate_all_pipelines(options)
    np.random.seed(42)
    test_indices = np.random.choice(len(all_pipelines), n_test_pipelines, replace=False)
    test_pipelines = [all_pipelines[i] for i in test_indices]
    
    for i, (vec, pipe) in enumerate(test_pipelines[:3]):
        print(f"  {i+1}. {pipe}")
    print(f"  ... ({len(test_pipelines)} total)")
    
    # Run test
    print(f"\nRunning test evaluation...")
    print(f"Total evaluations: {len(datasets) * len(test_pipelines)}")
    
    df = generate_performance_matrix_parallel(
        datasets=datasets,
        all_pipelines=test_pipelines,
        options=options,
        n_workers=2,
        checkpoint_file='test_checkpoint.pkl',
        save_every=5,
        output_csv='test_matrix.csv'
    )
    
    # Show results
    print(f"\n{'='*80}")
    print(f"TEST RESULTS")
    print(f"{'='*80}")
    print(f"Matrix shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"\nSample scores:")
    print(df.head())
    print(f"\nStatistics:")
    print(df.describe())
    
    return df


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance matrix in parallel')
    parser.add_argument('--test', action='store_true', help='Run test mode with subset')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--save-every', type=int, default=100, help='Save checkpoint frequency')
    parser.add_argument('--output', type=str, default='testing_performance_matrix_autogluon.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        test_parallel_system(n_test_pipelines=20, n_test_datasets=2)
    else:
        # Full mode
        print("Loading all datasets...")
        datasets = []
        for dataset_id in test_dataset_ids:
            dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
            if dataset:
                datasets.append(dataset)
                print(f"  ✓ {dataset['name']} (ID: {dataset['id']}, Shape: {dataset['X'].shape})")
        
        print(f"\nTotal datasets: {len(datasets)}")
        
        print("\nGenerating all pipeline configurations...")
        all_pipelines = generate_all_pipelines(options)
        print(f"Total pipelines: {len(all_pipelines)}")
        
        # Generate matrix
        df = generate_performance_matrix_parallel(
            datasets=datasets,
            all_pipelines=all_pipelines,
            options=options,
            n_workers=args.workers,
            checkpoint_file='performance_matrix_checkpoint.pkl',
            save_every=args.save_every,
            output_csv=args.output
        )
        
        # Save final
        df.to_csv(args.output)
        print(f"\n✓ Saved final matrix to {args.output}")
        print(f"  Shape: {df.shape}")
        print(f"  Missing: {df.isna().sum().sum()}")
        
        # Summary
        print(f"\nSummary Statistics:")
        print(f"  Mean: {df.mean().mean():.4f}")
        print(f"  Std: {df.std().mean():.4f}")
        print(f"  Min: {df.min().min():.4f}")
        print(f"  Max: {df.max().max():.4f}")


if __name__ == "__main__":
    main()
