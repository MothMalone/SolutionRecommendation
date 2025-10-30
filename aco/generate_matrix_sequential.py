"""
Sequential Performance Matrix Generator with Progress Tracking
More reliable than parallel version for AutoGluon
Includes checkpointing and detailed progress
"""

import os
import numpy as np
import pandas as pd
import itertools
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
os.environ['AUTOGLUON_VERBOSITY'] = '0'

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tqdm'])
    from tqdm import tqdm

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


def encode_pipeline_to_vector(pipeline_dict):
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


def decode_vector_to_pipeline(vector):
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


def generate_all_pipelines():
    """Generate all possible pipeline configurations."""
    steps = ['imputation', 'scaling', 'encoding', 'feature_selection', 
             'outlier_removal', 'dimensionality_reduction']
    step_choices = [options[step] for step in steps]
    
    all_pipelines = []
    for combo in itertools.product(*step_choices):
        pipeline = dict(zip(steps, combo))
        vector = encode_pipeline_to_vector(pipeline)
        all_pipelines.append((vector, pipeline))
    
    return all_pipelines


def evaluate_pipeline_on_dataset(pipeline_dict, dataset, verbose=False):
    """
    Evaluate a single pipeline on a single dataset.
    Returns: score (float)
    """
    try:
        # Convert to config format
        pipeline_config = {'name': 'eval_pipeline'}
        pipeline_config.update(pipeline_dict)
        
        X, y = dataset['X'], dataset['y']
        
        # Apply preprocessing
        X_processed, y_processed = apply_preprocessing(X, y, pipeline_config)
        
        if X_processed.empty or len(y_processed) == 0:
            if verbose:
                print(f"    Empty after preprocessing")
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
        
        # Prepare data
        train_data = X_train.copy()
        train_data['target'] = y_train
        test_data = X_test.copy()
        
        # Create temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"autogluon_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Train AutoGluon
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
            
            # Evaluate
            predictions = predictor.predict(test_data)
            
            if problem_type == "regression":
                score = r2_score(y_test, predictions)
            else:
                score = accuracy_score(y_test, predictions)
            
            return float(score)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return 0.0


def load_checkpoint(checkpoint_file):
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return {}
    return {}


def save_checkpoint(results, checkpoint_file):
    """Save checkpoint."""
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def generate_performance_matrix_sequential(
    datasets,
    all_pipelines,
    checkpoint_file='performance_matrix_checkpoint.pkl',
    save_every=50,
    output_file='testing_performance_matrix_autogluon.csv'
):
    """
    Generate performance matrix sequentially with progress tracking.
    More reliable than parallel version for AutoGluon.
    """
    
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL PERFORMANCE MATRIX GENERATION")
    print(f"{'='*80}")
    print(f"Datasets: {len(datasets)}")
    print(f"Pipelines: {len(all_pipelines)}")
    print(f"Total evaluations: {len(datasets) * len(all_pipelines)}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Save every: {save_every} evaluations")
    print(f"{'='*80}\n")
    
    # Load existing results
    results = load_checkpoint(checkpoint_file)
    print(f"Loaded {len(results)} existing results from checkpoint\n")
    
    # Prepare dataset names
    dataset_names = [d['name'] for d in datasets]
    
    # Create list of all tasks
    all_tasks = []
    for dataset in datasets:
        for vector, pipeline in all_pipelines:
            pipeline_id = vector_to_string(vector)
            all_tasks.append((pipeline_id, dataset['name'], pipeline, dataset))
    
    # Filter out completed tasks
    remaining_tasks = [
        task for task in all_tasks 
        if (task[0], task[1]) not in results
    ]
    
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Already completed: {len(all_tasks) - len(remaining_tasks)}")
    print(f"Remaining: {len(remaining_tasks)}\n")
    
    if len(remaining_tasks) == 0:
        print("All evaluations already completed!")
    else:
        # Estimate time
        avg_time_per_eval = 30  # seconds
        est_time = len(remaining_tasks) * avg_time_per_eval / 60
        print(f"Estimated time: {est_time:.1f} minutes ({est_time/60:.1f} hours)\n")
        
        # Process tasks with progress bar
        start_time = time.time()
        eval_times = []
        
        with tqdm(total=len(remaining_tasks), desc="Evaluating", unit="eval") as pbar:
            for i, (pipeline_id, dataset_name, pipeline, dataset) in enumerate(remaining_tasks):
                eval_start = time.time()
                
                # Evaluate
                score = evaluate_pipeline_on_dataset(pipeline, dataset, verbose=False)
                results[(pipeline_id, dataset_name)] = score
                
                eval_time = time.time() - eval_start
                eval_times.append(eval_time)
                
                # Update progress bar
                avg_time = np.mean(eval_times[-20:])  # Use last 20 for moving average
                pbar.set_postfix({
                    'score': f'{score:.3f}',
                    'avg_time': f'{avg_time:.1f}s',
                    'dataset': dataset_name[:10]
                })
                pbar.update(1)
                
                # Save checkpoint periodically
                if (i + 1) % save_every == 0:
                    save_checkpoint(results, checkpoint_file)
                    
                    # Also save intermediate CSV
                    if (i + 1) % (save_every * 5) == 0:
                        save_intermediate_matrix(results, all_pipelines, dataset_names, 
                                                f"{output_file}.partial")
        
        # Final save
        save_checkpoint(results, checkpoint_file)
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Evaluation complete!")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Average time per evaluation: {np.mean(eval_times):.1f} seconds")
        print(f"{'='*80}\n")
    
    # Convert to DataFrame
    print("Converting results to DataFrame...")
    matrix_data = []
    pipeline_indices = []
    
    for vector, pipeline in tqdm(all_pipelines, desc="Building matrix"):
        pipeline_id = vector_to_string(vector)
        pipeline_indices.append(pipeline_id)
        
        row = []
        for dataset_name in dataset_names:
            score = results.get((pipeline_id, dataset_name), np.nan)
            row.append(score)
        
        matrix_data.append(row)
    
    df = pd.DataFrame(matrix_data, index=pipeline_indices, columns=dataset_names)
    
    # Save final matrix
    df.to_csv(output_file)
    print(f"\n✓ Saved final matrix to {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    print(f"  Mean score: {df.mean().mean():.4f}")
    
    return df


def save_intermediate_matrix(results, all_pipelines, dataset_names, filename):
    """Save intermediate matrix during processing."""
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
    df.to_csv(filename)
    print(f"\n  ✓ Saved intermediate checkpoint to {filename}")


def main(n_datasets=None, n_pipelines=None):
    """Main execution."""
    
    # Load datasets
    print("Loading datasets...")
    datasets = []
    dataset_limit = n_datasets if n_datasets else len(test_dataset_ids)
    
    for i, dataset_id in enumerate(test_dataset_ids):
        if i >= dataset_limit:
            break
        dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
        if dataset:
            datasets.append(dataset)
            print(f"  ✓ {dataset['name']} (ID: {dataset['id']}, Shape: {dataset['X'].shape})")
    
    print(f"\nTotal datasets: {len(datasets)}")
    
    # Generate pipelines
    print("\nGenerating pipeline configurations...")
    all_pipelines = generate_all_pipelines()
    
    if n_pipelines:
        import random
        random.seed(42)
        all_pipelines = random.sample(all_pipelines, min(n_pipelines, len(all_pipelines)))
    
    print(f"Total pipelines: {len(all_pipelines)}")
    print(f"Total evaluations: {len(datasets) * len(all_pipelines)}")
    
    # Generate matrix
    performance_matrix = generate_performance_matrix_sequential(
        datasets=datasets,
        all_pipelines=all_pipelines,
        checkpoint_file='performance_matrix_checkpoint.pkl',
        save_every=50,
        output_file='testing_performance_matrix_autogluon.csv'
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Matrix shape: {performance_matrix.shape}")
    print(f"Mean score: {performance_matrix.mean().mean():.4f}")
    print(f"Std score: {performance_matrix.std().mean():.4f}")
    print(f"Best pipeline overall: {performance_matrix.mean(axis=1).idxmax()}")
    print(f"Best score: {performance_matrix.mean(axis=1).max():.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate performance matrix sequentially')
    parser.add_argument('--test', action='store_true', help='Run small test (2 datasets, 20 pipelines)')
    parser.add_argument('--datasets', type=int, default=None, help='Number of datasets to use')
    parser.add_argument('--pipelines', type=int, default=None, help='Number of pipelines to use')
    
    args = parser.parse_args()
    
    if args.test:
        print("\n🧪 Running TEST mode (2 datasets, 20 pipelines)")
        main(n_datasets=2, n_pipelines=20)
    else:
        main(n_datasets=args.datasets, n_pipelines=args.pipelines)
