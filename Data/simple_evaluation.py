#!/usr/bin/env python3
"""
Simple Performance Matrix Generator
====================================
Evaluates all 12 preprocessing pipelines on new datasets.
No train/val/test splits - uses entire dataset with AutoGluon's internal CV.

Usage:
    python simple_evaluation.py
"""

import pandas as pd
import numpy as np
import os
import warnings
import tempfile
import shutil
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from autogluon.tabular import TabularPredictor
import json

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')

# Import from test.py
import sys
sys.path.insert(0, '/drive1/nammt/SoluRec')
from test import (
    AG_ARGS_FIT,
    STABLE_MODELS,
    pipeline_configs,
    Preprocessor
)

# Load NEW dataset IDs only
with open('dataset_lists/new_training_datasets.json', 'r') as f:
    NEW_TRAIN_IDS = json.load(f)['dataset_ids']

print(f"📊 Will evaluate {len(NEW_TRAIN_IDS)} NEW datasets × 12 pipelines")
print(f"   Estimated time: ~{len(NEW_TRAIN_IDS) * 12 * 30 / 3600:.1f} hours")

def load_dataset(dataset_id):
    """Load dataset from OpenML"""
    try:
        dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X, y = dataset.data, dataset.target
        
        # Handle categorical
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(['category']).columns
            if len(categorical_cols) > 0:
                X = X.copy()
                X.loc[:, categorical_cols] = X.loc[:, categorical_cols].astype(object)
        
        # Encode target
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = pd.Series(LabelEncoder().fit_transform(y), name=y.name)
        
        # Remove NaN targets
        valid_indices = y.dropna().index
        X = X.loc[valid_indices].reset_index(drop=True)
        y = y.loc[valid_indices].reset_index(drop=True)
        
        # Subsample if too large
        if len(X) > 5000:
            X, y = shuffle(X, y, n_samples=5000, random_state=42)
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        
        if len(X) < 20:
            return None
            
        print(f"   ✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features, {y.nunique()} classes")
        return X, y
        
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return None

def evaluate_pipeline(X, y, pipeline_config, dataset_id):
    """Evaluate one pipeline on dataset using AutoGluon"""
    temp_dir = None
    try:
        # Apply preprocessing
        if pipeline_config['name'] == 'baseline':
            X_processed = X.copy()
            y_processed = y.copy()
        else:
            preprocessor = Preprocessor(pipeline_config)
            preprocessor.fit(X, y)
            X_processed, y_processed = preprocessor.transform(X, y)
        
        if len(X_processed) < 20:
            print(f"      ⚠️  Too little data after preprocessing")
            return np.nan
        
        # Reset column names
        X_processed = X_processed.copy()
        X_processed.columns = [f"col_{i}" for i in range(X_processed.shape[1])]
        
        # Create training data
        train_data = X_processed.copy()
        train_data['target'] = y_processed.values
        
        # Train AutoGluon
        temp_dir = tempfile.mkdtemp(prefix=f"ag_{dataset_id}_{pipeline_config['name']}_")
        
        problem_type = 'binary' if y_processed.nunique() <= 2 else 'multiclass'
        predictor = TabularPredictor(
            label='target',
            path=temp_dir,
            problem_type=problem_type,
            eval_metric='accuracy',
            verbosity=0
        )
        
        predictor.fit(
            train_data,
            time_limit=600,
            presets='medium_quality',
            included_model_types=STABLE_MODELS,
            hyperparameter_tune_kwargs=None,
            feature_generator=None,
            ag_args_fit=AG_ARGS_FIT,
            raise_on_no_models_fitted=False
        )
        
        # Get validation score (AutoGluon uses internal CV)
        leaderboard = predictor.leaderboard()
        best_score = leaderboard['score_val'].max()
        
        return best_score
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return np.nan
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    print("\n" + "="*80)
    print("SIMPLE PERFORMANCE MATRIX GENERATION")
    print("="*80)
    
    # Initialize results dataframe
    pipeline_names = [config['name'] for config in pipeline_configs]
    results = pd.DataFrame(
        index=pipeline_names,
        columns=[f'D_{ds_id}' for ds_id in NEW_TRAIN_IDS],
        dtype=float
    )
    
    # Checkpoint system
    checkpoint_file = 'checkpoints/simple_eval_checkpoint.csv'
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Loading checkpoint...")
        results = pd.read_csv(checkpoint_file, index_col=0)
        completed = results.notna().any(axis=0).sum()
        print(f"   ✅ Resumed: {completed}/{len(NEW_TRAIN_IDS)} datasets completed")
    
    # Evaluate each dataset
    for idx, dataset_id in enumerate(NEW_TRAIN_IDS, 1):
        col_name = f'D_{dataset_id}'
        
        # Skip if already completed
        if results[col_name].notna().all():
            print(f"\n[{idx}/{len(NEW_TRAIN_IDS)}] Dataset {dataset_id} - ALREADY COMPLETED ✓")
            continue
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(NEW_TRAIN_IDS)}] Evaluating Dataset {dataset_id}")
        print(f"{'='*80}")
        
        # Load dataset
        data = load_dataset(dataset_id)
        if data is None:
            print(f"   ❌ Skipping dataset {dataset_id}")
            continue
        
        X, y = data
        
        # Evaluate each pipeline
        for i, config in enumerate(pipeline_configs, 1):
            pipeline_name = config['name']
            
            # Skip if already done
            if pd.notna(results.loc[pipeline_name, col_name]):
                print(f"   [{i}/12] {pipeline_name:30s} - CACHED ✓")
                continue
            
            print(f"   [{i}/12] {pipeline_name:30s} - ", end='', flush=True)
            
            score = evaluate_pipeline(X, y, config, dataset_id)
            results.loc[pipeline_name, col_name] = score
            
            if not np.isnan(score):
                print(f"✅ {score:.4f}")
            else:
                print(f"❌ FAILED")
        
        # Save checkpoint after each dataset
        results.to_csv(checkpoint_file)
        print(f"\n   💾 Checkpoint saved")
    
    # Save final results
    output_file = 'new_training_performance_106datasets.csv'
    results.to_csv(output_file)
    
    print(f"\n" + "="*80)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"="*80)
    print(f"Output: {output_file}")
    print(f"Shape: {results.shape} (12 pipelines × {results.shape[1]} datasets)")
    print(f"Valid scores: {results.notna().sum().sum()} / {results.size}")
    print(f"Mean accuracy: {results.mean().mean():.4f}")
    
    print(f"\n📋 Next steps:")
    print(f"1. Merge with old performance matrix:")
    print(f"   python merge_performance_matrices.py \\")
    print(f"       --old preprocessed_performance.csv \\")
    print(f"       --new {output_file} \\")
    print(f"       --output expanded_preprocessed_performance.csv")
    print(f"\n2. Use merged matrix for training:")
    print(f"   cp expanded_preprocessed_performance.csv preprocessed_performance.csv")
    print(f"   python compare_all_recommenders.py")

if __name__ == '__main__':
    main()
