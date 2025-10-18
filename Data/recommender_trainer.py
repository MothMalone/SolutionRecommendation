import pandas as pd
import numpy as np
import os
import warnings
import tempfile
import shutil
import uuid
import argparse
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier, AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import zscore
import scipy.stats as st
import xgboost as xgb
from autogluon.tabular import TabularPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders import (
    PmmRecommender, HybridMetaRecommender, BayesianSurrogateRecommender,
    AutoGluonPipelineRecommender, RandomRecommender, AverageRankRecommender,
    L1Recommender, BasicRecommender, KnnRecommender, RFRecommender,
    NNRecommender, RegressorRecommender, AdaBoostRegressorRecommender,
    set_pipeline_configs, set_ag_config
)

from evaluation_utils import (
    Preprocessor, can_stratify, safe_train_test_split, load_openml_dataset,
    get_metafeatures, run_autogluon_evaluation, run_experiment_for_dataset,
    analyze_recommendations
)

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')

# ==============================================================================
# REPRODUCIBILITY - Set random seeds
# ==============================================================================
def set_all_seeds(seed=42):
    """Fix all random seeds for reproducible results."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# AutoGluon settings
AG_ARGS_FIT = {
    "ag.max_memory_usage_ratio": 0.3,
    'num_gpus': 1,
    'num_cpus': min(10, os.cpu_count() if os.cpu_count() else 4)
}

# AutoGluon supported models
STABLE_MODELS = [
    "GBM", "CAT", "XGB", "RF", "XT", "KNN", "LR", "NN_TORCH", "FASTAI",
    "NN_MXNET", "TABPFN", "DUMMY", "NB"
]

# Dataset splits
train_dataset_ids = [40, 41, 60, 187, 308, 310, 464, 468]  # Train recommender
test_dataset_ids = [1503, 23517, 1551, 1552, 183, 255, 546, 475, 481, 516, 3, 6, 8, 10, 12, 14, 9, 11, 5]  # Evaluate

# Preprocessing pipelines to evaluate
pipeline_configs = [        problem_type = 'binary' if y_train.nunique() <= 2 else 'multiclass'
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
        
        preds = predictor.predict(X_test_ag)
        return accuracy_score(y_test, preds)
        
    except Exception as e:
        print(f"      Warning: AutoGluon evaluation failed: {e}")
        
        # Fallback to a simple RandomForest
        try:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            return accuracy_score(y_test, preds)
        except Exception as e2:
            print(f"      Warning: Fallback classifier also failed: {e2}")
            return np.nan
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


# =============================================================================
# MAIN FUNCTION - For training and evaluating recommenders
# =============================================================================

def paper_pmm(meta_features_df=None, performance_matrix=None, recommender_type='paper_pmm'):
    if recommender_type == 'paper_pmm':
        print("\nTraining Paper-Style PMM Recommender (Dataset+Pipeline pairs)...")
        from pmm_paper_style import PaperPmmRecommender
        recommender = PaperPmmRecommender(
            hidden_dim=128,
            embedding_dim=64,
            margin=0.8,
            batch_size=64,
            num_epochs=500,
            learning_rate=0.001,
            use_influence_weighting=args.use_influence,
            influence_method=args.influence_method if args.use_influence else 'performance_variance'
        )
        if args.use_influence:
            print(f"  Using influence weighting with method: {args.influence_method}")
        recommender_success = recommender.fit(performance_matrix, meta_features_df, verbose=True)
    
    else:  # Default to baseline (no actual recommender needed)
        print("\nUsing baseline recommender (best local performance)")
        recommender_success = True
    
    print("\n" + "="*80)
    print("EVALUATION ON TEST DATASET")
    print("="*80)
    
    # NEW: For paper_pmm only, evaluate on ALL test datasets using ground truth
    if recommender_type == 'paper_pmm' and recommender_success and recommender is not None:
        try:
            ground_truth_perf_matrix = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
            print(f"✅ Loaded test ground truth: {ground_truth_perf_matrix.shape}")
            print(f"   Test datasets: {list(ground_truth_perf_matrix.columns)}")
            
            test_summaries = []

            avg_score = 0
            
            for ds_col in ground_truth_perf_matrix.columns:
                # Extract dataset ID
                if ds_col.startswith('D_'):
                    ds_id = int(ds_col[2:])
                else:
                    ds_id = ds_col
                
                # Check if dataset has metafeatures
                if ds_id not in meta_features_df.index:
                    print(f"  ⚠️  Dataset {ds_id} not in metafeatures, skipping")
                    continue
                
                # Get metafeatures
                dataset_metafeats = meta_features_df.loc[ds_id].values
                
                # Get ground truth performances
                ground_truth_perf = ground_truth_perf_matrix[ds_col].dropna()
                
                if len(ground_truth_perf) == 0:
                    continue
                
                print(f"\n  Evaluating on dataset {ds_id}...")
                
                # Get recommendation from paper_pmm
                top_k_pipelines = recommender.recommend(dataset_metafeats, top_k=5)
                recommended_pipeline = top_k_pipelines[0]
                
                # Evaluate recommendation
                best_pipeline = ground_truth_perf.idxmax()
                best_score = ground_truth_perf.max()
                recommended_score = ground_truth_perf[recommended_pipeline]
                avg_score += recommended_score
                baseline_score = ground_truth_perf.get('baseline', np.nan)
                
                # Calculate rank
                sorted_pipelines = ground_truth_perf.sort_values(ascending=False)
                rank = list(sorted_pipelines.index).index(recommended_pipeline) + 1
                
                print(f"    Recommended: {recommended_pipeline} (rank {rank}/{len(ground_truth_perf)})")
                print(f"    Score: {recommended_score:.4f} (best: {best_score:.4f}, gap: {best_score-recommended_score:.4f})")
                
                test_summaries.append({
                    'dataset': ds_col,
                    'recommended': recommended_pipeline,
                    'best': best_pipeline,
                    'rank': rank,
                    'recommended_score': recommended_score,
                    'best_score': best_score,
                    'baseline_score': baseline_score,
                    'score_gap': best_score - recommended_score,
                    'better_than_baseline': 'yes' if recommended_score > baseline_score else ('equal' if np.isclose(recommended_score, baseline_score) else 'no')
                })
            
    

            # Show aggregate results
            if test_summaries:
                summary_df = pd.DataFrame(test_summaries)
                
                print("\n" + "="*80)
                print("AGGREGATE TEST RESULTS")
                print("="*80)
                print(f"Total test datasets evaluated: {len(summary_df)}")
                print(f"Average rank: {summary_df['rank'].mean():.2f}")
                print(f"Top-1 accuracy: {(summary_df['rank'] == 1).mean()*100:.1f}%")
                print(f"Top-3 accuracy: {(summary_df['rank'] <= 3).mean()*100:.1f}%")
                print(f"Avg recommended score:  {(avg_score/19.0):.4f}")
                print(f"Average score gap: {summary_df['score_gap'].mean():.4f}")
                print(f"Better than baseline: {(summary_df['better_than_baseline'] == 'yes').mean()*100:.1f}%")
                
                # Save results
                summary_df.to_csv('paper_pmm_test_results.csv', index=False)
                print(f"\n✅ Saved detailed results to 'paper_pmm_test_results.csv'")
                
                # Show some example recommendations
                print("\n" + "="*80)
                print("SAMPLE RECOMMENDATIONS")
                print("="*80)
                for i, row in summary_df.head(5).iterrows():
                    print(f"{row['dataset']}: Recommended={row['recommended']} (rank {row['rank']}), Best={row['best']}, Score={row['recommended_score']:.3f}")
                
                # Done - skip the old single-dataset evaluation
                print("\n[SUCCESS] Paper-PMM evaluation completed on all test datasets!")
                print("\n[FINISH] RECOMMENDER TRAINING & EVALUATION COMPLETED [FINISH]")
                return
                
        except Exception as e:
            print(f"⚠️  Could not load test ground truth: {e}")
            print(f"   Falling back to single-dataset evaluation...")
    
    # OLD: Single dataset evaluation (for baseline or if ground truth not available)
    
    # Try to load a test dataset for evaluation
    test_dataset_ids = [1503, 23517, 1551, 1552, 255, 546, 475, 481, 516, 6, 8, 10, 12, 14, 9, 11, 5]
    
    # Let's take the first test dataset ID that exists in the metafeatures
    test_dataset_id = None
    
    # For PMM recommenders, we need the test dataset to be in both metafeatures and performance_matrix
    if recommender_type in ['pmm']:
        for dataset_id in meta_features_df.index:
            # Try to find this dataset ID in the performance matrix (looking for D_X format)
            formatted_id = f"D_{dataset_id}"
            if formatted_id in performance_matrix.columns:
                test_dataset_id = dataset_id
                print(f"Found dataset {dataset_id} in both metafeatures and performance matrix (as {formatted_id})")
                break
    
    # Fallback to the traditional approach
    if test_dataset_id is None:
        for dataset_id in test_dataset_ids:
            if dataset_id in meta_features_df.index:
                test_dataset_id = dataset_id
                break
    
    if test_dataset_id is None:
        print("❌ No suitable test dataset found in metafeatures.")
        return
    
    print(f"\nEvaluating recommender on dataset ID {test_dataset_id}...")
    
    # Get recommendations from the recommender
    recommendations = {}
    scores = {}
    similarity_info = {}
    
    if recommender_success and recommender_type != 'baseline':
        try:
            # Handle different return formats based on recommender type
            if recommender_type in ['pmm']:
                # PMM recommenders return a dictionary
                print(f"\nGetting recommendation from {recommender_type.upper()} recommender for dataset {test_dataset_id}...")
                recommendation_result = recommender.recommend(test_dataset_id, performance_matrix)
                
                if recommendation_result and recommendation_result.get('pipeline'):
                    recommendation = recommendation_result['pipeline']
                    score_dict = recommendation_result.get('performance_scores', {})
                    similar_datasets = recommendation_result.get('similar_datasets', [])
                    similarity_scores = recommendation_result.get('similarity_scores', {})
                    
                    recommendations[recommender_type] = recommendation
                    scores[recommender_type] = score_dict
                    similarity_info[recommender_type] = {
                        'similar_datasets': similar_datasets,
                        'similarity_scores': similarity_scores
                    }
                    
                    # Print similar datasets information
                    print(f"\nMost similar datasets to {test_dataset_id}:")
                    for i, dataset in enumerate(similar_datasets[:5]):
                        sim_score = similarity_scores.get(dataset, 0.0)
                        print(f"  {i+1}. Dataset {dataset} (similarity: {sim_score:.4f})")
                    
                    # Print detailed scores for top 5 pipelines
                    print("\nTop 5 pipeline scores:")
                    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                    for pipeline, score in sorted_scores:
                        print(f"  {pipeline}: {score:.4f}")
                else:
                    print(f"\n❌ PMM recommender failed for dataset {test_dataset_id}")
                    print(f"Result: {recommendation_result}")
            
            elif recommender_type == 'paper_pmm':
                # Paper-style PMM takes dataset metafeatures and returns list of pipelines
                print(f"\nGetting recommendation from PAPER-PMM recommender for dataset {test_dataset_id}...")
                
                # Get dataset metafeatures
                if test_dataset_id in meta_features_df.index:
                    dataset_metafeats = meta_features_df.loc[test_dataset_id].values
                else:
                    # Try with string conversion
                    test_dataset_id_str = str(test_dataset_id)
                    if test_dataset_id_str in meta_features_df.index:
                        dataset_metafeats = meta_features_df.loc[test_dataset_id_str].values
                    else:
                        print(f"❌ Dataset {test_dataset_id} not found in metafeatures")
                        dataset_metafeats = None
                
                if dataset_metafeats is not None:
                    # Get recommendations (returns list of pipeline names)
                    top_k_pipelines = recommender.recommend(dataset_metafeats, top_k=5)
                    
                    if top_k_pipelines and len(top_k_pipelines) > 0:
                        recommendation = top_k_pipelines[0]
                        recommendations[recommender_type] = recommendation
                        
                        print(f"  ✅ Paper-PMM recommended: {recommendation}")
                        print(f"  Top-5 recommendations: {top_k_pipelines}")
                        
                        # Create score dict (paper-pmm doesn't return explicit scores, use rank)
                        score_dict = {pipeline: 1.0 / (i+1) for i, pipeline in enumerate(top_k_pipelines)}
                        scores[recommender_type] = score_dict
                        
                        # Show influence information if available
                        if args.use_influence and hasattr(recommender, 'dataset_influence_scores'):
                            print(f"\n  🎯 Used influence weighting (method: {args.influence_method})")
                            # Show top 5 most influential datasets
                            top_influential = sorted(
                                recommender.dataset_influence_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:5]
                            print(f"  Most influential training datasets:")
                            for ds, score in top_influential:
                                print(f"    - Dataset {ds}: influence score {score:.3f}")
                    else:
                        print(f"❌ Paper-PMM returned empty recommendations")
                else:
                    print(f"❌ Could not get metafeatures for dataset {test_dataset_id}")
            
            else:
                # Original format for other recommenders
                recommendation, score_dict = recommender.recommend(test_dataset_id)
                if recommendation:
                    recommendations[recommender_type] = recommendation
                    scores[recommender_type] = score_dict
        except Exception as e:
            import traceback
            print(f"\n❌ Error in recommendation: {e}")
            traceback.print_exc()
            print("Using baseline recommender as fallback")
    
    # Print comparison of recommendations
    print("\n" + "="*80)
    print("RECOMMENDER COMPARISON")
    print("="*80)
    
    for recommender_name, pipeline in recommendations.items():
        print(f"{recommender_name.capitalize()} Recommender: {pipeline}")
        
        # Show similarity info for PMM recommenders
        if recommender_name in ['pmm'] and recommender_name in similarity_info:
            similar_datasets = similarity_info[recommender_name].get('similar_datasets', [])
            similarity_scores = similarity_info[recommender_name].get('similarity_scores', {})
            
            if similar_datasets:
                print("\nTop similar datasets used for recommendation:")
                for i, dataset in enumerate(similar_datasets[:3]):
                    sim_score = similarity_scores.get(dataset, 0.0)
                    col_name = None
                    
                    # Try to find the corresponding column in performance matrix
                    if dataset in performance_matrix.columns:
                        col_name = dataset
                    elif f"D_{dataset}" in performance_matrix.columns:
                        col_name = f"D_{dataset}"
                        
                    if col_name:
                        best_pipeline = performance_matrix[col_name].idxmax()
                        best_score = performance_matrix[col_name].max()
                        print(f"  {i+1}. Dataset {dataset} (similarity: {sim_score:.4f}, best pipeline: {best_pipeline} with score {best_score:.4f})")
                    else:
                        print(f"  {i+1}. Dataset {dataset} (similarity: {sim_score:.4f}, not found in performance matrix)")
        
        # Show top 5 pipelines if we have score information
        if recommender_name in scores and scores[recommender_name]:
            print("\nTop 5 pipelines:")
            sorted_scores = sorted(scores[recommender_name].items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (pipe, score) in enumerate(sorted_scores):
                print(f"  {i+1}. {pipe} (score: {score:.4f})")
                
            # Print score distribution statistics
            score_values = list(scores[recommender_name].values())
            if score_values:
                print(f"\nScore statistics:")
                print(f"  Min: {min(score_values):.4f}")
                print(f"  Max: {max(score_values):.4f}")
                print(f"  Mean: {sum(score_values) / len(score_values):.4f}")
                print(f"  Std Dev: {np.std(score_values):.4f}")
                
                # Calculate number of pipelines with non-zero scores
                non_zero_scores = sum(1 for s in score_values if abs(s) > 1e-6)
                print(f"  Non-zero scores: {non_zero_scores}/{len(score_values)} ({non_zero_scores/len(score_values)*100:.1f}%)")
    
    # Find any column in performance matrix that corresponds to this dataset ID
    test_column = None
    for col in performance_matrix.columns:
        if col == str(test_dataset_id) or col == f"D_{test_dataset_id}":
            test_column = col
            break
        elif col.startswith('D_') and col[2:] == str(test_dataset_id):
            test_column = col
            break
    
    if test_column:
        print("\nActual performances in training data:")
        actual_scores = performance_matrix[test_column].sort_values(ascending=False)
        for i, (pipeline, score) in enumerate(actual_scores.items()):
            if not np.isnan(score) and i < 10:  # Show top 10
                print(f"  {pipeline}: {score:.4f}")
    
    print("\n" + "="*80)
    print("SAVING MODELS AND RESULTS")
    print("="*80)
    
    # Save recommender comparison to file
    with open('recommender_comparison.txt', 'w') as f:
        f.write("RECOMMENDER COMPARISON\n")
        f.write("=" * 40 + "\n\n")
        
        for recommender_name, pipeline in recommendations.items():
            f.write(f"{recommender_name.capitalize()} Recommender: {pipeline}\n")
            
            # Write top 5 recommendations
            if recommender_name in scores and scores[recommender_name]:
                f.write("\nTop 5 pipelines:\n")
                sorted_scores = sorted(scores[recommender_name].items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (pipe, score) in enumerate(sorted_scores):
                    f.write(f"  {i+1}. {pipe} (score: {score:.4f})\n")
                    
                # Write score distribution statistics
                score_values = list(scores[recommender_name].values())
                if score_values:
                    f.write(f"\nScore statistics:\n")
                    f.write(f"  Min: {min(score_values):.4f}\n")
                    f.write(f"  Max: {max(score_values):.4f}\n")
                    f.write(f"  Mean: {sum(score_values) / len(score_values):.4f}\n")
                    f.write(f"  Std Dev: {np.std(score_values):.4f}\n")
            f.write("\n")
        
        if test_column:
            f.write("\nActual performances in training data:\n")
            actual_scores = performance_matrix[test_column].sort_values(ascending=False)
            for pipeline, score in actual_scores.items():
                if not np.isnan(score):
                    f.write(f"  {pipeline}: {score:.4f}\n")
    
    print("[SUCCESS] Saved recommender comparison to 'recommender_comparison.txt'")
    
    print("\n[FINISH] RECOMMENDER TRAINING & EVALUATION COMPLETED [FINISH]")


def run_evaluation(meta_features_df, performance_matrix=None, is_quick_test=False, recommender_type='baseline', args=None):
    """Run a comprehensive evaluation of pipeline recommendations on test datasets."""
    import time
    
    # Set default args if not provided
    if args is None:
        class DefaultArgs:
            use_influence = False
            influence_method = 'performance_variance'
            explain = False
            explain_samples = 10
        args = DefaultArgs()
    
    print(f"STARTING PIPELINE RECOMMENDER EVALUATION WITH {recommender_type.upper()} RECOMMENDER")
    
    # Show enabled enhancements
    if args.explain:
        print(f"  ✓ Explainability (analyzing {args.explain_samples} samples)")
    
    # Try to load ground truth performance data to skip evaluation
    ground_truth_perf_matrix = None
    try:
        ground_truth_perf_matrix = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
        print(f"✅ SUCCESS: Loaded ground truth performance matrix with shape {ground_truth_perf_matrix.shape}")
        print(f"   This will skip the expensive pipeline evaluation step!")
    except Exception as e:
        print(f"INFO: No ground truth performance file found ({e})")
        print(f"   Will evaluate pipelines from scratch (this will be slow)...")
    
    # Dictionary to store comprehensive metrics
    evaluation_result = {
        'recommender_name': recommender_type,
        'recommender_type': recommender_type,
        'training_success': False,
        'training_time': 0.0,
        'recommendation_success': 0,
        'total_tests': 0,
        'errors': [],
        'test_summaries': []
    }
    
    # Train the recommender
    print("\n" + "="*80)
    print(f"TRAINING {recommender_type.upper()} RECOMMENDER")
    print("="*80)
    
    recommender = None
    training_start = time.time()
    
    try:
        # Create and train the appropriate recommender
        if recommender_type == 'hybrid':
            recommender = HybridMetaRecommender(
                performance_matrix, 
                meta_features_df,
                use_influence_weighting=args.use_influence,
                influence_method=args.influence_method
            )
            evaluation_result['training_success'] = recommender.fit()
        
        elif recommender_type == 'surrogate':
            recommender = BayesianSurrogateRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
        
        elif recommender_type == 'autogluon':
            recommender = AutoGluonPipelineRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
        
        elif recommender_type == 'random':
            recommender = RandomRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'average-rank':
            recommender = AverageRankRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'l1':
            recommender = L1Recommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'basic':
            recommender = BasicRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'knn':
            recommender = KnnRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'rf':
            recommender = RFRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'nn':
            recommender = NNRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            explanation_log = open('nn_prediction_explanations.txt', 'w')
            explanation_log.write("NEURAL NETWORK PREDICTION EXPLANATIONS\n")
            explanation_log.write("="*80 + "\n\n")
            
        elif recommender_type == 'regressor':
            recommender = RegressorRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'adaboost':
            recommender = AdaBoostRegressorRecommender(performance_matrix, meta_features_df)
            evaluation_result['training_success'] = recommender.fit()
            
        elif recommender_type == 'pmm':
            recommender = PmmRecommender()
            evaluation_result['training_success'] = recommender.fit(performance_matrix, meta_features_df)
            
        
        elif recommender_type == 'baseline':
            evaluation_result['training_success'] = True
            
        else:
            print(f"Unknown recommender type: {recommender_type}")
            evaluation_result['training_success'] = False
            
    except Exception as e:
        print(f"❌ Error training {recommender_type} recommender: {e}")
        import traceback
        traceback.print_exc()
        evaluation_result['errors'].append(f"Training error: {str(e)}")
        evaluation_result['training_success'] = False
    
    evaluation_result['training_time'] = time.time() - training_start
    
    if not evaluation_result['training_success']:
        print(f"❌ Training failed for {recommender_type} recommender")
        print_evaluation_summary(evaluation_result)
        return None
    
    print(f"✅ Training completed in {evaluation_result['training_time']:.2f}s")
    
    # Evaluate on test datasets
    print("\n" + "="*80)
    print("EVALUATING ALL PIPELINES ON TEST DATASETS")
    print("="*80)
    
    test_local_perfs = []
    test_summaries = []
    successful_test = 0
    failed_test = 0
    
    # Use fewer datasets for quick test
    test_ids = test_dataset_ids[:3] if is_quick_test else test_dataset_ids
    evaluation_result['total_tests'] = len(test_ids)
    
    for i, ds_id in enumerate(test_ids):
        print(f"\n[{i+1}/{len(test_ids)}] Processing TEST dataset {ds_id}...")
        
        dataset = load_openml_dataset(ds_id)
        if dataset is None:
            failed_test += 1
            evaluation_result['errors'].append(f"Failed to load dataset {ds_id}")
            continue
            
        try:
            local_perf, _, summary = run_experiment_for_dataset(
                dataset, 
                meta_features_df,
                global_performance_matrix=performance_matrix,
                recommender_type=recommender_type,
                ground_truth_perf_matrix=ground_truth_perf_matrix,
                use_influence=args.use_influence,
                influence_method=args.influence_method
            )
            
            if local_perf is not None: 
                test_local_perfs.append(local_perf)
            if summary is not None: 
                test_summaries.append(summary)
                evaluation_result['test_summaries'].append(summary)
                successful_test += 1
                evaluation_result['recommendation_success'] += 1

                if recommender_type == 'nn' and recommender is not None:
                    ground_truth_best = summary.get('ground_truth_best')
                    recommended_score = summary.get('recommended_score')
                    
                    # Log to console
                    print("\n" + "🔍 Generating prediction explanation...")
                    explanation = recommender.log_prediction_evidence(
                        ds_id, 
                        ground_truth_best=ground_truth_best,
                        actual_score=recommended_score
                    )
                    
                    # Log to file
                    if explanation and explanation_log:
                        explanation_log.write(f"\nDataset {ds_id}\n")
                        explanation_log.write("-"*80 + "\n")
                        explanation_log.write(f"Recommended: {summary.get('recommendation')}\n")
                        explanation_log.write(f"Ground Truth Best: {ground_truth_best}\n")
                        explanation_log.write(f"Rank: {summary.get('rank')}\n")
                        explanation_log.write(f"Score: {recommended_score:.4f}\n")
                        explanation_log.write(f"Confidence: {explanation['confidence_metrics']['top1_confidence']:.4f}\n")
                        
                        explanation_log.write("\nTop 5 Predictions:\n")
                        for pred in explanation['top_predictions']:
                            explanation_log.write(f"  {pred['rank']}. {pred['pipeline']}: {pred['confidence_pct']}\n")
                        
                        explanation_log.write("\nTop 5 Important Features:\n")
                        for feat_name, feat_data in list(explanation['feature_importance'].items())[:5]:
                            explanation_log.write(f"  {feat_name}: {feat_data['normalized_importance']*100:.2f}%\n")
                        
                        explanation_log.write("\n" + "="*80 + "\n\n")
                        explanation_log.flush()
            else:
                failed_test += 1
                evaluation_result['errors'].append(f"No summary for dataset {ds_id}")
                
        except Exception as e:
            print(f"  ERROR: Experiment failed for test dataset {ds_id}: {e}")
            evaluation_result['errors'].append(f"Dataset {ds_id}: {str(e)}")
            failed_test += 1
            continue
    
    if recommender_type == 'nn' and 'explanation_log' in locals():
        explanation_log.close()
        print(f"\n✅ Detailed explanations saved to 'nn_prediction_explanations.txt'")
    
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETION SUMMARY")
    print(f"{'='*80}")
    print(f"Test datasets: {successful_test} successful, {failed_test} failed")
    
    if test_local_perfs:
        test_local_df = pd.concat(test_local_perfs, axis=1)
        test_local_df.to_csv('test_local_performance.csv')
        print(f"\nSUCCESS: Saved test local performances to 'test_local_performance.csv' ({test_local_df.shape})")
        
        # We're using the same test_local_df as the ground truth performances
        test_local_df.to_csv('test_ground_truth_performance.csv')
        print(f"SUCCESS: Saved test ground truth performances to 'test_ground_truth_performance.csv' ({test_local_df.shape})")
        
    if test_summaries:
        test_summary_df = pd.DataFrame(test_summaries).set_index('dataset')
        test_summary_df.to_csv('test_evaluation_summary.csv')
        print(f"SUCCESS: Saved test summary to 'test_evaluation_summary.csv' ({test_summary_df.shape})")
        
        print(f"\n{'='*80}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        print(test_summary_df[['recommendation', 'ground_truth_best', 'rank', 'baseline_rank', 'better_than_baseline', 'recommended_score',  'baseline_score', 'best_score']])
        
        valid_ranks = test_summary_df[test_summary_df['rank'] > 0]['rank']
        if not valid_ranks.empty:
            avg_rank = valid_ranks.mean()
            top1_accuracy = (valid_ranks == 1).mean() * 100
            top3_accuracy = (valid_ranks <= 3).mean() * 100
            
            print(f"\nTEST PERFORMANCE METRICS:")
            print(f"- Average Rank: {avg_rank:.2f}")
            print(f"- Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"- Top-3 Accuracy: {top3_accuracy:.2f}%")
            print(f"- Valid Recommendations: {len(valid_ranks)}/{len(test_summary_df)}")
            
            # Show score differences
            valid_summaries = test_summary_df[test_summary_df['rank'] > 0]
            if not valid_summaries.empty:
                score_diff = valid_summaries['best_score'] - valid_summaries['recommended_score']
                avg_score_diff = score_diff.mean()
                print(f"- Average Score Gap: {avg_score_diff:.4f}")
                print(f"- Score Gap Std: {score_diff.std():.4f}")
        else:
            print("WARNING: No valid recommendations were made on test datasets")
    else:
        print("WARNING: No successful test experiments completed")

    analyze_recommendations(test_summary_df, meta_features_df)
    
    # Print comprehensive evaluation summary
    print_evaluation_summary(evaluation_result)
    
    # ========================================================================
    # HANDLER: Explainability Report
    # ========================================================================
    if args.explain and recommender_type == 'nn' and recommender is not None:
        print("\n" + "="*80)
        print("🔍 GENERATING EXPLAINABILITY REPORT")
        print("="*80)
        
        try:
            from explainability import NeuralNetworkExplainer
            
            # Get the actual NN model (unwrap from active learning if needed)
            if hasattr(recommender, 'base_recommender'):
                nn_model = recommender.base_recommender.model
            else:
                nn_model = recommender.model
            
            # Prepare data for explainability
            feature_names = meta_features_df.columns.tolist()
            pipeline_names = [config['name'] for config in pipeline_configs]
            
            # Get training data
            X_train = []
            y_train = []
            for idx in meta_features_df.index:
                if idx in [int(col.split('_')[1]) for col in performance_matrix.columns if '_' in col]:
                    X_train.append(meta_features_df.loc[idx].values)
                    col_name = f'D_{idx}'
                    if col_name in performance_matrix.columns:
                        y_train.append(np.nanargmax(performance_matrix[col_name].values))
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Get test data (limit to explain_samples)
            X_test = []
            y_test = []
            for ds_id in test_ids[:args.explain_samples]:
                if ds_id in meta_features_df.index:
                    X_test.append(meta_features_df.loc[ds_id].values)
                    # Get actual best pipeline from test results
                    # test_summary_df has 'dataset' as index (e.g., 'D_1503')
                    ds_col = f'D_{ds_id}'
                    if ds_col in test_summary_df.index:
                        # Try to get the actual best pipeline index
                        if 'ground_truth_best' in test_summary_df.columns:
                            best_pipeline_name = test_summary_df.loc[ds_col, 'ground_truth_best']
                            # Convert pipeline name to index
                            try:
                                y_test.append(pipeline_configs.index(next(p for p in pipeline_configs if p['name'] == best_pipeline_name)))
                            except:
                                y_test.append(0)  # Fallback
                        else:
                            y_test.append(0)  # Fallback
                    else:
                        y_test.append(0)  # Fallback
            
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            if len(X_test) > 0:
                print(f"\n  Creating explainer...")
                explainer = NeuralNetworkExplainer(nn_model, feature_names, pipeline_names)
                
                print(f"  Generating report for {len(X_test)} test samples...")
                explainer.generate_report(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    save_dir='explainability_report'
                )
                
                print("\n  ✅ Explainability report complete!")
                print("     Check explainability_report/ for:")
                print("       - global_feature_importance.png")
                print("       - prediction_comparison.png")
                print("       - summary_report.txt")
            else:
                print("  ⚠️  No test data available for explainability analysis")
                
        except ImportError as e:
            print(f"  ⚠️  Explainability module not available: {e}")
            print("     Make sure explainability.py is in the same directory")
        except Exception as e:
            print(f"  ⚠️  Could not generate explainability report: {e}")
            import traceback
            traceback.print_exc()
    
    return test_summary_df


def print_evaluation_summary(result):
    """Print comprehensive evaluation summary in the requested format."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    
    # Calculate metrics from test summaries
    accuracy = 0.0
    avg_degradation = 999.0
    avg_rank = 999.0
    better_than_baseline_pct = 0.0
    equal_to_baseline_pct = 0.0
    worse_than_baseline_pct = 0.0
    average_performance = 0.0
    
    if result['test_summaries']:
        summaries_df = pd.DataFrame(result['test_summaries'])
        
        # Calculate accuracy (Top-1)
        if 'rank' in summaries_df.columns:
            valid_ranks = summaries_df[summaries_df['rank'] > 0]['rank']
            if not valid_ranks.empty:
                accuracy = (valid_ranks == 1).mean() * 100
                avg_rank = valid_ranks.mean()
        
        # Calculate degradation (percentage worse than best)
        if 'recommended_score' in summaries_df.columns and 'best_score' in summaries_df.columns:
            valid_scores = summaries_df[(summaries_df['recommended_score'].notna()) & (summaries_df['best_score'].notna())]
            if not valid_scores.empty:
                degradation = ((valid_scores['best_score'] - valid_scores['recommended_score']) / valid_scores['best_score']) * 100
                average_performance = valid_scores['recommended_score'].mean()
                avg_degradation = degradation.mean()
        
        # Calculate baseline comparison
        if 'better_than_baseline' in summaries_df.columns:
            total = len(summaries_df)
            better_count = (summaries_df['better_than_baseline'] == 'yes').sum()
            equal_count = (summaries_df['better_than_baseline'] == 'equal').sum()
            worse_count = (summaries_df['better_than_baseline'] == 'no').sum()
            
            better_than_baseline_pct = (better_count / total) * 100 if total > 0 else 0.0
            equal_to_baseline_pct = (equal_count / total) * 100 if total > 0 else 0.0
            worse_than_baseline_pct = (worse_count / total) * 100 if total > 0 else 0.0
    
    # Print formatted summary
    print(f"\nRecommender: {result['recommender_name']}")
    print(f"Type: {result['recommender_type']}")
    print(f"Training Success: {'✅' if result['training_success'] else '❌'}")
    print(f"Training Time (s): {result['training_time']:.2f}")
    print(f"Successful Recs: {result['recommendation_success']}/{result['total_tests']}")
    print(f"Accuracy (%): {accuracy:.2f}")
    print(f"Average Performace: {average_performance:.2f}")
    print(f"Avg Degradation (%): {avg_degradation:.2f}")
    print(f"Avg Rank: {avg_rank:.2f}")
    print(f"Better than Baseline (%): {better_than_baseline_pct:.2f}")
    print(f"Equal to Baseline (%): {equal_to_baseline_pct:.2f}")
    print(f"Worse than Baseline (%): {worse_than_baseline_pct:.2f}")
    print(f"Errors: {len(result['errors'])}")
    
    if result['errors']:
        print("\nError Details:")
        for i, error in enumerate(result['errors'][:5], 1):  # Show first 5 errors
            print(f"  {i}. {error}")
        if len(result['errors']) > 5:
            print(f"  ... and {len(result['errors']) - 5} more errors")
    
    # Save to CSV for easy comparison
    summary_data = {
        'Recommender': result['recommender_name'],
        'Type': result['recommender_type'],
        'Training Success': '✅' if result['training_success'] else '❌',
        'Training Time (s)': f"{result['training_time']:.2f}",
        'Successful Recs': f"{result['recommendation_success']}/{result['total_tests']}",
        'Accuracy (%)': f"{accuracy:.2f}",
        'Average Performace': f"{average_performance:.4f}",
        'Avg Degradation (%)': f"{avg_degradation:.2f}",
        'Avg Rank': f"{avg_rank:.4f}",
        'Better than Baseline (%)': f"{better_than_baseline_pct:.2f}",
        'Equal to Baseline (%)': f"{equal_to_baseline_pct:.2f}",
        'Worse than Baseline (%)': f"{worse_than_baseline_pct:.2f}",
        'Errors': len(result['errors'])
    }
    
    # Append to CSV file
    summary_df = pd.DataFrame([summary_data])
    csv_file = 'recommender_evaluation_results.csv'
    
    try:
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        summary_df.to_csv(csv_file, index=False)
        print(f"\n✅ Evaluation summary appended to '{csv_file}'")
    except Exception as e:
        print(f"\n⚠️ Could not save evaluation summary to CSV: {e}")
    
    print("="*80)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SoluRec - Pipeline recommender system')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on test datasets')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with fewer datasets')
    parser.add_argument('--debug', action='store_true', help='Run in deep debugging mode with detailed analysis')
    parser.add_argument('--debug-datasets', type=int, default=5, help='Number of datasets to analyze in debug mode')
    parser.add_argument('--recommender', type=str, default='baseline', 
                       choices=['baseline', 'autogluon', 'hybrid', 'surrogate', 'random', 
                               'avgrank', 'l1', 'basic', 'knn', 'rf', 'nn', 
                               'regressor', 'adaboost', 'pmm', 'paper_pmm'], 
                       help='Type of recommender to use')
    parser.add_argument('--use-influence', action='store_true', 
                       help='Enable DPO-style influence weighting for PMM recommenders')
    parser.add_argument('--influence-method', type=str, default='performance_variance',
                       choices=['performance_variance', 'dataset_diversity', 'prediction_confidence', 'combined'],
                       help='Method for calculating influence scores')
    
    # NEW: Feature Engineering flag
    parser.add_argument('--feature-engineering', action='store_true',
                       help='Use TRUE meta-features without data leakage (removes performance-based features)')
    
    # NEW: Explainability flag
    parser.add_argument('--explain', action='store_true',
                       help='Generate explainability report (SHAP, attention, feature importance)')
    parser.add_argument('--explain-samples', type=int, default=10,
                       help='Number of samples to explain in detail (default: 10)')
    
    args = parser.parse_args()
    
    # Add default values for removed active learning flags (for backward compatibility)
    args.active_learning = False
    args.active_k = 5
    args.quick_eval_time = 60
    
    # Pass configurations to recommenders module
    set_pipeline_configs(pipeline_configs)
    set_ag_config(AG_ARGS_FIT, STABLE_MODELS)
    
    # ========================================================================
    # HANDLER: Feature Engineering Flag
    # ========================================================================
    if args.feature_engineering:
        print("\n" + "="*80)
        print("🔧 FEATURE ENGINEERING MODE: Using TRUE meta-features")
        print("="*80)
        
        # Check if TRUE meta-features exist
        if os.path.exists('true_metafeatures.csv'):
            print("✓ Loading TRUE meta-features from true_metafeatures.csv")
            meta_features_df = pd.read_csv('true_metafeatures.csv', index_col=0)
            print(f"  - Datasets: {len(meta_features_df)}")
            print(f"  - Features: {len(meta_features_df.columns)} (no data leakage!)")
        else:
            print("❌ TRUE meta-features not found!")
            print("\nPlease extract them first:")
            print("  python enhanced_nn_trainer.py --extract-features")
            print("\nThis will create true_metafeatures.csv with 42 features")
            exit(1)
    else:
        # Load original meta-features with potential leakage
        try:
            meta_features_df = pd.read_csv('dataset_feats.csv', index_col=0)
            print(f"SUCCESS: Loaded {len(meta_features_df)} dataset metafeatures")
            if not args.evaluate:
                print("  ⚠️  Note: Using original features (may contain data leakage)")
                print("      Use --feature-engineering for TRUE features")
        except Exception as e:
            print(f"ERROR: Could not load meta-features: {e}")
            exit(1)
    
    # Load saved meta-features and preprocessed performance matrix if they exist
    try:
        
        if args.debug:
            # Deep debugging mode
            print("\n🔬 ENTERING DEBUG MODE")
            print("="*80)
            
            # Load performance matrix
            try:
                performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
            except Exception as e:
                print(f"ERROR: Performance matrix required for debug mode: {e}")
                exit(1)
            
            # Load ground truth if available
            try:
                ground_truth_matrix = pd.read_csv('test_ground_truth_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded test ground truth with shape {ground_truth_matrix.shape}")
            except Exception as e:
                print(f"INFO: No test ground truth found: {e}")
                ground_truth_matrix = None
            
            # Determine test datasets
            if ground_truth_matrix is not None:
                # Use datasets from ground truth file
                test_dataset_ids = [int(col.split('_')[1]) for col in ground_truth_matrix.columns if col.startswith('D_')]
            else:
                # Use a default set of test datasets
                test_dataset_ids = [1503, 1551, 255, 183, 1552]
            
            print(f"\nTest datasets: {test_dataset_ids[:args.debug_datasets]}")
            
            # Train recommenders to debug
            recommender_configs = []
            
            if args.recommender == 'pmm':
                influence_status = "WITH" if args.use_influence else "WITHOUT"
                print(f"\n Training PMM recommender {influence_status} influence weighting...")
                if args.use_influence:
                    print(f"  Influence method: {args.influence_method}")
                pmm_rec = PmmRecommender(
                    num_epochs=20, 
                    batch_size=64,
                    use_influence_weighting=args.use_influence,
                    influence_method=args.influence_method
                )
                if pmm_rec.fit(performance_matrix, meta_features_df):
                    recommender_configs.append((pmm_rec, 'pmm'))
                else:
                    print("ERROR: PMM recommender training failed")
                    exit(1)
                    
    
            elif args.recommender == 'hybrid':
                print("\nTraining Hybrid recommender...")
                hybrid_rec = HybridMetaRecommender(
                    performance_matrix, 
                    meta_features_df,
                    use_influence_weighting=args.use_influence,
                    influence_method=args.influence_method
                )
                if hybrid_rec.fit():
                    recommender_configs.append((hybrid_rec, 'hybrid'))
                else:
                    print("ERROR: Hybrid recommender training failed")
                    exit(1)
            else:
                print(f"ERROR: Debug mode currently only supports pmm, and hybrid recommenders")
                exit(1)
            
            # Run deep debugging analysis
            from debug_analysis import run_debug_mode
            output_dir = run_debug_mode(
                recommender_configs=recommender_configs,
                test_dataset_ids=test_dataset_ids,
                metafeatures_df=meta_features_df,
                performance_matrix=performance_matrix,
                ground_truth_matrix=ground_truth_matrix,
                max_datasets=args.debug_datasets
            )
            
            print(f"\n🎉 Debug analysis complete! Check the '{output_dir}' directory for detailed results.")
            
        elif args.evaluate:
            # If performance matrix exists, use it; otherwise, build it
            try:
                performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
                run_evaluation(meta_features_df, performance_matrix, args.quick, args.recommender, args)
            except Exception as e:
                print(f"INFO: No existing performance matrix found: {e}")
                print("Building performance matrix from scratch...")
                run_evaluation(meta_features_df, None, args.quick, args.recommender, args)
        else:
            # Regular recommender training flow
            try:
                performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
                print(f"SUCCESS: Loaded performance matrix with shape {performance_matrix.shape}")
                paper_pmm(meta_features_df, performance_matrix, args.recommender)
            except Exception as e:
                print(f"ERROR: Error loading performance matrix: {e}")
                print("Please run preprocess_metafeatures.py first to generate the required files.")
    except Exception as e:
        print(f"ERROR: Error loading saved data: {e}")
        print("Please run preprocess_metafeatures.py first to generate the required files.")