#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Heuristic Pipeline Recommender based on Average Performance
-----------------------------------------------------------
This script implements a simple heuristic approach to recommending
preprocessing pipelines based solely on their average performance
across training datasets. The recommendations are then evaluated
on test datasets and compared with the hybrid recommender approach.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Load the preprocessed performance matrix (training data)
print("Loading training performance matrix...")
train_perf = pd.read_csv('preprocessed_performance.csv', index_col=0)

# Load test ground truth performance
print("Loading test ground truth performance...")
test_perf = pd.read_csv('test_ground_truth_performance.csv', index_col=0)

# Calculate average performance for each pipeline across training datasets
print("Calculating average performance for each pipeline...")
pipeline_avg_perf = train_perf.mean(axis=1)
print("\nAverage Pipeline Performance (Training Data):")
print(pipeline_avg_perf.sort_values(ascending=False))

# Always recommend the pipeline with highest average performance
best_pipeline = pipeline_avg_perf.idxmax()
print(f"\nBest average pipeline: {best_pipeline} (avg score: {pipeline_avg_perf[best_pipeline]:.4f})")

# Evaluate how this recommendation performs on test datasets
print("\nEvaluating heuristic recommendation on test datasets...")

results = []
for dataset in test_perf.columns:
    # Get test scores for this dataset
    dataset_scores = test_perf[dataset].dropna()
    
    if dataset_scores.empty:
        print(f"  Skipping dataset {dataset} - no valid test scores")
        continue
        
    # Find the best pipeline for this dataset
    ground_truth_best = dataset_scores.idxmax()
    best_score = dataset_scores[ground_truth_best]
    
    # Find recommended pipeline's performance
    if best_pipeline in dataset_scores:
        recommended_score = dataset_scores[best_pipeline]
    else:
        print(f"  Warning: Recommended pipeline {best_pipeline} has no score for dataset {dataset}")
        continue
        
    # Find baseline performance
    if 'baseline' in dataset_scores:
        baseline_score = dataset_scores['baseline']
    else:
        baseline_score = np.nan
        
    # Handle ties for ranking (equal performance gets same rank)
    sorted_scores = sorted(dataset_scores.items(), key=lambda x: x[1], reverse=True)
    current_rank = 1
    last_score = None
    rank_mapping = {}
    skip = 0
    
    for i, (pipeline, score) in enumerate(sorted_scores):
        if last_score is not None and score != last_score:
            current_rank += skip
            skip = 0
        rank_mapping[pipeline] = current_rank
        skip += 1
        last_score = score
        
    recommended_rank = rank_mapping.get(best_pipeline, float('inf'))
    baseline_rank = rank_mapping.get('baseline', float('inf'))
    
    # Determine if recommended is better than baseline
    if np.isnan(baseline_score):
        better_than_baseline = "unknown"
    elif np.isclose(recommended_score, baseline_score, rtol=1e-5, atol=1e-8):
        better_than_baseline = "equal"
    elif recommended_score > baseline_score:
        better_than_baseline = "yes"
    else:
        better_than_baseline = "no"
        
    # Calculate performance gap
    score_gap = best_score - recommended_score
    
    # Store result
    results.append({
        'dataset': dataset,
        'ground_truth_best': ground_truth_best,
        'recommended_pipeline': best_pipeline,
        'rank': recommended_rank,
        'baseline_rank': baseline_rank,
        'better_than_baseline': better_than_baseline,
        'recommended_score': recommended_score,
        'baseline_score': baseline_score,
        'best_score': best_score,
        'score_gap': score_gap,
        'num_pipelines': len(dataset_scores)
    })
    
# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.set_index('dataset', inplace=True)

# Calculate summary metrics
valid_ranks = results_df['rank']
avg_rank = valid_ranks.mean()
top1_accuracy = (valid_ranks == 1).mean() * 100
top3_accuracy = (valid_ranks <= 3).mean() * 100

better_than_baseline = (results_df['better_than_baseline'] == 'yes').mean() * 100
equal_to_baseline = (results_df['better_than_baseline'] == 'equal').mean() * 100
worse_than_baseline = (results_df['better_than_baseline'] == 'no').mean() * 100

valid_gaps = results_df['score_gap'].dropna()
avg_gap = valid_gaps.mean()

# Display evaluation summary
print("\nHeuristic Recommender Evaluation Summary:")
print(f"Average Rank: {avg_rank:.2f}")
print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")
print(f"Better than baseline: {better_than_baseline:.1f}%")
print(f"Equal to baseline: {equal_to_baseline:.1f}%")
print(f"Worse than baseline: {worse_than_baseline:.1f}%")
print(f"Average Performance Gap: {avg_gap:.4f}")

# Create comparison table with hybrid recommender results
print("\nCreating comparison table...")
hybrid_metrics = {
    'Average Rank': 4.74,
    'Top-1 Accuracy': 15.79,
    'Top-3 Accuracy': 31.58,
    'Better than baseline': 42.1,
    'Equal to baseline': 36.8,
    'Worse than baseline': 21.1
}

heuristic_metrics = {
    'Average Rank': avg_rank,
    'Top-1 Accuracy': top1_accuracy,
    'Top-3 Accuracy': top3_accuracy,
    'Better than baseline': better_than_baseline,
    'Equal to baseline': equal_to_baseline,
    'Worse than baseline': worse_than_baseline
}

comparison_df = pd.DataFrame({
    'Metric': list(hybrid_metrics.keys()),
    'Hybrid Recommender': list(hybrid_metrics.values()),
    'Heuristic (Avg Performance)': list(heuristic_metrics.values())
})

# Save detailed results to CSV
print("\nSaving results to CSV files...")
results_df.to_csv('heuristic_recommender_results.csv')
comparison_df.to_csv('recommender_comparison.csv', index=False)

print("\nDetailed results for heuristic recommendation by dataset:")
print(results_df[['recommended_pipeline', 'ground_truth_best', 'rank', 'baseline_rank', 'better_than_baseline', 'recommended_score', 'baseline_score', 'best_score']])

print("\nComparison between Hybrid and Heuristic recommenders:")
print(comparison_df.to_string(index=False))
