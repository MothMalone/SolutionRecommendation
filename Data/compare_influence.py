#!/usr/bin/env python3
"""
Compare recommender performance with and without influence weighting.
Shows internal differences even when final recommendations are the same.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender_trainer import PmmRecommender, HybridMetaRecommender, load_data


def compare_pmm_influence():
    """Compare PMM recommender with and without influence weighting"""
    print("="*80)
    print("COMPARING PMM RECOMMENDER: WITH vs WITHOUT INFLUENCE WEIGHTING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    performance_matrix, metafeatures_df = load_data()
    
    # Test datasets (from ground truth)
    test_datasets = [1503, 23517, 1551]
    
    results_comparison = []
    
    for dataset_id in test_datasets:
        print(f"\n{'='*80}")
        print(f"DATASET {dataset_id}")
        print(f"{'='*80}")
        
        # WITHOUT influence weighting
        print("\n1️⃣  WITHOUT INFLUENCE WEIGHTING:")
        recommender_no_inf = PmmRecommender(
            num_epochs=20, 
            batch_size=64,
            use_influence_weighting=False
        )
        recommender_no_inf.fit(performance_matrix, metafeatures_df)
        result_no_inf = recommender_no_inf.recommend(dataset_id, performance_matrix)
        
        if result_no_inf and 'pipeline_ranking' in result_no_inf:
            top_3_no_inf = result_no_inf['pipeline_ranking'][:3]
            scores_no_inf = result_no_inf['performance_scores']
            similar_datasets_no_inf = result_no_inf.get('similar_datasets', [])[:3]
            similarity_scores_no_inf = result_no_inf.get('similarity_scores', {})
            
            print(f"   Recommended: {result_no_inf['pipeline']}")
            print(f"   Top-3: {top_3_no_inf}")
            print(f"   Similar datasets: {similar_datasets_no_inf}")
            print(f"   Similarities:")
            for ds in similar_datasets_no_inf:
                print(f"     - Dataset {ds}: {similarity_scores_no_inf.get(ds, 0):.4f}")
        else:
            print("   ❌ Recommendation failed")
            continue
        
        # WITH influence weighting (default method)
        print("\n2️⃣  WITH INFLUENCE WEIGHTING (performance_variance):")
        recommender_inf = PmmRecommender(
            num_epochs=20, 
            batch_size=64,
            use_influence_weighting=True,
            influence_method='performance_variance'
        )
        recommender_inf.fit(performance_matrix, metafeatures_df)
        result_inf = recommender_inf.recommend(dataset_id, performance_matrix)
        
        if result_inf and 'pipeline_ranking' in result_inf:
            top_3_inf = result_inf['pipeline_ranking'][:3]
            scores_inf = result_inf['performance_scores']
            similar_datasets_inf = result_inf.get('similar_datasets', [])[:3]
            similarity_scores_inf = result_inf.get('similarity_scores', {})
            influence_scores_inf = result_inf.get('influence_scores', {})
            
            print(f"   Recommended: {result_inf['pipeline']}")
            print(f"   Top-3: {top_3_inf}")
            print(f"   Similar datasets: {similar_datasets_inf}")
            print(f"   Similarities & Influence:")
            for ds in similar_datasets_inf:
                sim = similarity_scores_inf.get(ds, 0)
                inf = influence_scores_inf.get(ds, 1.0)
                weighted = sim * inf
                print(f"     - Dataset {ds}: sim={sim:.4f}, influence={inf:.3f}, weighted={weighted:.4f}")
        else:
            print("   ❌ Recommendation failed")
            continue
        
        # COMPARISON
        print("\n3️⃣  COMPARISON:")
        print(f"   Recommendation changed: {result_no_inf['pipeline'] != result_inf['pipeline']}")
        
        if result_no_inf['pipeline'] != result_inf['pipeline']:
            print(f"     WITHOUT: {result_no_inf['pipeline']}")
            print(f"     WITH:    {result_inf['pipeline']}")
        
        # Compare top-3 rankings
        top_3_same = top_3_no_inf == top_3_inf
        print(f"   Top-3 ranking changed: {not top_3_same}")
        if not top_3_same:
            print(f"     WITHOUT: {top_3_no_inf}")
            print(f"     WITH:    {top_3_inf}")
        
        # Compare pipeline scores
        print(f"\n   Pipeline Score Differences:")
        all_pipelines = set(scores_no_inf.keys()) | set(scores_inf.keys())
        score_diffs = []
        for pipeline in sorted(all_pipelines):
            score_no_inf = scores_no_inf.get(pipeline, 0)
            score_inf = scores_inf.get(pipeline, 0)
            diff = score_inf - score_no_inf
            score_diffs.append({
                'pipeline': pipeline,
                'without_influence': score_no_inf,
                'with_influence': score_inf,
                'difference': diff
            })
        
        # Sort by absolute difference
        score_diffs.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        print(f"   Top 5 biggest changes:")
        for i, item in enumerate(score_diffs[:5], 1):
            print(f"     {i}. {item['pipeline']}: {item['without_influence']:.4f} → {item['with_influence']:.4f} (Δ {item['difference']:+.4f})")
        
        # Store results
        results_comparison.append({
            'dataset_id': dataset_id,
            'recommendation_changed': result_no_inf['pipeline'] != result_inf['pipeline'],
            'top3_changed': not top_3_same,
            'recommended_no_inf': result_no_inf['pipeline'],
            'recommended_inf': result_inf['pipeline'],
            'max_score_diff': max(abs(x['difference']) for x in score_diffs)
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results_comparison)
    print(f"\nDatasets where recommendation changed: {df['recommendation_changed'].sum()}/{len(df)}")
    print(f"Datasets where top-3 changed: {df['top3_changed'].sum()}/{len(df)}")
    print(f"Average max score difference: {df['max_score_diff'].mean():.4f}")
    
    print("\nDetailed results:")
    print(df.to_string(index=False))


def compare_hybrid_influence():
    """Compare Hybrid recommender with and without influence weighting"""
    print("\n\n")
    print("="*80)
    print("COMPARING HYBRID RECOMMENDER: WITH vs WITHOUT INFLUENCE WEIGHTING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    performance_matrix, metafeatures_df = load_data()
    
    # Test datasets (from ground truth)
    test_datasets = [1503, 23517, 1551]
    
    results_comparison = []
    
    for dataset_id in test_datasets:
        print(f"\n{'='*80}")
        print(f"DATASET {dataset_id}")
        print(f"{'='*80}")
        
        # WITHOUT influence weighting
        print("\n1️⃣  WITHOUT INFLUENCE WEIGHTING:")
        recommender_no_inf = HybridMetaRecommender(
            performance_matrix,
            metafeatures_df,
            use_influence_weighting=False
        )
        recommender_no_inf.fit()
        result_no_inf = recommender_no_inf.recommend(dataset_id)
        
        if isinstance(result_no_inf, tuple):
            pipeline_no_inf, scores_no_inf = result_no_inf
        else:
            pipeline_no_inf = result_no_inf
            scores_no_inf = {}
        
        print(f"   Recommended: {pipeline_no_inf}")
        if scores_no_inf:
            top_3_no_inf = sorted(scores_no_inf.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top-3 scores:")
            for pipeline, score in top_3_no_inf:
                print(f"     - {pipeline}: {score:.4f}")
        
        # WITH influence weighting
        print("\n2️⃣  WITH INFLUENCE WEIGHTING (combined):")
        recommender_inf = HybridMetaRecommender(
            performance_matrix,
            metafeatures_df,
            use_influence_weighting=True,
            influence_method='combined'
        )
        recommender_inf.fit()
        result_inf = recommender_inf.recommend(dataset_id)
        
        if isinstance(result_inf, tuple):
            pipeline_inf, scores_inf = result_inf
        else:
            pipeline_inf = result_inf
            scores_inf = {}
        
        print(f"   Recommended: {pipeline_inf}")
        if scores_inf:
            top_3_inf = sorted(scores_inf.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top-3 scores:")
            for pipeline, score in top_3_inf:
                print(f"     - {pipeline}: {score:.4f}")
            
            # Show influence scores if available
            if hasattr(recommender_inf, 'dataset_influence_scores'):
                print(f"   Sample influence scores:")
                sample_datasets = list(recommender_inf.dataset_influence_scores.keys())[:5]
                for ds in sample_datasets:
                    inf = recommender_inf.dataset_influence_scores.get(ds, 1.0)
                    print(f"     - Dataset {ds}: {inf:.3f}")
        
        # COMPARISON
        print("\n3️⃣  COMPARISON:")
        print(f"   Recommendation changed: {pipeline_no_inf != pipeline_inf}")
        
        if pipeline_no_inf != pipeline_inf:
            print(f"     WITHOUT: {pipeline_no_inf}")
            print(f"     WITH:    {pipeline_inf}")
        
        # Compare scores
        if scores_no_inf and scores_inf:
            print(f"\n   Score Differences:")
            all_pipelines = set(scores_no_inf.keys()) | set(scores_inf.keys())
            score_diffs = []
            for pipeline in sorted(all_pipelines):
                score_no_inf = scores_no_inf.get(pipeline, 0)
                score_inf = scores_inf.get(pipeline, 0)
                diff = score_inf - score_no_inf
                score_diffs.append({
                    'pipeline': pipeline,
                    'without_influence': score_no_inf,
                    'with_influence': score_inf,
                    'difference': diff
                })
            
            # Sort by absolute difference
            score_diffs.sort(key=lambda x: abs(x['difference']), reverse=True)
            
            print(f"   Top 5 biggest changes:")
            for i, item in enumerate(score_diffs[:5], 1):
                print(f"     {i}. {item['pipeline']}: {item['without_influence']:.4f} → {item['with_influence']:.4f} (Δ {item['difference']:+.4f})")
            
            max_diff = max(abs(x['difference']) for x in score_diffs)
        else:
            max_diff = 0
        
        # Store results
        results_comparison.append({
            'dataset_id': dataset_id,
            'recommendation_changed': pipeline_no_inf != pipeline_inf,
            'recommended_no_inf': pipeline_no_inf,
            'recommended_inf': pipeline_inf,
            'max_score_diff': max_diff
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results_comparison)
    print(f"\nDatasets where recommendation changed: {df['recommendation_changed'].sum()}/{len(df)}")
    print(f"Average max score difference: {df['max_score_diff'].mean():.4f}")
    
    print("\nDetailed results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare influence weighting impact")
    parser.add_argument('--recommender', choices=['pmm', 'hybrid', 'both'], default='both',
                       help='Which recommender to compare')
    
    args = parser.parse_args()
    
    if args.recommender in ['pmm', 'both']:
        compare_pmm_influence()
    
    if args.recommender in ['hybrid', 'both']:
        compare_hybrid_influence()
