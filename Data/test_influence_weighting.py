#!/usr/bin/env python3
"""
Test script to verify influence weighting is working in PMM recommender
"""
import pandas as pd
import numpy as np
from recommender_trainer import PmmRecommender

def test_influence_weighting():
    print("="*80)
    print("TESTING INFLUENCE WEIGHTING IN PMM RECOMMENDER")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    metafeatures_df = pd.read_csv("dataset_feats.csv", index_col=0)
    performance_matrix = pd.read_csv("preprocessed_performance.csv", index_col=0)
    
    print(f"Metafeatures shape: {metafeatures_df.shape}")
    print(f"Performance matrix shape: {performance_matrix.shape}")
    
    # Test different influence methods
    methods = ['none', 'performance_gap', 'data_diversity']
    
    for method in methods:
        print(f"\n{'-'*80}")
        print(f"Testing influence method: {method}")
        print(f"{'-'*80}")
        
        # Create recommender with this method
        recommender = PmmRecommender(
            num_epochs=5,  # Quick test
            batch_size=64,
            influence_method=method,
            use_influence_weighting=True
        )
        
        # Train
        print(f"\nTraining PMM recommender with influence method: {method}...")
        success = recommender.fit(performance_matrix, metafeatures_df)
        
        if success:
            print(f"✅ Training successful!")
            
            # Check influence scores
            if hasattr(recommender, 'dataset_influence_scores'):
                scores = recommender.dataset_influence_scores
                print(f"\nInfluence scores calculated for {len(scores)} datasets")
                
                # Show statistics
                values = list(scores.values())
                print(f"  Mean: {np.mean(values):.3f}")
                print(f"  Std:  {np.std(values):.3f}")
                print(f"  Min:  {np.min(values):.3f}")
                print(f"  Max:  {np.max(values):.3f}")
                
                # Show top 5 most influential
                top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"\n  Top 5 most influential datasets:")
                for ds_id, score in top_5:
                    print(f"    Dataset {ds_id}: {score:.3f}")
                
                # Show bottom 5 least influential
                bottom_5 = sorted(scores.items(), key=lambda x: x[1])[:5]
                print(f"\n  Bottom 5 least influential datasets:")
                for ds_id, score in bottom_5:
                    print(f"    Dataset {ds_id}: {score:.3f}")
            else:
                print("⚠️ No influence scores found!")
                
            # Test recommendation
            test_dataset_id = 1503  # Use a test dataset
            if test_dataset_id in metafeatures_df.index:
                print(f"\n\nTesting recommendation for dataset {test_dataset_id}...")
                result = recommender.recommend(test_dataset_id, performance_matrix)
                
                if result:
                    print(f"  Recommended pipeline: {result['pipeline']}")
                    print(f"  Influence weighted: {result.get('influence_weighted', False)}")
                    
                    if 'similar_datasets' in result and 'influence_scores' in result:
                        print(f"\n  Similar datasets with influence scores:")
                        for ds in result['similar_datasets'][:5]:
                            sim = result['similarity_scores'].get(ds, 0.0)
                            inf = result['influence_scores'].get(ds, 1.0)
                            print(f"    Dataset {ds}: similarity={sim:.4f}, influence={inf:.3f}, combined={sim*inf:.4f}")
        else:
            print(f"❌ Training failed!")
    
    print("\n" + "="*80)
    print("INFLUENCE WEIGHTING TEST COMPLETED")
    print("="*80)

if __name__ == "__main__":
    test_influence_weighting()
