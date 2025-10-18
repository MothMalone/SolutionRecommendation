#!/usr/bin/env python3
"""
Script to evaluate all recommenders sequentially and collect comprehensive metrics.
This will run each recommender with the --evaluate flag and save results to a CSV file.
"""

import subprocess
import sys
import os
import pandas as pd
from datetime import datetime

# List of all recommenders to evaluate
RECOMMENDERS = [
    "baseline",
    "random",
    "basic",
    # "average-rank",
    "l1",
    "knn",
    "rf",
    "nn",
    "regressor",
    "adaboost",
    "surrogate",
    # "autogluon",
    "hybrid",
    # "pmm",
    # "balancedpmm",
]

def main():
    print("=" * 80)
    print("EVALUATING ALL RECOMMENDERS")
    print("=" * 80)
    print()
    print("This script will evaluate all recommenders on the test datasets.")
    print("Results will be saved to: recommender_evaluation_results.csv")
    print()
    
    # Remove previous results file to start fresh
    results_file = "recommender_evaluation_results.csv"
    if os.path.exists(results_file):
        print(f"Removing previous results file: {results_file}")
        os.remove(results_file)
        print()
    
    total = len(RECOMMENDERS)
    successful = 0
    failed = 0
    failed_recommenders = []
    
    # Evaluate each recommender
    for i, recommender in enumerate(RECOMMENDERS, 1):
        print()
        print("=" * 80)
        print(f"[{i}/{total}] Evaluating: {recommender}")
        print("=" * 80)
        print()
        
        try:
            # Run the evaluation
            result = subprocess.run(
                ["python", "recommender_trainer.py", "--recommender", recommender, "--evaluate"],
                check=True,
                capture_output=False,
                text=True
            )
            
            print(f"✅ {recommender} evaluation completed successfully")
            successful += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {recommender} evaluation failed with exit code {e.returncode}")
            failed += 1
            failed_recommenders.append(recommender)
        except Exception as e:
            print(f"❌ {recommender} evaluation failed with error: {e}")
            failed += 1
            failed_recommenders.append(recommender)
        
        print()
        print("-" * 80)
        print()
    
    # Print summary
    print()
    print("=" * 80)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 80)
    print()
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {failed}/{total}")
    
    if failed_recommenders:
        print(f"\nFailed recommenders: {', '.join(failed_recommenders)}")
    
    print()
    
    # Display the final results
    if os.path.exists(results_file):
        print(f"Final Results saved to: {results_file}")
        print()
        print("Summary:")
        print("-" * 80)
        
        try:
            df = pd.read_csv(results_file)
            # Display key columns in a readable format
            display_cols = [
                'Recommender', 
                'Training Success', 
                'Training Time (s)',
                'Successful Recs',
                'Accuracy (%)',
                'Avg Degradation (%)',
                'Avg Rank',
                'Better than Baseline (%)',
                'Errors'
            ]
            
            # Filter to only existing columns
            display_cols = [col for col in display_cols if col in df.columns]
            
            print(df[display_cols].to_string(index=False))
            print()
            
            # Print some aggregate statistics
            print("-" * 80)
            print("AGGREGATE STATISTICS:")
            print("-" * 80)
            
            if 'Accuracy (%)' in df.columns:
                try:
                    df['Accuracy_numeric'] = pd.to_numeric(df['Accuracy (%)'], errors='coerce')
                    print(f"Average Accuracy: {df['Accuracy_numeric'].mean():.4f}%")
                    print(f"Best Accuracy: {df['Accuracy_numeric'].max():.4f}% ({df.loc[df['Accuracy_numeric'].idxmax(), 'Recommender']})")
                except:
                    pass
            
            if 'Avg Rank' in df.columns:
                try:
                    df['Rank_numeric'] = pd.to_numeric(df['Avg Rank'], errors='coerce')
                    print(f"Average Rank: {df['Rank_numeric'].mean():.2f}")
                    print(f"Best Rank: {df['Rank_numeric'].min():.2f} ({df.loc[df['Rank_numeric'].idxmin(), 'Recommender']})")
                except:
                    pass
            
        except Exception as e:
            print(f"Could not display results: {e}")
            print("\nRaw CSV content:")
            with open(results_file, 'r') as f:
                print(f.read())
    else:
        print("⚠️ No results file found!")
    
    print()
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
