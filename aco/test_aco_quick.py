"""
Quick test of ACO system with progress indicators
Shows expected behavior and timing
"""

import warnings
warnings.filterwarnings('ignore')

from setting_aco import load_openml_dataset, test_dataset_ids
from aco_main import optimize_dataset_pipeline
import time

print("="*80)
print("ACO SYSTEM - QUICK TEST")
print("="*80)

# Load ONE small dataset for quick test
print("\nLoading test dataset...")
dataset = load_openml_dataset(test_dataset_ids[0], test_dataset_ids=test_dataset_ids)

if dataset:
    print(f"\n✓ Dataset loaded: {dataset['name']} (ID: {dataset['id']})")
    print(f"  Shape: {dataset['X'].shape}")
    print(f"  Task: {dataset.get('task_type', 'Unknown')}")
    
    print("\n" + "="*80)
    print("Running ACO with minimal settings for QUICK TEST")
    print("(5 iterations × 8 ants = 40 evaluations)")
    print("Expected time: ~20-40 minutes")
    print("="*80)
    
    start = time.time()
    
    results = optimize_dataset_pipeline(
        dataset=dataset,
        aco_params={
            'n_ants': 8,           # Fewer ants
            'n_iterations': 5,     # Only 5 iterations
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1,
            'q0': 0.9,
            'verbose': True
        },
        verbose=True
    )
    
    elapsed = time.time() - start
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Total Time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Avg time per evaluation: {elapsed/40:.1f} seconds")
    print(f"\nBest Pipeline:")
    for step, operator in results['best_pipeline'].items():
        print(f"  {step:30s}: {operator}")
    
    print("\n" + "="*80)
    print("WHAT YOU SAW:")
    print("="*80)
    print("✓ Progress bar showing iterations")
    print("✓ Stats updated after each iteration (best, avg, global)")
    print("✓ AutoGluon 'path already exists' warnings (NORMAL - not errors!)")
    print("✓ Final results with best pipeline")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. For real optimization: Use 15-20 ants, 30+ iterations")
    print("2. For parallel matrix generation: See generate_matrix_parallel.py")
    print("3. The warnings are NORMAL - AutoGluon just being verbose")
    print("="*80)
    
else:
    print("❌ Failed to load dataset")
