"""
Run ACO on All Test Datasets
Quick configuration: 5s AutoGluon, 8 ants, 5 iterations per dataset
"""

from setting_aco import test_dataset_ids, load_openml_dataset
from aco_main import run_aco_on_test_datasets

# Configuration
AUTOGLUON_TIME_LIMIT = 5  # 5 seconds per AutoGluon evaluation
ACO_PARAMS = {
    'n_ants': 8,
    'n_iterations': 5,
    'alpha': 1.0,
    'beta': 2.0,
    'rho': 0.1,
    'q0': 0.9,
    'verbose': True
}

print("="*80)
print("ACO PIPELINE OPTIMIZATION - ALL TEST DATASETS")
print("="*80)
print(f"Configuration:")
print(f"  AutoGluon time limit: {AUTOGLUON_TIME_LIMIT}s per evaluation")
print(f"  Ants per iteration: {ACO_PARAMS['n_ants']}")
print(f"  Iterations: {ACO_PARAMS['n_iterations']}")
print(f"  Total test datasets: {len(test_dataset_ids)}")
print("="*80)

# Update AutoGluon config
from aco_main import AUTOGLUON_CONFIG
AUTOGLUON_CONFIG['time_limit'] = AUTOGLUON_TIME_LIMIT

# Load all test datasets
print("\nLoading test datasets...")
test_datasets = []
for dataset_id in test_dataset_ids:
    dataset = load_openml_dataset(dataset_id, test_dataset_ids=test_dataset_ids)
    if dataset:
        test_datasets.append(dataset)
        print(f"  ✓ {dataset['name']} (ID: {dataset['id']}, Shape: {dataset['X'].shape})")

print(f"\nSuccessfully loaded {len(test_datasets)}/{len(test_dataset_ids)} datasets")

# Estimate time
evaluations_per_dataset = ACO_PARAMS['n_ants'] * ACO_PARAMS['n_iterations']
total_evaluations = evaluations_per_dataset * len(test_datasets)
estimated_time_per_eval = AUTOGLUON_TIME_LIMIT + 5  # AutoGluon + preprocessing overhead
estimated_total_time = total_evaluations * estimated_time_per_eval / 60

print(f"\nEstimated execution time:")
print(f"  Evaluations per dataset: {evaluations_per_dataset}")
print(f"  Total evaluations: {total_evaluations}")
print(f"  Estimated total time: {estimated_total_time:.1f} minutes ({estimated_total_time/60:.1f} hours)")

# Confirm
response = input("\nProceed with optimization? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("Aborted.")
    exit(0)

# Run optimization
print("\n" + "="*80)
print("STARTING OPTIMIZATION")
print("="*80)

results = run_aco_on_test_datasets(
    test_datasets=test_datasets,
    aco_params=ACO_PARAMS,
    save_results=True,
    output_prefix="aco_results_all_datasets"
)

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\nResults saved to:")
print(f"  - aco_results_all_datasets_summary.csv")
print(f"  - aco_results_all_datasets_detailed.csv")

# Show summary
print(f"\nSummary:")
print(f"  Average best score: {results['summary']['best_score'].mean():.4f}")
print(f"  Std best score: {results['summary']['best_score'].std():.4f}")
print(f"  Average time per dataset: {results['summary']['total_time'].mean():.1f}s")
