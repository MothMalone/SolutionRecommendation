# ACO Pipeline Optimizer - Quick Reference

## 🚀 Installation & Setup

No additional installation needed! Uses your existing environment with:
- AutoGluon
- scikit-learn  
- pandas, numpy
- matplotlib (for visualizations)

## ⚡ Quick Commands

### Single Dataset Optimization
```python
from setting import load_openml_dataset, test_dataset_ids
from aco_main import optimize_dataset_pipeline

dataset = load_openml_dataset(test_dataset_ids[0])
results = optimize_dataset_pipeline(dataset)

print(f"Best Score: {results['best_score']:.4f}")
print(f"Pipeline: {results['best_pipeline']}")
```

### Batch Optimization
```python
from aco_main import run_aco_on_test_datasets

datasets = [load_openml_dataset(id) for id in test_dataset_ids[:5]]
results = run_aco_on_test_datasets(datasets, save_results=True)
```

### With Custom Parameters
```python
results = optimize_dataset_pipeline(
    dataset,
    aco_params={
        'n_ants': 20,
        'n_iterations': 40,
        'alpha': 1.5,
        'beta': 2.5
    }
)
```

## 📊 Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_ants` | 20 | 10-50 | More = better coverage, slower |
| `n_iterations` | 50 | 10-100 | More = better results, slower |
| `alpha` | 1.0 | 0.5-3.0 | Pheromone importance |
| `beta` | 2.0 | 1.0-5.0 | Heuristic importance |
| `rho` | 0.1 | 0.05-0.3 | Evaporation rate |
| `q0` | 0.9 | 0.7-0.99 | Exploitation level |

## 🎯 Common Use Cases

### Fast Prototyping (5-10 min)
```python
aco_params = {'n_ants': 10, 'n_iterations': 15}
```

### Balanced Search (15-25 min)
```python
aco_params = {'n_ants': 15, 'n_iterations': 30}
```

### Thorough Search (30-60 min)
```python
aco_params = {'n_ants': 20, 'n_iterations': 50}
```

### Production Quality (1-2 hours)
```python
aco_params = {'n_ants': 30, 'n_iterations': 60}
```

## 📈 Results Structure

```python
results = {
    'best_pipeline': {
        'imputation': 'median',
        'scaling': 'robust',
        'encoding': 'onehot',
        'feature_selection': 'k_best',
        'outlier_removal': 'iqr',
        'dimensionality_reduction': 'none'
    },
    'best_score': 0.8543,
    'iteration_best_scores': [0.72, 0.74, ..., 0.85],
    'total_time': 1234.56,
    'pheromone_summary': DataFrame,
    'operator_statistics': DataFrame
}
```

## 🎨 Visualization

### Auto-generate All Plots
```python
from aco_visualization import create_full_visualization_suite

create_full_visualization_suite(
    results, 
    dataset['name'], 
    output_dir="./plots"
)
```

### Individual Plots
```python
from aco_visualization import (
    plot_convergence,
    plot_pheromone_heatmap,
    plot_operator_usage
)

plot_convergence(results, save_path="convergence.png")
plot_pheromone_heatmap(results, save_path="pheromone.png")
plot_operator_usage(results, save_path="operators.png")
```

## 🔧 Troubleshooting

### Issue: Slow Execution
**Fix:** Reduce AutoGluon time limit
```python
AUTOGLUON_CONFIG['time_limit'] = 180  # 3 minutes instead of 5
```

### Issue: Poor Convergence
**Fix:** More iterations or different parameters
```python
aco_params = {'n_iterations': 50, 'beta': 3.0}
```

### Issue: All Ants Same Pipeline
**Fix:** Increase exploration
```python
aco_params = {'q0': 0.7, 'rho': 0.2}
```

## 📁 Output Files

### From `run_aco_on_test_datasets`:
- `{prefix}_summary.csv` - Performance overview
- `{prefix}_detailed.csv` - Full pipeline details

### From `create_full_visualization_suite`:
- `{dataset}_convergence.png` - Optimization curve
- `{dataset}_pheromone.png` - Pheromone levels
- `{dataset}_operator_usage.png` - Operator statistics
- `{dataset}_report.txt` - Text summary

## 💡 Tips

1. **Start small**: Test with 1-2 datasets first
2. **Check convergence**: Look at convergence plot to see if more iterations needed
3. **Compare baselines**: Use `compare_aco_with_baseline()` to validate improvement
4. **Save results**: Always use `save_results=True` for batch runs
5. **Tune parameters**: Use Example 4 to find best parameters for your datasets

## 🎯 Complete Workflow

```python
# 1. Load datasets
from setting import load_openml_dataset, test_dataset_ids
datasets = [load_openml_dataset(id) for id in test_dataset_ids]

# 2. Run optimization
from aco_main import run_aco_on_test_datasets
results = run_aco_on_test_datasets(
    datasets,
    aco_params={'n_ants': 15, 'n_iterations': 30},
    save_results=True,
    output_prefix="aco_final"
)

# 3. Analyze results
print(results['summary'])

# 4. Visualize
from aco_visualization import plot_multi_dataset_comparison
plot_multi_dataset_comparison(results['summary'], save_path="comparison.png")

# 5. Compare with baselines
from aco_main import compare_aco_with_baseline
from setting import pipeline_configs
comparison = compare_aco_with_baseline(
    datasets, 
    pipeline_configs,
    aco_params={'n_ants': 15, 'n_iterations': 30}
)
comparison.to_csv('aco_vs_baselines.csv')
```

## 📞 Getting Help

1. Run examples: `python aco_examples.py`
2. Read full docs: `ACO_README.md`
3. Check summary: `ACO_SUMMARY.md`
4. View code: All files have detailed docstrings

## ✅ Checklist

Before running on all test datasets:

- [ ] Tested on 1-2 datasets
- [ ] Verified results look reasonable
- [ ] Checked convergence plots
- [ ] Tuned parameters if needed
- [ ] Set appropriate `save_results` path
- [ ] Estimated total runtime
- [ ] Have backup/checkpoint strategy

## 🏁 Ready to Run!

```bash
cd /drive1/nammt/gfacs/solurec
python aco_examples.py
```

Choose option 1 for quick test, or option 2 for batch optimization!
