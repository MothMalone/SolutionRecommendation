# ACO-based Preprocessing Pipeline Optimizer

A complete Ant Colony Optimization (ACO) system for automatically finding the optimal combination of preprocessing operators for machine learning datasets.

## 📋 Overview

This system uses ACO to search through the space of preprocessing pipeline configurations and find the best combination for each dataset. Instead of relying on limited training data or meta-features, ACO actively explores the pipeline space through iterative evaluation.

### The Problem

Given a dataset, we need to select **one operator** from each preprocessing category:

- **Imputation**: `none`, `mean`, `median`, `most_frequent`, `knn`, `constant`
- **Scaling**: `none`, `standard`, `minmax`, `robust`, `maxabs`
- **Encoding**: `none`, `onehot`
- **Feature Selection**: `none`, `variance_threshold`, `k_best`, `mutual_info`
- **Outlier Removal**: `none`, `iqr`, `zscore`, `lof`, `isolation_forest`
- **Dimensionality Reduction**: `none`, `pca`, `svd`

This creates a search space of **6 × 5 × 2 × 4 × 5 × 3 = 3,600 possible pipelines**!

### The Solution

ACO uses artificial "ants" that construct complete pipelines by probabilistically selecting operators. Good pipelines deposit more "pheromone," making their operator choices more likely to be selected by future ants. Over iterations, the system converges to high-performing pipeline configurations.

## 🚀 Quick Start

### Basic Usage

```python
from setting import load_openml_dataset, test_dataset_ids, options
from aco_main import optimize_dataset_pipeline

# Load a dataset
dataset = load_openml_dataset(test_dataset_ids[0])

# Run ACO optimization
results = optimize_dataset_pipeline(
    dataset=dataset,
    aco_params={
        'n_ants': 15,
        'n_iterations': 30,
        'verbose': True
    }
)

# Get best pipeline
print(f"Best Score: {results['best_score']:.4f}")
print(f"Best Pipeline: {results['best_pipeline']}")
```

### Batch Optimization

```python
from aco_main import run_aco_on_test_datasets

# Load multiple datasets
test_datasets = [load_openml_dataset(id) for id in test_dataset_ids[:5]]

# Optimize all datasets
results = run_aco_on_test_datasets(
    test_datasets=test_datasets,
    aco_params={'n_ants': 15, 'n_iterations': 30},
    save_results=True,
    output_prefix="results/aco_batch"
)
```

### With Visualization

```python
from aco_visualization import create_full_visualization_suite

# After optimization
create_full_visualization_suite(
    results=results,
    dataset_name=dataset['name'],
    output_dir="./visualizations"
)
```

## 📁 File Structure

```
solurec/
├── aco_pipeline_optimizer.py    # Core ACO implementation
├── aco_main.py                   # Main execution and evaluation
├── aco_visualization.py          # Visualization and analysis
├── aco_examples.py               # Tutorial and examples
├── setting.py                    # Your existing preprocessing code
└── aco_output/                   # Generated results (created automatically)
```

## 🎯 Key Features

### 1. **Adaptive Heuristic Learning**
- Tracks performance of each operator across evaluations
- Updates heuristic information to guide future searches
- Operators that perform well get higher selection probabilities

### 2. **Elitist Pheromone Update**
- Best solutions get extra pheromone deposit
- Balances global best vs iteration best
- Prevents premature convergence

### 3. **Pheromone Bounds**
- Min/max pheromone levels prevent stagnation
- Ensures continuous exploration even late in search

### 4. **Pseudo-Random Proportional Rule**
- Parameter `q0` controls exploitation vs exploration
- With probability `q0`: select best operator (exploit)
- With probability `1-q0`: probabilistic selection (explore)

## 🔧 ACO Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_ants` | 20 | Number of ants per iteration |
| `n_iterations` | 50 | Number of optimization iterations |
| `alpha` | 1.0 | Pheromone importance (higher = follow pheromone more) |
| `beta` | 2.0 | Heuristic importance (higher = follow heuristic more) |
| `rho` | 0.1 | Pheromone evaporation rate (0-1) |
| `q0` | 0.9 | Exploitation probability (0-1) |
| `elite_weight` | 2.0 | Weight multiplier for best solution pheromone |

### Parameter Tuning Guidelines

**For faster convergence:**
```python
aco_params = {
    'alpha': 2.0,      # Follow pheromone trails more
    'q0': 0.95,        # Exploit more, explore less
    'elite_weight': 3.0  # Emphasize best solutions
}
```

**For better exploration:**
```python
aco_params = {
    'beta': 3.0,       # Follow heuristic more
    'q0': 0.7,         # Explore more
    'rho': 0.2         # Faster pheromone decay
}
```

**For limited time:**
```python
aco_params = {
    'n_ants': 25,      # More ants per iteration
    'n_iterations': 15  # Fewer iterations
}
```

**For thorough search:**
```python
aco_params = {
    'n_ants': 15,      # Moderate ants
    'n_iterations': 50  # More iterations
}
```

## 📊 Output and Results

### Results Dictionary

```python
results = {
    'best_pipeline': {...},           # Best pipeline configuration
    'best_score': 0.8543,             # Best performance score
    'iteration_best_scores': [...],   # Best score per iteration
    'all_solutions': [...],           # All evaluated solutions
    'total_time': 234.56,             # Total optimization time (seconds)
    'final_pheromone': {...},         # Final pheromone levels
    'final_heuristic': {...},         # Final heuristic values
    'pheromone_summary': DataFrame,   # Summary of pheromone levels
    'operator_statistics': DataFrame  # Performance stats per operator
}
```

### Generated Files

**Batch optimization:**
- `{prefix}_summary.csv` - Summary of results across datasets
- `{prefix}_detailed.csv` - Detailed results with pipeline components

**Visualizations:**
- `{dataset}_convergence.png` - Convergence curve
- `{dataset}_pheromone.png` - Final pheromone levels
- `{dataset}_operator_usage.png` - Operator usage and performance
- `{dataset}_report.txt` - Text report

## 💡 Examples

### Example 1: Quick Test

```python
from aco_examples import example_1_single_dataset

# Runs ACO on one dataset with full visualization
example_1_single_dataset()
```

### Example 2: Compare with Baselines

```python
from aco_examples import example_3_compare_with_baselines

# Compares ACO with your predefined pipelines
example_3_compare_with_baselines()
```

### Example 3: Parameter Sensitivity

```python
from aco_examples import example_4_custom_aco_parameters

# Tests different ACO parameter configurations
example_4_custom_aco_parameters()
```

### Example 4: Full Analysis

```python
from aco_examples import example_5_analyze_pipeline_components

# Analyzes which operators perform best
example_5_analyze_pipeline_components()
```

## 🎓 How ACO Works

### Algorithm Overview

1. **Initialization**
   - Set all pheromone levels to initial value
   - Initialize heuristic information (uniform)

2. **For each iteration:**
   
   a. **Construction Phase** (for each ant):
      - For each preprocessing step:
        - Calculate selection probabilities based on pheromone and heuristic
        - Select operator using pseudo-random proportional rule
      - Evaluate complete pipeline on dataset
   
   b. **Update Phase**:
      - Evaporate all pheromone trails
      - Deposit pheromone on paths used by ants (weighted by performance)
      - Extra pheromone for iteration best and global best
      - Update heuristic information based on operator performance

3. **Return best solution found**

### Why ACO Works Well Here

✅ **No need for meta-features** - Direct evaluation on target dataset  
✅ **Handles combinatorial search** - Efficiently explores large spaces  
✅ **Adaptive** - Learns which operators work well during search  
✅ **Balances exploration/exploitation** - Avoids local optima  
✅ **Parallelizable** - Ants can be evaluated in parallel  
✅ **Anytime algorithm** - Can stop early if time-limited  

## 🔬 Advanced Usage

### Custom Evaluation Function

```python
def my_evaluation_function(dataset, pipeline_dict):
    """Custom evaluation with your own logic"""
    # Apply preprocessing
    config = pipeline_dict_to_config(pipeline_dict)
    X_proc, y_proc = apply_preprocessing(dataset['X'], dataset['y'], config)
    
    # Your custom evaluation
    score = my_model.evaluate(X_proc, y_proc)
    
    return score

# Use in optimization
results = optimizer.optimize(
    dataset=dataset,
    evaluate_func=my_evaluation_function
)
```

### Warm Start from Previous Results

```python
# First optimization
optimizer1 = ACOPipelineOptimizer(options=options, n_ants=15, n_iterations=30)
results1 = optimizer1.optimize(dataset, evaluate_func)

# Continue with learned pheromones
optimizer2 = ACOPipelineOptimizer(options=options, n_ants=15, n_iterations=20)
optimizer2.pheromone = results1['final_pheromone'].copy()
optimizer2.heuristic = results1['final_heuristic'].copy()
results2 = optimizer2.optimize(dataset, evaluate_func)
```

### Constrained Search

```python
# Modify options to constrain search space
constrained_options = {
    'imputation': ['mean', 'median'],  # Only these two
    'scaling': ['standard', 'robust'],
    'encoding': ['onehot'],            # Fixed
    'feature_selection': options['feature_selection'],  # All
    'outlier_removal': ['none', 'iqr'],
    'dimensionality_reduction': ['none']  # Fixed
}

optimizer = ACOPipelineOptimizer(
    options=constrained_options,
    n_ants=15,
    n_iterations=20
)
```

## 📈 Performance Tips

### For Large Datasets (>5000 samples)

```python
# Use faster AutoGluon settings
AUTOGLUON_CONFIG['time_limit'] = 180  # 3 minutes instead of 5
AUTOGLUON_CONFIG['presets'] = 'good_quality'  # Faster preset

# Reduce ACO iterations
aco_params = {'n_ants': 12, 'n_iterations': 20}
```

### For Many Datasets

```python
# Use multiprocessing (implement parallel ant evaluation)
# Or run datasets sequentially but save checkpoints

for i, dataset in enumerate(datasets):
    results = optimize_dataset_pipeline(dataset, aco_params)
    # Save after each dataset
    save_checkpoint(results, f'checkpoint_{i}.pkl')
```

### For Time-Limited Scenarios

```python
# Aggressive parameters for quick search
aco_params = {
    'n_ants': 20,       # More ants
    'n_iterations': 10,  # Fewer iterations
    'q0': 0.95,         # More exploitation
    'elite_weight': 3.0
}
```

## 🐛 Troubleshooting

### Issue: Low scores / Poor convergence

**Solutions:**
- Increase `n_iterations`
- Increase `beta` (rely more on heuristic)
- Decrease `q0` (more exploration)
- Check that evaluation function returns valid scores

### Issue: Slow execution

**Solutions:**
- Reduce AutoGluon `time_limit`
- Use fewer ants or iterations
- Use simpler AutoGluon preset
- Reduce dataset size if very large

### Issue: All ants select same pipeline

**Solutions:**
- Decrease `alpha` (less pheromone influence)
- Increase `rho` (faster evaporation)
- Decrease `q0` (less exploitation)
- Check `min_pheromone` isn't too high

### Issue: Results not better than baseline

**Solutions:**
- Run more iterations
- Try different ACO parameters
- Check that baseline isn't already optimal
- Verify evaluation function is correct

## 📚 References

### ACO Algorithm
- Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
- Dorigo, M., & Gambardella, L. M. (1997). Ant colony system. *IEEE Transactions on Evolutionary Computation*.

### Applications to ML Pipeline Optimization
- Feurer, M., et al. (2015). Efficient and robust automated machine learning. *NeurIPS*.
- De Sá, A. G., et al. (2017). RECIPE: A grammar-based framework for automatically evolving classification pipelines. *EuroGP*.

## 📝 License

This code is part of the gfacs project. See main repository for license details.

## 🤝 Contributing

To extend this system:

1. **Add new preprocessing operators**: Update `options` in `setting.py`
2. **Add new evaluation metrics**: Modify `evaluate_pipeline_with_autogluon`
3. **Implement parallel evaluation**: Use multiprocessing in ant construction phase
4. **Add new ACO variants**: Extend `ACOPipelineOptimizer` class

## 📧 Support

For issues or questions:
1. Check this README
2. Run examples in `aco_examples.py`
3. Review visualization outputs
4. Check generated reports

---

**Ready to optimize your pipelines? Start with:**

```bash
python aco_examples.py
```

Then select an example to run!
