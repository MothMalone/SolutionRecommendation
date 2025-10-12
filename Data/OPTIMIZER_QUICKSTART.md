# Quick Start Guide: Optimization-Based Recommender

This guide shows you how to use the new optimization-based pipeline recommender.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Make sure SMAC3 is installed
pip install smac
# or
pip install -r requirements.txt
```

### 2. Train the Optimizer

```bash
# Train the optimizer as a recommender type
python recommender_trainer.py --recommender optimizer
```

**What this does:**
- Loads `preprocessed_performance.csv` (12 pipelines × 90 datasets)
- Loads `dataset_feats.csv` (metafeatures)
- Trains a Random Forest surrogate model that predicts:
  ```
  performance = f(dataset_metafeatures, pipeline_configuration)
  ```
- Saves the trained model for later use

**Expected output:**
```
TRAINING OPTIMIZER RECOMMENDER
  This recommender searches the configuration space to construct custom pipelines
  
Training surrogate model on 1080 examples...
✅ Surrogate model trained successfully!
  Model can predict performance for any (dataset, pipeline) combination
```

### 3. Evaluate the Optimizer

```bash
# Evaluate on test datasets
python evaluate_optimizer.py
```

**What this does:**
1. Shows educational explanation of how optimization works
2. Loads test ground truth data
3. For each test dataset:
   - Runs Bayesian optimization (50 evaluations)
   - Finds best custom pipeline
   - Compares against 12 predefined pipelines
4. Outputs discovery analysis and performance metrics

**Expected output:**
```
HOW THE OPTIMIZATION-BASED RECOMMENDER WORKS
================================================================

[Educational explanation...]

Press Enter to continue with evaluation...

OPTIMIZER EVALUATION ON TEST DATASETS
================================================================

[1/19] Dataset 1503
  🔍 Running SMAC optimization (50 evaluations)...
  ✅ Optimizer REDISCOVERED predefined pipeline: 'robust_preprocess'
     Configuration:
       - imputation: median
       - scaling: robust
       - encoding: onehot
       - feature_selection: none
       - outlier_removal: iqr
       - dimensionality_reduction: none
  📊 Performance:
     - Rank: 2/12
     - Score: 0.8234
     - Best score: 0.8456
     - Gap to best: 0.0222

[2/19] Dataset 23517
  🔍 Running SMAC optimization (50 evaluations)...
  🆕 Optimizer found NOVEL pipeline: median_maxabs_onehot_k_best_lof_none
     Configuration:
       - imputation: median
       - scaling: maxabs
       - encoding: onehot
       - feature_selection: k_best
       - outlier_removal: lof
       - dimensionality_reduction: none
  ⚠️  Novel pipeline - would need evaluation to get true score

...

EVALUATION SUMMARY
================================================================

✅ Successfully evaluated 19 test datasets

🔍 Discovery Analysis:
  - Novel pipelines: 7 (36.8%)
  - Rediscovered pipelines: 12 (63.2%)
  
  Most rediscovered pipelines:
    - robust_preprocess: 4 times
    - baseline: 3 times
    - simple_preprocess: 2 times

📊 Performance Metrics (on 12 datasets with rankings):
  - Average rank: 3.42
  - Top-1 accuracy: 3/12 (25.0%)
  - Top-3 accuracy: 8/12 (66.7%)
  - Average score gap: 0.0345
  - Better than baseline: 10/12 (83.3%)

💾 Results saved to: optimizer_evaluation_results.csv
```

### 4. Analyze Results

```bash
# View the detailed results
cat optimizer_evaluation_results.csv
```

**Columns in results:**
- `dataset_id`: OpenML dataset ID
- `recommended_pipeline`: The custom pipeline constructed by optimizer
- `matching_predefined`: Which predefined pipeline it matches (or "NOVEL")
- `is_novel`: Whether it's a novel combination
- `rank`: Where it ranks among the 12 predefined pipelines
- `optimizer_score`: Its accuracy score
- `best_pipeline`: The best performing predefined pipeline
- `best_score`: The highest score achieved
- `score_gap`: Difference between optimizer and best

---

## 🎓 Understanding the Output

### Rediscovered Pipelines

When the optimizer outputs:
```
✅ Optimizer REDISCOVERED predefined pipeline: 'robust_preprocess'
```

**What this means:**
- The optimizer found that a predefined pipeline is optimal for this dataset
- This validates that our predefined set includes good options
- We can directly compare its rank and score

**Example interpretation:**
```
Rank: 2/12  →  The optimizer's choice is the 2nd best among 12 options
Score: 0.8234  →  It achieves 82.34% accuracy
Gap to best: 0.0222  →  Only 2.2% worse than the best pipeline
```

### Novel Pipelines

When the optimizer outputs:
```
🆕 Optimizer found NOVEL pipeline: median_maxabs_onehot_k_best_lof_none
```

**What this means:**
- The optimizer found a combination NOT in the predefined set
- This is a genuinely new pipeline configuration
- To know its true performance, we'd need to evaluate it on the dataset

**Why is this interesting?**
- Shows the optimizer is exploring beyond our predefined options
- Might discover better combinations we didn't think of
- Reveals gaps in our predefined pipeline set

**Example novel pipeline breakdown:**
```
median_maxabs_onehot_k_best_lof_none
  ↓
  imputation: median           (fill missing with median)
  scaling: maxabs              (scale to [-1, 1])
  encoding: onehot             (one-hot encode categoricals)
  feature_selection: k_best    (select top-k features)
  outlier_removal: lof          (remove outliers via Local Outlier Factor)
  dimensionality_reduction: none (no PCA/SVD)
```

### Performance Metrics

**Average Rank**: Lower is better
- `1.0` = Always finds the best pipeline (perfect!)
- `3.42` = On average, 2-3 pipelines perform better
- `12.0` = Always picks the worst (very bad)

**Top-1 Accuracy**: Higher is better
- `100%` = Always finds the #1 best pipeline
- `25%` = Finds the best pipeline 1 out of 4 times
- `0%` = Never finds the best

**Top-3 Accuracy**: Higher is better
- `100%` = Always in top 3
- `66.7%` = In top 3 about 2 out of 3 times
- Easier than Top-1, shows "good enough" performance

**Score Gap**: Lower is better
- `0.0` = Matches the best score (perfect)
- `0.0345` = On average 3.45% worse than best
- How much performance we're losing by not picking the best

---

## 🔧 Advanced Usage

### Adjust Optimization Budget

```bash
# More evaluations = better optimization (but slower)
python evaluate_optimizer.py --n-trials 100

# Fewer evaluations = faster (but might miss optimal)
python evaluate_optimizer.py --n-trials 25
```

**Trade-off:**
- More trials: Better chance of finding optimal pipeline
- Fewer trials: Faster evaluation, but may get stuck in local optimum

**Recommendation:**
- Start with 50 (default) for reasonable results
- Try 100 if you want higher quality
- Use 25 for quick testing

### Skip the Explanation

```bash
# Skip the educational explanation (faster start)
python evaluate_optimizer.py --no-explain
```

### Use Custom Paths

```bash
# If your files are in different locations
python evaluate_optimizer.py \
  --performance-matrix path/to/performance.csv \
  --metafeatures path/to/features.csv \
  --test-ground-truth path/to/test.csv
```

---

## 📊 Interpreting Different Scenarios

### Scenario 1: High Rediscovery Rate (>80%)

**Interpretation:**
- Our predefined set is comprehensive
- Most optimal pipelines are already included
- Less benefit from searching larger space

**Action:**
- Focus on improving existing recommenders' selection accuracy
- Predefined set is already good

### Scenario 2: High Novel Rate (>50%)

**Interpretation:**
- Optimizer is finding many pipelines not in our predefined set
- Significant unexplored space with potentially better options
- Our predefined set may have gaps

**Action:**
- Evaluate the novel pipelines to see if they're actually better
- Consider adding successful novel pipelines to predefined set
- Expand predefined set based on discoveries

### Scenario 3: Novel Pipelines Rank Better

**What to look for:**
If you manually evaluate novel pipelines and find they rank higher than predefined:

**Interpretation:**
- Strong evidence that optimization approach is beneficial
- Searching larger space yields better solutions
- Predefined set was suboptimal

**Action:**
- Use optimizer as primary recommender
- Build library of successful novel pipelines
- Update predefined set with discovered patterns

### Scenario 4: Consistent Top-3 Performance

**Interpretation:**
- Optimizer reliably finds good (if not best) pipelines
- Demonstrates robust recommendation capability
- May have slight calibration issues (predicts well, doesn't optimize to best)

**Action:**
- Consider this "good enough" for practical use
- Could try increasing n_trials to improve to Top-1
- May indicate surrogate model needs improvement

---

## 🐛 Troubleshooting

### Error: "No module named 'smac'"

```bash
pip install smac
```

### Error: "File not found: preprocessed_performance.csv"

You need the training data first:
```bash
# Make sure these files exist:
ls preprocessed_performance.csv
ls dataset_feats.csv
ls test_ground_truth_performance.csv
```

### Optimizer Takes Too Long

```bash
# Reduce number of trials
python evaluate_optimizer.py --n-trials 25
```

### Want to See Detailed SMAC Output

Edit `optimized_pipeline_recommender.py`:
```python
# Line ~150: Change logging=logging.WARNING to logging=logging.INFO
scenario = Scenario(
    configspace=self.config_space,
    deterministic=True,
    n_trials=self.n_trials,
    seed=self.random_state,
    logging_level=logging.INFO  # ← Change this
)
```

---

## 📈 Next Steps

### 1. Evaluate Novel Pipelines

For pipelines marked as "NOVEL", you can evaluate them:

```python
from evaluation_utils import load_openml_dataset, Preprocessor
from autogluon.tabular import TabularPredictor

# Load the dataset
dataset = load_openml_dataset(dataset_id)

# Create the novel pipeline
novel_pipeline_config = {
    'imputation': 'median',
    'scaling': 'maxabs',
    'encoding': 'onehot',
    'feature_selection': 'k_best',
    'outlier_removal': 'lof',
    'dimensionality_reduction': 'none'
}

# Preprocess with this pipeline
preprocessor = Preprocessor(**novel_pipeline_config)
X_processed, y = preprocessor.fit_transform(dataset['X'], dataset['y'])

# Train AutoGluon and get score
predictor = TabularPredictor(label='target').fit(...)
score = predictor.leaderboard()['score_val'].max()

print(f"Novel pipeline score: {score:.4f}")
```

### 2. Compare Multiple Recommenders

```bash
# Evaluate different recommender types
python recommender_trainer.py --recommender pmm --evaluate
python recommender_trainer.py --recommender rf --evaluate
python recommender_trainer.py --recommender optimizer --evaluate

# Compare their test_evaluation_summary.csv files
```

### 3. Expand the Configuration Space

Edit `optimized_pipeline_recommender.py` to add more options:

```python
self.config_space.add(
    ConfigurationSpace.CategoricalHyperparameter(
        'imputation',
        choices=['none', 'mean', 'median', 'most_frequent', 'knn', 'constant', 'iterative']  # ← Add 'iterative'
    )
)
```

### 4. Use Optimizer in Production

```python
from optimized_pipeline_recommender import OptimizedPipelineRecommender
import pandas as pd

# Load trained optimizer
performance_matrix = pd.read_csv('preprocessed_performance.csv', index_col=0)
metafeatures = pd.read_csv('dataset_feats.csv', index_col=0)

optimizer = OptimizedPipelineRecommender()
optimizer.fit(performance_matrix, metafeatures)

# Get recommendation for new dataset
recommended_pipeline = optimizer.recommend(new_dataset_id)

print(f"Recommended pipeline: {recommended_pipeline}")
# Output: {'imputation': 'median', 'scaling': 'robust', ...}
```

---

## 💡 Key Insights

### What Makes This Powerful

1. **Learns Patterns**: Understands what preprocessing helps which datasets
2. **Searches Efficiently**: Finds good pipelines in 50 tries vs 3,600 total
3. **Generalizes**: Applies knowledge from 90 training datasets to new ones
4. **Flexible**: Not limited to predefined options

### When to Use Optimizer vs Traditional Recommenders

**Use Optimizer When:**
- Dataset characteristics are unusual
- Predefined pipelines all perform poorly
- You want to explore beyond fixed options
- You have time for 50+ evaluations per dataset

**Use Traditional Recommenders When:**
- Need instant recommendations (no optimization time)
- Predefined set already works well
- Want simpler, more interpretable choices
- Limited computational budget

### The Big Picture

The optimizer represents a **shift in philosophy**:

**Old Approach**: "Which of these 12 options is best?"
**New Approach**: "What is the optimal configuration for this dataset?"

This is more flexible but also more complex. The choice between them depends on your specific use case and constraints.

---

## 🎉 Summary

You now have a working optimization-based recommender that:
- ✅ Searches a space of ~3,600 pipeline combinations
- ✅ Uses Bayesian optimization for efficient search
- ✅ Learns from existing performance data
- ✅ Can discover novel pipeline configurations
- ✅ Compares results against predefined baselines

Try it out and see what interesting pipelines it discovers for your datasets!
