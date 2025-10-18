# Code Comments Guide - SoluRec

A quick reference for understanding code comments across the repository.

---

## 📝 Comment Style Guide

### Principle: Only Comment When Necessary

If the code is self-explanatory, **don't add comments**. Good variable names and function names replace comments.

```python
# BAD - Obvious comment:
# Increment counter
counter += 1

# GOOD - No comment needed:
counter += 1

# GOOD - Comment only if non-obvious:
# Exponential backoff: double wait time, max 30 seconds
wait_time = min(wait_time * 2, 30)
```

### Rule of Thumb

Ask yourself:
- **"Is the code obvious?"** → No comment needed
- **"Would someone understand WHY this is done?"** → Add comment
- **"Is this preventing a subtle bug?"** → Add CRITICAL tag
- **"Is this an unexpected behavior?"** → Add comment

### Good Comments Explain WHY

```python
# BAD - Explains WHAT (obvious from code):
# Get the maximum value from the list
max_val = max(values)

# GOOD - Explains WHY:
# Use max() instead of sorted()[0] for O(n) vs O(n log n) performance
max_val = max(values)
```

### When to Comment

1. **WHY not WHAT** - Explain reasoning, not obvious actions
2. **Complex logic** - Non-obvious algorithms or business rules
3. **Critical sections** - Data leakage prevention, important validations
4. **Gotchas** - Unexpected behaviors or edge cases

### Section Pattern

```python
# ==============================================================================
# SECTION_NAME - One-line purpose
# ==============================================================================
# Code here - no comment if self-explanatory

def load_openml_dataset(dataset_id):
    """One-line: what it does and returns."""
    # Only comment if NOT obvious from code
    dataset = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
    X, y = dataset.data, dataset.target
    
    # CRITICAL: Handle NaN before normalization (prevents errors)
    valid_indices = y.dropna().index
    X = X.loc[valid_indices].reset_index(drop=True)
    y = y.loc[valid_indices].reset_index(drop=True)
    
    return X, y
```

---

## 🎯 Comment Examples

### ❌ OVER-COMMENTED (Don't do this)
```python
# Loop through each dataset
for dataset_id in dataset_ids:
    # Load the dataset
    dataset = load_openml_dataset(dataset_id)
    # Get X and y
    X, y = dataset['X'], dataset['y']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create preprocessor
    preprocessor = Preprocessor(config)
```

### ✅ WELL-COMMENTED (Do this)
```python
for dataset_id in dataset_ids:
    dataset = load_openml_dataset(dataset_id)
    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # CRITICAL: Fit on train only to prevent data leakage
    preprocessor = Preprocessor(config)
    preprocessor.fit(X_train, y_train)
    X_test = preprocessor.transform(X_test, y_test)
```

### 🔍 Complex Logic - Comment the WHY
```python
# CRITICAL: Exponential backoff with jitter prevents thundering herd
# when AutoGluon recovers from failure
retry_delay = random.uniform(0.5, 2) * (2 ** attempt)
if retry_delay > 60:
    retry_delay = 60  # Cap at 1 minute
```

---

## 🏷️ Comment Tags

### `recommender_trainer.py` - Main Orchestration
```
ROLE: Train recommenders and evaluate on test datasets

SECTIONS:
1. REPRODUCIBILITY - Fix random seeds
2. CONFIGURATION - Dataset IDs, pipelines, AutoGluon settings  
3. EVALUATION FUNCTION - AutoGluon wrapper
4. MAIN FUNCTION - Entry point, trains recommenders
```

**Run**:
```bash
python recommender_trainer.py --recommender_type all
```

---

### `evaluation_utils.py` - Dataset & Pipeline Evaluation
```
ROLE: Load datasets, apply preprocessing, evaluate with AutoGluon

CLASSES:
- Preprocessor: Stateful pipeline application
- Preprocessor.fit(): Learn transformations from train data
- Preprocessor.transform(): Apply to test data (no data leakage)

FUNCTIONS:
- run_autogluon_evaluation(X_train, y_train, X_test, y_test) → score
- run_experiment_for_dataset(dataset, ...) → (matrix, results, summary)
- safe_train_test_split() → Smart stratified split
```

**Key Insight**: Preprocessor is stateful to prevent data leakage between train/test.

---

### `recommenders.py` - Recommender Implementations
```
ROLE: All recommender models

CLASSES:
- NNRecommender: 3-layer neural network (MAIN)
- KnnRecommender: KNN on meta-features
- RFRecommender: Random Forest regressor
- PmmRecommender: Embedding-based PMM
- *Recommender: Other variants
- HybridMetaRecommender: Ensemble method

KEY METHOD: recommend(dataset_id) → pipeline_name
```

**Neural Network Architecture**:
```
Input (meta-features) 
  ↓
Dense(hidden) + BatchNorm + ReLU
  ↓
Dense(hidden/2) + BatchNorm + ReLU
  ↓
Dense(n_pipelines) + Softmax
  ↓
Output (pipeline probabilities)
```

---

### `pmm_paper_style.py` - Paper's PMM Implementation
```
ROLE: Siamese network for performance prediction

ARCHITECTURE:
- LamNet: Shared embedding network
- SiameseNet: Twin towers for pair comparison
- ContrastiveLoss: Learn to rank pipeline pairs

TRAINING:
1. Sample dataset + pipeline pairs
2. Get their performances
3. Learn to predict relative ordering
4. Recommend pipeline with highest predicted score
```

**Use**: Experimental/slow. Use `pmm` from `recommenders.py` instead.

---

### `data.py` - Interactive Dashboard
```
ROLE: Streamlit app for dataset exploration

PAGES:
1. Overview - Dataset statistics
2. Preview - Show sample data
3. Missing Values - Patterns and percentages
4. Categorical Features - Value distributions
5. Statistics - Numeric summaries
6. Correlations - Feature relationships
```

**Run**: `streamlit run data.py`

---

### `test.py` - Quick Validation
```
ROLE: Test core functionality on 3 datasets

CHECKS:
- Load datasets
- Evaluate baseline pipeline  
- Print results
- ~1 minute runtime
```

**Run**: `python test.py`

---

## 🎯 Understanding Key Concepts

### Data Flow
```
OpenML Dataset ID
        ↓
[load_openml_dataset] → Raw data (X, y)
        ↓
[safe_train_test_split] → (X_train, X_test, y_train, y_test) [80/20]
        ↓
[Preprocessor.fit(X_train, y_train)] → Learn transformations
        ↓
[Preprocessor.transform(X_test, y_test)] → X_test_processed
        ↓
[run_autogluon_evaluation] → Score
```

### Train-Test Split Logic
```
TRAINING DATASETS (90):
  - Use ALL data (no split)
  - Build performance matrix

TEST DATASETS (19):
  - Split 80/20
  - Evaluate pipelines
  - Compare recommender vs ground truth
```

### Recommender Training
```
Performance Matrix (pipelines × datasets)
        ↓
Extract meta-features for each dataset
        ↓
[Recommender.fit(performance_matrix, meta_features)]
        ↓
Learn mapping: meta_features → best_pipeline
        ↓
[Recommender.recommend(dataset_id)] → Pipeline name
```

---

## 🏷️ Comment Tags

Use tags only when they add value:

### `# <- CRITICAL`
Features preventing data leakage or bugs:
```python
# CRITICAL: Fit on train, transform on test (no leakage)
preprocessor.fit(X_train, y_train)
X_test_processed = preprocessor.transform(X_test, y_test)
```

### `# <- TODO`
Future improvements:
```python
# <- TODO: Support regression tasks
if problem_type == 'regression':
    pass  # Not yet implemented
```

### `# <- HACK`
Workarounds needing revision:
```python
# <- HACK: Rename columns to avoid AutoGluon issues
X.columns = [f"col_{i}" for i in range(len(X.columns))]
```

### No tag needed
For obvious code, don't add any comment:
```python
# Good - no comment:
score = accuracy_score(y_test, predictions)

# Not needed:
# Calculate accuracy (too obvious!)
score = accuracy_score(y_test, predictions)
```

---

## � Key Files & Their Purpose

| File | Contains | Format |
|------|----------|--------|
| `preprocessed_performance.csv` | Eval results on 90 training datasets | rows=pipelines, cols=D_ID |
| `test_evaluation_summary.csv` | Aggregated metrics by recommender | rows=recommenders, cols=metrics |
| `recommender_evaluation_results.csv` | Per-dataset predictions | rows=predictions, cols=details |

---

## 🧪 Common Code Patterns

### Loading a Dataset
```python
from evaluation_utils import load_openml_dataset
dataset = load_openml_dataset(dataset_id=1503)
# Returns: {'id': 1503, 'name': 'D_1503', 'X': DataFrame, 'y': Series}
```

### Evaluating a Pipeline
```python
from evaluation_utils import Preprocessor, run_autogluon_evaluation

preprocessor = Preprocessor(config)
preprocessor.fit(X_train, y_train)
X_train_p, y_train_p = preprocessor.transform(X_train, y_train)
X_test_p, y_test_p = preprocessor.transform(X_test, y_test)

score = run_autogluon_evaluation(X_train_p, y_train_p, X_test_p, y_test_p)
```

### Training a Recommender
```python
from recommenders import NNRecommender

recommender = NNRecommender(performance_matrix, meta_features_df)
recommender.fit()
pipeline = recommender.recommend(dataset_id=1503)
```

---

## ⚡ Performance Tips

1. **Use baseline first**: Test logic without AutoGluon wait
2. **Skip paper_pmm**: Experimental, very slow - use `pmm` instead
3. **Enable GPU**: `export CUDA_VISIBLE_DEVICES=0` for PyTorch
4. **Reduce time_limit**: Change `AUTOGLUON_CONFIG['time_limit']` to 300 for quick testing

---

## 🐛 Debugging

### Check Data Leakage
```python
# This is BAD (leakage):
scaler.fit(X_full)  # Fits on entire data
X_train = scaler.transform(X_train)  # Test info in X_full!

# This is GOOD (no leakage):
scaler.fit(X_train)  # Fits only on train
X_test = scaler.transform(X_test)  # Only seen after fitting
```

### Verify Recommender
```python
pipeline = recommender.recommend(1503)
print(f"Recommended: {pipeline}")
# Should be one of: baseline, simple_preprocess, robust_preprocess, ...
```

### Check Performance Matrix
```python
import pandas as pd
perf = pd.read_csv('preprocessed_performance.csv', index_col=0)
print(perf.iloc[:, :3])  # Check first 3 datasets
print(perf.describe())   # Statistics
print(perf.isna().sum())  # Missing values
```

---

## 🎓 Learning Path

**Beginner**: Read this document + `README.md`  
**Intermediate**: Run `python test.py`, explore output files  
**Advanced**: Modify `pipeline_configs`, add new recommender class  
**Expert**: Understand `pmm_paper_style.py` architecture  

---

**Last Updated**: October 18, 2025
