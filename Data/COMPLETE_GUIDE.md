# Complete Guide to Recommenders and Preprocessing Pipelines

## Table of Contents
1. [Overview](#overview)
2. [Preprocessing Pipelines Explained](#preprocessing-pipelines-explained)
3. [Recommender Systems Explained](#recommender-systems-explained)
4. [When to Use Each Recommender](#when-to-use-each-recommender)
5. [Performance Metrics Explained](#performance-metrics-explained)

---

## Overview

This system recommends preprocessing pipelines for machine learning datasets. The goal is to automatically select the best preprocessing strategy based on dataset characteristics (meta-features) without running expensive experiments.

**The Problem:** Given a new dataset, which preprocessing pipeline will give the best classification accuracy?

**The Solution:** Train a recommender system on historical performance data to predict which pipeline works best.

---

## Preprocessing Pipelines Explained

We have **12 preprocessing pipelines**, each combining different data cleaning and transformation techniques. Here's what each component does:

### Pipeline Components

#### 1. **Imputation** (Handling Missing Values)
Missing values are common in real-world data and must be handled before training.

- **none**: No imputation (fails if missing values exist)
- **mean**: Replace missing values with column mean
  - *Good for*: Normally distributed numerical data
  - *Example*: Age with a few missing values → use average age
  
- **median**: Replace missing values with column median
  - *Good for*: Skewed data or data with outliers
  - *Example*: Income data where billionaires skew the mean
  
- **most_frequent**: Replace with most common value
  - *Good for*: Categorical data
  - *Example*: Missing "Country" → use most frequent country
  
- **constant**: Replace with a constant value (e.g., -1)
  - *Good for*: When "missing" is informative
  - *Example*: "Years of experience" missing for unemployed → use 0
  
- **knn**: Use K-Nearest Neighbors to predict missing values
  - *Good for*: Complex patterns in missing data
  - *Expensive*: Slower but more accurate

#### 2. **Scaling** (Normalizing Feature Ranges)
Machine learning algorithms often work better when features are on similar scales.

- **none**: No scaling (raw values)
  - *Good for*: Tree-based models (Random Forest, XGBoost)
  
- **standard**: Z-score normalization (mean=0, std=1)
  - *Formula*: `(x - mean) / std`
  - *Good for*: Neural networks, SVM, linear models
  - *Assumes*: Gaussian distribution
  
- **minmax**: Scale to [0, 1] range
  - *Formula*: `(x - min) / (max - min)`
  - *Good for*: Neural networks, bounded data
  - *Problem*: Sensitive to outliers
  
- **robust**: Use median and IQR instead of mean and std
  - *Formula*: `(x - median) / IQR`
  - *Good for*: Data with outliers
  - *Example*: Income data with extreme values
  
- **maxabs**: Scale by maximum absolute value to [-1, 1]
  - *Formula*: `x / max(|x|)`
  - *Good for*: Sparse data, preserves zeros

#### 3. **Encoding** (Converting Categorical to Numerical)

- **none**: No encoding (assumes all numerical)
- **onehot**: Create binary columns for each category
  - *Example*: "Color: Red, Blue, Green" → 3 columns [1,0,0], [0,1,0], [0,0,1]
  - *Good for*: Low cardinality (<20 categories)
  - *Problem*: High dimensionality for many categories

#### 4. **Feature Selection** (Removing Irrelevant Features)
Too many features can cause overfitting and slow training.

- **none**: Keep all features
  
- **variance_threshold**: Remove low-variance features
  - *Rationale*: Features that don't vary much are not informative
  - *Example*: Remove "Gender" if 99% of data is male
  
- **k_best**: Keep top K features based on statistical tests
  - *Method*: ANOVA F-test for classification
  - *Good for*: Reducing dimensionality quickly
  - *Keeps*: Top 50% of features
  
- **mutual_info**: Select features with high mutual information with target
  - *Better than*: Correlation-based methods
  - *Captures*: Non-linear relationships
  - *Example*: Quadratic relationships that correlation misses

#### 5. **Outlier Removal** (Handling Extreme Values)
Outliers can distort model training and reduce accuracy.

- **none**: Keep all data points
  
- **iqr**: Remove data outside 1.5 × IQR from quartiles
  - *Formula*: Remove if x < Q1 - 1.5×IQR or x > Q3 + 1.5×IQR
  - *Standard*: Boxplot rule
  - *Good for*: Normally distributed data
  
- **zscore**: Remove data with |z-score| > 3
  - *Formula*: Remove if |(x - mean) / std| > 3
  - *Assumes*: Gaussian distribution
  - *99.7% rule*: Keeps 99.7% of normal data
  
- **isolation_forest**: ML-based outlier detection
  - *Method*: Isolates outliers using random trees
  - *Good for*: High-dimensional data, complex outliers
  - *Better than*: Statistical methods for complex patterns
  
- **lof**: Local Outlier Factor (density-based)
  - *Method*: Compares local density to neighbors
  - *Good for*: Clustered data with varying densities
  - *Example*: City populations (some dense areas, some sparse)

#### 6. **Dimensionality Reduction** (Compressing Features)
Reduces features while preserving information.

- **none**: Keep original dimensions
  
- **pca**: Principal Component Analysis
  - *Method*: Linear combinations that capture maximum variance
  - *Good for*: Correlated features, visualization
  - *Keeps*: 95% of variance
  - *Example*: 100 correlated stock prices → 10 principal components
  
- **svd**: Singular Value Decomposition
  - *Similar to*: PCA but works on sparse matrices
  - *Good for*: Text data, sparse features
  - *Faster than*: PCA for sparse data

---

### The 12 Pipelines Explained

#### 1. **baseline**
```python
{'imputation': 'none', 'scaling': 'none', 'encoding': 'none', 
 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Minimal preprocessing - use data as-is.

**When it works:**
- Clean, complete, numerical data
- Tree-based models (don't need scaling)
- Small datasets where preprocessing might overfit

**When it fails:**
- Missing values (will crash)
- Different feature scales (bad for neural nets)
- Categorical variables

**Example use case:** UCI Iris dataset - clean, no missing values, already numerical.

---

#### 2. **simple_preprocess**
```python
{'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot',
 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Standard ML textbook preprocessing.

**When it works:**
- Most standard datasets
- Normally distributed features
- Neural networks and SVMs

**When it fails:**
- Heavy outliers (mean imputation is affected)
- Skewed distributions
- High-dimensional categorical variables

**Example use case:** Student performance prediction - some missing test scores, mix of numerical and categorical features.

---

#### 3. **robust_preprocess**
```python
{'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot',
 'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Defense against outliers.

**When it works:**
- Real-world messy data
- Income, price, or measurement data with extremes
- When data quality is questionable

**Why it's better than simple:**
- Median imputation: Not affected by outliers
- Robust scaling: Uses IQR instead of std
- IQR outlier removal: Removes extreme points

**Example use case:** Housing prices - some outliers (mansions), some missing values.

---

#### 4. **feature_selection**
```python
{'imputation': 'median', 'scaling': 'standard', 'encoding': 'onehot',
 'feature_selection': 'k_best', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Focus on most important features.

**When it works:**
- High-dimensional data (many features)
- Irrelevant features present
- Risk of overfitting

**Why it's useful:**
- Reduces computation time
- Reduces overfitting
- Improves interpretability

**Example use case:** Gene expression data - 20,000 genes but only 50 are relevant to disease.

---

#### 5. **dimension_reduction**
```python
{'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot',
 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'}
```
**Philosophy:** Compress correlated features.

**When it works:**
- Highly correlated features
- Visualization needed
- Large feature space

**How PCA helps:**
- Removes redundancy
- Captures 95% variance with fewer features
- Can reveal hidden patterns

**Example use case:** Image pixels - 784 pixels for MNIST, but can use 50 principal components.

---

#### 6. **conservative**
```python
{'imputation': 'median', 'scaling': 'minmax', 'encoding': 'onehot',
 'feature_selection': 'variance_threshold', 'outlier_removal': 'none', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Gentle preprocessing - remove obviously useless features.

**When it works:**
- Small datasets (aggressive preprocessing might overfit)
- When you want to preserve most information
- Neural networks (minmax scaling good for bounded activations)

**Why "conservative":**
- Only removes zero-variance features
- Doesn't remove outliers (might be important)
- Gentle scaling

**Example use case:** Medical diagnosis - don't want to lose any potentially important features.

---

#### 7. **aggressive**
```python
{'imputation': 'mean', 'scaling': 'standard', 'encoding': 'onehot',
 'feature_selection': 'k_best', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'pca'}
```
**Philosophy:** Maximum preprocessing - throw everything at it.

**When it works:**
- Large datasets (can afford to remove data)
- Noisy data
- Many irrelevant features
- Outliers are likely errors, not important cases

**What it does:**
1. Imputes missing values
2. Removes outliers (cleaner data)
3. Selects best features (removes noise)
4. Reduces dimensions (captures patterns)

**When it fails:**
- Small datasets (over-preprocessing)
- Outliers are important (fraud detection)

**Example use case:** Customer churn prediction - large dataset, many irrelevant features, some data entry errors.

---

#### 8. **knn_impute_pca**
```python
{'imputation': 'knn', 'scaling': 'standard', 'encoding': 'onehot',
 'feature_selection': 'none', 'outlier_removal': 'none', 'dimensionality_reduction': 'pca'}
```
**Philosophy:** Sophisticated imputation + dimensionality reduction.

**When it works:**
- Missing data has patterns (MAR - Missing At Random)
- Correlated features
- Enough data for KNN to work well

**Why KNN imputation:**
- More accurate than mean/median
- Uses similarity to predict missing values
- Example: Missing age? Use age of similar people

**Trade-off:**
- Much slower than mean/median
- Needs enough complete cases

**Example use case:** Survey data - people who skip question A also skip B (pattern in missingness).

---

#### 9. **mutual_info_zscore**
```python
{'imputation': 'median', 'scaling': 'robust', 'encoding': 'onehot',
 'feature_selection': 'mutual_info', 'outlier_removal': 'zscore', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Smart feature selection + statistical outlier removal.

**When it works:**
- Non-linear relationships between features and target
- Gaussian-ish data (z-score works)
- Complex feature interactions

**Why mutual information:**
- Captures non-linear relationships
- Better than correlation for complex patterns
- Example: Captures XOR relationships

**Example use case:** Bioinformatics - complex gene interactions, some measurement errors.

---

#### 10. **constant_maxabs_iforest**
```python
{'imputation': 'constant', 'scaling': 'maxabs', 'encoding': 'onehot',
 'feature_selection': 'variance_threshold', 'outlier_removal': 'isolation_forest', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Sparse data + ML-based outlier detection.

**When it works:**
- Sparse data (many zeros)
- Complex outlier patterns
- Missing values are meaningful

**Why this combination:**
- Constant imputation: Preserves sparsity
- MaxAbs scaling: Preserves zeros
- Isolation Forest: Finds complex outliers

**Example use case:** Text classification - sparse word counts, some spam documents as outliers.

---

#### 11. **mean_minmax_lof_svd**
```python
{'imputation': 'mean', 'scaling': 'minmax', 'encoding': 'onehot',
 'feature_selection': 'k_best', 'outlier_removal': 'lof', 'dimensionality_reduction': 'svd'}
```
**Philosophy:** Density-based outlier detection + SVD for sparse data.

**When it works:**
- Clustered data with varying densities
- Sparse matrices
- Text or count data

**Why LOF:**
- Finds outliers in variable-density clusters
- Better than distance-based methods for clustered data

**Why SVD over PCA:**
- Works better with sparse data
- Computationally efficient

**Example use case:** Document classification - clusters of topics with different sizes, need compression.

---

#### 12. **mostfreq_standard_iqr**
```python
{'imputation': 'most_frequent', 'scaling': 'standard', 'encoding': 'onehot',
 'feature_selection': 'none', 'outlier_removal': 'iqr', 'dimensionality_reduction': 'none'}
```
**Philosophy:** Categorical-friendly imputation + standard preprocessing.

**When it works:**
- Many categorical features with missing values
- Mixed data types
- Standard Gaussian-ish numerical features

**Why most_frequent:**
- Best for categorical features
- Preserves category distribution

**Example use case:** Customer database - missing product categories, some outlier purchase amounts.

---

## Recommender Systems Explained

Now that we understand the pipelines, let's understand how different recommender systems predict which pipeline to use.

### 1. **baseline** (No ML)
**Type:** Heuristic

**How it works:**
```
For each new dataset:
  Run all pipelines
  Pick the one with best accuracy
```

**Philosophy:** No prediction - just evaluate everything.

**Pros:**
- Always picks the truly best pipeline
- No risk of wrong recommendation

**Cons:**
- Extremely expensive (must run all pipelines)
- Not practical for large datasets
- Defeats the purpose of recommendation

**When to use:** Benchmark comparison only.

**Training time:** 0s (no training)
**Prediction time:** Hours (runs all pipelines)

---

### 2. **random** (Random Selection)
**Type:** Baseline

**How it works:**
```
Pick a random pipeline from the 12 options
```

**Philosophy:** Sanity check - any recommender should beat this.

**Expected accuracy:** 8.33% (1/12 chance of picking best)

**Why it exists:** Baseline for comparison. If your recommender doesn't beat random, something is wrong.

---

### 3. **average-rank** (Popularity-Based)
**Type:** Simple Heuristic

**How it works:**
```python
# Training:
for each pipeline:
    calculate average rank across all training datasets
    
# Recommendation:
recommend pipeline with best average rank
```

**Philosophy:** "Which pipeline works best on average?"

**Example:**
```
Pipeline A: Ranks [1, 3, 2, 1] across 4 datasets → Avg rank = 1.75
Pipeline B: Ranks [2, 1, 3, 4] across 4 datasets → Avg rank = 2.50
→ Recommend Pipeline A
```

**Pros:**
- Simple and interpretable
- No overfitting risk
- Fast

**Cons:**
- Ignores dataset characteristics
- Same recommendation for every dataset
- Can't learn from meta-features

**When to use:** Quick baseline when you have no meta-features.

**Example:** Pipeline "simple_preprocess" works well on 60% of datasets → recommend it by default.

---

### 4. **l1** (Distance-Based)
**Type:** Instance-Based Learning (k-NN variant)

**How it works:**
```python
# Training: Store all (meta-features, best_pipeline) pairs

# Recommendation:
1. Find K most similar training datasets (using L1 distance)
2. Look at which pipeline worked best on those similar datasets
3. Recommend the most common best pipeline
```

**Philosophy:** "If dataset A is similar to dataset B, the same pipeline should work."

**L1 Distance Formula:**
```
distance(A, B) = |A_feature1 - B_feature1| + |A_feature2 - B_feature2| + ...
```

**Example:**
```
New dataset: [n_samples=1000, n_features=50, n_classes=3]

Find 5 nearest neighbors:
  Dataset_42: [n_samples=1200, n_features=45, n_classes=3] → best: robust_preprocess
  Dataset_17: [n_samples=900,  n_features=55, n_classes=2] → best: robust_preprocess
  Dataset_89: [n_samples=1100, n_features=48, n_classes=3] → best: simple_preprocess
  Dataset_34: [n_samples=950,  n_features=52, n_classes=3] → best: robust_preprocess
  Dataset_71: [n_samples=1050, n_features=47, n_classes=4] → best: robust_preprocess

Vote: robust_preprocess wins 4-1 → Recommend it
```

**Pros:**
- No training required (lazy learning)
- Works well with enough training data
- Interpretable (can see similar datasets)

**Cons:**
- Slow prediction (must compute all distances)
- Sensitive to irrelevant features
- Needs many training examples

**When to use:** 
- Many training datasets (>100)
- Clear similarity structure
- Want explainable recommendations

---

### 5. **basic** (Average Performance)
**Type:** Simple Heuristic

**How it works:**
```python
# Training:
for each pipeline:
    calculate average accuracy across all datasets
    
# Recommendation:
recommend pipeline with highest average accuracy
```

**Philosophy:** "Which pipeline has the best average performance?"

**Difference from average-rank:**
- Uses actual accuracy scores instead of ranks
- More sensitive to magnitude of differences

**Example:**
```
Pipeline A: Accuracies [0.85, 0.70, 0.90, 0.88] → Avg = 0.8325
Pipeline B: Accuracies [0.80, 0.75, 0.85, 0.82] → Avg = 0.8050
→ Recommend Pipeline A
```

**Pros:**
- Simple
- Considers actual performance values
- Fast

**Cons:**
- Ignores dataset characteristics
- One recommendation for all datasets

---

### 6. **knn** (K-Nearest Neighbors Classifier)
**Type:** Machine Learning - Classification

**How it works:**
```python
# Training:
X = dataset meta-features (n_samples, n_classes, n_features, etc.)
y = best pipeline for each dataset

model = KNeighborsClassifier(k=5)
model.fit(X, y)

# Recommendation:
new_features = [1000, 50, 3, ...]  # new dataset characteristics
prediction = model.predict(new_features)  # predicted best pipeline
```

**Philosophy:** "Learn from similar datasets using scikit-learn's KNN."

**How it predicts:**
1. Find K=5 nearest training datasets (using Euclidean distance)
2. Look at their best pipelines
3. Vote: Pick most common pipeline

**Difference from L1 recommender:**
- Uses Euclidean distance (L2) instead of Manhattan (L1)
- More sophisticated voting mechanism
- Can use distance-weighted voting

**Pros:**
- Proven ML algorithm
- Good for non-linear relationships
- Naturally handles multi-class output (12 pipelines)

**Cons:**
- Slow for large training sets
- Sensitive to feature scaling
- Curse of dimensionality

**When to use:**
- Medium-sized training data (50-500 datasets)
- Clear neighborhood structure
- Want robust predictions

**Training complexity:** O(1) (just stores data)
**Prediction complexity:** O(n) where n = training size

---

### 7. **rf** (Random Forest Classifier)
**Type:** Machine Learning - Ensemble

**How it works:**
```python
# Training:
X = dataset meta-features
y = best pipeline for each dataset

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Recommendation:
prediction = model.predict(new_features)
probabilities = model.predict_proba(new_features)  # confidence scores
```

**Philosophy:** "Build 100 decision trees and let them vote."

**Decision Tree Example:**
```
if n_features > 100:
    if n_samples < 1000:
        recommend: dimension_reduction
    else:
        recommend: feature_selection
else:
    if has_missing_values:
        recommend: simple_preprocess
    else:
        recommend: baseline
```

**Why Random Forest:**
- Combines 100 such trees (reduces overfitting)
- Each tree trained on random subset of data & features
- Final prediction: majority vote

**Pros:**
- Handles non-linear relationships
- Robust to overfitting
- Feature importance available
- Works with imbalanced classes
- No feature scaling needed

**Cons:**
- Black box (hard to interpret)
- Can be slow with many trees
- Needs tuning (n_estimators, max_depth, etc.)

**When to use:**
- Complex relationships between meta-features and pipelines
- Want robust predictions
- Have enough training data (>50 datasets)

**Key hyperparameters:**
- `n_estimators=100`: Number of trees (more = better but slower)
- `max_depth`: Tree depth (deeper = more complex)

---

### 8. **nn** (Neural Network Classifier)
**Type:** Deep Learning

**How it works:**
```python
# Training data:
X = dataset_metafeatures  # ONLY dataset characteristics (20 features)
y = best_pipeline_id      # Which pipeline is best (0-11)

# Architecture:
Input (dataset meta-features ONLY) 
  → BatchNorm 
  → Dense(256) → ReLU → Dropout(0.3)
  → BatchNorm
  → Dense(128) → ReLU → Dropout(0.2)
  → Dense(12) → Softmax
  
# Output: Probability distribution over 12 pipelines
```

**Key Point:** The NN sees **ONLY dataset characteristics**, NOT pipeline features.
- Input: `[n_samples, n_features, n_classes, mean_attr, ...]` (20 meta-features)
- Output: Pipeline ID (0 = baseline, 1 = simple_preprocess, ..., 11 = mostfreq_standard_iqr)

**Philosophy:** "Learn which dataset characteristics indicate which pipeline works best."

**Training process:**
```python
for epoch in range(300):
    for batch in training_data:
        # Forward pass
        predictions = model(batch_features)
        loss = CrossEntropyLoss(predictions, true_labels, class_weights)
        
        # Backward pass
        gradients = compute_gradients(loss)
        update_weights(gradients, learning_rate=0.001)
        
    # Validation
    val_accuracy = evaluate(validation_data)
    if val_accuracy > best_val_accuracy:
        save_model()  # Save best model
```

**Key components:**

1. **BatchNorm:** Normalizes inputs to each layer
   - Speeds up training
   - Reduces internal covariate shift
   
2. **Dropout (0.3, 0.2):** Randomly drops 30%, 20% of neurons
   - Prevents overfitting
   - Forces network to learn robust features
   
3. **Class Weights:** Handles imbalanced pipeline frequencies
   - If pipeline A is best for 50 datasets and pipeline B for only 5
   - Give pipeline B higher weight to avoid bias

4. **Adam Optimizer:** Adaptive learning rate
   - Faster convergence than SGD
   - Less sensitive to learning rate choice

**Training vs Validation:**
```
Training (80% of data): Model learns from this
Validation (20% of data): Model is tested on this (not used for learning)

Goal: Training accuracy ≈ Validation accuracy
Problem: If training >> validation → Overfitting
Solution: More dropout, regularization, or simpler model
```

**Your NN's current performance:**
```
Training: 95.83%
Validation: 27.78%
→ SEVERE OVERFITTING
```

**Why this happens:**
- 90 training datasets / 12 pipelines = only ~7-8 examples per pipeline
- Model memorizes training data instead of learning patterns
- Neural network is too powerful for small dataset

**Solutions:**
1. ✅ Already implemented: Dropout, BatchNorm, Class weights
2. Could add: More training data, simpler architecture, stronger regularization

**Pros:**
- Can learn very complex patterns
- State-of-the-art performance on large datasets
- Flexible architecture

**Cons:**
- Needs LOTS of data (typically 1000s of examples)
- Computationally expensive
- Prone to overfitting on small data
- Hard to interpret

**When to use:**
- Large training data (>500 datasets)
- Complex non-linear relationships
- High-dimensional meta-features

**When NOT to use:**
- Small data (<100 datasets) ← Your current situation
- Need interpretability
- Limited computation

---

### 9. **regressor** (Neural Network Regressor)
**Type:** Deep Learning - Regression

**How it works:**
```python
# Training data (DIFFERENT from NN Classifier):
# Creates 12x more training examples - one for EACH (dataset, pipeline) pair

for each dataset:
    for each pipeline:
        X = [dataset_metafeatures + pipeline_features]  # 20 + 23 = 43 features
        y = actual_accuracy_of_this_pipeline_on_this_dataset  # 0-1

# Example:
Dataset A + Pipeline "baseline" → X=[1000,50,3,...,0,0,0,...], y=0.72
Dataset A + Pipeline "simple" → X=[1000,50,3,...,1,0,1,...], y=0.85
...

# Architecture:
Input (dataset features + pipeline features)  # 43 features total
  → BatchNorm
  → Dense(128) → ReLU
  → Dense(64) → ReLU
  → Dense(1) → Linear
  
# Output: Predicted accuracy (continuous value 0-1)
```

**Key Difference from NN Classifier:**
- **NN Classifier:** Input = dataset only (20 features), Output = best pipeline ID
- **NN Regressor:** Input = dataset + pipeline (43 features), Output = accuracy score

**Training data size:**
- **NN Classifier:** 90 examples (one per dataset)
- **NN Regressor:** 1080 examples (90 datasets × 12 pipelines)

**Example prediction process:**
```
New dataset meta-features: [1000 samples, 50 features, 3 classes, ...]

For each of 12 pipelines:
  Step 1: Combine dataset features + pipeline features
    baseline features: [0,0,0,0,0,0,...]  (all zeros - no preprocessing)
    X = [1000,50,3,... + 0,0,0,0,0,0,...] → Predict: 0.72
    
    simple_preprocess features: [1,1,1,0,0,0,...]  (mean, standard, onehot)
    X = [1000,50,3,... + 1,1,1,0,0,0,...] → Predict: 0.85
    
    robust_preprocess features: [0,1,1,0,1,0,...]  (median, robust, onehot, iqr)
    X = [1000,50,3,... + 0,1,1,0,1,0,...] → Predict: 0.88  ← BEST
    ...

Recommendation: robust_preprocess (highest predicted accuracy)
```

**Critical Difference from NN Classifier:**

| Aspect | NN Classifier | NN Regressor |
|--------|--------------|--------------|
| **Input** | Dataset only (20 features) | Dataset + Pipeline (43 features) |
| **Output** | Pipeline ID (0-11) | Accuracy score (0-1) |
| **Training examples** | 90 (one per dataset) | 1,080 (90 × 12) |
| **Prediction speed** | Fast (one forward pass) | Slow (12 forward passes) |
| **What it learns** | Dataset → Best pipeline | Dataset+Pipeline → Accuracy |
| **Advantage** | Faster, simpler | Sees pipeline characteristics |
| **Disadvantage** | Doesn't see pipeline features | 12x more computation |

**Pros:**
- Can estimate confidence (predicted accuracy)
- Provides ranking of all pipelines
- Can optimize for specific accuracy threshold

**Cons:**
- Must evaluate ALL 12 pipelines for each prediction (12x slower)
- Harder to train (regression is harder than classification)
- Needs even more data

**When to use:**
- Need accuracy estimates, not just best pipeline
- Want to see top-K pipelines
- Have lots of training data

---

### 10. **adaboost** (AdaBoost Regressor)
**Type:** Machine Learning - Boosting

**How it works:**
```python
# Sequential ensemble: Each model focuses on mistakes of previous models

model = AdaBoostRegressor(n_estimators=200, learning_rate=0.05)

# Training:
for i in range(200):  # 200 weak learners
    # Train model i on data, weighted by previous errors
    model_i.fit(X, y, sample_weight=error_weights)
    
    # Update weights: Give more weight to poorly predicted examples
    errors = abs(predictions - true_values)
    error_weights = update_weights(errors)

# Prediction: Weighted combination of all 200 models
```

**Philosophy:** "Focus on hard cases - learn from mistakes."

**Example:**
```
Round 1: Train simple model → Some datasets predicted poorly
Round 2: Train another model, focusing more on poorly predicted datasets
Round 3: Focus even more on the remaining hard cases
...
Round 200: Combine all models

Final prediction = weighted_sum(all_models)
```

**Like Random Forest but different:**
- Random Forest: Trees are independent, trained in parallel
- AdaBoost: Trees are sequential, each learns from previous mistakes

**Pros:**
- Often outperforms single models
- Reduces bias (learns complex patterns)
- Less prone to overfitting than single decision trees

**Cons:**
- Slower training (sequential)
- Sensitive to noisy data
- Can overfit if too many estimators

**Key hyperparameters:**
- `n_estimators=200`: Number of weak learners (increased from 50)
- `learning_rate=0.05`: How much each model contributes (lowered from 1.0)
- `loss='exponential'`: How to weight errors (changed from 'linear')

**When to use:**
- Tabular data (works well on meta-features)
- Want to improve beyond single model
- Have medium-sized training data

---

### 11. **surrogate** (Bayesian Surrogate Model)
**Type:** Bayesian Optimization

**How it works:**
```python
# Uses Random Forest to model uncertainty

model = RandomForestRegressor(n_estimators=100)

# For each pipeline:
predictions = []
for tree in model.trees:
    predictions.append(tree.predict(new_dataset))

# Estimate: mean and standard deviation
predicted_accuracy = mean(predictions)
uncertainty = std(predictions)

# Recommendation: Balance exploitation vs exploration
score = predicted_accuracy + beta * uncertainty
```

**Philosophy:** "Recommend pipelines with high predicted performance OR high uncertainty."

**Why uncertainty matters:**
```
Pipeline A: Predicted=0.85, Uncertainty=0.02 (confident)
Pipeline B: Predicted=0.83, Uncertainty=0.10 (uncertain)

→ Maybe try B? It might actually be 0.93 (0.83 + 0.10)
→ Exploration: Try uncertain options that might be great
→ Exploitation: Stick with known good options
```

**Bayesian approach:**
- Maintains belief distribution over performance
- Updates beliefs as more data arrives
- Quantifies uncertainty in predictions

**Pros:**
- Provides confidence intervals
- Good for active learning
- Naturally handles exploration/exploitation trade-off

**Cons:**
- More complex than standard ML
- Slower inference
- Needs careful tuning of beta parameter

**When to use:**
- Interactive scenario (can get feedback)
- Want to quantify uncertainty
- Sequential decision making

---

### 12. **autogluon** (AutoML)
**Type:** Automated Machine Learning

**How it works:**
```python
# AutoGluon automatically tries many models and ensembles them

predictor = TabularPredictor(label='best_pipeline')





# Then: Creates weighted ensemble of best models
```

**Philosophy:** "Why choose one model when you can try them all?"

**Training process:**
```
1. Tries 10-15 different model types
2. Hyperparameter tuning for each
3. Creates multi-layer ensembles
4. Returns best combination

Total models trained: ~50-100
```

**Pros:**
- State-of-the-art performance (wins Kaggle competitions)
- No model selection needed
- Automatic feature engineering
- Ensemble learning

**Cons:**
- VERY slow training (tries many models)
- High memory usage
- Black box (hard to understand)
- Overkill for small datasets

**When to use:**
- Need best possible performance
- Have time for training (can be hours)
- Have enough data (>100 datasets)

**When NOT to use:**
- Need fast training
- Limited resources
- Want interpretability

---

### 13. **hybrid** (Hybrid Meta-Learning)
**Type:** Ensemble of Multiple Approaches

**How it works:**
```python
# Combines multiple recommender strategies

# 1. Similarity-based (like L1)
similar_datasets = find_similar_datasets(new_dataset, k=5)
similarity_vote = most_common_pipeline(similar_datasets)

# 2. Performance-based (like basic)
avg_performance_vote = best_average_pipeline()

# 3. Meta-learning (Random Forest on meta-features)
meta_learner = RandomForestClassifier()
ml_vote = meta_learner.predict(meta_features)

# Final recommendation: Weighted combination
final_score = w1 * similarity_vote + w2 * avg_performance_vote + w3 * ml_vote
```

**Philosophy:** "Different strategies work for different datasets - combine them."

**Components:**

1. **K-NN similarity:** Good for datasets similar to training data
2. **Average performance:** Good when no clear similar datasets
3. **Meta-learner:** Good for complex patterns

**Influence Weighting (Optional):**
```python
# Give more weight to "influential" training datasets

influence_score = calculate_influence(dataset)
# High influence = dataset teaches us a lot
# Low influence = dataset is redundant/noisy

weighted_vote = sum(influence_score[i] * vote[i] for i in datasets)
```

**Influence methods:**
- `performance_variance`: Datasets where pipelines differ a lot
- `data_diversity`: Unique, representative datasets
- `discriminative_power`: Clear winner among pipelines

**Pros:**
- Robust (if one strategy fails, others compensate)
- Adaptable (uses best strategy per dataset)
- Can leverage influence weighting

**Cons:**
- Complex to tune (many hyperparameters)
- Slower than single strategy
- Harder to debug

**When to use:**
- Diverse test datasets
- Want robust predictions
- Have computational resources

---

### 14. **pmm** (Pairwise Metric Matching)
**Type:** Siamese Neural Network

**How it works:**
```python
# Siamese Network: Two identical networks that learn similarity

# Training: Learn which datasets are similar
for dataset_a, dataset_b in pairs:
    embedding_a = network(metafeatures_a)
    embedding_b = network(metafeatures_b)
    
    distance = euclidean_distance(embedding_a, embedding_b)
    
    # If similar (same best pipeline): minimize distance
    # If dissimilar (different best pipelines): maximize distance
    if same_best_pipeline(a, b):
        loss = distance  # Pull together
    else:
        loss = max(0, margin - distance)  # Push apart

# Recommendation:
embedding_new = network(new_dataset)
most_similar_training = find_closest(embedding_new, training_embeddings)
recommend = best_pipeline(most_similar_training)
```

**Philosophy:** "Learn a metric space where similar datasets are close together."

**Example:**
```
Original space: Datasets described by 20 meta-features
Learned space: Datasets mapped to 32-dimensional embeddings

In learned space:
- Datasets needing "robust_preprocess" cluster together
- Datasets needing "simple_preprocess" cluster elsewhere
- Distance in embedding space = similarity in pipeline needs
```

**Contrastive Learning:**
```python
Positive pair: (Dataset A, Dataset B) where both work best with same pipeline
→ Pull embeddings closer

Negative pair: (Dataset A, Dataset C) where they need different pipelines
→ Push embeddings apart by margin
```

**Influence Weighting:**
```python
# Sample more pairs from influential datasets
influential_datasets = calculate_influence(datasets)

# Training pairs weighted by influence
prob_sample(dataset_i) ∝ influence_score[i]
```

**Pros:**
- Learns task-specific similarity (not just Euclidean)
- Can capture complex relationships
- Naturally extends to new datasets

**Cons:**
- Needs many training pairs (generates 10,000 pairs)
- Slow training
- Sensitive to pair selection

**When to use:**
- Large training data
- Complex similarity relationships
- Standard distance metrics don't work well

---

### 15. **balancedpmm** (Balanced PMM)
**Type:** Improved Pairwise Metric Matching

**How it differs from PMM:**
```python
# PMM problem: Imbalanced pair generation
# - Common pipelines get many pairs
# - Rare pipelines get few pairs
# - Network biased toward common pipelines

# BalancedPMM solution: Balance pair generation
for each_pipeline:
    generate_equal_number_of_pairs()

# Result: Each pipeline gets equal representation
```

**Why this matters:**
```
Example dataset distribution:
- Pipeline A: best for 50 datasets
- Pipeline B: best for 5 datasets

PMM: Generates ~2500 pairs with A, ~25 pairs with B
→ Network learns A well, ignores B

BalancedPMM: Generates 500 pairs with A, 500 pairs with B
→ Network learns both equally well
```

**Pros:**
- Better for imbalanced datasets
- More robust to rare pipelines
- Fairer evaluation

**Cons:**
- Oversamples rare pipelines (may overfit to them)
- More complex pair generation
- Slower training

**When to use:**
- Imbalanced pipeline distribution
- Care about rare pipelines
- Want unbiased recommendations

---

### 16. **paper_pmm** (Paper-Style PMM)
**Type:** Advanced Siamese Network

**How it differs:**
```python
# Learns embeddings for BOTH datasets AND pipelines

# Network:
dataset_embedding = dataset_encoder(metafeatures)  # 64-dim
pipeline_embedding = pipeline_encoder(pipeline_features)  # 64-dim

# Similarity:
similarity = cosine_similarity(dataset_embedding, pipeline_embedding)

# Training: Triplet loss
anchor = dataset_embedding
positive = embedding(best_pipeline for this dataset)
negative = embedding(random other pipeline)

loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)
```

**Philosophy:** "Learn a joint space where datasets and pipelines can be compared."

**Advantage:**
```
Traditional: dataset → similar datasets → their best pipelines
Paper-style: dataset → directly compare with all pipelines in learned space

More direct optimization for the end task
```

**Pipeline features:**
```python
# One-hot encoding of pipeline components
pipeline_features = [
    has_mean_imputation,
    has_median_imputation,
    ...,
    has_pca,
    has_standard_scaling,
    ...
]  # 23 binary features
```

**Pros:**
- More principled approach
- Direct dataset-pipeline matching
- Can generalize to new pipelines

**Cons:**
- Even more complex
- Needs careful feature engineering for pipelines
- Requires lots of training data

**When to use:**
- Research setting
- Want state-of-the-art
- Have abundant resources

---

## When to Use Each Recommender

### Quick Decision Tree

```
START
│
├─ Need fast baseline? → **random** or **average-rank**
│
├─ Small data (<50 datasets)?
│  ├─ Want interpretable → **l1** or **basic**
│  └─ Want best performance → **knn** or **rf**
│
├─ Medium data (50-200 datasets)?
│  ├─ Need fast training → **rf** or **knn**
│  ├─ Want uncertainty → **surrogate**
│  └─ Want best performance → **hybrid** or **autogluon**
│
└─ Large data (>200 datasets)?
   ├─ Need interpretability → **rf** (with feature importance)
   ├─ Have GPU → **nn** or **pmm**
   ├─ Want state-of-the-art → **autogluon** or **hybrid**
   └─ Research setting → **paper_pmm** or **balancedpmm**
```

### By Use Case

**Production System (need fast, reliable recommendations):**
1. **rf** - Good balance of speed and accuracy
2. **hybrid** - Robust to different dataset types
3. **knn** - Fast, interpretable

**Research (need best possible performance):**
1. **autogluon** - State-of-the-art ensemble
2. **paper_pmm** - Advanced deep learning
3. **hybrid** - Multiple strategies

**Limited Resources (small training data):**
1. **l1** - No training needed
2. **average-rank** - Simple baseline
3. **basic** - Fast and simple

**Need Explainability:**
1. **l1** - Can show similar datasets
2. **rf** - Feature importance available
3. **knn** - Can trace to nearest neighbors

---

## Performance Metrics Explained

### Understanding the Metrics

#### 1. **Training Success (✅/❌)**
Did the recommender train without errors?

#### 2. **Training Time**
How long to train the model (in seconds).
- **Baseline:** 0s (no training)
- **Simple:** <1s (l1, basic, average-rank)
- **ML Models:** 1-10s (knn, rf, nn)
- **Heavy:** 10-60s (autogluon, pmm)

#### 3. **Successful Recs / Total Tests**
How many test datasets got a valid recommendation?
- Should be 100% for most recommenders
- Lower might indicate errors

#### 4. **Accuracy (Top-1)**
Percentage of times the recommended pipeline was THE BEST.
```python
accuracy = (times_recommended_best_pipeline / total_datasets) * 100
```

**Interpretation:**
- 100%: Perfect (unrealistic)
- 50-70%: Very good
- 30-50%: Decent
- 8.33%: Random guessing (1/12 pipelines)

#### 5. **Average Degradation**
How much worse is the recommended pipeline compared to the best?
```python
degradation = ((best_score - recommended_score) / best_score) * 100
```

**Example:**
```
Best pipeline: 0.90 accuracy
Recommended: 0.81 accuracy
Degradation = (0.90 - 0.81) / 0.90 = 10%
```

**Interpretation:**
- 0%: Perfect (recommended IS the best)
- <5%: Excellent (very close to best)
- 5-15%: Good (acceptable loss)
- >20%: Poor (significant degradation)

**Why it matters more than accuracy:**
- Rank 2 pipeline might be only 1% worse than rank 1
- Still counts as "wrong" for accuracy
- But practically, 1% difference is negligible

#### 6. **Average Rank**
Average position of recommended pipeline.
```python
rank = 1 (best), 2 (second best), ..., 12 (worst)
```

**Interpretation:**
- 1.0: Perfect
- 1-2: Excellent
- 2-3: Very good
- 3-5: Acceptable
- >6: Poor

#### 7. **Better/Equal/Worse than Baseline**
How often does the recommender beat the simple baseline pipeline?

```
Better: Recommended pipeline > Baseline pipeline
Equal: Recommended pipeline = Baseline pipeline  
Worse: Recommended pipeline < Baseline pipeline
```

**Goal:** >60% better than baseline

---

## Conclusion

### Key Takeaways

1. **Pipelines:** 12 different preprocessing strategies, each suited for different data characteristics

2. **Simple Recommenders:** Fast but ignore dataset characteristics
   - random, average-rank, basic

3. **Distance-Based:** Use similarity to past datasets
   - l1, knn

4. **ML Classifiers:** Learn patterns from meta-features
   - rf, nn, autogluon

5. **ML Regressors:** Predict actual performance
   - regressor, adaboost, surrogate

6. **Advanced:** Deep learning and ensembles
   - pmm, balancedpmm, paper_pmm, hybrid

### For Your Current Situation

**Problem:** NN has 95% training but 28% validation accuracy = **SEVERE OVERFITTING**

**Why:** 90 datasets ÷ 12 pipelines = only 7-8 examples per class
- Not enough data for neural network
- NN memorizes instead of learns

**Better choices for small data:**
1. **rf** - Handles small data well
2. **knn** - No training, uses all data
3. **l1** - Simple, interpretable
4. **hybrid** - Combines multiple strategies

**If you want to use NN:**
- Need >500 training datasets
- Or use simpler architecture
- Or stronger regularization

### Next Steps

1. **Run quick evaluation:**
   ```bash
   python recommender_trainer.py --recommender rf --evaluate --quick
   ```

2. **Compare multiple recommenders:**
   ```bash
   python evaluate_all_recommenders.py
   ```

3. **Analyze results:**
   - Which has best accuracy?
   - Which has lowest degradation?
   - Which is fastest?

4. **Choose based on your needs:**
   - Speed → rf, knn
   - Accuracy → hybrid, autogluon
   - Interpretability → l1, rf

---

**Created:** October 12, 2025  
**For:** SoluRec Recommender System  
**Author:** Analysis of recommender_trainer.py and evaluation_utils.py
