# Sample-Level Influence Weighting for Pipeline Recommendation

## Overview

This document explains the **Sample-Level Influence Weighting** mechanism implemented in the PMM (Pipeline Meta-Model) Recommender to improve recommendation quality by prioritizing more informative training datasets.

## Motivation

Similar to how a movie recommender should prioritize popular and highly-rated movies when making suggestions, our pipeline recommender should give more weight to training datasets that are:
1. **More informative** - Can better distinguish between good and bad pipelines
2. **More discriminative** - Show clear performance differences between pipelines
3. **More diverse** - Represent unique metafeature profiles

## Implementation

### Three Influence Scoring Methods

#### 1. Performance Variance (`performance_variance`)
**Concept**: Datasets with higher variance in pipeline performance are more informative.

**Rationale**: If all pipelines perform similarly on a dataset (low variance), that dataset doesn't help us distinguish between good and bad pipelines. Datasets with high variance provide more signal about which pipelines work best.

**Formula**:
```python
influence_score = Var(pipeline_performances)
```

**Example**:
- Dataset A: All pipelines get 0.80-0.82 accuracy → Low influence (variance ≈ 0.0001)
- Dataset B: Pipelines range from 0.60-0.95 accuracy → High influence (variance ≈ 0.01)

#### 2. Discriminative Power (`discriminative_power`)
**Concept**: Datasets where the best pipeline significantly outperforms others are more valuable.

**Rationale**: Clear winners help us make confident recommendations. If the top-3 pipelines are very close but far from bottom-3, the dataset provides clear guidance.

**Formula**:
```python
gap = max(performances) - min(performances)
clarity = gap / (std(top_3) + std(bottom_3) + ε)
influence_score = clarity
```

**Example**:
- Dataset A: Best=0.90, Worst=0.88, Top-3 similar → Low influence
- Dataset B: Best=0.95, Worst=0.65, Clear separation → High influence

#### 3. Data Diversity (`data_diversity`)
**Concept**: Datasets with unique metafeature profiles contribute more novel information.

**Rationale**: If many training datasets are similar to each other, they provide redundant information. Unique datasets expand our knowledge coverage.

**Formula**:
```python
for each dataset:
    distance_to_nearest_neighbor = min(euclidean_distances(dataset, all_other_datasets))
    influence_score = distance_to_nearest_neighbor
```

**Example**:
- Dataset A: Very similar to 5 other datasets → Low influence
- Dataset B: Unlike any other dataset → High influence

### Score Normalization

After calculating raw influence scores, they are normalized to [0.5, 1.5] range:

```python
# Z-score normalization
scores = (scores - mean(scores)) / std(scores)

# Exponential scaling to emphasize differences
scores = exp(scores)

# Normalize to [0, 1]
scores = (scores - min(scores)) / (max(scores) - min(scores))

# Scale to [0.5, 1.5] 
# This ensures all samples contribute, but influential ones get more weight
scores = 0.5 + scores
```

This ensures:
- All training datasets contribute (minimum weight = 0.5)
- Influential datasets get up to 3x more weight (maximum weight = 1.5)
- The weighting is smooth and continuous

### Integration with Similarity-Based Recommendation

The final weight for each dataset combines **similarity** and **influence**:

```python
final_weight = similarity_score × influence_score
```

**Effect**:
- A dataset that is both **similar** to the target AND **influential** gets maximum weight
- A dataset that is similar but not influential gets moderate weight
- A dataset that is influential but not similar gets low weight
- This creates a balanced approach that considers both relevance and informativeness

## Usage

### Enabling Influence Weighting

```python
# Create PMM recommender with influence weighting
recommender = PmmRecommender(
    use_influence_weighting=True,
    influence_method='performance_variance'  # or 'discriminative_power' or 'data_diversity'
)

# Train the recommender
recommender.fit(performance_matrix, metafeatures_df)

# Get recommendation (influence scores are automatically applied)
result = recommender.recommend(new_dataset_id, performance_matrix)

# Result includes influence scores
print(result['influence_scores'])  # {dataset_id: influence_score, ...}
print(result['influence_weighted'])  # True/False
```

### Choosing the Right Method

| Method | Best For | Computational Cost |
|--------|----------|-------------------|
| `performance_variance` | General use, fast | Low |
| `discriminative_power` | When you need clear winners | Low |
| `data_diversity` | When training data might be redundant | Medium (requires distance calculations) |

**Recommendation**: Start with `performance_variance` as it's fast and effective for most cases.

## Benefits

### 1. Improved Recommendation Quality
By prioritizing informative datasets, the recommender learns from the most valuable examples, leading to better pipeline suggestions.

### 2. Robustness to Noisy Data
Datasets with inconsistent or unreliable performance (low discriminative power) get lower weights, reducing their impact on recommendations.

### 3. Better Use of Training Data
Identifies which datasets in your training set are most valuable, helping you understand your data better.

### 4. Adaptive Learning
The influence scores adapt to your specific performance matrix, automatically identifying which datasets are most informative for your use case.

## Example Output

```
Calculating influence scores using method: performance_variance
✅ Calculated influence scores for 90 datasets
   Mean: 1.000, Std: 0.289, Min: 0.500, Max: 1.500
   Top 5 most influential datasets:
     Dataset 469: 1.487
     Dataset 338: 1.394
     Dataset 277: 1.356
     Dataset 453: 1.298
     Dataset 342: 1.267

PMM recommender suggests: constant_maxabs_iforest
🎯 Using influence weighting (method: performance_variance)
Based on similar datasets: [469, 338, 277]
  - Dataset 469 (similarity: 0.9006, influence: 1.487)
  - Dataset 338 (similarity: 0.8923, influence: 1.394)
  - Dataset 277 (similarity: 0.8845, influence: 1.356)
```

## Comparison with DPO

This approach is inspired by **Direct Preference Optimization (DPO)** from LLM fine-tuning, but adapted for our use case:

| Aspect | DPO (LLMs) | Our Approach |
|--------|------------|--------------|
| **Goal** | Prefer outputs aligned with human preferences | Prefer datasets that provide more information |
| **Weighting** | Preferred vs. rejected examples | Influential vs. less influential datasets |
| **Signal** | Human feedback | Performance variance/discrimination |
| **Application** | Training loss weighting | Recommendation weighting |

## Future Enhancements

Potential improvements:
1. **Dynamic Weighting**: Update influence scores as more data becomes available
2. **Multi-Method Ensemble**: Combine multiple influence scoring methods
3. **Dataset Clustering**: Group similar datasets and weight clusters
4. **Active Learning**: Use influence scores to select which datasets to evaluate next

## References

- Direct Preference Optimization (DPO): Rafailov et al., 2023
- Data Valuation: Ghorbani & Zou, 2019
- Influence Functions: Koh & Liang, 2017
