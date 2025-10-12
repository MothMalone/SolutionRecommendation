# DPO-Style Influence Weighting for Pipeline Recommenders

## Overview

We've implemented **DPO-style (Direct Preference Optimization) influence weighting** for the PMM and BalancedPMM recommenders. This feature helps identify and prioritize more influential training datasets when making recommendations.

## Concept

Similar to how DPO prioritizes more informative samples in preference learning, our influence weighting system:

1. **Identifies influential datasets**: Some datasets are more "informative" than others for making recommendations
2. **Weights similarities**: Instead of treating all similar datasets equally, we give more weight to influential ones
3. **Improves recommendations**: More influential datasets have stronger impact on final pipeline selection

## Influence Methods

### 1. Performance Variance (`performance_variance`)
- **Idea**: Datasets where pipelines show more variation are more informative
- **Why**: If all pipelines perform similarly, the dataset doesn't help distinguish between them
- **Use when**: You want to focus on datasets that reveal clear pipeline preferences

### 2. Dataset Diversity (`dataset_diversity`)
- **Idea**: Datasets that are more unique/different from others are more valuable
- **Why**: Diverse datasets provide unique information not captured by similar datasets
- **Use when**: You want to avoid over-relying on clusters of similar datasets

### 3. Prediction Confidence (`prediction_confidence`)
- **Idea**: Datasets where the model is more confident are more reliable
- **Why**: High-confidence predictions are more trustworthy
- **Use when**: You want to prioritize datasets the model understands well

### 4. Combined (`combined`)
- **Idea**: Combines all three methods with equal weight
- **Why**: Balances multiple aspects of influence
- **Use when**: You want a comprehensive influence measure

## Usage

### Command Line

**Without influence weighting (default):**
```bash
python3 recommender_trainer.py --evaluate --recommender pmm
```

**With influence weighting:**
```bash
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence
```

**With specific influence method:**
```bash
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence --influence-method dataset_diversity
```

**All influence methods:**
```bash
# Performance variance (default)
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence --influence-method performance_variance

# Dataset diversity
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence --influence-method dataset_diversity

# Prediction confidence
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence --influence-method prediction_confidence

# Combined approach
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence --influence-method combined
```

### With Hybrid Recommender

The influence weighting also works with the Hybrid recommender:

```bash
# Without influence weighting
python3 recommender_trainer.py --evaluate --recommender hybrid

# With influence weighting (default method)
python3 recommender_trainer.py --evaluate --recommender hybrid --use-influence

# With specific influence method
python3 recommender_trainer.py --evaluate --recommender hybrid --use-influence --influence-method dataset_diversity
```

### In Code

```python
# Create PMM recommender with influence weighting
recommender = PmmRecommender(
    num_epochs=20,
    batch_size=64,
    use_influence_weighting=True,
    influence_method='performance_variance'  # or 'dataset_diversity', 'prediction_confidence', 'combined'
)

# Train the recommender
recommender.fit(performance_matrix, metafeatures_df)

# Get recommendations (influence scores are automatically used)
result = recommender.recommend(test_dataset_id, performance_matrix)

# Access influence information
if result:
    print(f"Recommended pipeline: {result['pipeline']}")
    print(f"Influence weighted: {result.get('influence_weighted', False)}")
    
    # See influence scores for similar datasets
    influence_scores = result.get('influence_scores', {})
    for dataset_id, score in influence_scores.items():
        print(f"  Dataset {dataset_id}: influence = {score:.3f}")
```

## How It Works

### Training Phase
1. Calculate influence scores for all training datasets
2. Store scores in `recommender.dataset_influence_scores`
3. Influence scores are normalized to [0, 1] range

### Recommendation Phase
1. Find k most similar datasets using embeddings
2. For each similar dataset:
   - Get cosine similarity: `s`
   - Get influence score: `w`
   - Compute weighted similarity: `s * w`
3. Use weighted similarities to aggregate pipeline performances
4. Recommend pipeline with highest weighted performance

### Mathematical Formula

**Without influence weighting:**
```
recommended_score = Σ(similarity_i × performance_i) / Σ(similarity_i)
```

**With influence weighting:**
```
recommended_score = Σ(similarity_i × influence_i × performance_i) / Σ(similarity_i × influence_i)
```

## Expected Impact

### When Influence Weighting Helps
- **Noisy training data**: Filters out unreliable datasets
- **Diverse dataset collection**: Prioritizes most informative examples
- **Imbalanced similarity**: Prevents over-reliance on clusters

### When It May Not Help
- **Small training set**: All datasets may be equally important
- **Uniform dataset quality**: No clear distinction in influence
- **Already good baseline**: Marginal improvements only

## Debugging Influence Scores

To see influence scores during recommendation:

```python
result = recommender.recommend(dataset_id, performance_matrix)

print(f"Influence method: {recommender.influence_method}")
print(f"Influence weighted: {result.get('influence_weighted', False)}")

# Compare standard vs. influence-weighted similarities
similarity_scores = result.get('similarity_scores', {})
influence_scores = result.get('influence_scores', {})

for ds_id in similarity_scores:
    sim = similarity_scores[ds_id]
    inf = influence_scores.get(ds_id, 1.0)
    weighted = sim * inf
    print(f"Dataset {ds_id}:")
    print(f"  Similarity: {sim:.4f}")
    print(f"  Influence:  {inf:.4f}")
    print(f"  Weighted:   {weighted:.4f}")
```

## Experiments

To compare performance with and without influence weighting:

```bash
# Baseline (no influence)
python3 recommender_trainer.py --evaluate --recommender pmm > results_no_influence.txt

# With influence weighting
python3 recommender_trainer.py --evaluate --recommender pmm --use-influence > results_with_influence.txt

# Compare different methods
for method in performance_variance dataset_diversity prediction_confidence combined; do
    python3 recommender_trainer.py --evaluate --recommender pmm --use-influence --influence-method $method > results_$method.txt
done
```

## Future Enhancements

Potential improvements to the influence weighting system:

1. **Learned influence weights**: Train a meta-model to predict influence
2. **Task-specific influence**: Different influence for different types of recommendations
3. **Temporal influence**: Weight recent datasets more heavily
4. **Ensemble influence**: Combine multiple influence signals
5. **Gradient-based influence**: Use influence functions from the model's gradients

## References

- DPO (Direct Preference Optimization): Prioritizing informative samples in preference learning
- Influence Functions: Understanding model predictions through training data influence
- Meta-learning: Learning to weight training examples for better generalization
