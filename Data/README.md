# SoluRec/Data - Core Scripts Guide

## 📋 Quick Reference

| Script | Purpose | Run with |
|--------|---------|----------|
| `recommender_trainer.py` | Train/evaluate all recommenders | `python recommender_trainer.py --recommender_type all` |
| `evaluation_utils.py` | Dataset loading, AutoGluon eval | Imported by other scripts |
| `recommenders.py` | Recommender implementations | Imported by trainer |
| `pmm_paper_style.py` | Siamese network PMM | Imported by trainer |
| `data.py` | Interactive dataset dashboard | `streamlit run data.py` |
| `test.py` | Quick validation | `python test.py` |

---

## 📂 Data Files

| File | Purpose | Format |
|------|---------|--------|
| `dataset_feats.csv` | Meta-features for 109 datasets | rows=datasets, cols=features |
| `preprocessed_performance.csv` | Performance of 9 pipelines on 90 training datasets | rows=pipelines, cols=D_ID |
| `test_ground_truth_performance.csv` | Baseline performance on 19 test datasets | rows=pipelines, cols=dataset_names |

---

## 🎯 Main Workflow

### 1️⃣ Train Recommenders
```bash
python recommender_trainer.py --recommender_type all
```

**Outputs**:
- Console: Training progress
- `recommender_evaluation_results.csv`: Per-dataset results
- `test_evaluation_summary.csv`: Summary metrics

### 2️⃣ View Results
```bash
# View summary
head -20 test_evaluation_summary.csv

# Or use Python
python -c "import pandas as pd; print(pd.read_csv('test_evaluation_summary.csv'))"
```

### 3️⃣ Inspect Data
```bash
streamlit run data.py
```

Open browser to `localhost:8501` for interactive exploration.

---

## 🔑 Key Concepts

### Training vs Test Splits
- **Training (90 datasets)**: Build recommender model
- **Test (19 datasets)**: Evaluate recommender accuracy
- Each dataset split 80/20 for pipeline evaluation

### Recommender Types
- `baseline`: Best local pipeline (no ML)
- `knn`: Meta-feature similarity
- `rf`: Random Forest regression
- `nn`: Neural Network (3 layers)
- `adaboost`: Boosting ensemble
- `pmm`: Embedding-based PMM
- `paper_pmm`: Siamese network (slow, experimental)

### Metrics
- **Accuracy**: % of correct top-pipeline predictions
- **Avg Rank**: Average ranking of recommended pipeline (1=best)
- **Better than Baseline**: % improvement over simple baseline

---

## ⚡ Common Commands

```bash
# Train only Neural Network
python recommender_trainer.py --recommender_type nn

# Train with influence weighting
python recommender_trainer.py --recommender_type all --use_influence

# Use specific influence method
python recommender_trainer.py --recommender_type pmm --influence_method discriminative_power

# Quick test on 3 datasets
python test.py
```

---

## 🧠 Algorithm Overview

### AutoGluon Evaluation (evaluation_utils.py)
```
Dataset → Split 80/20 → Apply Pipeline → 
Train AutoGluon (10min, medium_quality) → 
Predict & Evaluate → Score
```

### Recommender Training (recommender_trainer.py)
```
Meta-features + Performance Matrix → 
Train Recommender Model → 
Save Model
```

### Recommender Evaluation
```
Test Dataset → Extract Meta-features →
Predict Pipeline → Evaluate → 
Compare with Ground Truth
```

---

## 🔧 Configuration Locations

| Config | File | Lines |
|--------|------|-------|
| AutoGluon settings | `evaluation_utils.py` | 24-35 |
| Pipeline definitions | `recommender_trainer.py` | ~80-120 |
| Train/test datasets | `recommender_trainer.py` | ~200-250 |
| Neural net architecture | `recommenders.py` | ~2480-2540 |

---

## 📊 Understanding Output Files

### `test_evaluation_summary.csv`
```
recommender_type, accuracy, avg_rank, better_than_baseline, ...
baseline, 0.35, 4.5, 0.0, ...
nn, 0.55, 3.2, 0.45, ...
```

### `recommender_evaluation_results.csv`
```
dataset, recommender_type, recommendation, rank, better_than_baseline, ...
D_3, baseline, baseline, 1, yes, ...
D_3, nn, simple_preprocess, 2, yes, ...
```

---

## 🧪 Testing Pipeline

1. **Quick test** (1 min):
   ```bash
   python test.py
   ```

2. **Full evaluation** (30+ min):
   ```bash
   python recommender_trainer.py --recommender_type all
   ```

3. **Specific recommender** (varies):
   ```bash
   python recommender_trainer.py --recommender_type nn  # ~5 min
   ```

---

## 💡 Tips & Tricks

### Monitor Progress
Watch files in real-time:
```bash
watch -n 5 'ls -lh *.csv'
```

### Analyze Performance
```python
import pandas as pd
df = pd.read_csv('test_evaluation_summary.csv')

# Best recommender
best = df.loc[df['accuracy'].idxmax()]
print(f"Best: {best['recommender_type']} ({best['accuracy']:.1%})")

# Group by recommender
by_type = df.groupby('recommender_type').agg({
    'accuracy': 'mean',
    'avg_rank': 'mean'
}).round(3)
print(by_type)
```

### Debug Failed Datasets
```python
import pandas as pd
results = pd.read_csv('recommender_evaluation_results.csv')
failed = results[results['better_than_baseline'] == 'no']
print(f"Failed on {len(failed)} predictions")
print(failed[['dataset', 'recommender_type', 'recommendation']])
```

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| "AutoGluon failed" | Check internet (OpenML), verify disk space (5+ GB) |
| "Not enough training pairs" | Training data too sparse - check `preprocessed_performance.csv` |
| "CUDA out of memory" | Set `CUDA_VISIBLE_DEVICES=""` for CPU mode |
| "Module not found" | Run `pip install -r requirements.txt` |
| "Slow performance" | `paper_pmm` is experimental/slow; use `pmm` instead |

---

## 🚀 Next Steps

1. Run training: `python recommender_trainer.py --recommender_type all`
2. Check results: `head -20 test_evaluation_summary.csv`
3. Explore data: `streamlit run data.py`
4. Analyze output: See tips above

---

**Last Updated**: October 18, 2025
