# SoluRec - Solution Recommender System

A machine learning system that recommends optimal data preprocessing pipelines for classification datasets using meta-learning and neural network-based recommenders.

## 🎯 Project Overview

**Goal**: Automatically recommend the best preprocessing pipeline for a given dataset to maximize classification performance.

**Approach**: 
- Train on 90 datasets with known pipeline performances
- Use meta-features to build recommender models
- Evaluate on 19 test datasets to validate recommendations

**Key Innovation**: Neural network + ensemble recommenders that learn pipeline performance patterns from dataset meta-features.

---

## 📁 Repository Structure

```
SoluRec/
├── Data/
│   ├── evaluation_utils.py          # AutoGluon evaluation, preprocessing, train-test logic
│   ├── recommender_trainer.py       # Main training script for all recommender types
│   ├── recommenders.py              # Recommender class definitions
│   ├── pmm_paper_style.py          # Paper-style PMM with Siamese networks
│   ├── data.py                      # Streamlit app for dataset inspection
│   ├── test.py                      # Quick test script
│   ├── dataset_feats.csv            # Meta-features for all datasets
│   ├── preprocessed_performance.csv # Training dataset performances
│   └── test_ground_truth_performance.csv # Test set baseline scores
├── settings/                         # Configuration templates
├── code/                            # Utility functions
└── README.md                        # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# 1. Navigate to SoluRec/Data directory
cd SoluRec/Data

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python test.py
```

### Key Dependencies
- `autogluon>=1.0` - AutoML framework
- `torch` - Deep learning
- `pandas`, `numpy` - Data processing
- `scikit-learn` - ML algorithms

---

## 🚀 Quick Start

### 1. Train Recommenders

Train all recommender types on training data:

```bash
python recommender_trainer.py --recommender_type all
```

**Available recommender types**:
- `baseline` - Best local pipeline (no learning)
- `knn` - KNN using meta-feature similarity
- `rf` - Random Forest regressor
- `nn` - 3-layer Neural Network
- `adaboost` - AdaBoost regressor
- `pmm` - Performance Matrix Mapping (embedding-based)
- `paper_pmm` - Siamese network PMM from research paper
- `all` - Train all models

### 2. Evaluate on Test Datasets

Automatically included when running training. Results saved to:
- `recommender_evaluation_results.csv` - Performance on each test dataset
- `test_evaluation_summary.csv` - Aggregated metrics

### 3. Inspect Results

```bash
# View test results summary
head -20 test_evaluation_summary.csv

# Run Streamlit dashboard
streamlit run data.py
```

---

## 📊 Key Scripts

### `recommender_trainer.py`
**Purpose**: Main orchestration script
- Loads training/test data
- Trains recommenders
- Evaluates on test datasets
- Generates performance reports

**Usage**:
```bash
python recommender_trainer.py --recommender_type nn --use_influence
```

**Options**:
- `--recommender_type`: Which recommender(s) to train
- `--use_influence`: Enable influence weighting for datasets
- `--influence_method`: `performance_variance`, `discriminative_power`, `data_diversity`, `combined`

### `evaluation_utils.py`
**Purpose**: Core evaluation pipeline
- Loads datasets from OpenML
- Applies preprocessing pipelines
- Runs AutoGluon for evaluation
- Splits train/test sets appropriately

**Key Functions**:
- `run_autogluon_evaluation()` - Evaluate single pipeline with AutoGluon
- `run_experiment_for_dataset()` - Full evaluation with recommender
- `safe_train_test_split()` - Smart train-test splitting

### `recommenders.py`
**Purpose**: All recommender implementations
- Base recommender classes
- KNN, Random Forest, Neural Network, etc.
- PMM algorithms
- Ensemble methods

### `pmm_paper_style.py`
**Purpose**: Research paper implementation
- Siamese network architecture
- Contrastive loss learning
- Direct performance prediction
- Influence weighting support

### `data.py`
**Purpose**: Interactive exploration
- Streamlit dashboard
- Dataset overview tabs
- Missing value analysis
- Feature statistics and correlations

---

## 🔄 Workflow

### Training Phase
```
1. Load 90 training datasets with known performances
2. Extract meta-features from each dataset
3. Train recommender on (meta-features → best-pipeline)
4. Save trained model
```

### Evaluation Phase
```
1. Load 19 test datasets
2. Extract meta-features
3. Use trained recommender to predict best pipeline
4. Evaluate predicted pipeline on test set
5. Compare vs. ground truth performance
6. Report metrics:
   - Accuracy: % predictions matching ground truth
   - Avg Rank: Average rank of recommended pipeline
   - Better than Baseline: % improvement over baseline
```

---

## 📈 Performance Metrics

Evaluated on 19 test datasets (excluding those with >85% baseline performance):

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | # correct predictions / total | How often recommendation is best |
| **Avg Rank** | Average ranking of recommended pipeline | Lower is better (1 = best) |
| **Better than Baseline** | % where recommendation > baseline | Improvement potential |
| **Performance Gap** | Best - Recommended score | 0 = perfect prediction |

---

## 🧠 Recommender Comparison

| Recommender | Type | Complexity | Speed | Notes |
|-------------|------|-----------|-------|-------|
| Baseline | Rule-based | Low | Very Fast | Best local pipeline (no learning) |
| KNN | Instance-based | Low | Fast | Cosine similarity on meta-features |
| Random Forest | Tree ensemble | Medium | Medium | Regression on meta-features → pipeline performance |
| Neural Network | Deep learning | High | Medium | 3-layer MLP with batch norm |
| AdaBoost | Boosting | Medium | Medium | Adaptive ensemble learning |
| PMM | Embedding | High | Medium | Learns dataset-pipeline embeddings |
| Paper PMM | Siamese network | Very High | Slow | Contrastive learning on performance pairs |

---

## 🔧 Configuration

### AutoGluon Settings
Edit in `evaluation_utils.py`:
```python
AUTOGLUON_CONFIG = {
    "time_limit": 600,           # 10 minutes per dataset
    "presets": "medium_quality", # Preset complexity level
    "verbosity": 4,              # Output verbosity
    "ag_args_fit": {
        "ag.max_memory_usage_ratio": 0.9,
    }
}
```

### Preprocessing Pipelines
Edit in `recommender_trainer.py`:
```python
pipeline_configs = [
    {
        'name': 'baseline',
        'imputation': 'none',
        'scaling': 'none',
        'encoding': 'onehot',
        # ... other parameters
    },
    # Add more pipelines here
]
```

---

## 📝 Input Data Format

### Meta-features (`dataset_feats.csv`)
CSV with dataset IDs as index and meta-features as columns:
```
dataset_id, NumberOfFeatures, NumberOfInstances, ...
22, 10, 1000, ...
28, 15, 500, ...
```

### Performance Matrix (`preprocessed_performance.csv`)
CSV with pipelines as rows and datasets as columns:
```
pipeline, D_22, D_28, D_30, ...
baseline, 0.85, 0.90, 0.88, ...
simple_preprocess, 0.87, 0.91, 0.89, ...
```

---

## 🎓 Understanding Key Concepts

### Meta-features
Statistical/information-theoretic properties of datasets:
- Number of instances, features, classes
- Feature correlations, missing value ratios
- Class imbalance, dimensionality

### Preprocessing Pipelines
Sequence of transformations:
1. **Imputation**: Handle missing values (mean, median, KNN, etc.)
2. **Encoding**: Convert categorical to numeric
3. **Scaling**: Normalize numeric features
4. **Feature Selection**: Choose important features
5. **Outlier Removal**: Detect and remove anomalies
6. **Dimensionality Reduction**: PCA/SVD for high-dimensional data

## 🧠 Recommender Learning
Maps meta-features → optimal pipeline:
- **Supervised**: Learn from known performances
- **Meta-learning**: Across many datasets
- **Transfer**: Use knowledge to recommend for new datasets

---

## 📝 Code Style

**Comments**: Only when necessary
- Function name clear? → No comment needed
- Logic obvious? → No comment needed
- Non-obvious algorithm? → Comment WHY, not WHAT
- Preventing bugs? → Mark with `# <- CRITICAL`

See `COMMENTS_GUIDE.md` for detailed style guide.

## 🧪 Testing & Validation

### Run Quick Test
```bash
python test.py
```
Evaluates baseline on 3 test datasets.

### Run Full Evaluation
```bash
python recommender_trainer.py --recommender_type all
```
Trains all recommenders and evaluates on full test set.

### Expected Results
- **Baseline Accuracy**: ~20-30% (random guess = 9%)
- **Best Recommender**: 40-60% accuracy
- **Average Rank**: 3-5 out of 9 pipelines

---

## 📊 Analyzing Results

### View Summary
```bash
python -c "
import pandas as pd
df = pd.read_csv('test_evaluation_summary.csv')
print(df[['recommender_type', 'accuracy', 'avg_rank', 'better_than_baseline']])
"
```

### Plot Comparison
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test_evaluation_summary.csv')
df.plot(x='recommender_type', y='accuracy', kind='bar')
plt.show()
```

---

## 🐛 Troubleshooting

### Issue: "AutoGluon evaluation failed"
**Solution**: Ensure AutoGluon is installed and dataset isn't corrupted
```bash
python -c "from autogluon.tabular import TabularPredictor; print('OK')"
```

### Issue: "Not enough training pairs for PMM"
**Solution**: Need sufficient performance variance. Check training data:
```bash
python -c "
import pandas as pd
perf = pd.read_csv('preprocessed_performance.csv')
print(f'Pipelines: {len(perf)}')
print(f'Datasets: {len(perf.columns)}')
print(f'Non-null: {perf.notna().sum().sum()}')
"
```

### Issue: CUDA memory error
**Solution**: Reduce batch size or use CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python recommender_trainer.py
```

---

## 📚 Additional Resources

### Files to Read First
1. **COMPLETE_GUIDE.md** - Detailed mathematical formulations
2. **DPO_INFLUENCE_EXPLAINED.md** - Influence weighting methodology

### Key Papers Referenced
- Kadioglu et al. (2017) - Algorithm Selection / Meta-learning
- Brazdil et al. (2009) - Metalearning for Algorithm Recommendation

### Related Work
- AutoML systems (Auto-WEKA, Auto-sklearn)
- Meta-learning approaches
- Algorithm portfolios

---

## 🤝 Contributing

To extend the system:

1. **Add new recommender**: Implement class in `recommenders.py`
2. **Add preprocessing pipeline**: Add config to `pipeline_configs` in `recommender_trainer.py`
3. **Add new metric**: Modify `analyze_recommendations()` in `evaluation_utils.py`

---

## 📄 License

[Add your license here]

---

## ✅ Checklist for Running Full Pipeline

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] OpenML API working
- [ ] AutoGluon functional
- [ ] Training data ready (`preprocessed_performance.csv`, `dataset_feats.csv`)
- [ ] Test data ready (`test_ground_truth_performance.csv`)
- [ ] Disk space available (5-10 GB for models)

---

## 🚀 Getting Started (TL;DR)

```bash
cd SoluRec/Data
pip install -r requirements.txt
python recommender_trainer.py --recommender_type all
```

Results will be in `test_evaluation_summary.csv`

---

**Last Updated**: October 18, 2025  
**Status**: Production Ready ✅
