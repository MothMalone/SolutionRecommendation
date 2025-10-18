# QUICKSTART.md - Get Running in 5 Minutes

## 🚀 Installation (1 min)

```bash
cd SoluRec/Data
pip install -r requirements.txt
```

## ⚡ Quick Test (1 min)

```bash
python test.py
```

Should complete in ~1 minute. Evaluates baseline on 3 test datasets.

## 🎯 Full Training (30+ min)

```bash
python recommender_trainer.py --recommender_type all
```

Trains all recommenders and evaluates on 19 test datasets.

## 📊 View Results

```bash
# Summary metrics
head -20 test_evaluation_summary.csv

# Or in Python
python -c "import pandas as pd; df = pd.read_csv('test_evaluation_summary.csv'); print(df[['recommender_type', 'accuracy', 'avg_rank']])"
```

## 🎨 Explore Data

```bash
streamlit run data.py
```

Open `http://localhost:8501` in browser.

---

## 🔧 Common Commands

```bash
# Train only Neural Network (faster)
python recommender_trainer.py --recommender_type nn

# Train with influence weighting
python recommender_trainer.py --recommender_type all --use_influence

# Use specific influence method
python recommender_trainer.py --recommender_type pmm --influence_method discriminative_power

# Train KNN recommender
python recommender_trainer.py --recommender_type knn
```

---

## 📁 Output Files

After running, you'll get:

- `test_evaluation_summary.csv` - Metrics by recommender
- `recommender_evaluation_results.csv` - Per-dataset predictions
- `preprocessed_performance.csv` - Training dataset performance matrix

---

## ❓ Stuck?

1. **"AutoGluon failed"** → Check internet (OpenML) + disk space (5+ GB)
2. **"Module not found"** → Run `pip install -r requirements.txt` again
3. **"CUDA out of memory"** → Set `export CUDA_VISIBLE_DEVICES=""` for CPU
4. **"Still confused?"** → Read `README.md` for full details

---

## 📖 Next Steps

1. ✅ Run `python test.py` to verify setup
2. ✅ Run `python recommender_trainer.py --recommender_type nn` to train one model
3. ✅ Check results: `head test_evaluation_summary.csv`
4. ✅ Run `streamlit run data.py` to explore data
5. ✅ Read `README.md` for deep dive

---

**Estimated Times**:
- Setup: 5 minutes
- Quick test: 1 minute  
- Train one recommender: 5-10 minutes
- Train all recommenders: 30-60 minutes

---

**Last Updated**: October 18, 2025
