# QUICKSTART

## Local (fast setup)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a quick ACO recommendation:

```bash
python -m scripts.run_recommend \
  --performance-matrix aco/training_performance_matrix_autogluon.csv \
  --metafeatures aco/dataset_feats.csv \
  --pipeline-configs aco/pipeline_configs.json \
  --dataset-source openml \
  --dataset-id 2 \
  --use-aco \
  --verbose
```

Outputs:
- `outputs/recommendation.json`
- `outputs/aco_history.csv`
- `outputs/aco_progress.png` (if `matplotlib` installed)

## Kaggle

```bash
!git clone <your-repo-url>
%cd SolutionRecommendation
!pip -q install -r requirements-kaggle.txt
```

```bash
!python -m scripts.run_recommend \
  --kaggle \
  --dataset-source openml \
  --dataset-id 2 \
  --use-aco \
  --verbose
```

Outputs go to `/kaggle/working`.
