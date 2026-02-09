# AutoML ACO Recommender

This repository is a production-ready, research-friendly refactor of the original notebook. It preserves the notebook behavior (preprocessing operators, ACO logic, meta-learning flow, and evaluation APIs) while providing a modular, testable codebase.

## How to run (local)

Train the siamese regression metric:

```bash
python -m scripts.train_metric \
  --performance-matrix path/to/performance_matrix.csv \
  --metafeatures path/to/metafeatures.csv \
  --output outputs/siamese_metric.pt
```

Run a recommendation:

```bash
python -m scripts.run_recommend \
  --performance-matrix path/to/performance_matrix.csv \
  --metafeatures path/to/metafeatures.csv \
  --pipeline-configs path/to/pipeline_configs.json \
  --dataset-source csv \
  --dataset-csv path/to/dataset.csv \
  --target-column target \
  --dataset-id 1387 \
  --use-aco
```

Run tests:

```bash
pytest -q
```

## Kaggle quickstart

From a Kaggle notebook cell:

```bash
!git clone <your-repo-url>
%cd SolutionRecommendation
!pip -q install -r requirements-kaggle.txt
```

Run ACO + final AutoGluon evaluation (outputs saved to `/kaggle/working`):

```bash
!python -m scripts.run_recommend \
  --kaggle \
  --dataset-source openml \
  --dataset-id 2 \
  --use-aco \
  --verbose
```

Artifacts saved:
- `/kaggle/working/recommendation.json`
- `/kaggle/working/aco_history.csv`
- `/kaggle/working/aco_progress.png` (if `matplotlib` is installed)

If AutoGluon is not available, the final evaluation will fall back to proxy scores and the output will note `autogluon_unavailable`.

## Design notes

- `src/automl_aco/preprocessing/`: Implements the notebook `Preprocessor` and operator helpers. Execution order matches the notebook to preserve behavior; use `DEFAULT_PREPROCESSOR_ORDER` to change it if needed.
- `src/automl_aco/metalearning/`: Siamese regression metric model (train/save/load) and the `MetaPipelineRecommender` class.
- `src/automl_aco/search/`: ACO search, heuristic construction, and evaluation functions (simple models + optional AutoGluon).
- `src/automl_aco/data/`: Dataset schemas, metafeature lookup, and leak-free train/val/test split.
- `scripts/`: CLI entrypoints for training the metric and generating recommendations.

Extension points:
- Add new preprocessing operators by extending `OPERATORS` and updating the `Preprocessor`.
- Swap the ACO evaluation function in `search/aco.py` to test alternative scoring strategies.
- Replace the metafeature extractor in `data/metafeatures.py` with a custom extractor.

## Notes on ordering

The ACO search ordering and preprocessor execution order are preserved from the notebook to keep results identical. The domain layer order is still available in `config.DOMAIN_LAYER_ORDER` if you want to re-align search or preprocessing.
