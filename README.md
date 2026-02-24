# AutoML ACO Recommender

This repository is a production-ready, research-friendly refactor of the original notebook. It preserves the notebook behavior (preprocessing operators, ACO logic, meta-learning flow, and evaluation APIs) while providing a modular, testable codebase.

## How to run (local)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the siamese regression metric:

```bash
python -m scripts.train_metric \
  --performance-matrix path/to/performance_matrix.csv \
  --metafeatures path/to/metafeatures.csv \
  --output outputs/siamese_metric.pt
```

Run a recommendation (CSV source):

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

Run a recommendation (OpenML source):

```bash
python -m scripts.run_recommend \
  --performance-matrix path/to/performance_matrix.csv \
  --metafeatures path/to/metafeatures.csv \
  --pipeline-configs path/to/pipeline_configs.json \
  --dataset-source openml \
  --dataset-id 2 \
  --use-aco \
  --verbose
```

Run with ordering search (CtxPipe-style outer loop over valid step orders):

```bash
python -m scripts.run_recommend \
  --dataset-source openml \
  --dataset-id 2 \
  --use-aco \
  --search-ordering \
  --num-orders 10 \
  --order-strategy heuristic \
  --n-ants 10 \
  --n-iterations 10 \
  --seed 42 \
  --verbose
```

Outputs (local):
- `outputs/recommendation.json`
- `outputs/aco_history.csv`
- `outputs/aco_progress.png` (if `matplotlib` is installed)

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

If your uploaded dataset uses a different root, pass `--kaggle-root /kaggle/input/<your-dataset-name>`.

Artifacts saved:
- `/kaggle/working/recommendation.json`
- `/kaggle/working/aco_history.csv`
- `/kaggle/working/aco_progress.png` (if `matplotlib` is installed)

If AutoGluon is not available, the final evaluation will fall back to proxy scores and the output will note `autogluon_unavailable`.

Notes:
- You do not need to set `PYTHONPATH`; the scripts add `src/` automatically.
- Warnings are suppressed by default. Use `--show-warnings` to re-enable them.
- If you see a NumPy 2.x compatibility error from AutoGluon/PyTorch, reinstall with `numpy<2` (already pinned in `requirements.txt`).

## Local vs Kaggle mode (path behavior)

The CLI resolves input files differently depending on mode:

- **Local (default)**: uses local training matrix and metafeatures if you don’t pass paths:
  - `aco/training_performance_matrix_autogluon.csv`
  - `aco/dataset_feats.csv`
  - `aco/pipeline_configs.json`
- **Kaggle**: enable with `--kaggle` (or when running inside `/kaggle/working`), then it uses the repo root at `/kaggle/input/acorec` by default and saves outputs to `/kaggle/working`.
  - Override the root with `--kaggle-root /kaggle/input/your_dataset`.

You can always override any file with `--performance-matrix`, `--metafeatures`, and `--pipeline-configs`.

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

By default, search uses the notebook-fixed order from `DEFAULT_PIPELINE_OPTIONS`. You can enable order search with:
- `--search-ordering`
- `--num-orders K`
- `--order-strategy {fixed,random,heuristic,scored,all}`

Order search enforces precedence constraints from `DEFAULT_ORDERING_CONSTRAINTS` and evaluates each proposed order by running the existing ACO operator search within that order. Each candidate pipeline stores `step_order`, and preprocessing executes in that exact order.
