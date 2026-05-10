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

Run dataset-378 conservative profile (toggle to reduce over-processing on OpenML 378):

```bash
python -m scripts.run_recommend \
  --dataset-source openml \
  --dataset-id 378 \
  --use-aco \
  --optimizer tpe \
  --sample-budget 100 \
  --proxy-profile robust \
  --final-autogluon-topk 5 \
  --seed 42 \
  --verbose
```

Proxy robustness knobs:
- `--proxy-profile robust` enables multi-seed proxy scoring + over-processing penalties.
- `--proxy-seeds 42,52,62` overrides split seeds.
- `--final-autogluon-topk 5` re-ranks top-5 proxy pipelines with final AutoGluon.
- `--dataset378-profile {off,conservative,scaling_only}` keeps optional dataset-378-only search-space constraints.

Run optimizer ablation with 100 sampled configs per optimizer (same split/order constraints/eval flow):

```bash
python -m scripts.run_recommend \
  --performance-matrix aco/training_performance_matrix_autogluon.csv \
  --metafeatures aco/dataset_feats.csv \
  --pipeline-configs aco/pipeline_configs.json \
  --dataset-source openml \
  --dataset-ids 248,1066,1164,1047,862,2,40663,1054,1387,876,18,1520,1548,184,378,381,382,993,1485,14 \
  --use-aco \
  --optimizer ga \
  --sample-budget 100 \
  --search-ordering \
  --num-orders 10 \
  --order-strategy heuristic \
  --seed 42 \
  --verbose
```

Supported optimizers:
- `aco`
- `dqn` (context-gated DQN with warm-start priors and online proxy-reward updates)
- `random`
- `ga`
- `sa`
- `greedy`
- `mcts`
- `beam`
- `tpe` (lightweight categorical TPE-style model-based search)
- `exhaustive` (exact only when full space size <= sample budget, otherwise random fallback)

Run the DQN-enhanced search (warm-start + context gate inspired by CtxPipe):

```bash
python -m scripts.run_recommend \
  --performance-matrix aco/training_performance_matrix_autogluon.csv \
  --metafeatures aco/dataset_feats.csv \
  --pipeline-configs aco/pipeline_configs.json \
  --dataset-source openml \
  --dataset-id 2 \
  --use-aco \
  --optimizer dqn \
  --sample-budget 120 \
  --dqn-order-policy ctxpipe \
  --dqn-num-logic-orders 6 \
  --dqn-updates-per-episode 2 \
  --dqn-replay-warmup 20 \
  --dqn-loss-fn huber \
  --dqn-grad-clip-norm 5.0 \
  --dqn-target-q-clip 5.0 \
  --dqn-gamma 0.95 \
  --dqn-warmstart-weight 0.5 \
  --verbose
```

`--dqn-order-policy ctxpipe` uses an internal RL logical-order selector (CtxPipe-style)
instead of outer-loop `--search-ordering`.

For batch runs, summary metrics are saved to `recommendations_summary.json`:
- average elapsed time
- average proxy score
- average final score
- average AutoGluon score (runs where final method is `autogluon`)

Explain ablation outputs (quantization / fallback / tie-vs-baseline diagnostics):

```bash
python -m scripts.rq3_explain_sensitivity \
  --suite-dir outputs/ablation_budget/rq3_budget
```

This generates:
- `ablation_explain_rows.csv` (dataset-level diagnostics and explanation labels)
- `ablation_explain_summary.csv` (variant-level win/tie/loss + AutoGluon usage rate)

For long Kaggle runs, you can shard/resume sensitivity suites:

```bash
python -m scripts.rq3_sensitivity_budget \
  --dataset-ids 27 1047 248 1387 1054 \
  --variants ants5_iter5 ants10_iter10 \
  --resume \
  --output-root /kaggle/working/rq3_sensitivity
```

`--variants` runs only named variants. `--resume` loads existing `sensitivity_results.*` and skips completed variants.
Results are flushed after each variant, so progress survives session timeouts.

Run multiple datasets in one command:

```bash
python -m scripts.run_recommend \
  --performance-matrix aco/training_performance_matrix_autogluon.csv \
  --metafeatures aco/dataset_feats.csv \
  --pipeline-configs aco/pipeline_configs.json \
  --dataset-source openml \
  --dataset-ids 2,14,18,46 \
  --use-aco \
  --search-ordering \
  --num-orders 10 \
  --order-strategy heuristic \
  --seed 42 \
  --n-ants 10 \
  --n-iterations 10 \
  --verbose
```

`--dataset-ids` accepts comma-separated and/or space-separated ids.

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
- If AutoGluon hits an XGBoost compatibility error (`'XGBClassifier' object has no attribute 'n_classes_'`), evaluation now retries automatically without XGB models.

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
- `--ordering-quick-time-limit SECONDS` (quick AutoGluon score per ordering iteration)

Order search enforces precedence constraints from `DEFAULT_ORDERING_CONSTRAINTS` and evaluates each proposed order by running the existing ACO operator search within that order. Each candidate pipeline stores `step_order`, and preprocessing executes in that exact order.
When order search is enabled, each ordering iteration also runs a quick AutoGluon check on that order's best pipeline, and history uses this per-iteration AutoGluon score (falling back to proxy only if quick AutoGluon fails).
