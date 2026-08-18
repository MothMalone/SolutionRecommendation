# ACORec on the AutoDP36 operator space

`autodp36` is an additive runner profile. The default profile remains `ours`.
It selects the matching 36-pipeline performance matrix and pipeline configs,
uses the six paper-style AutoDP operator families, and executes them with the
same fixed order/model-input adapter used when the matrix was generated.

## Kaggle setup

Enable Internet for the notebook, then run:

```bash
git clone --branch feature/acorec-autodp-space --single-branch \
  https://github.com/MothMalone/SolutionRecommendation.git \
  /kaggle/working/SolutionRecommendation
cd /kaggle/working/SolutionRecommendation
python -m pip install -r requirements-kaggle.txt
```

If the evaluation CSVs are attached as a Kaggle Dataset, place or mount them as
`<folder>/<openml_id>.csv` (or `.csv.zip`) and pass that folder through
`--openml-local-folder`. The loader prefers these local snapshots and only
falls back to the OpenML API.

## One-dataset smoke run

This diagnostic skips AutoGluon and runs a tiny ACO budget:

```bash
python scripts/run_recommend.py \
  --operator-space autodp36 \
  --dataset-source local \
  --dataset-id 36 \
  --openml-local-folder /kaggle/input/autodp60-csv \
  --optimizer aco \
  --n-ants 1 \
  --n-iterations 1 \
  --no-autogluon \
  --no-train-metric-inline \
  --output-dir /kaggle/working/autodp36-smoke
```

## Experiment run

Remove the diagnostic flags, train the metric on the AutoDP36 matrix, and let
the normal AutoGluon final evaluator run:

```bash
python scripts/run_recommend.py \
  --operator-space autodp36 \
  --dataset-source local \
  --dataset-ids 36 728 735 737 \
  --openml-local-folder /kaggle/input/autodp60-csv \
  --optimizer aco \
  --n-ants 10 \
  --n-iterations 10 \
  --train-metric-inline \
  --metric-epochs 100 \
  --time-limit 300 \
  --output-dir /kaggle/working/acorec-autodp36 \
  --tar-outputs
```

Use `--shard i/n` to distribute the dataset IDs across independent Kaggle
Save-Version runs. Explicit `--performance-matrix` and `--pipeline-configs`
arguments still override the profile defaults when needed.
