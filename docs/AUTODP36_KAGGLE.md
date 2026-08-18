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

The runner supports three download modes through `--openml-backend`:

- `gitlab` downloads the DataGit/OpenML Parquet mirror and caches it in
  `--openml-local-folder`; no Kaggle Dataset upload is required;
- `openml` uses the official `openml-python` package, then sklearn's OpenML
  loader, without falling back to GitLab;
- `auto` tries local/OpenML first and falls back to GitLab.

For reproducible Kaggle runs, use `gitlab`. OpenML remains selectable for
connectivity tests and environments where its API is responsive.

## One-dataset smoke run

This diagnostic skips AutoGluon and runs a tiny ACO budget:

```bash
python scripts/run_recommend.py \
  --operator-space autodp36 \
  --dataset-source openml \
  --openml-backend gitlab \
  --dataset-id 36 \
  --openml-local-folder /kaggle/working/autodp60_cache \
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
  --dataset-source openml \
  --openml-backend gitlab \
  --dataset-ids 36 728 735 737 \
  --openml-local-folder /kaggle/working/autodp60_cache \
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

The ready-to-import notebook is
`notebooks/run-acorec-autodp36-kaggle.ipynb`. Its smoke mode validates one
dataset without AutoGluon; final mode runs the normal AutoGluon evaluator.

The notebook defaults to `DATASET_SUITE = "ours30"` and splits the canonical
30 test datasets into ten shards of three datasets. For the 17 DiffPrep
snapshots it scans an attached Kaggle input first, then downloads the frozen
files from the DiffPrep GitHub repository. Dataset `100000` (`google`) is not
an OpenML ID: it is loaded from DiffPrep's `google/data.csv`, whose label
`Rating>4.2` is normalized to `target`. Set `DATASET_SUITE = "autodp60"` only
when switching back to AutoDP's 60-dataset suite.
