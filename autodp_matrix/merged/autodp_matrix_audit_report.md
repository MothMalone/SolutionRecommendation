# AutoDP performance-matrix audit

## Result

- Source shards: 32/32
- Full matrix: 36 pipelines x 912 datasets = 32,832 jobs
- Successful jobs: 30,004 (91.39%)
- Missing jobs: 2,828 (8.61%)
- Conflicting duplicate values: 0
- Complete datasets: 793
- Completely empty datasets: 68

## Leakage and quality filtering

- Removed 26 AutoDP60 holdout columns present in the historical corpus.
- Kept every non-holdout dataset with at least one successful evaluation; only completely empty datasets were removed.
- ACORec-ready matrix: 36 pipelines x 818 datasets.
- Remaining missing cells: 380 (1.29%); ACORec's row-mean performance imputer can handle them.
- Metafeature overlap: 818/818 ready datasets.

## Performance signal

- Baseline mean accuracy: 0.7707.
- Best average pipeline: autodp_ofat_normalization_zscore (0.7790).
- Per-dataset oracle mean accuracy: 0.8126.
- Mean oracle lift over baseline: 0.0408.
- At least one pipeline beats baseline on 595/818 datasets (72.7%).
- Median within-dataset score range: 0.1283.

## Conclusion

The filtered matrix is ready for ACORec. Use `training_performance_matrix_autodp36_ready.csv`
with `aco/pipeline_configs_autodp36.json` and `data/openml/dataset_feats.csv`.
