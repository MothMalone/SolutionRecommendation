# ACORec similarity and search ablation

All experimental options are opt-in. The legacy defaults and Markov weight `0`
remain unchanged. The fixed manifest is `data/openml/meta_dev18.json`; it is
disjoint from paper test30 and AutoDP-60. Every query dataset is removed from the
performance matrix, metafeatures, metric training, and heuristic transfer.

For a fast diagnostic of retrieval quality and information loss during Eq-7
aggregation, run:

```bash
python scripts/audit_similarity_transfer.py \
  --output-dir outputs/similarity_transfer_audit
```

It reports behavioral top-k overlap, effective neighbor count, operator-quality
correlation, recombination expansion, and the initial probability mass retained
on the exact transferred pipelines.

## 1. Rank similarity models offline

Run the 18 leave-one-dataset-out folds (the example uses one Kaggle shard):

```bash
python scripts/evaluate_similarity_meta_dev.py \
  --num-shards 18 --shard-index 0 \
  --output-dir outputs/similarity_meta_dev
```

After collecting the shards:

```bash
python scripts/summarize_similarity_meta_dev.py outputs/similarity_meta_dev
```

`similarity_finalists.json` contains the two variants admitted to TPOT screening.

## 2. Sequential A/B screening

Use `scripts/run_acorec_meta_dev_ablation.py` or the Kaggle notebook
`notebooks/run-acorec-meta-dev-ablation-kaggle.ipynb`. A run is one similarity ×
search combination and is resumable by copying its output directory back into
`/kaggle/working`.

```bash
python scripts/run_acorec_meta_dev_ablation.py \
  --similarity-variant rank_listwise \
  --search-variant improvement_only \
  --num-shards 6 --shard-index 0
```

Follow the configured sequence: update policy; MMAS; exploration; beta; then
evaporation/top-k. Screening is 10 ants × 10 iterations, ACO seeds 42/43/44,
and estimator-only TPOT seed 1 for one minute. Aggregate a combination with
`--stage aggregate`, then compare all completed combinations:

```bash
python scripts/summarize_acorec_ablation.py outputs/acorec_meta_dev_ablation
```

The summarizer applies the 0.005 accuracy tie threshold, paired bootstrap CI,
and reports the 2×2 retrieval/search factorial. Re-run the two leaders with
`--confirmation` for five-minute TPOT.

## 3. Final protocol

Freeze similarity and ACO hyperparameters before test30. Primary test30 uses ACO
seed 42 and TPOT seed 1 once; seeds 43/44 are sensitivity only. Compare paired
outer-test accuracy against DiffPrep with identical split fingerprints. If the
mean gain is below 0.03, return to meta-dev instead of tuning on test30.

Each run stores its git commit, seeds, held-out query, protocol config, ACO
history, split fingerprints, TPOT budget, accuracy, and failure JSON.
