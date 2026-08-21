#!/usr/bin/env python3
"""Move AutoDP's evaluation protocol onto ours, without touching its search.

RUNS IN THE PINNED ``.venv-autodp`` ENVIRONMENT, imported by scripts/run_autodatapre.py.

The cross-comparison arms (docs/ARMS.md) only mean something if everything *except the
pipeline-selection logic* is identical on both sides. AutoDP ships four defects that sit in the
other layer -- split, leakage discipline, scorer determinism, scorer objective -- and this module
fixes exactly those, in memory. `autodatapre`'s files on disk are never modified.

The dividing line, and it is the whole design:

  MOVED TO OUR SETTING (this module)          LEFT AS THEIRS (disclosed, never patched)
  ----------------------------------          -----------------------------------------
  the train/val split the search scores on     MCTS tree policy, UCB, backup, pruning
  which labels an operator may consume         the pretrained value estimate
  scorer seeding (reproducibility)             progressive subsampling (Is_BatchTraining)
  scorer objective (same metric for NB/LDA/RF) operator semantics (per-split statistics)

Four patches:

1. ``build_search_dataset`` -- replaces ``MCTS_DATA.read_dataset``, whose
   ``train_test_split(X, Y, test_size=0.2)`` has no ``random_state`` and no stratify. Their search
   signal was therefore non-reproducible run to run, and on the full frame its internal "test"
   contained our held-out rows. We hand the search our seed-42 0.6 train / 0.2 val instead.
   This needs no monkeypatching: MCTS never calls ``read_dataset``, the dict is built by the
   caller and passed in.

2. ``install_leakfree_cbe`` -- ``Encoding.transform`` fits ``CatBoostEncoder`` on
   ``concat(target, target_test)``, so the rows being scored get encodings computed from their own
   labels. Refit on the train block only, then transform the rest. The feature-side union
   (``concat(train, test)``) is left alone: that is operator semantics, and it plausibly helps them.

3. ``install_scorer_patches`` -- seeds ``RandomForestClassifier`` and ``ExtraTreesClassifier``
   (both unseeded, and the RF *is* the search signal), and makes ``LDA_classification`` score the
   holdout split. Their LDA returned a 10-fold CV mean over train while NB and RF returned holdout
   accuracy; MCTS compares those numbers directly to choose ``order[0]``.

4. ``ExceptionCounter`` -- ``CLA_Without_TimeBudget``'s loop body is wrapped in a bare ``except:``
   that can swallow every iteration and still report convergence. Counting is not a method change;
   it makes a silently-empty search visible in the results row.
"""
from __future__ import annotations

import traceback
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

#: Recorded in autodp_meta.json so a number can be traced to the protocol that produced it.
SEARCH_SPLIT_TAG = "ours-seed42-0.6train/0.2val"


# ---------------------------------------------------------------------------------------- split


def build_search_dataset(
    df: pd.DataFrame,
    target: str,
    tr: np.ndarray,
    val: np.ndarray,
) -> Dict[str, Any]:
    """The dict AutoDP's MCTS scores on, built from OUR split instead of ``read_dataset``.

    ``tr`` / ``val`` are ORIGINAL row positions from ``_split_positions`` (seed-42 0.6/0.2/0.2).
    The test positions are not a parameter here on purpose: they must not be reachable.

    Index preservation is load-bearing and is why this uses ``.iloc[positions]`` on the original
    frame rather than ``split_train_val_test``'s return values, which ``reset_index(drop=True)``:

      * their operators look labels up as ``dataset['target'].loc[X.index]``;
      * ``scripts/autodp_our_space.py`` reads surviving row identity off the index;
      * ``__adp_row__`` -- how stage 3 re-attaches the original y -- is that index.

    Their ``read_dataset`` also LabelEncodes the target, and we keep that: it is the signal their
    classifiers were built around. Stage 3 re-attaches the ORIGINAL y regardless.
    """
    from sklearn.preprocessing import LabelEncoder

    if set(np.asarray(tr).tolist()) & set(np.asarray(val).tolist()):
        raise AssertionError("train and val positions overlap; the split is not a partition")

    y_enc = pd.Series(LabelEncoder().fit_transform(df[target]), index=df.index, name=target)
    feats = df.drop(columns=[target])

    dataset = {
        "train": feats.iloc[tr].copy(),
        "target": y_enc.iloc[tr].to_frame(),
        "test": feats.iloc[val].copy(),
        "target_test": y_enc.iloc[val].to_frame(),
    }
    assert_dict_aligned(dataset)
    return dataset


def assert_dict_aligned(dataset: Dict[str, Any]) -> None:
    """``get_part_dataset`` subsamples 'train' and 'target' INDEPENDENTLY.

    ``MCTS.get_part_dataset`` calls ``.sample(frac=rate, random_state=42)`` on each of them
    separately. Two independent draws land on the same rows only while both frames have identical
    length and identical index -- otherwise every profit evaluation in the search scores features
    against the wrong labels, silently. Cheap to assert, catastrophic to miss.
    """
    for x_key, y_key in (("train", "target"), ("test", "target_test")):
        X, y = dataset.get(x_key), dataset.get(y_key)
        if X is None or y is None or isinstance(X, dict) or isinstance(y, dict):
            continue
        if len(X) != len(y):
            raise AssertionError(
                f"{x_key}/{y_key} length mismatch: {len(X)} vs {len(y)}; get_part_dataset would "
                f"subsample them onto different rows"
            )
        if not np.array_equal(np.asarray(X.index), np.asarray(y.index)):
            raise AssertionError(
                f"{x_key}/{y_key} index mismatch; .loc lookups and get_part_dataset both break"
            )


# ------------------------------------------------------------------------------------- CBE leak


def install_leakfree_cbe(verbose: bool = False) -> None:
    """Fit ``CatBoostEncoder`` on the train block only, not on ``concat(target, target_test)``.

    Their ``Encoding.transform`` (encoding.py:79-95) concatenates the two splits, fits the
    supervised encoder on the concatenated target, then slices the result back apart. Whichever
    rows sit in ``test`` -- our val rows during the search, our real test rows during the apply
    step -- are encoded using their own labels.

    The replacement keeps the shape of their method exactly, including the feature-side
    ``concat`` for the three unsupervised encoders, and diverges only in the label channel. The
    encoder itself comes from ``automl_aco.preprocessing.autodp_ops.build_encoder("CBE")``, so the
    reimplemented `theirs` operator space and this patch cannot drift apart.

    Direction of the effect, for disclosure: this REMOVES an advantage. Expect CBE to be selected
    less often and their scores to fall.
    """
    from autodatapre.Search_Space import encoding as E

    from automl_aco.preprocessing.autodp_ops import build_encoder

    def transform(self):
        normd = self.dataset
        X = pd.concat([self.dataset["train"], self.dataset["test"]], axis=0)
        trainlen = len(self.dataset["train"])
        totallen = len(X)

        if self.strategy == "OE":
            dn = self.ordinal_encoding(X)
        elif self.strategy == "BE":
            dn = self.binary_encoding(X)
        elif self.strategy == "FE":
            dn = self.frequency_encoding(X)
        elif self.strategy == "CBE":
            dn = _catboost_fit_on_train(build_encoder, X, trainlen, self.dataset["target"])
        else:
            raise ValueError(f"Unknown AutoDP encoding code: {self.strategy}")

        normd["train"] = dn.head(trainlen)
        normd["test"] = dn.tail(totallen - trainlen)
        return normd

    E.Encoding.transform = transform
    if verbose:
        print("[protocol] CBE now fits on the TRAIN block only (was: concat(target, target_test))",
              flush=True)


def _catboost_fit_on_train(build_encoder, X: pd.DataFrame, trainlen: int, d_target) -> pd.DataFrame:
    """Their ``CatBoost_encoding`` body, fit on the first ``trainlen`` rows only."""
    cat = X.select_dtypes(["object"])
    if cat.empty:
        return X
    num = X.select_dtypes(["number"])
    dt = X.select_dtypes(["datetime64"])

    y_train = d_target.iloc[:, 0] if isinstance(d_target, pd.DataFrame) else pd.Series(d_target)
    cat_train = cat.head(trainlen)
    if len(y_train) != len(cat_train):
        # Row-dropping operators upstream can desynchronise these; their code would have silently
        # broadcast or truncated. Refuse instead -- a wrong-length target here is a wrong encoding.
        raise RuntimeError(
            f"CBE: {len(cat_train)} train rows but {len(y_train)} train labels; refusing to fit"
        )

    enc = build_encoder("CBE")
    enc.fit(cat_train, np.asarray(y_train).ravel())
    obtained = enc.transform(cat)
    obtained.index = cat.index
    return obtained.join(num).join(dt)


# ------------------------------------------------------------------------------- scorer patches


def install_scorer_patches(seed: int = 42, verbose: bool = False) -> None:
    """Seed the search signal, and give all three classifiers the same objective.

    Their scorer is what MCTS selects on, so its determinism and its metric are evaluation logic,
    not method. Three changes:

      * ``RandomForestClassifier()`` (classifier.py:90) -- unseeded, and it IS the search signal.
      * ``ExtraTreesClassifier(n_estimators=10)`` (feature_selector.py:92) -- unseeded. Their
        ``n_estimators`` is kept; only ``random_state`` is added, for reproducibility.
      * ``LDA_classification`` returned ``results['mean_test_score']``, a 10-fold CV mean over
        TRAIN, while NB and RF returned holdout accuracy on the internal test split. Their
        ``if target in X_test.columns`` escape to a holdout score is dead code -- the target is
        dropped from the features long before. MCTS compares the three numbers directly to pick
        ``order[0]``, so they must measure the same thing. LDA now scores the holdout too.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    from autodatapre.Search_Space import classifier as C
    from autodatapre.Search_Space import feature_selector as F

    def _xy(dataset, target, k):
        """Their own framing rules: numeric columns, drop NaN rows, align labels by index.

        Kept verbatim from their NB path, including the ``len(X_train) < k_folds -> None`` guard,
        because those rules are how their scorer sees a frame and are not ours to change.
        """
        X_train = dataset["train"].select_dtypes(["number"]).dropna()
        if len(X_train.columns) < 1 or len(X_train) < k:
            return None
        y_train = dataset["target"].loc[X_train.index]
        X_test = dataset["test"].select_dtypes(["number"]).dropna()
        y_test = dataset["target_test"].loc[X_test.index]
        if target in X_train.columns.values:
            X_train = X_train.drop(columns=[target])
        if target in X_test.columns.values:
            X_test = X_test.drop(columns=[target])
        if len(X_test) < 1:
            return None
        return X_train, y_train, X_test, y_test

    def LDA_classification(self, dataset, target):
        parts = _xy(dataset, target, self.k_folds)
        if parts is None:
            return None
        X_train, y_train, X_test, y_test = parts
        model = LinearDiscriminantAnalysis(n_components=1)
        model.fit(X_train, y_train.values.ravel())
        return model.score(X_test, y_test)

    def RF_classification(self, dataset, target):
        parts = _xy(dataset, target, self.k_folds)
        if parts is None:
            return None
        X_train, y_train, X_test, y_test = parts
        model = RandomForestClassifier(random_state=seed)
        model.fit(X_train, y_train.values.ravel())
        return model.score(X_test, y_test)

    C.Classifier.LDA_classification = LDA_classification
    C.Classifier.RF_classification = RF_classification

    # Their TB selector does a FUNCTION-LOCAL `from sklearn.ensemble import ExtraTreesClassifier`
    # (feature_selector.py:83), which rebinds from sys.modules at call time -- so setting the
    # attribute on feature_selector would be shadowed. Patch sklearn.ensemble itself. `_original`
    # is captured before the swap, so the factory cannot recurse into itself.
    _original = ExtraTreesClassifier

    def _seeded_extra_trees(*args, **kwargs):
        kwargs.setdefault("random_state", seed)
        return _original(*args, **kwargs)

    import sklearn.ensemble as _sk_ensemble
    _sk_ensemble.ExtraTreesClassifier = _seeded_extra_trees
    if hasattr(F, "ExtraTreesClassifier"):
        F.ExtraTreesClassifier = _seeded_extra_trees

    if verbose:
        print(f"[protocol] scorer seeded (random_state={seed}); LDA now scores the holdout split "
              f"like NB and RF", flush=True)


# ---------------------------------------------------------------------------- exception counter


class ExceptionCounter:
    """Count what ``CLA_Without_TimeBudget``'s bare ``except:`` would otherwise hide.

    Their search loop is::

        try:
            ... monte_carlo_tree_search ...
        except:
            times.pop()

    If every iteration raises, the loop still exits through the convergence rule and reports a
    pipeline -- one that was never actually evaluated. This is most likely on small frames, where
    ``classifier.py``'s ``if len(X_train) < 10: accuracy = None`` combines with
    ``Is_BatchTraining``'s ``frac=(depth+1)/(MAX_DEPTH+1)`` to make ``get_profit_value`` add
    ``None`` to a float.

    Counting is not a method change: nothing about which pipeline is chosen differs. It only makes
    a silently-empty search visible in the results row.
    """

    def __init__(self) -> None:
        self.kinds: Counter = Counter()
        self.first_traceback: Optional[str] = None
        self.installed = False

    def install(self, verbose: bool = False) -> None:
        from autodatapre.Pipeline_Generation import MCTS

        original = MCTS.monte_carlo_tree_search
        counter = self

        def counting_mcts(node, dataset, datasetTarget, taskType):
            try:
                return original(node, dataset, datasetTarget, taskType)
            except BaseException as exc:
                counter.kinds[type(exc).__name__] += 1
                if counter.first_traceback is None:
                    counter.first_traceback = traceback.format_exc(limit=6)
                raise  # their loop still handles it; we only observe

        MCTS.monte_carlo_tree_search = counting_mcts
        self.installed = True
        if verbose:
            print("[protocol] counting search-iteration exceptions", flush=True)

    def report(self) -> dict:
        return {
            "search_iteration_exceptions": int(sum(self.kinds.values())),
            "search_iteration_exception_kinds": dict(self.kinds),
            "search_iteration_first_traceback": self.first_traceback,
        }
