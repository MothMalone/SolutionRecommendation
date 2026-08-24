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

import json
import os
import time
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional

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

    This dict is discarded once the search returns a pipeline (``_run_fair`` does ``del dataset``
    right after) -- unlike the APPLY dict built separately in ``_run_fair``, nothing downstream
    reads its row identity. So it does NOT preserve the original dataframe's row positions as its
    index; it gets a fresh contiguous ``0..len(tr)+len(val)-1`` range instead (train first, test
    second), and that turns out not to be optional:

    ``MCTS.default_policy`` -> ``merge_datasets`` concatenates train+test, sorts by index, and
    hands the result to ``MetaFeature.setMetaFeature``, which does ``data[0]`` -- a LABEL lookup
    for row-label ``0``, not a positional one. Their own ``read_dataset`` gets away with this
    because ``train_test_split`` partitions its FULL input, so ``train union test`` always covers
    every original label and ``0`` is always present after the sort. Preserving original row
    positions here would violate that invariant on purpose: ``tr union val`` is deliberately only
    80% of the dataset (``te`` is excluded so the search never sees test rows), so whenever
    original position 0 landed in ``te``, label ``0`` would be simply absent and ``data[0]`` would
    raise on EVERY search iteration -- observed on dataset 862 (87 rows): 1.7M swallowed
    exceptions, one per attempted iteration, and an empty pipeline. A gapless 0-based index sidesteps
    it entirely, since it is relabeling relative to the 80% actually being searched, not the original
    100%.

    Their operators still get what they need: ``dataset['target'].loc[X.index]``-style lookups only
    require train/target (and test/target_test) to share an index with EACH OTHER, which they do.

    Their ``read_dataset`` also LabelEncodes the target, and we keep that: it is the signal their
    classifiers were built around. Stage 3 re-attaches the ORIGINAL y regardless.
    """
    from sklearn.preprocessing import LabelEncoder

    tr = np.asarray(tr)
    val = np.asarray(val)
    if set(tr.tolist()) & set(val.tolist()):
        raise AssertionError("train and val positions overlap; the split is not a partition")

    y_enc = pd.Series(LabelEncoder().fit_transform(df[target]), index=df.index, name=target)
    feats = df.drop(columns=[target])

    train_labels = np.arange(len(tr))
    test_labels = np.arange(len(tr), len(tr) + len(val))

    dataset = {
        "train": feats.iloc[tr].set_axis(train_labels, axis=0).copy(),
        "target": y_enc.iloc[tr].to_frame().set_axis(train_labels, axis=0),
        "test": feats.iloc[val].set_axis(test_labels, axis=0).copy(),
        "target_test": y_enc.iloc[val].to_frame().set_axis(test_labels, axis=0),
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


# --------------------------------------------------------------------------- search checkpoint


class SearchCheckpoint:
    """Persist the best-so-far pipeline after every NODE EVALUATION.

    Iteration granularity was not enough. On datasets 378 and 722 the search never completed a
    single MCTS iteration in 5400s, so an iteration-level checkpoint stayed empty and the
    wall-clock kill produced nothing twice over. The cause is not slowness -- measured node
    evaluations on 378 take 0.0-0.9s -- it is that their own scorer returns ``accuracy = None``
    when a candidate leaves no usable numeric columns (``classifier.py``: ``if
    (len(X_train.columns) < 1) or (len(X_train) < k): accuracy = None``), and then:

      * ``Node.get_profit_value`` computes ``pre_profit + after_profit`` -> ``None + float``, and
      * ``drop_unpromising`` calls ``profit.sort()`` on a list mixing ``None`` with floats.

    Either raises TypeError on python 3, ``CLA_Without_TimeBudget``'s bare ``except:`` swallows it,
    ``gapcount`` never advances, and the loop spins forever at full CPU without evaluating another
    node. Observed on 378: 7 nodes evaluated in ~2s, then nothing for the rest of the budget.

    Hooking ``get_profit`` -- the atomic unit that actually calls ``mctsdata.getAcc`` -- captures
    those 7 evaluations, and the best of them (``['RF', 'CBE']``, profit 0.813) is a genuine,
    genuinely-evaluated AutoDP preference. It also runs during the initial expansion loop, before
    the ``while True`` is ever entered.

    Not a method change: which pipeline their loop would pick is untouched; this only records the
    running argmax over nodes THEY chose to evaluate and scored themselves.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.best_profit: Optional[float] = None
        self.pipeline: Optional[List[Any]] = None
        self.best_depth: Optional[int] = None
        self.n_node_evals = 0
        self.n_none_profits = 0

    def install(self, verbose: bool = False) -> None:
        from autodatapre.Pipeline_Generation import MCTS

        original = MCTS.get_profit
        ckpt = self

        def checkpointing_get_profit(node, dataset, taskType, datasetTarget):
            result = original(node, dataset, taskType, datasetTarget)
            try:
                ckpt.n_node_evals += 1
                profit = node.get_pre_profit()
                if profit is None:
                    # Their scorer's no-usable-numeric-columns path. Counting these is what turns
                    # the resulting spin into a diagnosis instead of a mystery slow dataset.
                    ckpt.n_none_profits += 1
                    return result
                profit = float(profit)
                if ckpt.best_profit is None or profit > ckpt.best_profit:
                    ckpt.best_profit = profit
                    ckpt.pipeline = list(node.get_state().cumulative_choices)
                    ckpt.best_depth = int(node.get_state().get_current_depth())
                    ckpt._write()
            except BaseException:
                pass  # bookkeeping must never break their search
            return result

        MCTS.get_profit = checkpointing_get_profit
        if verbose:
            print(f"[protocol] checkpointing best-so-far pipeline (per node eval) to {self.path}",
                  flush=True)

    def _write(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({
                "pipeline": self.pipeline,
                "profit": self.best_profit,
                # As of THIS checkpoint. The file is only rewritten when the best improves, so a
                # later non-improving iteration leaves this number behind -- which is what it
                # should mean: how much search the salvaged pipeline is backed by.
                "node_evals_completed": self.n_node_evals,
                "none_profit_evals": self.n_none_profits,
                "depth": self.best_depth,
                "written_at": time.time(),
            }, fh)
        os.replace(tmp, self.path)

    @staticmethod
    def read(path: str) -> Optional[dict]:
        """The parent-side reader: the checkpoint if one was written, else None."""
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                data = json.load(fh)
            return data if data.get("pipeline") else None
        except Exception:
            return None


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

    def __init__(self, checkpoint: "Optional[SearchCheckpoint]" = None,
                 spin_abort_after: int = 500) -> None:
        self.kinds: Counter = Counter()
        self.first_traceback: Optional[str] = None
        self.installed = False
        self.checkpoint = checkpoint
        self.spin_abort_after = int(spin_abort_after)
        self._evals_at_spin_start: Optional[int] = None
        self._consecutive = 0

    def install(self, verbose: bool = False) -> None:
        from autodatapre.Pipeline_Generation import MCTS

        original = MCTS.monte_carlo_tree_search
        counter = self

        def counting_mcts(node, dataset, datasetTarget, taskType):
            try:
                result = original(node, dataset, datasetTarget, taskType)
                counter._consecutive = 0
                counter._evals_at_spin_start = None
                return result
            except BaseException as exc:
                counter.kinds[type(exc).__name__] += 1
                if counter.first_traceback is None:
                    counter.first_traceback = traceback.format_exc(limit=6)
                counter._note_failure()
                raise  # their loop still handles it; we only observe

        MCTS.monte_carlo_tree_search = counting_mcts
        self.installed = True
        if verbose:
            print("[protocol] counting search-iteration exceptions"
                  + (f"; aborting if {self.spin_abort_after} consecutive raise with no new node "
                     f"evaluation" if self.checkpoint is not None else ""), flush=True)

    def _note_failure(self) -> None:
        """Detect the dead loop and get out of it, instead of burning the whole wall-clock cap.

        ``CLA_Without_TimeBudget``'s bare ``except:`` makes a permanently-failing iteration
        indistinguishable from a slow one: ``gapcount`` only advances on SUCCESS, so once every
        iteration raises, ``while True`` never terminates. That is what happened on 378 and 722 --
        7 nodes evaluated in ~2s, then hours of pure spin, killed at the cap with nothing to show.

        The signal for "dead, not slow" is precise: iterations keep raising while the node-eval
        counter does not move, i.e. no work is being done between failures. A genuinely slow
        dataset evaluates nodes between iterations and resets the counter.

        Escaping is the awkward part. Their ``except:`` is bare, so it catches every exception
        type, ``SystemExit`` and ``KeyboardInterrupt`` included -- there is no exception this can
        raise that reaches the caller. ``os._exit`` is the only exit that their handler cannot
        swallow. The checkpoint is already on disk (written per node evaluation), and
        ``run_autodatapre`` salvages on a non-zero exit code, so the result survives the abort.
        """
        if self.checkpoint is None:
            return
        evals = self.checkpoint.n_node_evals
        if self._evals_at_spin_start is None or evals != self._evals_at_spin_start:
            # Work happened since the last failure: slow, not stuck. Restart the count.
            self._evals_at_spin_start = evals
            self._consecutive = 1
            return
        self._consecutive += 1
        if self._consecutive < self.spin_abort_after:
            return

        import sys as _sys
        print(
            f"\n[protocol] ABORTING A DEAD SEARCH: {self._consecutive} consecutive iterations "
            f"raised {dict(self.kinds)} with no node evaluated between them.\n"
            f"[protocol] Their loop advances gapcount only on success, so this never terminates; "
            f"it would spin until the wall-clock cap.\n"
            f"[protocol] {evals} node(s) were evaluated before the spin "
            f"({self.checkpoint.n_none_profits} returned profit=None, which is what poisons "
            f"drop_unpromising/get_profit_value).\n"
            f"[protocol] best checkpointed pipeline: {self.checkpoint.pipeline} "
            f"(profit={self.checkpoint.best_profit}) -> salvaging apply-only.",
            flush=True)
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(3)   # their bare `except:` would swallow any exception, including SystemExit

    def report(self) -> dict:
        return {
            "search_iteration_exceptions": int(sum(self.kinds.values())),
            "search_iteration_exception_kinds": dict(self.kinds),
            "search_iteration_first_traceback": self.first_traceback,
        }
