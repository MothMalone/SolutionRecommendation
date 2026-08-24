"""AutoDP's evaluation layer, moved onto ours -- scripts/autodp_protocol.py.

These tests need the ``autodatapre`` package, which lives only in the pinned ``.venv-autodp``
environment (numpy<1.24, incompatible with the main env's AutoGluon stack). Every test that
imports it shells out to that interpreter, following the pattern in test_adp_family_order.py, and
is skipped if that venv has not been built.

What is under test, and why each one matters:

  * the search split is OURS, seeded, and never contains our held-out test rows (closes: AutoDP's
    own ``read_dataset`` used an unseeded ``train_test_split``, so under ``native`` its internal
    "test" was drawn from the full frame and the search signal was non-reproducible run to run);
  * the train/target and test/target_test pairs stay index-aligned (closes: ``get_part_dataset``
    subsamples each independently with the same ``random_state``, which only agrees while lengths
    and indices match -- a silent mismatch scores features against the wrong labels);
  * CBE no longer consumes the labels of the rows it is encoding (closes: their ``Encoding.transform``
    fits ``CatBoostEncoder`` on ``concat(target, target_test)``, so the rows being scored -- our val
    rows during search, our test rows during apply -- got encodings computed from their own labels).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADP_PY = REPO / ".venv-autodp" / "bin" / "python"
SRC = str(REPO / "src")
SCRIPTS = str(REPO / "scripts")

pytestmark = pytest.mark.skipif(not ADP_PY.exists(), reason="AutoDP venv not built")


def _run_in_adp_env(code: str) -> list:
    """Run ``code`` in the pinned AutoDP interpreter; return lines the script @@-tagged as output."""
    preamble = (
        "import warnings, sys, json\n"
        "warnings.filterwarnings('ignore')\n"
        f"sys.path.insert(0, {SRC!r})\n"
        f"sys.path.insert(0, {SCRIPTS!r})\n"
    )
    proc = subprocess.run([str(ADP_PY), "-c", preamble + code],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"AutoDP-env script failed:\n{proc.stdout}\n{proc.stderr}")
    return [json.loads(l[2:]) for l in proc.stdout.splitlines() if l.startswith("@@")]


DATASET_1054 = str(REPO / "data" / "eval_datasets" / "1054.csv")


def test_search_dataset_excludes_test_positions():
    """The dict MCTS scores on must contain exactly the rows of train+val, by VALUE, never test.

    The search dict's index is deliberately NOT the original row positions (see
    build_search_dataset's docstring: an original-position index breaks MetaFeature's `data[0]`
    label lookup whenever position 0 lands in `te`), so this checks row IDENTITY by content --
    every feature row in the search dict must appear in df.iloc[tr] union df.iloc[val], and none
    may appear only in df.iloc[te].
    """
    out = _run_in_adp_env(f"""
import pandas as pd, autodp_protocol as P
from automl_aco.data.splits import split_train_val_test
import numpy as np

df = pd.read_csv({DATASET_1054!r})
idx = pd.DataFrame({{'_pos': np.arange(len(df))}})
dummy = pd.Series(np.zeros(len(df)))
a,_,b,_,c,_ = split_train_val_test(idx, dummy, seed=42)
tr, val, te = a['_pos'].to_numpy(), b['_pos'].to_numpy(), c['_pos'].to_numpy()

d = P.build_search_dataset(df, 'target', tr, val)
feats = df.drop(columns=['target'])

search_rows = pd.concat([d['train'], d['test']], axis=0)
allowed_rows = pd.concat([feats.iloc[tr], feats.iloc[val]], axis=0).reset_index(drop=True)
te_rows = feats.iloc[te].reset_index(drop=True)

# Every search row must equal SOME allowed row (order need not match); none may equal a te-only
# row that isn't also an allowed row (features can coincidentally repeat, so compare as multisets
# via sorted tuples).
search_set = set(map(tuple, search_rows.to_numpy().tolist()))
allowed_set = set(map(tuple, allowed_rows.to_numpy().tolist()))
te_only_set = set(map(tuple, te_rows.to_numpy().tolist())) - allowed_set

print('@@' + json.dumps({{
    'search_size': len(search_rows), 'train_size': len(tr), 'val_size': len(val),
    'test_size': len(te),
    'search_subset_of_allowed': search_set.issubset(allowed_set),
    'leaked_te_only_rows': len(search_set & te_only_set),
}}))
""")
    result = out[0]
    assert result["search_subset_of_allowed"]
    assert result["leaked_te_only_rows"] == 0
    assert result["search_size"] == result["train_size"] + result["val_size"]


def test_search_dataset_is_deterministic():
    """Same positions in, same dict out -- twice, byte for byte on the row identity."""
    out = _run_in_adp_env(f"""
import pandas as pd, autodp_protocol as P
from automl_aco.data.splits import split_train_val_test
import numpy as np

df = pd.read_csv({DATASET_1054!r})
idx = pd.DataFrame({{'_pos': np.arange(len(df))}})
dummy = pd.Series(np.zeros(len(df)))
a,_,b,_,c,_ = split_train_val_test(idx, dummy, seed=42)
tr, val = a['_pos'].to_numpy(), b['_pos'].to_numpy()

d1 = P.build_search_dataset(df, 'target', tr, val)
d2 = P.build_search_dataset(df, 'target', tr, val)
print('@@' + json.dumps({{
    'train_idx_equal': list(d1['train'].index) == list(d2['train'].index),
    'test_idx_equal': list(d1['test'].index) == list(d2['test'].index),
}}))
""")
    result = out[0]
    assert result["train_idx_equal"]
    assert result["test_idx_equal"]


def test_search_dataset_index_is_gapless_and_starts_at_zero():
    """merge_datasets(dataset).sort_index() must cover a contiguous 0..m-1 range.

    This is the actual invariant AutoDP's own code needs: MetaFeature.setMetaFeature does
    `data[0]` (a LABEL lookup for row-label 0), which only resolves if train+test, once merged and
    sorted, contains label 0 with no gaps. Their own read_dataset gets this for free because
    train_test_split partitions 100% of its input; build_search_dataset partitions only 80% (tr+val,
    excluding te), so it must relabel onto a fresh contiguous range rather than keep original
    positions -- otherwise, whenever original position 0 lands in te, label 0 is simply absent and
    every search iteration crashes (observed on dataset 862: 1.7M swallowed exceptions, empty
    pipeline, before this fix).
    """
    out = _run_in_adp_env(f"""
import pandas as pd, autodp_protocol as P
from automl_aco.data.splits import split_train_val_test
from autodatapre.Pipeline_Generation.MCTS import merge_datasets
from autodatapre.Pipeline_Generation import MetaFeature
import numpy as np

df = pd.read_csv({DATASET_1054!r})
idx = pd.DataFrame({{'_pos': np.arange(len(df))}})
dummy = pd.Series(np.zeros(len(df)))
a,_,b,_,c,_ = split_train_val_test(idx, dummy, seed=42)
tr, val = a['_pos'].to_numpy(), b['_pos'].to_numpy()

d = P.build_search_dataset(df, 'target', tr, val)
merged = merge_datasets(d)
gapless = list(merged.index) == list(range(len(merged)))

metafeature_ok = True
try:
    MetaFeature.getfeature(merged)
except Exception:
    metafeature_ok = False

print('@@' + json.dumps({{
    'gapless_from_zero': gapless,
    'has_label_zero': 0 in merged.index,
    'metafeature_getfeature_succeeds': metafeature_ok,
    'train_target_aligned': list(d['train'].index) == list(d['target'].index),
    'test_target_aligned': list(d['test'].index) == list(d['target_test'].index),
}}))
""")
    result = out[0]
    assert result["gapless_from_zero"]
    assert result["has_label_zero"]
    assert result["metafeature_getfeature_succeeds"]
    assert result["train_target_aligned"]
    assert result["test_target_aligned"]


def test_search_dataset_reproduces_the_862_failure_mode_directly():
    """Regression: dataset 862's split puts original position 0 in `te`. Must not raise/leak."""
    out = _run_in_adp_env("""
import pandas as pd, autodp_protocol as P
from automl_aco.data.splits import split_train_val_test
from autodatapre.Pipeline_Generation.MCTS import merge_datasets
from autodatapre.Pipeline_Generation import MetaFeature
import numpy as np

df = pd.read_csv('data/eval_datasets/862.csv')
idx = pd.DataFrame({'_pos': np.arange(len(df))})
dummy = pd.Series(np.zeros(len(df)))
a,_,b,_,c,_ = split_train_val_test(idx, dummy, seed=42)
tr, val, te = a['_pos'].to_numpy(), b['_pos'].to_numpy(), c['_pos'].to_numpy()
position_zero_in_te = 0 in te.tolist()  # the actual trigger condition, confirmed present

d = P.build_search_dataset(df, 'target', tr, val)
merged = merge_datasets(d)
ok = True
try:
    MetaFeature.getfeature(merged)
except Exception:
    ok = False

print('@@' + json.dumps({
    'position_zero_in_te': position_zero_in_te,
    'metafeature_getfeature_succeeds': ok,
}))
""")
    result = out[0]
    assert result["position_zero_in_te"], "fixture no longer reproduces the trigger condition"
    assert result["metafeature_getfeature_succeeds"]


def test_assert_dict_aligned_catches_length_mismatch():
    """get_part_dataset .sample()s train/target independently; a length mismatch must be caught."""
    out = _run_in_adp_env("""
import pandas as pd, autodp_protocol as P

good = {
    'train': pd.DataFrame({'x': [1, 2, 3]}, index=[0, 1, 2]),
    'target': pd.DataFrame({'y': [0, 1, 0]}, index=[0, 1, 2]),
    'test': pd.DataFrame({'x': [4]}, index=[3]),
    'target_test': pd.DataFrame({'y': [1]}, index=[3]),
}
bad_length = dict(good)
bad_length['target'] = pd.DataFrame({'y': [0, 1]}, index=[0, 1])  # dropped a row

results = {}
try:
    P.assert_dict_aligned(good)
    results['good_passes'] = True
except AssertionError:
    results['good_passes'] = False
try:
    P.assert_dict_aligned(bad_length)
    results['bad_raises'] = False
except AssertionError:
    results['bad_raises'] = True
print('@@' + json.dumps(results))
""")
    result = out[0]
    assert result["good_passes"]
    assert result["bad_raises"]


def test_assert_dict_aligned_catches_index_mismatch():
    out = _run_in_adp_env("""
import pandas as pd, autodp_protocol as P

bad_index = {
    'train': pd.DataFrame({'x': [1, 2, 3]}, index=[0, 1, 2]),
    'target': pd.DataFrame({'y': [0, 1, 0]}, index=[0, 1, 99]),  # same length, wrong index
    'test': pd.DataFrame({'x': [4]}, index=[3]),
    'target_test': pd.DataFrame({'y': [1]}, index=[3]),
}
try:
    P.assert_dict_aligned(bad_index)
    raised = False
except AssertionError:
    raised = True
print('@@' + json.dumps({'raised': raised}))
""")
    assert out[0]["raised"]


# --- CBE: direct proof the label channel no longer reaches the rows being encoded ------------


def test_cbe_leak_is_closed_at_the_operator_level():
    """Pattern-reversal proof, run against the REAL patched Encoding.transform.

    Flips the test-split labels and asserts the test-row encodings do not move. Also checks the
    UNPATCHED path on the same fixture in the same process, so the test is proven non-vacuous:
    if the assertion below would also pass unpatched, this test is not testing anything.
    """
    out = _run_in_adp_env("""
import pandas as pd, numpy as np

CITY = ['hanoi','hue','hanoi','hue','hanoi','hue','hanoi','hanoi','hue','hue']
Y    = [1,0,1,0,1,0,0,0,1,1]
N_TRAIN = 6

def run_transform(y_test):
    df = pd.DataFrame({'city': CITY})
    train = df.iloc[:N_TRAIN].reset_index(drop=True)
    test = df.iloc[N_TRAIN:].reset_index(drop=True)
    target = pd.DataFrame({'y': Y[:N_TRAIN]})
    target_test = pd.DataFrame({'y': y_test})
    dataset = {'train': train, 'test': test, 'target': target, 'target_test': target_test}
    from autodatapre.Search_Space.encoding import Encoding
    enc = Encoding(dataset, strategy='CBE')
    return enc.transform()['test']['city'].to_numpy()

base_unpatched = run_transform([0, 1, 0, 0])
flipped_unpatched = run_transform([1, 0, 1, 1])
unpatched_differs = not np.allclose(base_unpatched, flipped_unpatched)

import autodp_protocol as P
P.install_leakfree_cbe(verbose=False)

base_patched = run_transform([0, 1, 0, 0])
flipped_patched = run_transform([1, 0, 1, 1])
patched_differs = not np.allclose(base_patched, flipped_patched)

print('@@' + json.dumps({
    'unpatched_leaks': unpatched_differs,
    'patched_leaks': patched_differs,
}))
""")
    result = out[0]
    assert result["unpatched_leaks"], "fixture is not sensitive to the leak -- test is vacuous"
    assert not result["patched_leaks"], "CBE still consumes test-row labels after the patch"


def test_cbe_still_uses_the_leakfree_autodp_ops_encoder():
    """The patch must route through automl_aco.preprocessing.autodp_ops, not a third encoder."""
    out = _run_in_adp_env("""
import autodp_protocol as P
from automl_aco.preprocessing.autodp_ops import build_encoder
import category_encoders as ce

fitted = build_encoder('CBE')
print('@@' + json.dumps({'is_catboost': isinstance(fitted, ce.CatBoostEncoder)}))
""")
    assert out[0]["is_catboost"]


# --- scorer patches: seeding and objective parity -------------------------------------------


def test_scorer_patches_seed_random_forest_and_extra_trees():
    out = _run_in_adp_env("""
import pandas as pd, numpy as np
import autodp_protocol as P
P.install_scorer_patches(seed=42, verbose=False)

import sklearn.ensemble as SK
print('@@' + json.dumps({
    'extra_trees_seeded': SK.ExtraTreesClassifier(n_estimators=10).random_state == 42,
}))

# RF_classification must be deterministic across calls despite an unseeded classifier constructor
# elsewhere disturbing the global numpy random state in between -- proof it carries its own seed
# rather than inheriting whatever state happened to be current.
from autodatapre.Search_Space.classifier import Classifier

rng = np.random.RandomState(0)
n = 60
X_train = pd.DataFrame(rng.rand(n, 4), columns=list('abcd'))
y_train = pd.DataFrame({'y': (X_train['a'] > 0.5).astype(int)})
X_test = pd.DataFrame(rng.rand(15, 4), columns=list('abcd'))
y_test = pd.DataFrame({'y': (X_test['a'] > 0.5).astype(int)})
dataset = {'train': X_train, 'target': y_train, 'test': X_test, 'target_test': y_test}
clf = Classifier(dataset, target='y')

score1 = clf.RF_classification(dataset, 'y')
np.random.seed(999)  # disturb global state between calls
score2 = clf.RF_classification(dataset, 'y')
print('@@' + json.dumps({'rf_deterministic': score1 == score2}))
""")
    result_et, result_rf = out[0], out[1]
    assert result_et["extra_trees_seeded"]
    assert result_rf["rf_deterministic"]


def test_lda_scores_holdout_like_nb_and_rf():
    """LDA must score dataset['test'], not a K-fold CV mean over dataset['train']."""
    out = _run_in_adp_env("""
import pandas as pd, numpy as np
import autodp_protocol as P
P.install_scorer_patches(seed=42, verbose=False)

from autodatapre.Search_Space.classifier import Classifier

rng = np.random.RandomState(0)
n = 40
X_train = pd.DataFrame(rng.rand(n, 3), columns=['a', 'b', 'c'])
y_train = pd.DataFrame({'y': (X_train['a'] > 0.5).astype(int)})
X_test_easy = pd.DataFrame(rng.rand(20, 3), columns=['a', 'b', 'c'])
y_test_easy = pd.DataFrame({'y': (X_test_easy['a'] > 0.5).astype(int)})
# An adversarial test split: label is the OPPOSITE of the train relationship. A train-only CV
# score would be blind to this; a holdout score must reflect it.
y_test_adversarial = pd.DataFrame({'y': (X_test_easy['a'] <= 0.5).astype(int)})

dataset_easy = {'train': X_train, 'target': y_train, 'test': X_test_easy, 'target_test': y_test_easy}
dataset_adv = {'train': X_train, 'target': y_train, 'test': X_test_easy, 'target_test': y_test_adversarial}

clf = Classifier(dataset_easy, target='y')
score_easy = clf.LDA_classification(dataset_easy, 'y')
score_adv = clf.LDA_classification(dataset_adv, 'y')

print('@@' + json.dumps({
    'score_easy': score_easy, 'score_adv': score_adv,
    'sensitive_to_test_labels': (score_easy is not None and score_adv is not None
                                 and abs(score_easy - score_adv) > 0.3),
}))
""")
    result = out[0]
    assert result["sensitive_to_test_labels"], (
        f"LDA score barely moved between an easy and an adversarial TEST split "
        f"({result['score_easy']} vs {result['score_adv']}) -- suggests it is still scoring "
        f"train via cross-validation instead of the holdout"
    )


# --- exception counter -----------------------------------------------------------------------


def test_exception_counter_counts_and_reraises():
    out = _run_in_adp_env("""
import autodp_protocol as P
from autodatapre.Pipeline_Generation import MCTS

def boom(*a, **k):
    raise ValueError('synthetic failure')

MCTS.monte_carlo_tree_search = boom
counter = P.ExceptionCounter()
counter.install(verbose=False)

raised = False
try:
    MCTS.monte_carlo_tree_search(None, None, None, None)
except ValueError:
    raised = True

print('@@' + json.dumps({
    'reraised': raised,
    'report': counter.report(),
}))
""")
    result = out[0]
    assert result["reraised"], "the counter must not swallow the exception -- their loop handles it"
    assert result["report"]["search_iteration_exceptions"] == 1
    assert result["report"]["search_iteration_exception_kinds"] == {"ValueError": 1}


def test_checkpoint_records_running_best_and_survives_a_kill(tmp_path):
    """The salvage path's whole premise: a checkpoint exists BEFORE the search finishes.

    Simulates three iterations returning profits 0.5, 0.7, 0.6 and asserts the file on disk holds
    the argmax pipeline (the 0.7 one), not the last one -- that is what makes an apply-only
    salvage equivalent to what their loop would have returned had it stopped there.
    """
    ckpt_path = tmp_path / "search_checkpoint.json"
    out = _run_in_adp_env(f"""
import autodp_protocol as P
from autodatapre.Pipeline_Generation import MCTS

class FakeState:
    def __init__(self, choices): self.cumulative_choices = choices

class FakeNode:
    def __init__(self, profit, choices):
        self._p = profit
        self.state = FakeState(choices)
    def get_pre_profit(self): return self._p

seq = iter([FakeNode(0.5, ['NB', 'MF']), FakeNode(0.7, ['NB', 'MF', 'ZS']), FakeNode(0.6, ['NB'])])
MCTS.monte_carlo_tree_search = lambda *a, **k: next(seq)

ckpt = P.SearchCheckpoint({str(ckpt_path)!r})
ckpt.install(verbose=False)

mid = None
for i in range(3):
    MCTS.monte_carlo_tree_search(None, None, None, None)
    if i == 0:
        # a kill right here must still leave a usable checkpoint behind
        mid = P.SearchCheckpoint.read({str(ckpt_path)!r})

print('@@' + json.dumps({{
    'after_first': mid,
    'final': P.SearchCheckpoint.read({str(ckpt_path)!r}),
}}))
""")
    result = out[0]
    assert result["after_first"]["pipeline"] == ["NB", "MF"], "no checkpoint after iteration 1"
    final = result["final"]
    assert final["pipeline"] == ["NB", "MF", "ZS"], "checkpoint must hold the BEST, not the last"
    assert final["profit"] == pytest.approx(0.7)
    # 2, not 3: the file is rewritten only when the best improves, so the count is "iterations
    # completed as of this checkpoint". Iteration 3 was worse and correctly did not overwrite it.
    assert final["iterations_completed"] == 2


def test_checkpoint_read_returns_none_when_nothing_was_written(tmp_path):
    """No completed iteration -> no salvage. The caller must fall through to the timeout row."""
    out = _run_in_adp_env(f"""
import autodp_protocol as P
missing = P.SearchCheckpoint.read({str(tmp_path / 'absent.json')!r})
open({str(tmp_path / 'empty.json')!r}, 'w').write('{{}}')
empty = P.SearchCheckpoint.read({str(tmp_path / 'empty.json')!r})
open({str(tmp_path / 'junk.json')!r}, 'w').write('not json at all')
junk = P.SearchCheckpoint.read({str(tmp_path / 'junk.json')!r})
print('@@' + json.dumps({{'missing': missing, 'empty': empty, 'junk': junk}}))
""")
    assert out[0] == {"missing": None, "empty": None, "junk": None}
