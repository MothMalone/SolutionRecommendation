"""AutoDP must get a fair shot at ACORec's operator space.

Their shipped task-order prior was learned over a different operator vocabulary: their family 5
is deduplication, used in only 10.2% of their 2000 training pipelines (1955 use dup_null), and it
maps onto our dimensionality_reduction. Transferring it therefore selects pca/svd in roughly one
search in ten while every other family is selected normally -- a WRONG prior, not a missing one,
and one that biases the arm toward ACORec, whose ACO searches all six families every run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ADP_PY = REPO / ".venv-autodp" / "bin" / "python"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def test_canonical_order_covers_every_family():
    """The neutral order must contain all six of our families, exactly once each."""
    from automl_aco.preprocessing.preprocessor import DEFAULT_PREPROCESSOR_ORDER
    import autodp_our_space as A

    assert set(DEFAULT_PREPROCESSOR_ORDER) == set(A.STEP_OPERATORS), (
        "canonical order and operator space disagree on the family set")
    assert len(DEFAULT_PREPROCESSOR_ORDER) == len(set(DEFAULT_PREPROCESSOR_ORDER)) == 6


def test_every_family_maps_to_a_their_index():
    """_neutral_order_fn drops families with no index; none may be dropped."""
    import autodp_our_space as A
    from automl_aco.preprocessing.preprocessor import DEFAULT_PREPROCESSOR_ORDER

    mapped = set(A.THEIR_FAMILY_TO_OUR_STEP.values())
    missing = [s for s in DEFAULT_PREPROCESSOR_ORDER if s not in mapped]
    assert not missing, f"families unreachable in the neutral order: {missing}"


def test_dimensionality_reduction_is_reachable_in_the_alias_fallback():
    """The fallback used to be [1,2,3,4,6], silently making pca/svd unreachable."""
    src = (REPO / "scripts" / "autodp_our_space.py").read_text()
    assert "return indices or [1, 2, 3, 4, 5, 6]" in src, (
        "alias fallback must include family 5 (dimensionality_reduction)")


@pytest.mark.skipif(not ADP_PY.exists(), reason="AutoDP venv not built")
def test_neutral_order_returns_all_six_families_in_canonical_order():
    import json
    import subprocess

    code = (
        "import warnings,sys,json;warnings.filterwarnings('ignore')\n"
        f"sys.path.insert(0,{str(REPO / 'src')!r});sys.path.insert(0,{str(REPO / 'scripts')!r})\n"
        "import pandas as pd, autodp_our_space as A\n"
        "A.install(family_order='all')\n"
        "from autodatapre.Pipeline_Generation import MCTS\n"
        f"o=MCTS.get_CLA_meta_task_order(pd.read_csv({str(REPO / 'data' / 'eval_datasets' / '1066.csv')!r}))\n"
        "print('@@'+json.dumps([f[0].split(':')[0].replace('AR_','') for f in o[1:]]))"
    )
    out = subprocess.run([str(ADP_PY), "-c", code], capture_output=True, text=True, check=True)
    got = json.loads([l for l in out.stdout.splitlines() if l.startswith("@@")][0][2:])

    from automl_aco.preprocessing.preprocessor import DEFAULT_PREPROCESSOR_ORDER
    assert got == list(DEFAULT_PREPROCESSOR_ORDER), got
