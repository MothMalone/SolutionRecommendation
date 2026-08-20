"""The corpus builder must get its metafeatures from AutoDP itself.

Guards the two things that would silently invalidate a retrained corpus: the vector must match
what AutoDP computes at search time, and row order must follow the caller's dataset order (row i
of Metafeature.csv is dataset i of label.csv -- their 1-NN indexes by position).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
ADP_PY = REPO / ".venv-autodp" / "bin" / "python"

pytestmark = pytest.mark.skipif(not ADP_PY.exists(), reason="AutoDP venv not built")
sys.path.insert(0, str(REPO / "src"))


def _fixtures():
    pkg = REPO / ".venv-autodp/lib/python3.10/site-packages/autodatapre/datasets"
    return sorted(pkg.glob("*.csv")) + sorted((REPO / "data" / "eval_datasets").glob("*.csv"))[:3]


def _reference(csv: Path) -> list:
    code = (
        "import warnings,json;warnings.filterwarnings('ignore')\n"
        "import pandas as pd\n"
        "from autodatapre.Pipeline_Generation import MetaFeature\n"
        f"m=MetaFeature.getfeature(pd.read_csv(r'{csv}'))\n"
        "print(json.dumps(m.numpy().mean(axis=0).tolist()))"
    )
    out = subprocess.run([str(ADP_PY), "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_batch_matches_autodp():
    from automl_aco.adp_metafeatures import batch_dataset_vectors

    files = _fixtures()
    got = batch_dataset_vectors(files, adp_python=ADP_PY)
    for csv in files:
        np.testing.assert_allclose(
            np.asarray(got[str(csv)], dtype=np.float32),
            np.asarray(_reference(csv), dtype=np.float32),
            rtol=1e-6, atol=1e-6,
            err_msg=f"metafeature mismatch on {csv.name}",
        )


def test_as_matrix_preserves_caller_order():
    from automl_aco.adp_metafeatures import as_matrix, batch_dataset_vectors

    files = _fixtures()[:3]
    vectors = batch_dataset_vectors(files, adp_python=ADP_PY)
    forward = as_matrix(vectors, files)
    reverse = as_matrix(vectors, list(reversed(files)))

    assert forward.shape == (3, 7)
    np.testing.assert_allclose(forward, reverse[::-1])


def test_failure_is_reported_not_raised(tmp_path):
    from automl_aco.adp_metafeatures import as_matrix, batch_dataset_vectors

    bad = tmp_path / "not_a_table.csv"
    bad.write_text('"unclosed\n')
    got = batch_dataset_vectors([bad], adp_python=ADP_PY)
    assert isinstance(got[str(bad)], dict) and "error" in got[str(bad)]
    with pytest.raises(ValueError, match="metafeatures failed"):
        as_matrix(got, [bad])


def test_worker_ignores_a_hostile_pythonpath(monkeypatch):
    """PYTHONPATH must not reach the AutoDP venv.

    On Kaggle the cell exports PYTHONPATH=/tmp/aglibs so AutoGluon is importable. That path holds
    numpy built for python 3.12 and outranks the venv's own site-packages, so the 3.10 worker died
    with "you should not try to import numpy from its source directory" -- after the whole corpus
    had already been scored. adp_bench.py scrubs it; this worker must too.
    """
    from automl_aco.adp_metafeatures import batch_dataset_vectors

    monkeypatch.setenv("PYTHONPATH", "/nonexistent/hostile/path")
    files = _fixtures()[:1]
    got = batch_dataset_vectors(files, adp_python=ADP_PY)
    assert isinstance(got[str(files[0])], list), got[str(files[0])]
    assert len(got[str(files[0])]) == 7
