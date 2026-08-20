"""AutoDP's 7 metafeatures, computed by AutoDP itself.

A retrained meta-learner corpus is only valid if its `Metafeature.csv` rows are produced by the
SAME computation that AutoDP uses for the query vector at search time -- the 1-NN compares them
directly. A pure-python port was tried first and abandoned: pandas 1.x/2.x disagree on
`is_string_dtype` (their pin returns True for any object dtype) and on `unique()` with NaN, and
sklearn's LabelEncoder orders NaN differently across the two versions. The residual was small
(max relative 4e-3) but a nearest-neighbour lookup can flip on that, silently.

So we shell into .venv-autodp and let their module answer, the same two-environment split
scripts/adp_bench.py already uses. Slower, and correct by construction.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ADP_PYTHON = REPO / ".venv-autodp" / "bin" / "python"

_WORKER = r"""
import warnings, json, sys
warnings.filterwarnings("ignore")
import pandas as pd
from autodatapre.Pipeline_Generation import MetaFeature

out = {}
for path in json.load(sys.stdin):
    try:
        m = MetaFeature.getfeature(pd.read_csv(path))
        v = m.numpy().mean(axis=0)
        out[path] = [float(x) for x in v]
    except Exception as exc:                      # one bad CSV must not lose the batch
        out[path] = {"error": f"{type(exc).__name__}: {exc}"}
print("@@JSON@@" + json.dumps(out))
"""


def batch_dataset_vectors(
    csv_paths: Sequence[Path],
    *,
    adp_python: Path | str = DEFAULT_ADP_PYTHON,
) -> Dict[str, List[float]]:
    """{csv path: 7-vector} computed by AutoDP. Failures map to {"error": ...} instead of raising."""
    adp_python = Path(adp_python)
    if not adp_python.exists():
        raise FileNotFoundError(
            f"AutoDP interpreter not found at {adp_python}. Build it with "
            "`ADP_VENV=... bash scripts/setup_autodp_env.sh`, or pass --adp-python."
        )
    # PYTHONPATH must NOT reach this interpreter. It outranks the venv's own site-packages, so a
    # caller exporting PYTHONPATH=/tmp/aglibs (AutoGluon's numpy, built for 3.12) makes this 3.10
    # venv import the wrong numpy and die with "you should not try to import numpy from its source
    # directory". scripts/adp_bench.py scrubs it for the same reason.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [str(adp_python), "-c", _WORKER],
        input=json.dumps([str(p) for p in csv_paths]),
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"AutoDP metafeature worker failed:\n{proc.stderr[-2000:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):           # their imports print banners; ignore them
            return json.loads(line[len("@@JSON@@"):])
    raise RuntimeError(f"AutoDP metafeature worker produced no result:\n{proc.stdout[-2000:]}")


def as_matrix(vectors: Dict[str, List[float]], order: Sequence[Path]) -> np.ndarray:
    """Stack in the caller's order, so row i of Metafeature.csv is dataset i of label.csv."""
    rows = []
    for p in order:
        v = vectors[str(p)]
        if isinstance(v, dict):
            raise ValueError(f"metafeatures failed for {p}: {v['error']}")
        rows.append(v)
    return np.asarray(rows, dtype=np.float64)
