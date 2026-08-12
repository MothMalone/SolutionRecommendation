#!/usr/bin/env bash
# Build .venv-autodp: the pinned python 3.10 environment for the AutoDP baseline.
#
# Kept apart from the main environment on purpose -- AutoDP needs numpy<1.24 / pandas<2.0, which
# AutoGluon cannot run on. See requirements-autodp.txt for why each pin is what it is.
#
# Needs a python 3.10 interpreter. It looks for one in this order:
#   1. $ADP_PYTHON if you set it
#   2. uv (installs one automatically; this is the Kaggle path -- Kaggle ships python 3.11, and
#      several of these pins have no 3.11 wheels)
#   3. a system python3.10
#   4. conda
#
# Usage:  bash scripts/setup_autodp_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv-autodp"

if [ -x "$VENV/bin/python" ]; then
  echo "[setup] $VENV already exists; verifying ..."
  if "$VENV/bin/python" -c "import autodatapre" 2>/dev/null; then
    echo "[setup] OK -- AutoDP already installed. Delete $VENV to rebuild."
    exit 0
  fi
  echo "[setup] incomplete env, rebuilding"
  rm -rf "$VENV"
fi

make_venv() {
  if [ -n "${ADP_PYTHON:-}" ]; then
    echo "[setup] using \$ADP_PYTHON=$ADP_PYTHON"
    "$ADP_PYTHON" -m venv "$VENV" && return 0
  fi
  if command -v uv >/dev/null 2>&1 || pip install -q uv; then
    echo "[setup] using uv (fetches a managed CPython 3.10 if needed)"
    # --seed: uv venvs ship without pip, and the rest of this script drives pip directly.
    uv venv --seed --python 3.10 "$VENV" && return 0
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    echo "[setup] using system python3.10"
    python3.10 -m venv "$VENV" && return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "[setup] using conda to create a python 3.10 prefix"
    conda create -y -p "$VENV" python=3.10 && return 0
  fi
  echo "[setup] FATAL: no python 3.10 available. Set ADP_PYTHON=/path/to/python3.10 and rerun." >&2
  return 1
}
make_venv

PY="$VENV/bin/python"
"$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "[setup] no pip in the venv, bootstrapping it"
  curl -sS https://bootstrap.pypa.io/get-pip.py | "$PY" - >/dev/null
fi
"$PY" -m pip install -q --upgrade pip wheel

echo "[setup] installing autodatapre without its unbuildable hard pins ..."
"$PY" -m pip install -q --no-deps autodatapre==0.1.12

echo "[setup] installing the pinned runtime ..."
grep -vE '^\s*(#|$)' requirements-autodp.txt | grep -vE '^(autodatapre|torch)' | \
  xargs "$PY" -m pip install -q

# AutoDP uses torch only for tensor math and its attention module, so take the CPU build: the
# default wheel drags in ~900MB of CUDA libraries that never get used.
echo "[setup] installing CPU torch ..."
"$PY" -m pip install -q torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu || \
  "$PY" -m pip install -q torch==1.13.1

# py-stringmatching / py-stringsimjoin build from source on python >= 3.10 and their setup.py
# imports pip, which is absent inside pip's isolated build env -- hence --no-build-isolation.
echo "[setup] building the string-similarity deps from source ..."
"$PY" -m pip install -q "setuptools<70" cython
"$PY" -m pip install -q --no-build-isolation "py-stringmatching==0.4.3"
"$PY" -m pip install -q --no-build-isolation "py-stringsimjoin==0.1.0"

echo "[setup] verifying ..."
MPLBACKEND=Agg "$PY" - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import numpy, pandas, sklearn, torch
from autodatapre.Pipeline_Generation import MCTS, MCTS_DATA  # exercises every operator import
print(f"  numpy {numpy.__version__} | pandas {pandas.__version__} | "
      f"sklearn {sklearn.__version__} | torch {torch.__version__}")
print("  AutoDP imports cleanly.")
EOF
echo "[setup] done -> $VENV"
