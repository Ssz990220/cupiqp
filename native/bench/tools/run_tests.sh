#!/usr/bin/env bash
# ctest + native pytest + upstream test_sparse_solver.py with both KKT
# backends (run from the repository root). Prints the summary lines.
set -uo pipefail
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
PY=${PY:-$HOME/Projects/cupiqp-native/.venv/bin/python}
(cd build/native && ctest -j 4 2>&1 | grep -E "tests passed|tests failed|Failed|\*\*\*")
CUPIQP_NATIVE_BUILD=$PWD/build/native $PY -m pytest -q -x native/tests/python 2>&1 | tail -1
for be in cudss reduced; do
  echo -n "upstream $be: "
  CUPIQP_NATIVE_KKT_BACKEND=$be PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$PWD/build/native \
    $PY -m pytest -p upstream_native -q tests/solver/test_sparse_solver.py 2>&1 | tail -1
done
