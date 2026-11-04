#!/usr/bin/env bash
# ctest + native pytest + upstream test_sparse_solver.py (run from the
# repository root). BUILD (default build/native) is the CMake build directory.
# Prints the summary lines.
set -uo pipefail
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
PY=${PY:-$HOME/Projects/cupiqp-native/.venv/bin/python}
BUILD=${BUILD:-build/native}
(cd "$BUILD" && ctest -j 4 2>&1 | grep -E "tests passed|tests failed|Failed|\*\*\*")
CUPIQP_NATIVE_BUILD=$PWD/$BUILD $PY -m pytest -q -x native/tests/python 2>&1 | tail -1
echo -n "upstream: "
PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$PWD/$BUILD \
  $PY -m pytest -p upstream_native -q tests/solver/test_sparse_solver.py 2>&1 | tail -1
