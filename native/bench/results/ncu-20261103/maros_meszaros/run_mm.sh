#!/usr/bin/env bash
# One upstream Maros-Meszaros run on the native solver with the reduced KKT
# backend, per-test durations included (run from the repository root):
#   bash native/bench/results/ncu-20261103/maros_meszaros/run_mm.sh <build dir> <label> <run index>
set -uo pipefail
BUILD=${1:?build dir}
LABEL=${2:?label}
RUN=${3:?run index}
D=native/bench/results/ncu-20261103/maros_meszaros
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
start=$(date +%s)
CUPIQP_NATIVE_KKT_BACKEND=reduced PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$BUILD \
  $HOME/Projects/cupiqp-native/.venv/bin/python -m pytest -p upstream_native -p no:cacheprovider -v --tb=line -rfs \
  --durations=0 tests/solver/test_maros_meszaros.py > "$D/$LABEL-run$RUN.log" 2>&1
echo "exit $? wall_s $(( $(date +%s) - start ))" >> "$D/$LABEL-run$RUN.log"
