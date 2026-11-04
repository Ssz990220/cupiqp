#!/usr/bin/env bash
# One upstream Maros-Meszaros run on the native solver (run from the repository root):
#   bash native/bench/results/reduced-kkt-backend-20261103/maros_meszaros/run_mm.sh <cudss|reduced> <run index>
set -uo pipefail
BE=${1:?backend}
RUN=${2:?run index}
D=native/bench/results/reduced-kkt-backend-20261103/maros_meszaros
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
start=$(date +%s)
CUPIQP_NATIVE_KKT_BACKEND=$BE PYTHONPATH=native/tests/python CUPIQP_NATIVE_BUILD=$PWD/build/native \
  .venv/bin/python -m pytest -p upstream_native -p no:cacheprovider -v --tb=line -rfs tests/solver/test_maros_meszaros.py \
  > "$D/$BE-run$RUN.log" 2>&1
echo "exit $? wall_s $(( $(date +%s) - start ))" >> "$D/$BE-run$RUN.log"
