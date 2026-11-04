#!/usr/bin/env bash
# End-to-end check of the default configuration (run from the repository root):
#   bash native/bench/tools/run_auto_graph_check.sh <out dir> [repeats]
# Per dataset (MPC, Trajopt): one native C++ replay with the library defaults
# (Auto graph mode = the whole solve in one CUDA graph), checked with the
# benchmark's own KKT checker. BUILD (default build/native) is the CMake
# build directory. Before 2026-11-04 the script ran three configurations per
# dataset (cuDSS Auto, reduced per-iteration graph, reduced Auto).
set -euo pipefail
OUT=${1:?usage: run_auto_graph_check.sh <out dir> [repeats]}
REPEATS=${2:-20}
BUILD=${BUILD:-build/native}
BENCH=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_utc.txt"
git rev-parse HEAD > "$OUT/native_commit.txt"
for ds in mpc trajopt; do
  if [ "$ds" = mpc ]; then bundle=$BENCH/qp_benchmark_v2_20261102; else bundle=$BENCH/datasets/trajopt-20261102; fi
  "$BUILD/bench/cupiqp_native_replay" --data "native/bench/data/$ds" --out "$OUT/$ds/auto" \
    --repeats "$REPEATS" > "$OUT/$ds-auto.log" 2>&1
  .venv/bin/python -I native/bench/tools/check_solutions.py --results "$OUT/$ds/auto" --bundle "$bundle" \
    >> "$OUT/$ds-auto.log" 2>&1
done
date -u +%FT%TZ > "$OUT/finished_utc.txt"
echo DONE
