#!/usr/bin/env bash
# Kernel-level Nsight Systems profile of one warmed native update+solve, with
# CUDA graph nodes traced individually, plus the per-iteration breakdown.
# Run from the repository root:
#   bash native/bench/tools/profile_iteration.sh <out dir> [mpc|trajopt] [iteration|device|none] [tag suffix] [extra flags]
# Extra flags go to profile_update_solve.py (e.g. --eps 1e-6).
set -euo pipefail
OUT=${1:?usage: profile_iteration.sh <out dir> [mpc|trajopt] [graph mode]}
DATASET=${2:-mpc}
MODE=${3:-iteration}
mkdir -p "$OUT"
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
SUFFIX=${4:-}
tag="$DATASET-native-$MODE${SUFFIX:+-$SUFFIX}"
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda --cuda-graph-trace=node \
  --force-overwrite=true -o "$OUT/$tag" \
  .venv/bin/python native/bench/harness/profile_update_solve.py --backend native --graph-mode "$MODE" \
    --dataset "$DATASET" --output "$OUT/$tag-out" "${@:5}" > "$OUT/$tag-console.txt" 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv --force-export=true -o "$OUT/$tag" \
  "$OUT/$tag.nsys-rep" > /dev/null 2>&1
python3 native/bench/tools/iteration_profile.py "$OUT/${tag}_cuda_gpu_trace.csv" --markdown "$OUT/$tag-iteration.md"
