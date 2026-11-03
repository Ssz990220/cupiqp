#!/usr/bin/env bash
# Nsight Systems profile of one warmed update+solve for each configuration,
# plus the CUDA API summary (cuda_api_sum CSV). Run from the repository root:
#   bash native/bench/tools/profile_all.sh <out dir> [mpc|trajopt]
set -euo pipefail
OUT=${1:?usage: profile_all.sh <out dir> [mpc|trajopt]}
DATASET=${2:-mpc}
mkdir -p "$OUT"
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
for cfg in "reference iteration" "native iteration" "native device"; do
  set -- $cfg
  tag="$DATASET-$1-$2"
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,osrt \
    --force-overwrite=true -o "$OUT/$tag" \
    .venv/bin/python native/bench/harness/profile_update_solve.py --backend "$1" --graph-mode "$2" \
      --dataset "$DATASET" --output "$OUT/$tag-out" > "$OUT/$tag-console.txt" 2>&1
  nsys stats --report cuda_api_sum --format csv --force-export=true -o "$OUT/$tag" "$OUT/$tag.nsys-rep" > /dev/null 2>&1
  echo "== $tag"
done
