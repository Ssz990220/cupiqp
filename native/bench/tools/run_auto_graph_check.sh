#!/usr/bin/env bash
# End-to-end check of the Auto graph mode (run from the repository root):
#   bash native/bench/tools/run_auto_graph_check.sh <out dir> [repeats]
# Per dataset: cuDSS with Auto (control: one graph per iteration), the reduced
# backend with the per-iteration graph forced, and the reduced backend with
# Auto (whole solve in one graph). Every run is checked with the benchmark's
# own KKT checker.
set -euo pipefail
OUT=${1:?usage: run_auto_graph_check.sh <out dir> [repeats]}
REPEATS=${2:-20}
BENCH=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_utc.txt"
git rev-parse HEAD > "$OUT/native_commit.txt"
for ds in mpc trajopt; do
  if [ "$ds" = mpc ]; then bundle=$BENCH/qp_benchmark_v2_20261102; else bundle=$BENCH/datasets/trajopt-20261102; fi
  for cfg in "cudss-auto:--kkt-backend cudss" \
             "reduced-iteration:--kkt-backend reduced --graph-mode iteration" \
             "reduced-auto:--kkt-backend reduced"; do
    name=${cfg%%:*}
    flags=${cfg#*:}
    # shellcheck disable=SC2086
    build/native/bench/cupiqp_native_replay --data "native/bench/data/$ds" --out "$OUT/$ds/$name" \
      --repeats "$REPEATS" $flags > "$OUT/$ds-$name.log" 2>&1
    .venv/bin/python -I native/bench/tools/check_solutions.py --results "$OUT/$ds/$name" --bundle "$bundle" \
      >> "$OUT/$ds-$name.log" 2>&1
  done
done
date -u +%FT%TZ > "$OUT/finished_utc.txt"
echo DONE
