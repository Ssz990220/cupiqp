#!/usr/bin/env bash
# Same-session comparison for the staged-input update (run from the repository root):
#   bash native/bench/tools/run_staged_comparison.sh <out dir> [repeats]
# For MPC and Trajopt: reference (harness protocol), native C++ default,
# native C++ --staged, native C++ --staged with whole-solve graph + custom SpMV,
# and the harness native backend with --staged. Native C++ runs are checked
# with the benchmark's own KKT checker.
set -euo pipefail
OUT=${1:?usage: run_staged_comparison.sh <out dir> [repeats]}
REPEATS=${2:-20}
BENCH=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
PY=.venv/bin/python
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_utc.txt"
for ds in mpc trajopt; do
  if [ "$ds" = mpc ]; then bundle=$BENCH/qp_benchmark_v2_20261102; else bundle=$BENCH/datasets/trajopt-20261102; fi
  D="$OUT/$ds"
  mkdir -p "$D"
  $PY native/bench/harness/replay_updates.py --backend reference --dataset "$ds" --repeats "$REPEATS" \
    --output "$D/harness-reference" > "$D/harness-reference.log" 2>&1
  $PY native/bench/harness/replay_updates.py --backend native --staged --dataset "$ds" --repeats "$REPEATS" \
    --output "$D/harness-native-staged" > "$D/harness-native-staged.log" 2>&1
  for variant in "default:" "staged:--staged" "staged-devloop-customspmv:--staged --graph-mode device --custom-spmv"; do
    name=${variant%%:*}
    flags=${variant#*:}
    # shellcheck disable=SC2086
    build/native/bench/cupiqp_native_replay --data "native/bench/data/$ds" --out "$D/native-cpp-$name" \
      --repeats "$REPEATS" $flags > "$D/native-cpp-$name.log" 2>&1
    $PY -I native/bench/tools/check_solutions.py --results "$D/native-cpp-$name" --bundle "$bundle" \
      >> "$D/native-cpp-$name.log" 2>&1
  done
done
date -u +%FT%TZ > "$OUT/finished_utc.txt"
$PY native/bench/tools/check_clocks.py --seconds 15 --out "$OUT/clocks-after-run.json" < /dev/null > /dev/null 2>&1 || true
echo DONE
