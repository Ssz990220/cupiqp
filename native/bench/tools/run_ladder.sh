#!/usr/bin/env bash
# Same-session optimization ladder (run from the repository root):
#   bash native/bench/tools/run_ladder.sh <out dir> <pre-multiblock replay binary> [repeats]
# Steps, each adding one optimization: reference (harness protocol), native
# default before multi-block reductions, native default (current), + staged
# inputs, + whole-solve graph, + custom SpMV; then the reduced KKT backend on
# the default configuration and on the fastest one (staged + whole-solve
# graph + custom SpMV). Every native run is checked with the benchmark's own
# KKT checker. MPC and Trajopt. DATA (default native/bench/data) is the
# snapshot export directory.
set -euo pipefail
OUT=${1:?usage: run_ladder.sh <out dir> <pre-multiblock replay binary> [repeats]}
OLD=${2:?pre-multiblock cupiqp_native_replay binary}
REPEATS=${3:-20}
BENCH=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
PY=.venv/bin/python
NEW=build/native/bench/cupiqp_native_replay
DATA=${DATA:-native/bench/data}
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_utc.txt"
git rev-parse HEAD > "$OUT/native_commit.txt"
for ds in mpc trajopt; do
  if [ "$ds" = mpc ]; then bundle=$BENCH/qp_benchmark_v2_20261102; else bundle=$BENCH/datasets/trajopt-20261102; fi
  D="$OUT/$ds"
  mkdir -p "$D"
  $PY native/bench/harness/replay_updates.py --backend reference --dataset "$ds" --repeats "$REPEATS" \
    --output "$D/1-reference" > "$D/1-reference.log" 2>&1
  run() {  # name binary flags...
    local name=$1 bin=$2
    shift 2
    "$bin" --data "$DATA/$ds" --out "$D/$name" --repeats "$REPEATS" "$@" > "$D/$name.log" 2>&1
    $PY -I native/bench/tools/check_solutions.py --results "$D/$name" --bundle "$bundle" >> "$D/$name.log" 2>&1
  }
  run 2-native-pre-multiblock "$OLD"
  run 3-native-default "$NEW"
  run 4-staged "$NEW" --staged
  run 5-staged-devloop "$NEW" --staged --graph-mode device
  run 6-staged-devloop-customspmv "$NEW" --staged --graph-mode device --custom-spmv
  run 7-native-default-reduced "$NEW" --kkt-backend reduced
  run 8-staged-devloop-customspmv-reduced "$NEW" --staged --graph-mode device --custom-spmv --kkt-backend reduced
done
date -u +%FT%TZ > "$OUT/finished_utc.txt"
$PY native/bench/tools/check_clocks.py --seconds 15 --out "$OUT/clocks-after-run.json" < /dev/null > /dev/null 2>&1 || true
echo DONE
