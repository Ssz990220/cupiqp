#!/usr/bin/env bash
# Same-session optimization ladder (run from the repository root):
#   bash native/bench/tools/run_ladder.sh <out dir> <previous replay binary> [repeats]
# Rows: reference (harness protocol), a previous native replay binary with its
# defaults (for a before/after comparison), the native default (whole-solve
# graph), the native default with the per-iteration graph, and + staged
# inputs. Every native run is checked with the benchmark's own KKT checker.
# MPC and Trajopt. DATA (default native/bench/data) is the snapshot export
# directory; BUILD (default build/native) the CMake build directory. Ladders
# recorded before 2026-11-04 had cuDSS / cuSPARSE rows (results/ladder-*).
set -euo pipefail
OUT=${1:?usage: run_ladder.sh <out dir> <previous replay binary> [repeats]}
OLD=${2:?previous cupiqp_native_replay binary}
REPEATS=${3:-20}
BENCH=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}
PY=.venv/bin/python
NEW=${BUILD:-build/native}/bench/cupiqp_native_replay
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
  run 2-native-previous "$OLD"
  run 3-native-default "$NEW"
  run 4-native-iteration-graph "$NEW" --graph-mode iteration
  run 5-staged "$NEW" --staged
done
date -u +%FT%TZ > "$OUT/finished_utc.txt"
$PY native/bench/tools/check_clocks.py --seconds 15 --out "$OUT/clocks-after-run.json" < /dev/null > /dev/null 2>&1 || true
echo DONE
