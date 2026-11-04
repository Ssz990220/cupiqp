#!/usr/bin/env bash
# Reduced-backend ladder rows of run_ladder.sh plus the cuDSS default row as
# a control and the whole-solve graph evaluated as the Auto choice for the
# reduced backend (run from the repository root):
#   bash native/bench/tools/run_ladder_reduced.sh <out dir> [repeats]
# Rows: 3-native-default (cuDSS, control), 7-native-default-reduced,
# 7d-native-default-reduced-devloop (default settings, reduced backend,
# --graph-mode device: what Auto would pick if it chose the whole-solve graph
# for kkt_backend = Reduced), 8-staged-devloop-customspmv-reduced.
# Every run is checked with check_solutions.py at 1e-6. DATA as in run_ladder.sh.
set -euo pipefail
OUT=${1:?usage: run_ladder_reduced.sh <out dir> [repeats]}
REPEATS=${2:-20}
BENCH=/mnt/c/Users/91354/Projects/gpu-qp-benchmark
PY=${PY:-$HOME/Projects/cupiqp-native/.venv/bin/python}
NEW=build/native/bench/cupiqp_native_replay
DATA=${DATA:-$HOME/Projects/cupiqp-native/native/bench/data}
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_utc.txt"
git rev-parse HEAD > "$OUT/native_commit.txt"
for ds in mpc trajopt; do
  if [ "$ds" = mpc ]; then bundle=$BENCH/qp_benchmark_v2_20261102; else bundle=$BENCH/datasets/trajopt-20261102; fi
  D="$OUT/$ds"
  mkdir -p "$D"
  run() {  # name flags...
    local name=$1
    shift
    "$NEW" --data "$DATA/$ds" --out "$D/$name" --repeats "$REPEATS" "$@" > "$D/$name.log" 2>&1
    $PY -I native/bench/tools/check_solutions.py --results "$D/$name" --bundle "$bundle" >> "$D/$name.log" 2>&1
  }
  run 3-native-default
  run 7-native-default-reduced --kkt-backend reduced
  run 7d-native-default-reduced-devloop --kkt-backend reduced --graph-mode device
  run 8-staged-devloop-customspmv-reduced --staged --graph-mode device --custom-spmv --kkt-backend reduced
done
date -u +%FT%TZ > "$OUT/finished_utc.txt"
echo DONE
