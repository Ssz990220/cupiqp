#!/usr/bin/env bash
# Per-optimization measurement (run from the repository root, GPU exclusive):
#   bash native/bench/tools/measure_step.sh <out dir>
# Microbench on the real MPC / Trajopt states, the synthetic generality
# sweep, and the real-state accuracy over every IPM iteration.
set -euo pipefail
OUT=${1:?out dir}
DATA=${DATA:-$HOME/Projects/cupiqp-native/native/bench/data}
B=build/native/bench
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/native_commit.txt"
git diff --stat > "$OUT/uncommitted.txt"
for ds in mpc trajopt; do
  $B/cupiqp_native_reduced_kkt_bench --data $DATA/$ds/group_0.bin --case 0 --reps 300 > "$OUT/microbench_$ds.jsonl"
done
$B/cupiqp_native_reduced_kkt_sweep --synthetic --reps 100 > "$OUT/sweep.jsonl"
for ds in mpc trajopt; do
  $B/cupiqp_native_reduced_kkt_sweep --data $DATA/$ds/group_0.bin --case 0 --label $ds > "$OUT/accuracy_$ds.jsonl"
done
