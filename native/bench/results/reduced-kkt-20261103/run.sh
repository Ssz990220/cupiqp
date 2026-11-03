#!/usr/bin/env bash
# Reproduces the raw data of this folder (run from the worktree root, after building
# cupiqp_native_reduced_kkt_probe). Runs are sequential (one GPU job at a time).
set -euo pipefail
BIN=build/native/bench/cupiqp_native_reduced_kkt_probe
DATA=${DATA:-$HOME/Projects/cupiqp-native/native/bench/data}
OUT=native/bench/results/reduced-kkt-20261103
$BIN --data $DATA/mpc/group_0.bin --label mpc --cases 0,21,42,63 --time-every 4 --reps 100 --out-dir $OUT 2> $OUT/log_mpc.txt
for gk in 0 1 2 3 4 5; do
  $BIN --data $DATA/trajopt/group_$gk.bin --label trajopt_g$gk --cases 0,4 --time-every 4 --reps 100 --out-dir $OUT 2> $OUT/log_trajopt_g$gk.txt
done
