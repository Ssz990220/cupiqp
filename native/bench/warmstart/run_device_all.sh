#!/usr/bin/env bash
# Phase-2 experiments on the TrajOpt SQP sequences (device warm start).
# Usage (repo root): native/bench/warmstart/run_device_all.sh <stage> [out root]
#   stage: main | robust | ring | all
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUT="${2:-$ROOT/native/bench/results/warmstart-device-20261104}"
export CUPIQP_NATIVE_BUILD="${CUPIQP_NATIVE_BUILD:-$ROOT/build/native-ws}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}"
STAGE="${1:-all}"
run() { "$PY" "$HERE/run_device.py" "$@" 2>&1 | grep -v -E "^ |Warp|Devices|Kernel"; }

if [ "$STAGE" = main ] || [ "$STAGE" = all ]; then
  rm -f "$OUT/main/records.jsonl"
  run --out "$OUT/main" --passes 4 --configs cold dev dev:match=pair_block dev:match=pair dev:match=pair_global dev:match=label \
    dev:s_min=1e-4 dev:match=pair,y=0
fi
if [ "$STAGE" = robust ] || [ "$STAGE" = all ]; then
  rm -f "$OUT/robust/records.jsonl"
  cfgs=(cold dev:match=pair)
  for kind in unmatched mismatch; do
    for frac in 0.1 0.3 0.6 1.0; do
      for seed in 0 1 2; do
        cfgs+=("dev:match=pair,corrupt=$kind,frac=$frac,seed=$seed")
      done
    done
  done
  run --out "$OUT/robust" --passes 1 --configs "${cfgs[@]}"
fi
if [ "$STAGE" = ring ] || [ "$STAGE" = all ]; then
  rm -f "$OUT/ring/records.jsonl"
  cfgs=(cold dev:match=pair)
  for K in 4 8 12; do
    for back in 0 1 $((K / 2)) $K; do
      cfgs+=("dev:match=pair,K=$K,back=$back")
    done
  done
  cfgs+=("dev:match=pair,K=8,back=8,s_min=1e-2,z_min=1e-8" "dev:match=pair,K=8,back=4,s_min=1e-2,z_min=1e-8")
  run --out "$OUT/ring" --passes 4 --configs "${cfgs[@]}"
fi
