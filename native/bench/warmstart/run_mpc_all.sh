#!/usr/bin/env bash
# Generality of the device warm start on the MPC replay dataset. Usage (repo root):
#   native/bench/warmstart/run_mpc_all.sh [out root]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUT="${1:-$ROOT/native/bench/results/warmstart-device-20261104}"
export CUPIQP_NATIVE_BUILD="${CUPIQP_NATIVE_BUILD:-$ROOT/build/native-ws}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$HOME/.cache/warp-cupiqp-native}"
cfgs=(cold dev)
for s in 1e-3 1e-2 1; do cfgs+=("dev:s_min=$s"); done
for z in 1e-6 1e-2; do cfgs+=("dev:z_min=$z"); done
cfgs+=("dev:s_min=1e-2,z_min=1e-2" "dev:s_min=1,z_min=1e-2" "dev:s_min=1e-4,z_min=1e-4" "dev:y=0" "dev:K=8,back=8")
for eps in 1e-6 1e-8; do
  rm -f "$OUT/mpc_eps$eps/records.jsonl"
  "$PY" "$HERE/run_mpc.py" --out "$OUT/mpc_eps$eps" --eps "$eps" --passes 4 --configs "${cfgs[@]}" 2>&1 \
    | grep -v -E "^ |Warp|Devices|Kernel"
done
