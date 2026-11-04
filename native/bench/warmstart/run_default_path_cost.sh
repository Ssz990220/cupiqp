#!/usr/bin/env bash
# Default-path cost of the hook: alternate cold solves of all QPs between the
# base build (a9f74da) and the hook build. Usage: run_default_path_cost.sh BASE_BUILD HOOK_BUILD OUT_DIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/Projects/cupiqp-native/.venv/bin/python"
BASE="$1"; HOOK="$2"; OUT="$3"
mkdir -p "$OUT"
for round in 1 2 3; do
  for tag in base hook; do
    if [ "$tag" = base ]; then B="$BASE"; else B="$HOOK"; fi
    echo "round $round $tag"
    CUPIQP_NATIVE_BUILD="$B" "$PY" "$HERE/default_path_cost.py" --reps 5 --out "$OUT/cold_${tag}_${round}.json" 2>&1 | grep -v -E "^ |Warp|Devices|Kernel"
  done
done
