#!/usr/bin/env bash
# Third tuning sweep (place run): neighbourhood of the best x0_duals safeguard.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/Projects/cupiqp-native/.venv/bin/python"
OUT="${1:?output dir}"
S=()
for ss in 3e-2 1e-1 3e-1 1; do for zs in 1e-5 1e-4 1e-3; do
  S+=("x0_duals:s_min=$ss,z_min=$zs")
done; done
"$PY" "$HERE/run_study.py" --out "$OUT" --reps 2 --runs 29-79b3-place-6f87a360 --strategies cold "${S[@]}"
