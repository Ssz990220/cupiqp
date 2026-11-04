#!/usr/bin/env bash
# Second tuning sweep (place run): asymmetric safeguards, active-set levels,
# Gondzio-Grothey earlier iterates.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/Projects/cupiqp-native/.venv/bin/python"
OUT="${1:?output dir}"
S=()
for ss in 1e-3 1e-2 1e-1; do for zs in 1e-8 1e-6 1e-4; do
  S+=("x0_duals:s_min=$ss,z_min=$zs")
done; done
S+=("active_set:small=1e-2,big=1e-1" "active_set:small=1e-3,big=1e-2" "active_set:small=1e-3,big=1e-1")
for back in 4 8 12; do
  S+=("gg_slack:back=$back,s_min=1e-8,z_min=1e-8" "gg_data:back=$back,s_min=1e-2,z_min=1e-8")
done
"$PY" "$HERE/run_study.py" --out "$OUT" --reps 2 --runs 29-79b3-place-6f87a360 --strategies cold "${S[@]}"
