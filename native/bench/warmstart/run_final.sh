#!/usr/bin/env bash
# Final evaluation on both runs (101 QPs, 99 transitions). Safeguard values
# were tuned on the place run only (sweep_tune*.sh); the pick run is held out.
# Usage: run_final.sh OUT_DIR [--chain]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/Projects/cupiqp-native/.venv/bin/python"
OUT="${1:?output dir}"; shift
S=(
  cold
  # options that exist today (settings only, no start)
  "cold_settings@rho=1e-8,delta=1e-6"
  "cold_settings@rho=1e-4,delta=1e-2"
  "cold_settings@eps_abs=1e-5,eps_duality_gap_abs=1e-5"
  # controls
  "full_prev"
  "full_prev_xdata:s_min=1e-1,z_min=1e-4"
  "x0_prev_slack"
  # x = 0 plus duals of QP k
  "x0_duals:s_min=1e-4,z_min=1e-4"
  "x0_duals:s_min=1e-1,z_min=1e-4"
  "x0_duals:s_min=1e-1,z_min=1e-4,map=label"
  "x0_duals:s_min=1e-1,z_min=1e-4,map=position"
  "x0_duals:s_min=1e-1,z_min=1e-4@rho=1e-8,delta=1e-6"
  "x0_duals_center:mu0=1e-4,beta=0.1"
  "active_set:small=1e-3,big=1e-2"
  "default_xy_prev_z"
  "gg_data:back=8,s_min=1e-2,z_min=1e-8"
)
"$PY" "$HERE/run_study.py" --out "$OUT" --reps 3 "$@" --strategies "${S[@]}"
