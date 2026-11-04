#!/usr/bin/env bash
# Parameter sweep of the warm-start safeguards on the tuning run
# (29-79b3-place); the other run is held out for the final evaluation.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HOME/Projects/cupiqp-native/.venv/bin/python"
OUT="${1:?output dir}"
S=()
for v in 1e-8 1e-6 1e-4 1e-2 1; do
  S+=("x0_duals:s_min=$v,z_min=$v")
done
for zs in 1e-6 1e-2; do for ss in 1e-6 1e-2; do
  S+=("x0_duals:s_min=$ss,z_min=$zs")
done; done
for rd in "rho=1e-8,delta=1e-6" "rho=1e-7,delta=1e-5" "rho=1e-9,delta=1e-7"; do
  S+=("x0_duals:s_min=1e-4,z_min=1e-4@$rd")
done
for mu in 1e-4 1e-3 1e-2 1e-1; do
  S+=("x0_duals_center:mu0=$mu,beta=0.1")
done
for sm in 1e-4 1e-2; do for bg in 1e-2 1; do
  S+=("active_set:small=$sm,big=$bg")
done; done
S+=("full_prev" "full_prev_xdata" "x0_prev_slack" "default_xy_prev_z")
"$PY" "$HERE/run_study.py" --out "$OUT" --reps 2 --runs 29-79b3-place-6f87a360 --strategies cold "${S[@]}"
