"""Summarize an optimization ladder run (native/bench/tools/run_ladder.sh).

Prints and writes <dir>/ladder.md: per dataset and step, KKT pass counts,
iterations, end-to-end wall p50/P95, update p50, solve p50, change vs the
previous step (for the reduced-backend rows: vs the same configuration with
cuDSS) and vs the reference. gpu_resident mode.
Usage: python3 native/bench/tools/summarize_ladder.py <ladder dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

STEPS = [
    ("1-reference", "Reference (Warp, harness protocol)", "trials.jsonl"),
    ("2-native-pre-multiblock", "Native default, before multi-block reductions", "checked-trials.jsonl"),
    ("3-native-default", "+ multi-block reductions (current default)", "checked-trials.jsonl"),
    ("4-staged", "+ staged preallocated inputs (update_staged)", "checked-trials.jsonl"),
    ("5-staged-devloop", "+ whole-solve CUDA graph (opt-in)", "checked-trials.jsonl"),
    ("6-staged-devloop-customspmv", "+ custom SpMV (opt-in)", "checked-trials.jsonl"),
    ("7-native-default-reduced", "Native default + reduced KKT backend (opt-in)", "checked-trials.jsonl"),
    ("8-staged-devloop-customspmv-reduced", "staged + whole-solve graph + custom SpMV + reduced KKT backend",
     "checked-trials.jsonl"),
]
# Row compared in "vs previous" (default: the row above).
BASE = {"7-native-default-reduced": "3-native-default",
        "8-staged-devloop-customspmv-reduced": "6-staged-devloop-customspmv"}


def main():
    root = Path(sys.argv[1])
    out = ["# Optimization ladder (same session, gpu_resident, 20 repeats)", ""]
    for ds in ("mpc", "trajopt"):
        out += [f"## {ds.upper()}", "",
                "| Step | KKT@1e-6 | iters p50/max/mean | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |",
                "|---|---|---|---|---|---|---|---|"]
        ref = prev = None
        p50_of = {}
        for key, label, fname in STEPS:
            f = root / ds / key / fname
            if not f.exists():
                out.append(f"| {label} | missing | | | | | | |")
                continue
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            rs = [r for r in rows if r["mode"] == "gpu_resident"]
            w = np.array([r["cpu_x_ready_wall_ms"] for r in rs])
            p50, p95 = np.percentile(w, 50), np.percentile(w, 95)
            ok = sum(r["strict_valid"] for r in rows)
            it = [r["iterations"] for r in rs]
            ref = p50 if ref is None else ref
            base = p50_of.get(BASE.get(key), prev)
            dprev = "" if base is None else f"{100 * (p50 / base - 1):+.1f}%"
            if key in BASE:
                dprev += f" (vs step {BASE[key].split('-')[0]})"
            out.append(f"| {label} | {ok}/{len(rows)} | {np.median(it):.1f}/{max(it)}/{np.mean(it):.2f} | {p50:.2f} / {p95:.2f} | "
                       f"{np.median([r['update_event_ms'] for r in rs]):.2f} | {np.median([r['solve_event_ms'] for r in rs]):.2f} | "
                       f"{dprev} | {ref / p50:.2f}x |")
            prev = p50
            p50_of[key] = p50
        out.append("")
    text = "\n".join(out) + "\n"
    (root / "ladder.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
