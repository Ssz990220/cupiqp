"""Summarize an optimization ladder run (native/bench/tools/run_ladder.sh).

Prints and writes <dir>/ladder.md: per dataset and step, KKT pass counts,
iterations, end-to-end wall p50/P95, update p50, solve p50, change vs the
previous step and vs the reference. gpu_resident mode. An extra row projects
the reduced-KKT backend (not integrated) from the measured per-iteration
linear-algebra savings of native/bench/results/reduced-kkt-20261103 and is
labelled as an estimate.
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
]
# Measured in reduced-kkt-20261103 (graph replay, us per IPM iteration):
# cuDSS 1 factor + 2 solves vs reduced (sparse assembly skipping zeros + tiled Cholesky + 2 solves).
REDUCED_SAVING_US = {"mpc": 1040 - 493, "trajopt": 995 - 677}


def main():
    root = Path(sys.argv[1])
    out = ["# Optimization ladder (same session, gpu_resident, 20 repeats)", ""]
    for ds in ("mpc", "trajopt"):
        out += [f"## {ds.upper()}", "",
                "| Step | KKT@1e-6 | iters p50/max | end-to-end p50 / P95 (ms) | update p50 | solve p50 | vs previous | vs reference |",
                "|---|---|---|---|---|---|---|---|"]
        ref = prev = None
        last = None
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
            dprev = "" if prev is None else f"{100 * (p50 / prev - 1):+.1f}%"
            out.append(f"| {label} | {ok}/{len(rows)} | {np.median(it):.1f}/{max(it)} | {p50:.2f} / {p95:.2f} | "
                       f"{np.median([r['update_event_ms'] for r in rs]):.2f} | {np.median([r['solve_event_ms'] for r in rs]):.2f} | "
                       f"{dprev} | {ref / p50:.2f}x |")
            prev = p50
            last = (p50, np.median(it))
        if last:
            p50, it = last
            est = p50 - REDUCED_SAVING_US[ds] * (it + 1) / 1000.0
            out.append(f"| *Estimate*: + reduced KKT solve (not integrated) | - | - | *~{est:.1f}* | | | "
                       f"*{100 * (est / p50 - 1):+.0f}%* | *{ref / est:.2f}x* |")
        out.append("")
    out.append(f"Estimate: measured best step minus ({REDUCED_SAVING_US} us per factorization) x (iterations + 1), "
               "from native/bench/results/reduced-kkt-20261103; not an end-to-end measurement.")
    text = "\n".join(out) + "\n"
    (root / "ladder.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
