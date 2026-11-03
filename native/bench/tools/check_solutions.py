"""Independent KKT check of native replay solutions with the benchmark's own checker.

Reads <results>/solutions-index.jsonl + solutions.bin (x, y, z = z_u - z_l per
timed transition, written by cupiqp_native_replay) and <results>/trials.jsonl,
loads each problem with the benchmark's ``qp_io.load_problem`` and runs
``qp_io.check_solution`` at 1e-6 (strict) and 3e-4, exactly like
gpu-qp-benchmark/benchmark/replay_mpc_updates.py. Writes checked-trials.jsonl
(trials + validation) and summary.json (medians / P95 and pass counts).

Usage:
    .venv/bin/python -I native/bench/tools/check_solutions.py \
        --results <dir> --bundle <qp bundle dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH_ROOT = Path("/mnt/c/Users/91354/Projects/gpu-qp-benchmark")


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def stats(values):
    v = [x for x in values if x is not None]
    return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99),
            "max": float(max(v)) if v else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--qp-io", type=Path, default=BENCH_ROOT / "qp_benchmark_v2_20261102" / "tools")
    args = ap.parse_args()
    sys.path.insert(0, str(args.qp_io))
    from qp_io import check_solution, load_problem  # noqa: E402

    trials = [json.loads(l) for l in (args.results / "trials.jsonl").read_text().splitlines() if l.strip()]
    index = [json.loads(l) for l in (args.results / "solutions-index.jsonl").read_text().splitlines() if l.strip()]
    assert len(trials) == len(index), (len(trials), len(index))
    blob = (args.results / "solutions.bin").read_bytes()
    problems = {}
    out = (args.results / "checked-trials.jsonl").open("w")
    for t, ix in zip(trials, index):
        assert t["to_case"] == ix["case"]
        cid = ix["case"]
        if cid not in problems:
            problems[cid] = load_problem(args.bundle / "cases" / cid)
        n, p, m = ix["n"], ix["p"], ix["m"]
        v = np.frombuffer(blob, dtype="<f8", count=n + p + m, offset=ix["offset"])
        x, y, z = v[:n].copy(), v[n:n + p].copy(), v[n + p:].copy()
        loose = check_solution(problems[cid], x, y, z, 3e-4)
        strict = check_solution(problems[cid], x, y, z, 1e-6)
        solved = t["status"] == "CUPIQP_SOLVED"
        t.update(validation=loose, strict_kkt_pass=bool(strict["kkt_pass"]),
                 valid=bool(solved and loose["kkt_pass"]), strict_valid=bool(solved and strict["kkt_pass"]))
        out.write(json.dumps(t) + "\n")
    out.close()

    summary = {"results": str(args.results), "bundle": str(args.bundle), "timed_transitions": len(trials),
               "setup": json.loads((args.results / "setup.json").read_text()), "modes": {}}
    for mode in sorted({t["mode"] for t in trials}):
        ts = [t for t in trials if t["mode"] == mode]
        summary["modes"][mode] = {
            "samples": len(ts),
            "valid_3e4": sum(t["valid"] for t in ts),
            "valid_1e6": sum(t["strict_valid"] for t in ts),
            "statuses": {s: sum(t["status"] == s for t in ts) for s in sorted({t["status"] for t in ts})},
            **{k: stats([t[k] for t in ts]) for k in ("cpu_x_ready_wall_ms", "upload_event_ms", "update_event_ms",
                                                      "solve_event_ms", "x_download_event_ms",
                                                      "gpu_x_ready_event_ms", "iterations")},
        }
    (args.results / "summary.json").write_text(json.dumps(summary, indent=1))
    for mode, s in summary["modes"].items():
        print(f"{mode}: {s['valid_1e6']}/{s['samples']} pass KKT@1e-6, statuses {s['statuses']}, "
              f"iters p50 {s['iterations']['p50']} max {s['iterations']['max']}, "
              f"wall p50 {s['cpu_x_ready_wall_ms']['p50']:.3f} ms p95 {s['cpu_x_ready_wall_ms']['p95']:.3f}, "
              f"update p50 {s['update_event_ms']['p50']:.3f}, solve p50 {s['solve_event_ms']['p50']:.3f}")


if __name__ == "__main__":
    main()
