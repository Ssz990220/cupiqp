#!/usr/bin/env python3
"""Compare two measure_step.sh folders: microbench ops, the synthetic sweep
(factor / solve / iteration us and accuracy per case), real-state accuracy.
Usage: compare_step.py <before dir> <after dir>"""
import json
import statistics
import sys
from pathlib import Path


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip().startswith("{")]


a, b = Path(sys.argv[1]), Path(sys.argv[2])
print("## microbench (graph replay median, us)\n\n| dataset | state | op | before | after | change |\n|---|---|---|---|---|---|")
for ds in ("mpc", "trajopt"):
    A = {(r["label"], r["op"]): r["graph_us"] for r in jl(a / f"microbench_{ds}.jsonl")}
    for r in jl(b / f"microbench_{ds}.jsonl"):
        k = (r["label"], r["op"])
        if k in A:
            print(f"| {ds} | {k[0]} | {k[1]} | {A[k]:.1f} | {r['graph_us']:.1f} | {100 * (r['graph_us'] / A[k] - 1):+.1f}% |")
print("\n## synthetic sweep\n\n| case | n | p | m | factor us | solve us | iter us (before -> after, change) | res before/after | cw before/after |\n|---|---|---|---|---|---|---|---|---|")
A = {r["case"]: r for r in jl(a / "sweep.jsonl")}
faster = same = slower = 0
for r in jl(b / "sweep.jsonl"):
    o = A.get(r["case"])
    if not o:
        continue
    ch = r["iter_us"] / o["iter_us"] - 1
    if ch < -0.03:
        faster += 1
    elif ch > 0.03:
        slower += 1
    else:
        same += 1
    print(f"| {r['case']} | {r['n']} | {r['p']} | {r['m']} | {o['factor_us']:.1f} -> {r['factor_us']:.1f} | "
          f"{o['solve_us']:.1f} -> {r['solve_us']:.1f} | {o['iter_us']:.1f} -> {r['iter_us']:.1f} ({100 * ch:+.1f}%) | "
          f"{o['res']:.1e} / {r['res']:.1e} | {o['cw']:.1e} / {r['cw']:.1e} |")
print(f"\nsweep iteration time: {faster} faster, {same} same (within 3%), {slower} slower")
print("\n## real-state accuracy (all IPM iterations of case 0)\n")
for ds in ("mpc", "trajopt"):
    O, N = jl(a / f"accuracy_{ds}.jsonl"), jl(b / f"accuracy_{ds}.jsonl")
    for k in ("res_random", "cw_random", "res_residual", "cw_residual"):
        o = [r[k] for r in O]
        n = [r[k] for r in N]
        rat = [y / x if x > 0 else 1.0 for x, y in zip(o, n)]
        print(f"- {ds} {k}: before max {max(o):.2e} median {statistics.median(o):.2e}; after max {max(n):.2e} "
              f"median {statistics.median(n):.2e}; after/before per state median {statistics.median(rat):.2f} max {max(rat):.2f}")
