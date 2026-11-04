#!/usr/bin/env python3
"""Compare real-state accuracy JSONL files (reduced_kkt_sweep --data) or the
synthetic sweep (--synthetic): max / median per metric, and the per-state
ratio new / old. Usage: accuracy_compare.py <old.jsonl> [<new.jsonl>]"""
import json
import statistics
import sys


def load(p):
    return [json.loads(l) for l in open(p) if l.strip().startswith("{")]


old = load(sys.argv[1])
new = load(sys.argv[2]) if len(sys.argv) > 2 else None
keys = [k for k in ("res_random", "cw_random", "res_residual", "cw_residual", "res", "cw") if k in old[0]]
for k in keys:
    o = [r[k] for r in old]
    line = f"{k:13s} old max {max(o):.3e} median {statistics.median(o):.3e}"
    if new:
        nv = [r[k] for r in new]
        ratios = [b / a if a > 0 else (1.0 if b == a else float('inf')) for a, b in zip(o, nv)]
        line += (f" | new max {max(nv):.3e} median {statistics.median(nv):.3e}"
                 f" | new/old per state: max {max(ratios):.2f} median {statistics.median(ratios):.2f}")
    print(line)
