"""Aggregate records.jsonl of run_study.py into per-strategy tables (markdown)."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", type=Path)
    ap.add_argument("--per-run", action="store_true")
    ap.add_argument("--skip-first", action="store_true", help="cold rows: drop k = 0 (no transition)")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.records)]
    groups = collections.OrderedDict()
    for r in recs:
        if r["strategy"] == "cold" and r["k"] == 0:
            continue  # only transitions k >= 1 are comparable
        key = (r["strategy"], r["run"] if args.per_run else "all")
        groups.setdefault(key, []).append(r)
    cold_ms = {k[1]: np.median([r["ms"] for r in v]) for k, v in groups.items() if k[0] == "cold"}
    fits = {}
    print("| strategy | run | QPs | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup (med ms) | fail | KKT pass | max dx_inf |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (s, run), v in groups.items():
        it = np.array([r["iter"] for r in v]); ms = np.array([r["ms"] for r in v])
        fail = sum(r["status"] != 0 for r in v); kp = sum(r["kkt_pass"] for r in v)
        dx = max(r["dx_inf"] for r in v)
        sp = cold_ms.get(run, np.nan) / np.median(ms)
        A = np.vstack([np.ones_like(it, dtype=float), it]).T
        coef = np.linalg.lstsq(A, ms, rcond=None)[0]
        fits[(s, run)] = coef
        print(f"| {s} | {run} | {len(v)} | {np.median(it):.0f} | {it.mean():.1f} | {it.max()} | {np.median(ms):.2f} | "
              f"{ms.mean():.2f} | {ms.max():.2f} | {sp:.2f} | {fail} | {kp}/{len(v)} | {dx:.1e} |")

    print("\n| strategy | run | fixed ms (intercept) | ms per IPM iteration |")
    print("|---|---|---:|---:|")
    for (s, run), (a, b) in fits.items():
        print(f"| {s} | {run} | {a:.2f} | {b:.3f} |")


if __name__ == "__main__":
    main()
