"""Tables of the phase-2 runs (run_device.py / run_mpc.py records).

Aggregates over the warm-started solves (k >= 1 of every chain): IPM
iterations, solve time (median over passes per QP), failures (status other
than Solved), KKT passes at 1e-6, graph launches and host syncs per solve,
max |x - x_cold|, and, for keyed matching, the realized fraction of
collision rows whose match differs from the (sample, pair) truth.

    python summarize_device.py records.jsonl [--per-run] [--group-prefix name]
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np


def fmt(v, p=2):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"


def table(recs, key=lambda r: r["config"], title_cold="cold"):
    by = collections.OrderedDict()
    for r in recs:
        if r["k"] == 0:
            continue
        by.setdefault(key(r), []).append(r)
    cold_ms = None
    for k, rs in by.items():
        if k == title_cold or (isinstance(k, tuple) and k[-1] == title_cold):
            cold_ms = np.median([r["ms"] for r in rs if r["ms"] is not None]) if rs[0]["ms"] is not None else None
    lines = ["| config | n | iter med | iter mean | iter max | ms med | ms mean | ms max | speedup | fail | KKT pass "
             "| launches/solve | syncs/solve | max dx | wrong match |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, rs in by.items():
        it = np.array([r["iter"] for r in rs])
        ms = np.array([r["ms"] for r in rs if r["ms"] is not None])
        fail = sum(r["status"] != 0 for r in rs)
        kp = sum(bool(r.get("kkt_pass")) for r in rs)
        la = sorted({r.get("launches") for r in rs if "launches" in r})
        sy = sorted({r.get("syncs") for r in rs if "syncs" in r})
        dx = max((r.get("dx_inf", 0.0) for r in rs), default=0.0)
        wm = [r["match"]["wrong"] for r in rs if r.get("match")]
        sp = (cold_ms / np.median(ms)) if (cold_ms and len(ms)) else None
        name = k if isinstance(k, str) else " / ".join(map(str, k))
        lines.append(f"| {name} | {len(rs)} | {np.median(it):.0f} | {it.mean():.2f} | {it.max()} | "
                     f"{fmt(np.median(ms) if len(ms) else None)} | {fmt(ms.mean() if len(ms) else None)} | "
                     f"{fmt(ms.max() if len(ms) else None)} | {fmt(sp)} | {fail} | {kp}/{len(rs)} | "
                     f"{','.join(map(str, la))} | {','.join(map(str, sy))} | {dx:.1e} | "
                     f"{fmt(100 * np.mean(wm), 1) + '%' if wm else '-'} |")
    return "\n".join(lines)


def fit(recs):
    """Least-squares ms = a + b * iter per config (fixed cost a, per-iteration b)."""
    by = collections.defaultdict(list)
    for r in recs:
        if r["ms"] is not None:
            by[r["config"]].append((r["iter"], r["ms"]))
    out = ["| config | fixed ms | ms / iteration |", "|---|---:|---:|"]
    for k, v in by.items():
        it, ms = np.array(v).T
        if np.ptp(it) == 0:
            continue
        b, a = np.polyfit(it, ms, 1)
        out.append(f"| {k} | {a:.3f} | {b:.4f} |")
    return "\n".join(out)


def robust(recs):
    """Corruption curve: aggregate over seeds per (kind, nominal fraction)."""
    import re
    rows = collections.OrderedDict()
    for r in recs:
        if r["k"] == 0:
            continue
        c = r["config"]
        if c == "cold":
            key = ("cold", 0.0)
        elif "corrupt=" in c:
            kind = re.search(r"corrupt=(\w+)", c).group(1)
            key = (kind, float(re.search(r"frac=([0-9.]+)", c).group(1)))
        else:
            key = ("none", 0.0)
        rows.setdefault(key, []).append(r)
    out = ["| corruption | nominal % | n | realized wrong % | of which unmatched % | iter med | iter mean | iter max "
           "| fail | KKT pass |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (kind, frac), rs in sorted(rows.items(), key=lambda kv: (kv[0][0] != "cold", kv[0][0], kv[0][1])):
        it = np.array([r["iter"] for r in rs])
        wm = [r["match"]["wrong"] for r in rs if r.get("match")]
        um = [r["match"]["unmatched"] for r in rs if r.get("match")]
        out.append(f"| {kind} | {100 * frac:.0f} | {len(rs)} | {fmt(100 * np.mean(wm), 1) if wm else '-'} | "
                   f"{fmt(100 * np.mean(um), 1) if um else '-'} | {np.median(it):.0f} | {it.mean():.2f} | {it.max()} | "
                   f"{sum(r['status'] != 0 for r in rs)} | {sum(bool(r.get('kkt_pass')) for r in rs)}/{len(rs)} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--per-run", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--robust", action="store_true")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.records)]
    if args.robust:
        print(robust(recs))
        for run in sorted({r["run"] for r in recs}):
            print(f"\n{run}\n")
            print(robust([r for r in recs if r["run"] == run]))
        return
    print(table(recs))
    if args.per_run:
        print()
        print(table(recs, key=lambda r: (r["run"], r["config"])))
    if args.fit:
        print()
        print(fit(recs))


if __name__ == "__main__":
    main()
