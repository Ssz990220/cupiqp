#!/usr/bin/env python3
"""Summarize run_ladder_reduced.sh: per dataset and row, KKT pass counts,
iterations, end-to-end p50 / P95, update p50, solve p50 (gpu_resident).
Usage: summarize_ladder_reduced.py <dir> [<dir before> for a before/after column]"""
import json
import sys
from pathlib import Path

import numpy as np

ROWS = ["3-native-default", "7-native-default-reduced", "7d-native-default-reduced-devloop",
        "8-staged-devloop-customspmv-reduced"]


def stats(d):
    f = d / "checked-trials.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rs = [r for r in rows if r["mode"] == "gpu_resident"]
    it = np.array([r["iterations"] for r in rs], float)
    w = np.array([r["cpu_x_ready_wall_ms"] for r in rs], float)
    return dict(ok=sum(r["strict_valid"] for r in rows), n=len(rows), it=(np.median(it), it.max(), it.mean()),
                e2e=(np.percentile(w, 50), np.percentile(w, 95)),
                up=np.median([r["update_event_ms"] for r in rs]), so=np.median([r["solve_event_ms"] for r in rs]))


def main():
    root = Path(sys.argv[1])
    before = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = []
    for ds in ("mpc", "trajopt"):
        out += [f"## {ds}", "", "| row | KKT@1e-6 | iters p50/max/mean | e2e p50 / P95 ms | update p50 | solve p50 |"
                + (" before e2e p50 | change |" if before else ""), "|---|---|---|---|---|---|" + ("---|---|" if before else "")]
        for r in ROWS:
            s = stats(root / ds / r)
            if s is None:
                continue
            line = (f"| {r} | {s['ok']}/{s['n']} | {s['it'][0]:.1f}/{s['it'][1]:.0f}/{s['it'][2]:.2f} | "
                    f"{s['e2e'][0]:.2f} / {s['e2e'][1]:.2f} | {s['up']:.2f} | {s['so']:.2f} |")
            if before:
                b = stats(before / ds / r)
                line += (f" {b['e2e'][0]:.2f} | {100 * (s['e2e'][0] / b['e2e'][0] - 1):+.1f}% |" if b else " | |")
            out.append(line)
        out.append("")
    text = "\n".join(out)
    (root / "ladder.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
