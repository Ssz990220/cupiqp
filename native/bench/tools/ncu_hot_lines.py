#!/usr/bin/env python3
"""Top CUDA source lines by warp-stall samples, from
`ncu -i <rep> --page source --csv --print-source cuda,sass -k <kernel> -c 1`.
Usage: ncu_hot_lines.py <source.csv> [top]"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
lines = []
for r in rows:
    if len(r) > 6 and r[0].isdigit() and r[2] == "-":
        try:
            lines.append((int(r[4]), int(r[0]), r[1].strip(), int(r[7]) if r[7].isdigit() else 0))
        except ValueError:
            pass
tot = sum(s for s, *_ in lines) or 1
print(f"total samples {tot}")
print("| line | samples % | warp insts | source |")
print("|---|---|---|---|")
for s, ln, src, ins in sorted(lines, reverse=True)[:top]:
    print(f"| {ln} | {100 * s / tot:.1f} | {ins} | `{src[:90]}` |")
