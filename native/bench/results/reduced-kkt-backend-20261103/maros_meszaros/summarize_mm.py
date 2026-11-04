"""Summarize the Maros-Meszaros runs of this directory (cudss-run*.log, reduced-run*.log)."""
import re
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
ROW = re.compile(r"test_maros_meszaros_problem\[(.+?)\.mat\] (PASSED|FAILED|SKIPPED)")


def outcomes(f):
    return {m.group(1): m.group(2) for m in ROW.finditer(f.read_text())}


runs = {be: [outcomes(f) for f in sorted(D.glob(f"{be}-run*.log"))] for be in ("cudss", "reduced")}
names = sorted(set().union(*[set(r) for rs in runs.values() for r in rs]))
print(f"problems: {len(names)}; runs: cudss {len(runs['cudss'])}, reduced {len(runs['reduced'])}\n")
for be, rs in runs.items():
    for i, r in enumerate(rs, 1):
        c = {k: sum(v == k for v in r.values()) for k in ("PASSED", "FAILED", "SKIPPED")}
        print(f"{be} run {i}: {c['PASSED']} passed, {c['FAILED']} failed, {c['SKIPPED']} skipped")
print()


def always(be, k):
    return {p for p in names if all(r.get(p) == k for r in runs[be])}


def ever(be, k):
    return {p for p in names if any(r.get(p) == k for r in runs[be])}


skipped = always("reduced", "SKIPPED")
evaluated = [p for p in names if p not in skipped]
print(f"reduced skips (n > limit, same in every run): {len(skipped)}: {', '.join(sorted(skipped))}\n")
print(f"cuDSS fails in every run: {len(always('cudss', 'FAILED'))}: {', '.join(sorted(always('cudss', 'FAILED')))}")
print(f"cuDSS fails in some run: {len(ever('cudss', 'FAILED'))}")
print(f"reduced fails in some run: {len(ever('reduced', 'FAILED'))}: {', '.join(sorted(ever('reduced', 'FAILED')))}")
print(f"reduced fails always, cuDSS always solves: "
      f"{sorted(always('reduced', 'FAILED') & always('cudss', 'PASSED'))}")
print(f"\non the {len(evaluated)} problems the reduced backend accepts:")
cf = sorted(always('cudss', 'FAILED') - skipped)
cflip = sorted((ever('cudss', 'FAILED') - always('cudss', 'FAILED')) - skipped)
print(f"  cuDSS fails always: {len(cf)}: {', '.join(cf)}")
print(f"  cuDSS flips: {len(cflip)}: {', '.join(cflip)}")
print(f"  reduced solves all of them in every run: {all(r.get(p) == 'PASSED' for r in runs['reduced'] for p in evaluated)}")
