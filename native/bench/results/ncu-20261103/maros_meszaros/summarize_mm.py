"""Summarize run_mm.sh logs of this directory: per label and run the
outcome counts; per problem the median test duration (pytest --durations)
of the 'baseline' and 'final' labels and the ratio."""
import re
import statistics
from pathlib import Path

D = Path(__file__).resolve().parent
ROW = re.compile(r"test_maros_meszaros_problem\[(.+?)\.mat\] (PASSED|FAILED|SKIPPED)")
DUR = re.compile(r"^([\d.]+)s call\s+\S*test_maros_meszaros_problem\[(.+?)\.mat\]", re.M)
runs = {}
for f in sorted(D.glob("*-run*.log")):
    label = f.name.rsplit("-run", 1)[0]
    t = f.read_text()
    runs.setdefault(label, []).append(({m.group(1): m.group(2) for m in ROW.finditer(t)},
                                       {m.group(2): float(m.group(1)) for m in DUR.finditer(t)}))
for label, rs in runs.items():
    for i, (o, _) in enumerate(rs, 1):
        c = {k: sum(v == k for v in o.values()) for k in ("PASSED", "FAILED", "SKIPPED")}
        print(f"{label} run {i}: {c['PASSED']} passed, {c['FAILED']} failed, {c['SKIPPED']} skipped")
if "baseline" in runs and "final" in runs:
    def med(label):
        names = set().union(*[set(d) for _, d in runs[label]])
        return {p: statistics.median([d[p] for _, d in runs[label] if p in d]) for p in names}
    b, f = med("baseline"), med("final")
    passed = [p for p in b if all(o.get(p) == "PASSED" for o, _ in runs["baseline"] + runs["final"])]
    r = sorted((f[p] / b[p], p) for p in passed if p in f and b[p] > 0)
    print(f"\nper-problem call duration final / baseline over {len(r)} problems passing in every run:")
    print(f"  median {statistics.median([x for x, _ in r]):.3f}, faster (<0.97) {sum(x < 0.97 for x, _ in r)}, "
          f"same {sum(0.97 <= x <= 1.03 for x, _ in r)}, slower (>1.03) {sum(x > 1.03 for x, _ in r)}")
    print("  slowest ratios:", ", ".join(f"{p} {x:.2f} ({b[p]:.2f}s -> {f[p]:.2f}s)" for x, p in r[-6:]))
    print("  fastest ratios:", ", ".join(f"{p} {x:.2f} ({b[p]:.2f}s -> {f[p]:.2f}s)" for x, p in r[:6]))
    print(f"  total call time baseline {sum(b[p] for _, p in r):.2f}s, final {sum(f[p] for _, p in r):.2f}s")
