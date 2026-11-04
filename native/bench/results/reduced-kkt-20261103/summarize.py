"""Summarize the reduced-KKT probe CSVs of this folder (stdlib only).

Usage: python3 summarize.py [folder]   (default: the folder of this script)
Prints markdown tables: timing per dataset (median over the timed states of
each op's per-state median), accuracy per dataset / config / rhs, the
last-iteration accuracy, and the pass / fail count of the accuracy criterion
  res_B <= 1e-10  and  res_B <= 10 * max(res_A, 2.2e-16)
on the normwise condensed residual (and the same on the componentwise
backward error, reported for information).
"""
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

folder = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
EPS = 2.220446049250313e-16


def dataset_of(label):
    return "mpc" if label.startswith("mpc") else "trajopt"


def f(x):
    return float(x) if x not in ("", "nan") else math.nan


# ---------------------------------------------------------------- timing
timing = defaultdict(lambda: defaultdict(lambda: {"eager": [], "graph": []}))
for path in sorted(glob.glob(os.path.join(folder, "timing_*.csv"))):
    for r in csv.DictReader(open(path)):
        d = timing[dataset_of(r["label"])][r["op"]]
        d["eager"].append(f(r["eager_p50_ms"]))
        d["graph"].append(f(r["graph_p50_ms"]))

order = [
    "cudss_factor", "cudss_solve", "cudss_iter_1f2s",
    "assemble_a_dense_syrk", "assemble_b_sparse", "assemble_c_sparse_skip0",
    "copyH", "copyH_potrf", "copyH_tiled_chol", "reduced_solve_recover",
    "reduced_iter_1a1f2s_a_dense_syrk", "reduced_iter_1a1f2s_b_sparse", "reduced_iter_1a1f2s_c_sparse_skip0",
    "reduced_iter_1a1f2s_b_sparse_tiled_chol", "reduced_iter_1a1f2s_c_sparse_skip0_tiled_chol",
]
for ds in sorted(timing):
    ops = timing[ds]
    nstates = len(ops["cudss_factor"]["graph"])
    print(f"\n### Timing, {ds} ({nstates} timed states; median over states of the per-state median, us)\n")
    print("| op | eager us | graph us | graph min..max over states |")
    print("|---|---:|---:|---|")
    for op in order:
        if op not in ops:
            continue
        e, g = ops[op]["eager"], ops[op]["graph"]
        gv = [x for x in g if not math.isnan(x)]
        rng = f"{min(gv)*1e3:.0f}..{max(gv)*1e3:.0f}" if gv else "n/a"
        gm = statistics.median(gv) * 1e3 if gv else math.nan
        print(f"| {op} | {statistics.median(e)*1e3:.0f} | {gm:.0f} | {rng} |")
    base = statistics.median(ops["cudss_iter_1f2s"]["graph"])
    print(f"\nPer-iteration speedup vs cuDSS (graph, median): ")
    for op in order:
        if op.startswith("reduced_iter") and op in ops:
            r = statistics.median(ops[op]["graph"])
            print(f"- {op}: {r*1e3:.0f} us, cuDSS/reduced = {base / r:.2f}x")

# ---------------------------------------------------------------- accuracy
rows = []
for path in sorted(glob.glob(os.path.join(folder, "accuracy_*.csv"))):
    rows += list(csv.DictReader(open(path)))

groups = defaultdict(list)
for r in rows:
    groups[(dataset_of(r["label"]), r["variant"], r["rhs"])].append(r)


def passes(rb, ra):
    return rb <= 1e-10 and rb <= 10 * max(ra, EPS)


print("\n### Accuracy over all states (normwise condensed residual; cw = componentwise backward error)\n")
print("| dataset | config | rhs | states | potrf fails | max res_A | max res_B | max res_B_ir | pass | "
      "max cw_A | max cw_B | max cw_B_ir | cw pass | max reldiff(A,B) |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for key in sorted(groups):
    g = groups[key]
    ra = [f(r["res_A"]) for r in g]
    rb = [f(r["res_B"]) for r in g]
    rbi = [f(r["res_B_ir"]) for r in g]
    ca = [f(r["cw_A"]) for r in g]
    cb = [f(r["cw_B"]) for r in g]
    cbi = [f(r["cw_B_ir"]) for r in g]
    fails = sum(int(r["potrf_info"]) != 0 for r in g)
    npass = sum(passes(b, a) for a, b in zip(ra, rb))
    cpass = sum(cb_ <= 10 * max(ca_, EPS) for ca_, cb_ in zip(ca, cb))
    rd = max(f(r["reldiff_AB"]) for r in g)
    print(f"| {key[0]} | {key[1]} | {key[2]} | {len(g)} | {fails} | {max(ra):.1e} | {max(rb):.1e} | {max(rbi):.1e} | "
          f"{npass}/{len(g)} | {max(ca):.1e} | {max(cb):.1e} | {max(cbi):.1e} | {cpass}/{len(g)} | {rd:.1e} |")

# Last iteration of every solve (smallest rho / delta).
print("\n### Last iteration of each solve (config b_sparse+potrf)\n")
print("| label | case | k | rho_f | delta_f | rhs | potrf | res_A | res_B | res_B_ir | cw_A | cw_B | cw_B_ir | "
      "reldiff | L diag ratio^2 |")
print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
last = {}
for r in rows:
    key = (r["label"], r["case"])
    last[key] = max(last.get(key, 0), int(r["k"]))
for r in rows:
    if r["variant"] != "b_sparse+potrf" or int(r["k"]) != last[(r["label"], r["case"])]:
        continue
    print(f"| {r['label']} | {r['case']} | {r['k']} | {f(r['rho_factored']):.0e} | {f(r['delta_factored']):.0e} | "
          f"{r['rhs']} | {r['potrf_info']} | {f(r['res_A']):.1e} | {f(r['res_B']):.1e} | {f(r['res_B_ir']):.1e} | "
          f"{f(r['cw_A']):.1e} | {f(r['cw_B']):.1e} | {f(r['cw_B_ir']):.1e} | {f(r['reldiff_AB']):.1e} | "
          f"{f(r['cholL_diag_ratio']):.1e} |")

# Per-iteration profile (worst over cases / groups) for b_sparse+potrf, both rhs.
print("\n### Worst over cases per iteration index (config b_sparse+potrf, both rhs)\n")
print("| dataset | k | max res_A | max res_B | max cw_A | max cw_B | max cw_B_ir | max reldiff | max L ratio^2 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
byk = defaultdict(list)
for r in rows:
    if r["variant"] == "b_sparse+potrf":
        byk[(dataset_of(r["label"]), int(r["k"]))].append(r)
for key in sorted(byk):
    g = byk[key]
    mx = lambda c: max(f(r[c]) for r in g)
    print(f"| {key[0]} | {key[1]} | {mx('res_A'):.1e} | {mx('res_B'):.1e} | {mx('cw_A'):.1e} | {mx('cw_B'):.1e} | "
          f"{mx('cw_B_ir'):.1e} | {mx('reldiff_AB'):.1e} | {mx('cholL_diag_ratio'):.1e} |")

nz = defaultdict(list)
for r in rows:
    nz[dataset_of(r["label"])].append(int(r["nnzM_nonzero"]))
for ds, v in sorted(nz.items()):
    print(f"\n{ds}: numerically nonzero stored [A; G] entries per state: min {min(v)}, max {max(v)}")
