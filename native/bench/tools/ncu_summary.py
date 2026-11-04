#!/usr/bin/env python3
"""Summarize Nsight Compute reports (exported with `ncu -i <rep> --page raw --csv`).

Usage: ncu_summary.py <raw.csv> [...] [--markdown out.md] [--csv out.csv]
One row per kernel name and grid size (median over instances): duration,
SM / memory throughput, FP64 pipe use, achieved / theoretical occupancy,
registers, spills, global load sectors per request, shared bank conflicts,
and the top warp-stall reasons (PC sampling shares).
"""
import csv
import statistics
import sys
from collections import defaultdict

M = {
    "dur_us": "gpu__time_duration.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "fp64_act": "sm__pipe_fp64_cycles_active.max.pct_of_peak_sustained_elapsed",
    "issue_act": "sm__issue_active.max.pct_of_peak_sustained_elapsed",
    "occ_ach": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_th": "sm__maximum_warps_per_active_cycle_pct",
    "regs": "launch__registers_per_thread",
    "spill_ld": "sass__inst_executed_register_spilling_mem_local_op_read",
    "spill_st": "sass__inst_executed_register_spilling_mem_local_op_write",
    "gld_sec": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "gld_req": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "bank_ld": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "bank_st": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
}
STALL = "smsp__pcsamp_warps_issue_stalled_"


def num(v):
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return float("nan")


def short(name):
    s = name.split("(")[0]
    return s.replace("void ", "").strip()


def main():
    args = sys.argv[1:]
    md = cs = None
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--markdown":
            md = args[i + 1]
            i += 2
        elif args[i] == "--csv":
            cs = args[i + 1]
            i += 2
        else:
            files.append(args[i])
            i += 1
    groups = defaultdict(list)
    order = []
    for f in files:
        with open(f) as fh:
            rows = list(csv.reader(fh))
        hdr, rows = rows[0], rows[2:]
        col = {h: k for k, h in enumerate(hdr)}
        stall_cols = [h for h in hdr if h.startswith(STALL) and not h.endswith("_not_issued")]
        for r in rows:
            name = short(r[col["Kernel Name"]])
            d = {k: num(r[col[v]]) if v in col else float("nan") for k, v in M.items()}
            st = {h[len(STALL):]: num(r[col[h]]) for h in stall_cols}
            key = (name, int(d["grid"]) if d["grid"] == d["grid"] else -1)
            if key not in groups:
                order.append(key)
            groups[key].append((d, st))
    out = []
    for key in order:
        inst = groups[key]
        med = {k: statistics.median([d[k] for d, _ in inst]) for k in M}
        tot = defaultdict(float)
        for _, st in inst:
            for k, v in st.items():
                if v == v:
                    tot[k] += v
        tot.pop("selected", None)
        s = sum(tot.values()) or 1.0
        top = sorted(tot.items(), key=lambda kv: -kv[1])[:4]
        med["stalls"] = "; ".join(f"{k} {100 * v / s:.0f}%" for k, v in top)
        med["spr"] = med["gld_sec"] / med["gld_req"] if med["gld_req"] else float("nan")
        out.append((key, len(inst), med))
    cols = ["dur_us", "sm_pct", "mem_pct", "fp64_act", "issue_act", "occ_ach", "occ_th", "regs", "spill_ld",
            "spill_st", "spr", "bank_ld", "bank_st"]
    if cs:
        with open(cs, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kernel", "grid", "block", "instances"] + cols + ["stalls"])
            for (name, grid), n, m in out:
                w.writerow([name, grid, int(m["block"]), n] + [f"{m[c]:.3g}" for c in cols] + [m["stalls"]])
    lines = ["| kernel | grid x block | n | us | SM % | mem % | FP64 pipe % (max SM) | issue % (max SM) | occ ach / th % | regs | spill ld/st | sectors/req (gld) | bank confl ld/st | top stalls |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (name, grid), n, m in out:
        lines.append(f"| {name} | {grid} x {int(m['block'])} | {n} | {m['dur_us']/1e3 if m['dur_us'] > 1e3 else m['dur_us']:.2f} | {m['sm_pct']:.1f} | {m['mem_pct']:.1f} | {m['fp64_act']:.1f} | {m['issue_act']:.1f} | {m['occ_ach']:.1f} / {m['occ_th']:.1f} | {int(m['regs'])} | {int(m['spill_ld'])}/{int(m['spill_st'])} | {m['spr']:.2f} | {int(m['bank_ld'])}/{int(m['bank_st'])} | {m['stalls']} |")
    text = "\n".join(lines) + "\n"
    if md:
        open(md, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
