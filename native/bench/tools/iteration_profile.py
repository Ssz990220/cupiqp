"""Per-iteration GPU profile from an Nsight Systems kernel trace.

Input: the cuda_gpu_trace CSV of a run recorded with
  nsys profile --capture-range=cudaProfilerApi --trace=cuda --cuda-graph-trace=node ...
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace --format csv -o <prefix> <rep>
(see native/bench/tools/profile_iteration.sh). Iterations are delimited by the
update_continue_flag kernel that ends every IPM iteration.

Prints, per IPM iteration (median over iterations):
  * GPU busy time by category (KKT assembly / Cholesky / solve, SpMV, IPM
    kernels, ...; the cuDSS and cuSPARSE categories classify traces recorded
    before those libraries were removed on 2026-11-04),
  * graph-internal GPU idle and the host round trip (continue-flag D2H, host
    sync and the next iteration-graph launch),
  * the largest idle gaps (which node follows which),
  * our kernels with grid size, block size, registers and median duration.
Writes the same as Markdown with --markdown <file>.
Usage: python3 native/bench/tools/iteration_profile.py <prefix>_cuda_gpu_trace.csv [--markdown out.md]
"""
import argparse
import collections
import csv
import statistics as st


def short(name: str) -> str:
    if name.startswith("[CUDA"):
        return name
    n = name.split("(")[0].replace("void ", "").replace("<unnamed>", "anon")
    depth, out = 0, ""
    for ch in n:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out += ch
    return out.split("::")[-1].strip()


def category(name: str) -> str:
    if any(k in name for k in ("cudss::factorize", "independent_ker", "copy_matrix_ker", "plain_map_ker")):
        return "cuDSS factorization"
    if "cudss::" in name:
        return "cuDSS solve"
    if "reduced_kkt" in name:
        if any(k in name for k in ("weights_kernel", "assemble_kernel", "compact_", "reduce_split_rows")):
            return "reduced: assembly"
        if "chol_" in name:
            return "reduced: Cholesky"
        return "reduced: solve"
    if any(k in name for k in ("cusparse", "mv_v3", "csrmv", "spmv_detail")):
        return "SpMV"
    if "kkt_kernels" in name or "update_kkt_diag" in name or "kkt_assembly" in name:
        return "KKT system kernels"
    if "ruiz" in name:
        return "Ruiz scaling"
    if name.startswith("[CUDA memcpy"):
        return "memcpy"
    if name.startswith("[CUDA memset"):
        return "memset"
    if "set_condition" in name:
        return "graph condition"
    if "cupiqp" in name:
        return "IPM kernels"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_csv")
    ap.add_argument("--markdown")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.trace_csv)))
    ev = sorted((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"], r["Strm"], r.get("GrdX", ""),
                 r.get("BlkX", ""), r.get("Reg/Trd", "")) for r in rows)
    idx = [i for i, e in enumerate(ev) if "update_continue_flag" in e[2]]
    n_it = len(idx) - 1
    busy = collections.defaultdict(list)
    host_rt, internal_idle, span = [], [], []
    gaps = collections.defaultdict(float)
    kern = collections.defaultdict(list)
    meta = {}
    for a, b in zip(idx, idx[1:]):
        seg = ev[a:b + 1]
        d2h = [i for i, e in enumerate(seg) if "Device-to-Host" in e[2]]
        j = d2h[0] if d2h else 0
        host_rt.append(((seg[j + 1][0] - (seg[j][0] + seg[j][1])) + (seg[j][0] - (seg[0][0] + seg[0][1]))) / 1e3)
        body = seg[j + 1:]
        c = collections.Counter()
        for s, d, n, *_ in body:
            c[category(n)] += d / 1e3
        for k in set(list(c) + list(busy)):
            busy[k].append(c.get(k, 0.0))
        t0, t1 = body[0][0], body[-1][0] + body[-1][1]
        span.append((t1 - t0) / 1e3)
        internal_idle.append(span[-1] - sum(c.values()))
        for e0, e1 in zip(body, body[1:]):
            tag = "stream switch" if e0[3] != e1[3] else ""
            gaps[(short(e0[2]), short(e1[2]), tag)] += (e1[0] - (e0[0] + e0[1])) / 1e3 / n_it
        for s, d, n, strm, gx, bx, regs in body:
            if "cupiqp" in n and "set_condition" not in n:
                kern[short(n)].append(d / 1e3)
                meta[short(n)] = (gx, bx, regs)

    med = {k: st.median(v) for k, v in busy.items()}
    idle, rt = st.median(internal_idle), st.median(host_rt)
    total = sum(med.values()) + idle + rt
    lines = [f"# IPM iteration profile ({n_it} iterations, medians per iteration)", "",
             f"Total per iteration: **{total:.1f} us** (GPU span {st.median(span):.1f} us + host round trip {rt:.1f} us)", "",
             "| Category | us / iteration | share |", "|---|---|---|"]
    for k in sorted(med, key=lambda k: -med[k]):
        lines.append(f"| {k} | {med[k]:.1f} | {100 * med[k] / total:.1f}% |")
    lines += [f"| GPU idle inside the iteration graph | {idle:.1f} | {100 * idle / total:.1f}% |",
              f"| Host round trip (flag D2H + sync + next graph launch) | {rt:.1f} | {100 * rt / total:.1f}% |", "",
              "## Largest idle gaps inside the iteration (us per iteration)", "",
              "| us | after | before | note |", "|---|---|---|---|"]
    for (a, b, tag), g in sorted(gaps.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"| {g:.1f} | {a} | {b} | {tag} |")
    lines += ["", "## Our kernels (per call)", "", "| kernel | calls / iteration | median us | us / iteration | grid | block | regs |",
              "|---|---|---|---|---|---|---|"]
    for k, v in sorted(kern.items(), key=lambda kv: -sum(kv[1])):
        gx, bx, regs = meta[k]
        lines.append(f"| {k} | {len(v) / n_it:.1f} | {st.median(v):.2f} | {sum(v) / n_it:.1f} | {gx} | {bx} | {regs} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        open(args.markdown, "w").write(text)


if __name__ == "__main__":
    main()
