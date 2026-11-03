"""Snapshot replay through the existing harness protocol, with a selectable backend.

Same protocol, timing boundaries and outputs as
gpu-qp-benchmark/benchmark/replay_mpc_updates.py (read-only reference; the
benchmark workspace is not modified): one persistent solver per stored CSR
pattern, explicit zeros kept, every matrix/vector value replaced through the
public update() in stored order, FP64, eps_abs = eps_duality_gap_abs = 1e-8,
zero relative tolerances, 10 Ruiz passes recomputed after matrix updates, no
numerical warm start. Two excluded warm-up passes; per repeat and trajectory
the first snapshot is an excluded seed, the following transitions are timed
in two modes with alternating order:
  gpu_resident  values already in GPU memory
  host_values   one packed pinned host buffer copied into a fixed GPU stage
Timing: host wall time from input values ready to x on the CPU (one explicit
completion wait), CUDA events split upload / update / solve / x download.
Correctness: the benchmark's own qp_io.check_solution at 3e-4 and 1e-6.

  --backend reference   pinned Warp cuPIQP (import cupiqp)
  --backend native      cupiqp_native (nanobind binding of the C++ solver)
  --dataset mpc         qp_benchmark_v2_20261102, cohort main (1 pattern, 16 trajectories x 4)
  --dataset trajopt     datasets/trajopt-20261102 (6 patterns, one per trajectory, 5 SQP iterates)

The output directory is compatible with benchmark/report_mpc_replay.py for
the MPC dataset (environment.json, trials.jsonl, seeds.jsonl, solutions/).
"""
import argparse
import collections
import dataclasses
import datetime
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def raw_ns():
    """Host clock for wall times: CLOCK_MONOTONIC_RAW (hardware TSC, not slewed
    by NTP). On this WSL host the NTP-disciplined CLOCK_MONOTONIC used by
    time.perf_counter ran ~6% slow on 2026-11-03 (checked against the Windows
    QPC clock, see native/bench/tools/check_clocks.py); CUDA events agreed with
    the Windows clock."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

import numpy as np

BENCH = Path(os.environ.get("QP_BENCH_ROOT", "/mnt/c/Users/91354/Projects/gpu-qp-benchmark"))
REPO = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("reference", "native"), required=True)
parser.add_argument("--dataset", choices=("mpc", "trajopt"), default="mpc")
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--native-build", type=Path, default=REPO / "build" / "native")
parser.add_argument("--graph-mode", choices=("iteration", "device", "none"), default="iteration",
                    help="native only: GraphMode (iteration = reference default)")
parser.add_argument("--custom-spmv", action="store_true", help="native only: deterministic custom SpMV")
args = parser.parse_args()

OUT = args.output
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(BENCH / "qp_benchmark_v2_20261102" / "tools"))
from qp_io import check_solution, load_problem  # noqa: E402  (benchmark-owned, read only)

import warp as wp  # noqa: E402

wp.config.quiet = True
wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", str(Path.home() / ".cache/warp-cupiqp-native"))
wp.init()

if args.backend == "native":
    sys.path.insert(0, str(args.native_build / "python"))
    import cupiqp_native as backend_mod  # noqa: E402
    from cupiqp_native import SparseSolver, Status  # noqa: E402
else:
    import cupiqp as backend_mod  # noqa: E402
    from cupiqp import SparseSolver, Status  # noqa: E402

if args.dataset == "mpc":
    BUNDLE = BENCH / "qp_benchmark_v2_20261102"
    cases = [c for c in json.loads((BUNDLE / "manifest.json").read_text())["cases"] if c["cohort"] == "main"]
    traj_key, step_key = "trajectory_name", "control_step"
else:
    BUNDLE = BENCH / "datasets" / "trajopt-20261102"
    cases = [c for c in json.loads((BUNDLE / "manifest.json").read_text())["cases"] if c.get("cohort", "main") == "main"]
    traj_key, step_key = "trajectory", "sqp_iteration"

trajectories = collections.defaultdict(list)
for c in cases:
    trajectories[c[traj_key]].append(c)
for group in trajectories.values():
    group.sort(key=lambda c: c[step_key])
problems = {c["id"]: load_problem(BUNDLE / "cases" / c["id"]) for c in cases}
csr = {cid: {k: p[k].tocsr() for k in ("P", "A", "C")} for cid, p in problems.items()}
for mats in csr.values():
    for m in mats.values():
        m.sort_indices()


def pattern_key(cid):
    return tuple((m.shape, m.indptr.tobytes(), m.indices.tobytes()) for m in csr[cid].values())


# Pattern groups: every trajectory belongs to one; MPC has a single group.
pattern_groups = collections.OrderedDict()
for name, cs in trajectories.items():
    keys = {pattern_key(c["id"]) for c in cs}
    assert len(keys) == 1, f"trajectory {name} mixes CSR patterns"
    pattern_groups.setdefault(keys.pop(), []).append((name, cs))
if args.dataset == "mpc":
    assert len(pattern_groups) == 1 and len(trajectories) == 16

names = ("P", "A", "C", "q", "b", "l", "u")


class Group:
    """One persistent solver, packets and buffers for one stored CSR pattern."""

    def __init__(self, gi, trajs):
        self.gi, self.trajs = gi, trajs
        first = trajs[0][1][0]["id"]
        self.template = csr[first]
        sizes = {k: self.template[k].nnz if k in self.template else problems[first][k].size for k in names}
        self.offsets, total = {}, 0
        for k in names:
            self.offsets[k] = total
            total += sizes[k]
        self.sizes, self.total = sizes, total
        self.packets = {}
        for _, cs in trajs:
            for c in cs:  # every conversion, packing and allocation happens here
                cid, p = c["id"], problems[c["id"]]
                flat = np.concatenate([csr[cid][k].data if k in self.template else p[k] for k in names])
                assert flat.size == total and flat.dtype == np.float64
                gpu = wp.array(flat, dtype=wp.float64, device="cuda:0")
                host = wp.array(flat, dtype=wp.float64, device="cpu", pinned=True)
                self.packets[cid] = dict(gpu=gpu, host=host, values=self.views(gpu))
        self.stage = wp.empty(total, dtype=wp.float64, device="cuda:0")
        self.stage_values = self.views(self.stage)
        n = problems[first]["q"].size
        self.x_stage = wp.empty((1, n), dtype=wp.float64, device="cuda:0")
        self.x_host = wp.empty((1, n), dtype=wp.float64, device="cpu", pinned=True)
        s = SparseSolver(dtype=wp.float64)
        s.settings.eps_abs = s.settings.eps_duality_gap_abs = 1e-8
        s.settings.eps_rel = s.settings.eps_duality_gap_rel = 0.0
        s.settings.enable_cuda_graph = True
        s.settings.verbose = False
        s.settings.preconditioner_reuse_on_update = False
        if args.backend == "native":
            s.settings.graph_mode = {"iteration": backend_mod.GraphMode.IterationGraphHostLoop,
                                     "device": backend_mod.GraphMode.SolveGraphDeviceLoop,
                                     "none": backend_mod.GraphMode.None_}[args.graph_mode]
            s.settings.enable_cuda_graph = args.graph_mode != "none"
            s.settings.custom_spmv = args.custom_spmv
        self.s = s
        v = self.packets[first]["values"]
        t = self.template
        start = raw_ns()
        s.setup(P=(t["P"].indptr, t["P"].indices, v["P"]), c=v["q"],
                A=(t["A"].indptr, t["A"].indices, v["A"]), b=v["b"],
                G=(t["C"].indptr, t["C"].indices, v["C"]), h_l=v["l"], h_u=v["u"])
        wp.synchronize_stream(s.stream)
        self.setup_ms = (raw_ns() - start) / 1e6
        # First solve: reference JIT (if any left) and first graph capture, reported separately.
        start = raw_ns()
        with wp.ScopedStream(s.stream):
            s.solve()
        wp.synchronize_stream(s.stream)
        self.first_solve_ms = (raw_ns() - start) / 1e6
        self.addresses = self.pointers()

    def views(self, packet):
        return {k: packet[self.offsets[k]:self.offsets[k] + self.sizes[k]] for k in names}

    def pointers(self):
        s = self
        base = dict(input_stage=s.stage.ptr, x_stage=s.x_stage.ptr, x_host=s.x_host.ptr,
                    x=s.s.result.x.ptr, y=s.s.result.y.ptr)
        if args.backend == "native":
            base.update({name: ptr for name, ptr, *_ in s.s.native.debug_buffers()})
        else:
            d = s.s.data
            base.update(P=d.P.data.ptr, A=d.A.data.ptr, C=d.G.data.ptr, q=d.c.ptr, b=d.b.ptr, l=d.h_l.ptr,
                        u=d.h_u.ptr)
        return base


events = [wp.Event(device="cuda:0", enable_timing=True) for _ in range(5)]


def run_step(g, cid, mode, measure):
    s = g.s
    if measure:
        wp.record_event(events[0])
    begin = raw_ns()
    begin_mono = time.perf_counter_ns()
    if mode == "host_values":
        wp.copy(g.stage, g.packets[cid]["host"])
        v = g.stage_values
    else:
        v = g.packets[cid]["values"]
    if measure:
        wp.record_event(events[1])
    s.update(P=v["P"], c=v["q"], A=v["A"], b=v["b"], G=v["C"], h_l=v["l"], h_u=v["u"], check_validity=False)
    if measure:
        wp.record_event(events[2])
    s.solve()
    if measure:
        wp.record_event(events[3])
    wp.copy(g.x_stage, s.result.x)
    wp.copy(g.x_host, g.x_stage)
    if measure:
        wp.record_event(events[4])
    wp.synchronize_stream(s.stream)
    end = raw_ns()
    end_mono = time.perf_counter_ns()
    assert g.pointers() == g.addresses
    result = dict(cpu_x_ready_wall_ms=(end - begin) / 1e6,
                  cpu_x_ready_wall_ms_perf_counter=(end_mono - begin_mono) / 1e6)
    if measure:
        for field, a, b in [("upload_event_ms", 0, 1), ("update_event_ms", 1, 2), ("solve_event_ms", 2, 3),
                            ("x_download_event_ms", 3, 4), ("gpu_x_ready_event_ms", 0, 4),
                            ("gpu_solution_ready_event_ms", 0, 3)]:
            result[field] = float(wp.get_event_elapsed_time(events[a], events[b], synchronize=False))
    r = s.result
    x = np.array(g.x_host.numpy().reshape(-1), copy=True)
    y = r.y.numpy().reshape(-1)
    z = r.z_u.numpy().reshape(-1) - r.z_l.numpy().reshape(-1)
    info = r.info.to_host()
    status = Status(int(info.status[0])).name
    validation = check_solution(problems[cid], x, y, z, 3e-4)
    strict = check_solution(problems[cid], x, y, z, 1e-6)
    result.update(status=status, iterations=int(np.asarray(r.info.iter.numpy())[0]), validation=validation,
                  strict_kkt_pass=strict["kkt_pass"], valid=status == "CUPIQP_SOLVED" and validation["kkt_pass"],
                  strict_valid=status == "CUPIQP_SOLVED" and strict["kkt_pass"],
                  primal_obj=float(info.primal_obj[0]), dual_obj=float(info.dual_obj[0]),
                  primal_res=float(info.primal_res[0]), dual_res=float(info.dual_res[0]),
                  duality_gap=float(info.duality_gap[0]))
    return result, x, y, z


def settings_dict(s):
    if args.backend == "reference":
        return {k: (v.__name__ if isinstance(v, type) else v) for k, v in dataclasses.asdict(s.settings).items()}
    nat = s.settings._native
    return {k: (str(getattr(nat, k)) if k == "graph_mode" else getattr(nat, k)) for k in dir(nat)
            if not k.startswith("_") and k != "verify"}


groups = [Group(gi, trajs) for gi, trajs in enumerate(pattern_groups.values())]
pairs = []
for g in groups:
    for trajectory, cs in g.trajs:
        for a, b in zip(cs, cs[1:]):
            pairs.append(dict(group=g.gi, trajectory=trajectory, from_case=a["id"], to_case=b["id"],
                              from_control_step=a[step_key], to_control_step=b[step_key],
                              step_gap=b[step_key] - a[step_key]))
metadata = dict(
    start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), repeats=args.repeats,
    backend=args.backend, dataset=args.dataset,
    solver=("cupiqp_native SparseSolver " + backend_mod.version()) if args.backend == "native" else "cuPIQP SparseSolver (Warp reference)",
    precision="float64", affinity=sorted(os.sched_getaffinity(0)), cases=len(cases), trajectories=len(trajectories),
    pattern_groups=len(groups), forward_pairs=len(pairs), pairs=pairs,
    expected_timed_transitions=len(pairs) * args.repeats * 2,
    packed_value_bytes=[g.total * 8 for g in groups],
    one_time_setup_ms=[g.setup_ms for g in groups],
    first_solve_incl_jit_and_capture_ms=[g.first_solve_ms for g in groups],
    settings=settings_dict(groups[0].s),
    native_graph_mode=args.graph_mode if args.backend == "native" else None,
    native_custom_spmv=args.custom_spmv if args.backend == "native" else None,
    protocol=("Same as benchmark/replay_mpc_updates.py: one fixed template and solver per stored pattern; "
              "explicit zeros retained; all numeric values replaced through public update(); two excluded "
              "warm-up passes; first snapshot of each trajectory solved as an excluded seed every repeat; mode "
              "order rotates; buffers, events, packets and pinned outputs preallocated. Setup and the first "
              "solve (JIT, first graph capture) are timed separately and excluded."),
    timing=("cpu_x_ready_wall_ms: host wall time (CLOCK_MONOTONIC_RAW) from input values ready to x on the CPU, "
            "one explicit wait; cpu_x_ready_wall_ms_perf_counter is the same interval on time.perf_counter (the "
            "original harness clock, NTP-slewed). "
            "Includes Python-side call overhead of update()/solve() for both backends; the pure native C++ "
            "path is measured separately by native/bench/mpc_replay.cpp."),
    numerical_warm_start=False,
    limitation="Sampled transitions (MPC gaps 13-24 ticks); frozen snapshots without closed-loop feedback.",
    gc="Disabled during replay; restored on exit.")
(OUT / "environment.json").write_text(json.dumps(metadata, indent=2))
counters_before = [g.s.native.counters() for g in groups] if args.backend == "native" else None

log = (OUT / "trials.jsonl").open("w", buffering=1)
seed_log = (OUT / "seeds.jsonl").open("w", buffering=1)
telemetry = (OUT / "gpu-telemetry.csv").open("w")
monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,clocks.sm,temperature.gpu,utilization.gpu,power.draw",
                            "--format=csv", "-l", "1"], stdout=telemetry, stderr=subprocess.STDOUT)
seed_counts = collections.Counter()
timed = 0
was_gc = gc.isenabled()
gc.collect()
gc.disable()
try:
    for g in groups:
        with wp.ScopedStream(g.s.stream):
            for warmup in range(2):
                for _, cs in g.trajs:
                    for i, c in enumerate(cs):
                        run_step(g, c["id"], "host_values" if (warmup + i) % 2 else "gpu_resident", False)
    for repeat in range(args.repeats):
        for g in groups:
            with wp.ScopedStream(g.s.stream):
                for ti, (trajectory, cs) in enumerate(g.trajs):
                    modes = ["gpu_resident", "host_values"] if (repeat + ti) % 2 == 0 else ["host_values", "gpu_resident"]
                    for mode in modes:
                        seed, *_ = run_step(g, cs[0]["id"], mode, False)
                        seed_counts["samples"] += 1
                        seed_counts["valid"] += seed["valid"]
                        seed_counts["strict_valid"] += seed["strict_valid"]
                        seed_log.write(json.dumps(dict(repeat=repeat, group=g.gi, trajectory=trajectory, mode=mode,
                                                       case=cs[0]["id"], **seed)) + "\n")
                        for a, b in zip(cs, cs[1:]):
                            result, x, y, z = run_step(g, b["id"], mode, True)
                            row = dict(repeat=repeat, group=g.gi, trajectory=trajectory, mode=mode,
                                       from_case=a["id"], to_case=b["id"], from_control_step=a[step_key],
                                       to_control_step=b[step_key], step_gap=b[step_key] - a[step_key], **result)
                            log.write(json.dumps(row) + "\n")
                            timed += 1
                            if repeat == 0:
                                target = OUT / "solutions" / mode / (b["id"] + ".npz")
                                target.parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(target, x=x, y=y, z=z)
        print(f"Completed replay {repeat + 1}/{args.repeats}; timed transitions={timed}", flush=True)
    assert timed == metadata["expected_timed_transitions"]
finally:
    if was_gc:
        gc.enable()
    metadata["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata["timed_transitions"] = timed
    metadata["seed_summary"] = dict(seed_counts)
    metadata["observed_buffer_addresses_stable"] = all(g.pointers() == g.addresses for g in groups)
    if args.backend == "native":
        after = [g.s.native.counters() for g in groups]
        metadata["native_counters_during_replay"] = [{k: a[k] - b[k] for k in a} for a, b in zip(after, counters_before)]
    (OUT / "environment.json").write_text(json.dumps(metadata, indent=2))
    log.close()
    seed_log.close()
    monitor.terminate()
    monitor.wait(timeout=10)
    telemetry.close()
