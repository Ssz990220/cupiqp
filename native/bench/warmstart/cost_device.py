"""Overhead of the device warm start, measured on the real solve graph.

1. Fixed cost of the start (no IPM iteration): settings.max_iter = 1 (0 is rejected), so a
   solve is the prelude (start or default initial guess, residuals, one iteration,
   termination check) plus the finalization. Repeated --reps times on one QP
   of the pick run: wall time (CLOCK_MONOTONIC_RAW, launch to stream sync) and
   GPU time (CUDA events of compute_timings). cold_if_not_solved is off so
   that the warm graphs keep taking the warm branch.
2. Ring-buffer cost per IPM iteration: full solves of the same QP from the
   same previous state with and without the snapshot ring (identical
   iterations, verified), difference of the median GPU time / iterations.
3. Counters: graph launches, captures and host syncs per steady-state solve;
   host-to-device bytes of the feature per solve (keys and groups).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_device import KeyPlan  # noqa: E402
from ws_common import RUNS, RunSolver, load_qp, raw_ns  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--reps", type=int, default=300)
ap.add_argument("--k", type=int, default=20)
args = ap.parse_args()
args.out.parent.mkdir(parents=True, exist_ok=True)

run = RUNS[1]
qps = [load_qp(run, k) for k in range(args.k + 1)]
rs = RunSolver(qps[0])
cn = rs.cn
rs.s.settings.compute_timings = True
plans = {m: KeyPlan(qps, m) for m in ("pair", "pair_block", "label")}
res = {}


def timed(fn, reps):
    wall, gpu, it = [], [], []
    for _ in range(reps):
        c0 = rs.s.native.counters()
        t0 = raw_ns()
        fn()
        rs.s.native.synchronize()
        wall.append((raw_ns() - t0) * 1e-6)
        c1 = rs.s.native.counters()
        gpu.append(rs.s.native.timings()["solve_ms"])
        it.append(rs.info()["iter"])
    d = {k: c1[k] - c0[k] for k in ("graph_launches", "graph_captures", "host_syncs")}
    return dict(wall_ms=float(np.median(wall)), gpu_ms=float(np.median(gpu)), iter=sorted(set(it)), **d)


def configure(mode, K=0, back=0):
    opts = dict(cold_if_not_solved=False, snapshot_slots=K, snapshot_back=back)
    kb = None
    if mode == "identity":
        opts["row_match"] = cn.RowMatch.Identity
    else:
        opts.update(plans[mode].options(cn))
    rs.s.set_warm_start(**opts)
    if mode != "identity":
        kb = rs.s.row_key_buffers(host=True)
        kb["keys"][:] = plans[mode].keys[args.k]
        if kb["groups"] is not None:
            kb["groups"][:] = plans[mode].groups[args.k]
    return kb


# 1. fixed cost (max_iter = 1: prelude + one IPM iteration + finalization)
rs.load(qps[args.k])
rs.s.settings.max_iter = 1
res["fixed:cold solve()"] = timed(lambda: rs.s.native.solve(), args.reps)
for mode in ("identity", "pair_block", "pair", "label"):
    kb = configure(mode)
    rs.s.native.solve_warm_start(kb is not None)  # cold-record graph
    rs.s.native.solve_warm_start(kb is not None)  # captures the warm graph
    key_bytes = (kb["keys"].nbytes + (kb["groups"].nbytes if kb["groups"] is not None else 0)) if kb else 0
    r = timed(lambda: rs.s.native.solve_warm_start(kb is not None), args.reps)
    r["h2d_bytes_per_solve"] = key_bytes
    res[f"fixed:warm {mode}"] = r
    if kb is not None:  # same keys from the device staging (already there): no H2D copy
        rs.s.native.solve_warm_start(False)
        r = timed(lambda: rs.s.native.solve_warm_start(False), args.reps)
        r["h2d_bytes_per_solve"] = 0
        res[f"fixed:warm {mode} (device keys)"] = r
rs.s.settings.max_iter = 250

# 2. ring cost per iteration: same QP, same previous state (the solution of QP k-1)
for K, back in ((0, 0), (8, 0), (8, 8)):
    kb = configure("pair", K, back)
    wall, gpu, its = [], [], []
    for rep in range(args.reps // 2):
        rs.s.reset_warm_start()
        rs.load(qps[args.k - 1])
        kb["keys"][:] = plans["pair"].keys[args.k - 1]
        kb["groups"][:] = plans["pair"].groups[args.k - 1]
        rs.s.native.solve_warm_start(True)
        rs.load(qps[args.k])
        kb["keys"][:] = plans["pair"].keys[args.k]
        kb["groups"][:] = plans["pair"].groups[args.k]
        rs.s.native.synchronize()
        t0 = raw_ns()
        rs.s.native.solve_warm_start(True)
        rs.s.native.synchronize()
        wall.append((raw_ns() - t0) * 1e-6)
        gpu.append(rs.s.native.timings()["solve_ms"])
        its.append(rs.info()["iter"])
    res[f"full:pair K={K} back={back}"] = dict(wall_ms=float(np.median(wall[1:])), gpu_ms=float(np.median(gpu[1:])),
                                              iter=sorted(set(its)))

base = res["full:pair K=0 back=0"]
ring = res["full:pair K=8 back=0"]
assert base["iter"] == ring["iter"] and len(base["iter"]) == 1
res["ring_us_per_iteration_gpu"] = 1e3 * (ring["gpu_ms"] - base["gpu_ms"]) / base["iter"][0]
res["ring_us_per_iteration_wall"] = 1e3 * (ring["wall_ms"] - base["wall_ms"]) / base["iter"][0]
args.out.write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, v)
