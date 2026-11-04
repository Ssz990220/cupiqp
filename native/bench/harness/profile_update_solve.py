"""Profile one warmed fixed-pattern update + solve (reference or native backend).

Same profile range as gpu-qp-benchmark/benchmark/profile_fixed_update.py
(read only; outputs go to --output, never into the benchmark workspace): two
snapshots with the same stored CSR pattern are pre-uploaded, four warm-up
update+solve rounds run, then exactly one update (all matrices and vectors) +
solve + device completion is bracketed by cuProfilerStart/Stop. Setup, JIT,
graph capture, status reads, downloads and validation are outside the range.
Settings follow the replay protocol (eps_abs = eps_duality_gap_abs = 1e-8,
zero relative tolerances) unless --eps is given.

Run under Nsight Systems, e.g.:
  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --trace=cuda,osrt -o <out>/mpc-native \
      .venv/bin/python native/bench/harness/profile_update_solve.py --backend native --dataset mpc --output <out>
  nsys stats --report cuda_api_sum --format csv -o <out>/mpc-native <out>/mpc-native.nsys-rep
"""
import argparse
import collections
import ctypes as ct
import json
import os
import sys
from pathlib import Path

import numpy as np

BENCH = Path(os.environ.get("QP_BENCH_ROOT", "/mnt/c/Users/91354/Projects/gpu-qp-benchmark"))
REPO = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=("reference", "native"), required=True)
ap.add_argument("--dataset", choices=("mpc", "trajopt"), default="mpc")
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--eps", type=float, default=1e-8)
ap.add_argument("--native-build", type=Path, default=REPO / "build" / "native")
ap.add_argument("--graph-mode", choices=("iteration", "device", "none"), default="iteration")
ap.add_argument("--kkt-backend", choices=("cudss", "reduced"), default="cudss")
ap.add_argument("--custom-spmv", action="store_true")
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BENCH / "qp_benchmark_v2_20261102" / "tools"))
from qp_io import check_solution, load_problem  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", str(Path.home() / ".cache/warp-cupiqp-native"))
wp.init()
if args.backend == "native":
    sys.path.insert(0, str(args.native_build / "python"))
    import cupiqp_native as mod  # noqa: E402
else:
    import cupiqp as mod  # noqa: E402
SparseSolver, Status = mod.SparseSolver, mod.Status

bundle = BENCH / ("qp_benchmark_v2_20261102" if args.dataset == "mpc" else "datasets/trajopt-20261102")
cases = [c for c in json.loads((bundle / "manifest.json").read_text())["cases"] if c.get("cohort", "main") == "main"]


def pattern(p):
    out = []
    for k in ("P", "A", "C"):
        m = p[k].tocsr()
        m.sort_indices()
        out.append((m.shape, m.indptr.tobytes(), m.indices.tobytes()))
    return tuple(out)


first = load_problem(bundle / "cases" / cases[0]["id"])
other, other_case = None, None
for c in cases[1:]:
    p = load_problem(bundle / "cases" / c["id"])
    if pattern(p) == pattern(first):
        other, other_case = p, c["id"]
        break
assert other is not None


def upload(p):
    gpu = lambda a: wp.array(np.ascontiguousarray(a), dtype=wp.float64, device="cuda:0")  # noqa: E731
    mats = {}
    for k in ("P", "A", "C"):
        m = p[k].tocsr()
        m.sort_indices()
        mats[k] = (m.indptr, m.indices, gpu(m.data))
    return dict(matrices=mats, **{k: gpu(p[k]) for k in ("q", "b", "l", "u")})


inputs = [upload(first), upload(other)]
s = SparseSolver(dtype=wp.float64)
s.settings.eps_abs = s.settings.eps_duality_gap_abs = args.eps
s.settings.eps_rel = s.settings.eps_duality_gap_rel = 0.0
s.settings.enable_cuda_graph = True
s.settings.verbose = False
if args.backend == "native":
    s.settings.graph_mode = {"iteration": mod.GraphMode.IterationGraphHostLoop,
                             "device": mod.GraphMode.SolveGraphDeviceLoop, "none": mod.GraphMode.None_}[args.graph_mode]
    s.settings.enable_cuda_graph = args.graph_mode != "none"
    s.settings.kkt_backend = args.kkt_backend
    s.settings.custom_spmv = args.custom_spmv
a = inputs[0]
s.setup(P=a["matrices"]["P"], c=a["q"], A=a["matrices"]["A"], b=a["b"], G=a["matrices"]["C"], h_l=a["l"], h_u=a["u"])


def update(a):
    s.update(P=a["matrices"]["P"][2], c=a["q"], A=a["matrices"]["A"][2], b=a["b"], G=a["matrices"]["C"][2],
             h_l=a["l"], h_u=a["u"], check_validity=False)


with wp.ScopedStream(s.stream):
    for i in (1, 0, 1, 0):
        update(inputs[i])
        s.solve()
        wp.synchronize()
    counters0 = s.native.counters() if args.backend == "native" else None
    driver = ct.CDLL("libcuda.so.1")
    assert driver.cuProfilerStart() == 0
    update(inputs[1])
    s.solve()
    wp.synchronize()
    assert driver.cuProfilerStop() == 0
    counters1 = s.native.counters() if args.backend == "native" else None
r = s.result
info = r.info.to_host()
status = Status(int(info.status[0])).name
x = r.x.numpy().reshape(-1)
y = r.y.numpy().reshape(-1)
z = r.z_u.numpy().reshape(-1) - r.z_l.numpy().reshape(-1)
v = check_solution(other, x, y, z, 1e-6)
record = dict(backend=args.backend, dataset=args.dataset, graph_mode=args.graph_mode if args.backend == "native" else None,
              kkt_backend=args.kkt_backend if args.backend == "native" else None,
              custom_spmv=args.custom_spmv if args.backend == "native" else None,
              eps=args.eps, source_cases=[cases[0]["id"], other_case], status=status,
              iterations=int(info.iter[0]), validation=v,
              profile_range="One warmed update(all matrices/vectors)+solve+device completion; inputs pre-uploaded; "
                            "setup, JIT, first capture, status reads, downloads and validation outside the range.",
              native_counters_in_range=({k: counters1[k] - counters0[k] for k in counters1} if counters0 else None))
(args.output / f"{args.dataset}-{args.backend}.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
assert status == "CUPIQP_SOLVED" and v["kkt_pass"]
