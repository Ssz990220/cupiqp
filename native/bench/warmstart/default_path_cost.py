"""Cold (default-path) solves of all captured QPs with a given build: prints
per-run median solve time, iterations and a hash of every x, so two builds
(e.g. a9f74da vs the hook branch) can be compared bitwise and in time.

    CUPIQP_NATIVE_BUILD=<build> python default_path_cost.py --reps 5 --out f.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_common import RUNS, RunSolver, load_qp, num_qps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()


class ColdOnly(RunSolver):
    def __init__(self, qp0):  # no warm buffers: the base build has no hook
        cn, wp = __import__("ws_common").import_native()
        self.cn, self.wp = cn, wp
        dev = lambda a: wp.array(np.ascontiguousarray(a), dtype=wp.float64, device="cuda:0")  # noqa: E731
        self.pattern = {k: (qp0[k][0].copy(), qp0[k][1].copy()) for k in ("P", "A", "G")}
        s = cn.SparseSolver()
        st = s.settings
        st.eps_abs, st.eps_rel, st.eps_duality_gap_abs, st.eps_duality_gap_rel, st.max_iter = 1e-6, 0.0, 1e-6, 0.0, 250
        keep = [dev(qp0[k][2]) for k in ("P", "A", "G")] + [dev(qp0[k]) for k in ("c", "b", "h_l", "h_u")]
        s.setup((qp0["P"][0], qp0["P"][1], keep[0]), keep[3], A=(qp0["A"][0], qp0["A"][1], keep[1]), b=keep[4],
                G=(qp0["G"][0], qp0["G"][1], keep[2]), h_l=keep[5], h_u=keep[6])
        s.native.synchronize()
        self.s = s
        self.host_in = s.host_input_buffers()
        self.res = s.native.result_ptrs()


out = {}
for run in RUNS:
    qps = [load_qp(run, k) for k in range(num_qps(run))]
    rs = ColdOnly(qps[0])
    rs.load(qps[0]); rs.solve()
    ms, iters, h = [], [], hashlib.sha256()
    for q in qps:
        rs.load(q)
        ts = [rs.solve() for _ in range(args.reps + 1)][1:]
        ms.append(float(np.median(ts)))
        iters.append(rs.info()["iter"])
        h.update(rs.result()["x"].tobytes())
    out[run] = dict(ms_median=float(np.median(ms)), ms_mean=float(np.mean(ms)), ms=ms, iters=iters,
                    x_sha256=h.hexdigest())
    print(run, f"median {np.median(ms):.3f} ms mean {np.mean(ms):.3f} ms", h.hexdigest()[:16])
args.out.write_text(json.dumps(out, indent=1))
