"""Generality check of the device warm start on the MPC replay dataset
(native/bench/data/mpc: one pattern, 16 trajectories x 4 control steps).

Each trajectory is a chain: its first QP cold, the next three warm started on
the device from the previous solve (identity row matching; the pattern is
fixed). One solver for all trajectories (reset_warm_start() between them).
--passes chains per config, the first a warm-up; time = median of the rest.

Config spec as in run_device.py (dev:s_min=..,z_min=..,K=..,back=..; cold).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpc_common import load_mpc  # noqa: E402
from run_device import parse_spec  # noqa: E402
from ws_common import RunSolver, kkt_check, kkt_pass, raw_ns, scipy_mats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rec_f = open(args.out / "records.jsonl", "a")
    trajs = load_mpc()
    chains = list(trajs.values())
    first = chains[0][0]
    mats = {q["id"]: scipy_mats(q) for c in chains for q in c}
    rs = RunSolver(first, settings=dict(eps_abs=args.eps, eps_duality_gap_abs=args.eps))
    cold_x = None
    for spec in args.configs:
        name, kw = parse_spec(spec)
        dev = name == "dev"
        if dev:
            rs.s.set_warm_start(s_min=float(kw.get("s_min", 0.1)), z_min=float(kw.get("z_min", 1e-4)),
                                z_unmatched=float(kw.get("z_unm", 1e-4)), previous_y=bool(int(kw.get("y", 1))),
                                snapshot_slots=int(kw.get("K", 0)), snapshot_back=int(kw.get("back", 0)))
        per = {}
        for ps in range(args.passes):
            for ci, chain in enumerate(chains):
                if dev:
                    rs.s.reset_warm_start()
                for k, q in enumerate(chain):
                    rs.load(q)
                    c0 = rs.s.native.counters()
                    t0 = raw_ns()
                    if dev:
                        rs.s.native.solve_warm_start(False)
                    else:
                        rs.s.native.solve()
                    rs.s.native.synchronize()
                    t = (raw_ns() - t0) * 1e-6
                    c1 = rs.s.native.counters()
                    i = rs.info()
                    d = per.setdefault((ci, k), dict(ms=[]))
                    if ps:
                        d["ms"].append(t)
                    if ps == 0:
                        d.update(status=i["status"], iter=i["iter"])
                    else:
                        assert d["iter"] == i["iter"], (spec, ci, k, "nondeterministic")
                    if ps == args.passes - 1:
                        d["launches"] = c1["graph_launches"] - c0["graph_launches"]
                        d["syncs"] = c1["host_syncs"] - c0["host_syncs"]
                        r = rs.result()
                        chk = kkt_check(q, r["x"], r["y"], r["z_l"], r["z_u"], mats[q["id"]])
                        d.update(kkt=chk, kkt_pass=kkt_pass(chk), x=r["x"])
        if spec == "cold":
            cold_x = {key: d["x"] for key, d in per.items()}
        warm_it = [d["iter"] for (ci, k), d in per.items() if k > 0]
        print(f"{spec:50s} iter med {np.median(warm_it):.0f} mean {np.mean(warm_it):.2f} max {max(warm_it)}",
              flush=True)
        for (ci, k), d in per.items():
            x = d.pop("x")
            ms = d.pop("ms")
            rec = dict(config=spec, run=chains[ci][0]["traj"], k=k, case=chains[ci][k]["id"],
                       ms=float(np.median(ms)) if ms else None, ms_all=ms, **d)
            if cold_x is not None:
                rec["dx_inf"] = float(np.max(np.abs(x - cold_x[(ci, k)])))
            rec_f.write(json.dumps(rec) + "\n")
        rec_f.flush()
    rec_f.close()


if __name__ == "__main__":
    main()
