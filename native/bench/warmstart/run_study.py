"""Warm-start study runner.

For each run: one native solver (pattern fixed), every QP solved cold (the
baseline and the source of the QP-k state), then every strategy on every
transition k -> k+1. Writes one JSON line per (strategy, run, k) to
<out>/records.jsonl and the cold solutions to <out>/cold_<run>.npz.

Strategy spec: name[:key=val,...][@rho=..,delta=..]; e.g.
    x0_duals:s_min=1e-4,z_min=1e-4@rho=1e-8,delta=1e-6
--chain: the QP-k state is the strategy's own solution of QP k (deployment);
otherwise the cold solution of QP k (default).

Usage:
    python run_study.py --out DIR --reps 3 --strategies cold x0_duals ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_common import RUNS, RunSolver, kkt_check, kkt_pass, load_qp, num_qps, scipy_mats  # noqa: E402
from ws_strategies import build, label_map  # noqa: E402

DEFAULT_RHO, DEFAULT_DELTA = 1e-6, 1e-4


def parse_spec(spec):
    name, _, sett = spec.partition("@")
    name, _, kws = name.partition(":")
    kw = {k: (v if k == "map" else float(v)) for k, v in (t.split("=") for t in kws.split(",") if t)}
    st = {k: float(v) for k, v in (t.split("=") for t in sett.split(",") if t)}
    return name, kw, st


_SETTING_NAMES = {"rho": "rho_init", "delta": "delta_init"}
_DEFAULTS = {"rho_init": DEFAULT_RHO, "delta_init": DEFAULT_DELTA, "eps_abs": 1e-6, "eps_duality_gap_abs": 1e-6}


def apply_settings(rs, st):
    """Strategy settings (rho, delta, eps_abs, eps_duality_gap_abs); others default."""
    want = dict(_DEFAULTS)
    for k, v in st.items():
        want[_SETTING_NAMES.get(k, k)] = v
    for k, v in want.items():
        setattr(rs.s.settings, k, v)


def timed_solves(rs, reps, warm=None, fields="xysz"):
    ts = []
    for r in range(reps + 1):
        t = rs.solve(warm if r == 0 else (True if warm is not None else None), fields)
        if r:
            ts.append(t)
    return float(np.median(ts)), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--max-qps", type=int, default=0)
    ap.add_argument("--strategies", nargs="+", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rec_f = open(args.out / "records.jsonl", "a")

    for run in args.runs:
        N = num_qps(run) if not args.max_qps else min(args.max_qps, num_qps(run))
        qps = [load_qp(run, k) for k in range(N)]
        mats = [scipy_mats(q) for q in qps]
        rs = RunSolver(qps[0])
        # warm-up: one untimed cold solve
        rs.load(qps[0]); rs.solve()
        cold = []
        for k, q in enumerate(qps):
            rs.load(q)
            rs.set_rho_delta(DEFAULT_RHO, DEFAULT_DELTA)
            t, ts = timed_solves(rs, args.reps)
            i, r = rs.info(), rs.result()
            chk = kkt_check(q, r["x"], r["y"], r["z_l"], r["z_u"], mats[k])
            cold.append(r)
            rec = dict(strategy="cold", run=run, k=k, status=i["status"], iter=i["iter"], ms=t, ms_all=ts,
                       kkt=chk, kkt_pass=kkt_pass(chk), dx_inf=0.0, dx_rel=0.0,
                       dx_rec=float(np.max(np.abs(r["x"] - q["rec_x"]))), rec_iter=q["rec_iter"])
            rec_f.write(json.dumps(rec) + "\n")
        np.savez(args.out / f"cold_{run}.npz", **{f"{k}_{key}": v for k, r in enumerate(cold) for key, v in r.items()})
        rec_f.flush()
        idx_cache = {}

        def idxs_for(mode):
            if mode not in idx_cache:
                idx_cache[mode] = [None] + [label_map(qps[k - 1]["labels"], qps[k]["labels"], mode)
                                            for k in range(1, N)]
            return idx_cache[mode]

        for spec in args.strategies:
            if spec == "cold":
                continue
            name, kw, st = parse_spec(spec)
            idxs = idxs_for(kw.pop("map", "pair"))
            prev = cold[0]
            cold_iter = [None] * N
            for r in (json.loads(l) for l in open(args.out / "records.jsonl")):
                if r["strategy"] == "cold" and r["run"] == run:
                    cold_iter[r["k"]] = r["iter"]
            for k in range(1, N):
                q = qps[k]
                if name.startswith("gg_"):  # truncated re-solve of QP k-1 (not timed)
                    rs.load(qps[k - 1])
                    rs.set_rho_delta(DEFAULT_RHO, DEFAULT_DELTA)
                    rs.s.settings.max_iter = max(1, cold_iter[k - 1] - int(kw.get("back", 6)))
                    rs.solve()
                    prev = rs.result()
                    rs.s.settings.max_iter = 250
                rs.load(q)
                ctx = dict(idx=idxs[k], G=mats[k][2])
                if name == "cold_settings":  # no start; only settings differ
                    start, fields = None, "xysz"
                else:
                    start, fields, _ = build(name, prev, q, ctx, **kw)
                apply_settings(rs, st)
                t, ts = timed_solves(rs, args.reps, start, fields)
                i, r = rs.info(), rs.result()
                chk = kkt_check(q, r["x"], r["y"], r["z_l"], r["z_u"], mats[k])
                xc = cold[k]["x"]
                rec = dict(strategy=spec, run=run, k=k, status=i["status"], iter=i["iter"], ms=t, ms_all=ts,
                           kkt=chk, kkt_pass=kkt_pass(chk), dx_inf=float(np.max(np.abs(r["x"] - xc))),
                           dx_rel=float(np.linalg.norm(r["x"] - xc) / max(np.linalg.norm(xc), 1e-300)),
                           obj_diff=float(chk["obj"] - kkt_check(q, xc, cold[k]["y"], cold[k]["z_l"], cold[k]["z_u"], mats[k])["obj"]),
                           start_mu=None)
                rec_f.write(json.dumps(rec) + "\n")
                prev = r if (args.chain and i["status"] == 0) else cold[k]
                apply_settings(rs, {})
            rec_f.flush()
        del rs
    rec_f.close()


if __name__ == "__main__":
    main()
