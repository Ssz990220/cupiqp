"""Loader of the MPC replay dataset (native/bench/data/mpc, written by
native/bench/tools/export_snapshots.py) in the form of ws_common.load_qp:
min 1/2 x'Px + q'x  s.t.  A x = b,  l <= C x <= u, bounds clamped to +/-1e20.
Cases are grouped by trajectory and ordered by control step."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from ws_common import INF, REPO

MPC = REPO / "native/bench/data/mpc"


def load_mpc():
    man = json.loads((MPC / "manifest.json").read_text())
    g = man["groups"][0]
    raw = np.fromfile(MPC / g["file"], dtype=np.uint8)
    assert raw[:8].tobytes() == b"CPQPSNP1"
    hdr = np.frombuffer(raw[8:36].tobytes(), dtype=np.int32)
    n, p, m, nP, nA, nC, ncase = (int(v) for v in hdr)
    off = 36

    def ints(k):
        nonlocal off
        a = np.frombuffer(raw[off:off + 4 * k].tobytes(), dtype=np.int32).astype(np.int64)
        off += 4 * k
        return a

    Pp, Pi = ints(n + 1), ints(nP)
    Ap, Ai = ints(p + 1), ints(nA)
    Cp, Ci = ints(m + 1), ints(nC)
    per = nP + nA + nC + n + p + m + m
    vals = np.frombuffer(raw[off:off + 8 * per * ncase].tobytes(), dtype=np.float64).reshape(ncase, per)
    meta = {c["id"]: c for c in g["case_meta"]}
    qps = []
    for ci, cid in enumerate(g["cases"]):
        v = vals[ci]
        o = np.cumsum([0, nP, nA, nC, n, p, m, m])
        lb, ub = v[o[5]:o[6]], v[o[6]:o[7]]
        qps.append(dict(id=cid, traj=meta[cid]["trajectory_name"], step=meta[cid]["control_step"], n=n, p=p, m=m,
                        P=(Pp, Pi, v[o[0]:o[1]].copy()), A=(Ap, Ai, v[o[1]:o[2]].copy()),
                        G=(Cp, Ci, v[o[2]:o[3]].copy()), c=v[o[3]:o[4]].copy(), b=v[o[4]:o[5]].copy(),
                        h_l=np.maximum(lb, -INF), h_u=np.minimum(ub, INF)))
    trajs = collections.OrderedDict()
    for q in qps:
        trajs.setdefault(q["traj"], []).append(q)
    for t in trajs.values():
        t.sort(key=lambda q: q["step"])
    return trajs
