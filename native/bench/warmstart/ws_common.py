"""Shared helpers of the warm-start study (native/docs/modules/warmstart_study.md).

Loads the captured TrajOpt SQP QPs, drives one native solver per run through
the staged-input path, and checks solutions against the QP in user
coordinates. Problem form handed to the solver (as the producer did):
    min 1/2 x'Px + c'x  s.t.  A x = b,  h_l <= G x <= h_u
with P = upper triangle of H mirrored, G = C, h_l/h_u = lb/ub clamped to
+/-1e20. Sign conventions of the solver: dual residual
    P x + c + A'y + G'(z_u - z_l),  s_u = h_u - G x,  s_l = G x - h_l.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "native/bench/data/warmstart/cupiqp_warmstart_sequences_20261104"
RUNS = ("29-79b3-place-6f87a360", "55-5d59-pick-d59c9e60")
INF = 1e20


def raw_ns():
    """CLOCK_MONOTONIC_RAW (see bench/harness/replay_updates.py)."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def num_qps(run):
    return len(list((DATA / run / "iterations").glob("iter_*")))


def _coo_to_csr(rows, cols, vals, shape):
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr), cols.astype(np.int64), vals.astype(np.float64)


def _csc_coo(z, key):
    data, indices, indptr = z[key + "_data"], z[key + "_indices"], z[key + "_indptr"]
    shape = tuple(int(v) for v in z[key + "_shape"])
    cols = np.repeat(np.arange(shape[1]), np.diff(indptr))
    return indices.astype(np.int64), cols.astype(np.int64), data, shape


def load_qp(run, k):
    z = np.load(DATA / run / "iterations" / f"iter_{k:03d}" / "problem.npz", allow_pickle=False)
    n = int(z["H_shape"][0])
    # P: every stored entry of H takes the value of its upper-triangle twin.
    r, c, v, _ = _csc_coo(z, "H")
    key = r * n + c
    up = r <= c
    ukey, uval = key[up], v[up]
    o = np.argsort(ukey)
    ukey, uval = ukey[o], uval[o]
    twin = np.where(up, key, c * n + r)
    pos = np.searchsorted(ukey, twin)
    assert np.all(ukey[pos] == twin), "H pattern not symmetric"
    P = _coo_to_csr(r, c, uval[pos], (n, n))
    r, c, v, sA = _csc_coo(z, "A")
    A = _coo_to_csr(r, c, v, sA)
    G = (z["C_indptr"].astype(np.int64), z["C_indices"].astype(np.int64), z["C_data"].astype(np.float64))
    m = int(z["C_shape"][0])
    lb, ub = z["lb"].astype(np.float64), z["ub"].astype(np.float64)
    return dict(
        run=run, k=k, n=n, p=int(sA[0]), m=m, P=P, A=A, G=G,
        c=z["g"].astype(np.float64), b=z["b"].astype(np.float64),
        h_l=np.maximum(lb, -INF), h_u=np.minimum(ub, INF), lb=lb, ub=ub,
        labels=z["ineq_labels"], rec_x=z["cupiqp_x"], rec_duals=z["cupiqp_ineq_duals"],
        rec_status=int(z["cupiqp_status"]), rec_iter=int(z["cupiqp_iterations"]))


def csr_matvec(M, nrows, x):
    indptr, indices, vals = M
    rows = np.repeat(np.arange(nrows), np.diff(indptr))
    out = np.zeros(nrows)
    np.add.at(out, rows, vals * x[indices])
    return out


def csr_rmatvec(M, nrows, ncols, y):
    indptr, indices, vals = M
    rows = np.repeat(np.arange(nrows), np.diff(indptr))
    out = np.zeros(ncols)
    np.add.at(out, indices, vals * y[rows])
    return out


def scipy_mats(qp):
    import scipy.sparse as sp
    n, p, m = qp["n"], qp["p"], qp["m"]
    mk = lambda M, s: sp.csr_matrix((M[2], M[1], M[0]), shape=s)  # noqa: E731
    return mk(qp["P"], (n, n)), mk(qp["A"], (p, n)), mk(qp["G"], (m, n))


def kkt_check(qp, x, y, z_l, z_u, mats=None):
    """Residuals of (x, y, z_l, z_u) on the QP in user coordinates."""
    P, A, G = mats if mats is not None else scipy_mats(qp)
    fl = qp["h_l"] > -INF
    fu = qp["h_u"] < INF
    Gx = G @ x
    eq = np.max(np.abs(A @ x - qp["b"])) if qp["p"] else 0.0
    viol = max(np.max(np.where(fu, Gx - qp["h_u"], 0.0)), np.max(np.where(fl, qp["h_l"] - Gx, 0.0)), 0.0)
    zl = np.where(fl, z_l, 0.0)
    zu = np.where(fu, z_u, 0.0)
    rd = P @ x + qp["c"] + A.T @ y + G.T @ (zu - zl)
    dual = np.max(np.abs(rd))
    xPx = x @ (P @ x)
    pobj = 0.5 * xPx + qp["c"] @ x
    dobj = -0.5 * xPx - qp["b"] @ y - np.sum(np.where(fu, qp["h_u"] * zu, 0.0)) + np.sum(np.where(fl, qp["h_l"] * zl, 0.0))
    gap = abs(pobj - dobj)
    zneg = max(-min(zl.min(), zu.min()), 0.0)
    return dict(eq=float(eq), ineq_viol=float(viol), dual=float(dual), gap=float(gap), z_neg=float(zneg),
                obj=float(pobj))


def kkt_pass(chk, tol=1e-6):
    return chk["eq"] <= tol and chk["ineq_viol"] <= tol and chk["dual"] <= tol and chk["gap"] <= tol \
        and chk["z_neg"] <= tol


def import_native(build=None):
    build = Path(build or os.environ.get("CUPIQP_NATIVE_BUILD", REPO / "build/native-ws"))
    sys.path.insert(0, str(build / "python"))
    import warp as wp
    wp.config.quiet = True
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", str(Path.home() / ".cache/warp-cupiqp-native"))
    wp.init()
    import cupiqp_native
    return cupiqp_native, wp


class RunSolver:
    """One native solver per SQP run (identical pattern), values replaced via
    the pinned host input buffers + update_staged(from_host=True)."""

    def __init__(self, qp0, settings=None):
        cn, wp = import_native()
        self.cn, self.wp = cn, wp
        dev = lambda a: wp.array(np.ascontiguousarray(a), dtype=wp.float64, device="cuda:0")  # noqa: E731
        self.pattern = {k: (qp0[k][0].copy(), qp0[k][1].copy()) for k in ("P", "A", "G")}
        s = cn.SparseSolver()
        st = s.settings
        st.eps_abs = 1e-6
        st.eps_rel = 0.0
        st.eps_duality_gap_abs = 1e-6
        st.eps_duality_gap_rel = 0.0
        st.max_iter = 250
        for k, v in (settings or {}).items():
            setattr(st, k, v)
        keep = [dev(qp0[k][2]) for k in ("P", "A", "G")] + [dev(qp0[k]) for k in ("c", "b", "h_l", "h_u")]
        s.setup((qp0["P"][0], qp0["P"][1], keep[0]), keep[3], A=(qp0["A"][0], qp0["A"][1], keep[1]), b=keep[4],
                G=(qp0["G"][0], qp0["G"][1], keep[2]), h_l=keep[5], h_u=keep[6])
        s.native.synchronize()
        self.s = s
        self.host_in = s.host_input_buffers()
        self.warm = s.warm_start_buffers(host=True)
        self.res = s.native.result_ptrs()

    def load(self, qp):
        for k in ("P", "A", "G"):
            assert np.array_equal(self.pattern[k][0], qp[k][0]) and np.array_equal(self.pattern[k][1], qp[k][1]), \
                f"pattern of {k} changed"
            self.host_in[k][:] = qp[k][2]
        for k in ("c", "b", "h_l", "h_u"):
            self.host_in[k][:] = qp[k]
        self.s.update_staged(from_host=True)
        self.s.native.synchronize()

    def set_rho_delta(self, rho, delta):
        self.s.settings.rho_init = rho
        self.s.settings.delta_init = delta

    def solve(self, warm=None, fields="xysz"):
        """Solve once; warm = dict(x, y, s_l, s_u, z_l, z_u) (already written
        if warm is True). Returns (wall ms of solve launch to completion)."""
        if isinstance(warm, dict):
            for k, v in warm.items():
                self.warm[k][:] = v
        t0 = raw_ns()
        if warm is None:
            self.s.native.solve()
        else:
            self.s.native.solve_warm(sum(self.s._WARM_BITS[ch] for ch in fields), True)
        self.s.native.synchronize()
        return (raw_ns() - t0) * 1e-6

    def info(self):
        return self.s.native.info()

    def result(self):
        out = {}
        for k in ("x", "y", "z_l", "z_u", "s_l", "s_u"):
            ptr, n = self.res[k]
            a = self.wp.array(ptr=ptr, dtype=self.wp.float64, shape=(n,), device="cuda:0") \
                if n else None
            out[k] = a.numpy().copy() if a is not None else np.zeros(0)
        return out
