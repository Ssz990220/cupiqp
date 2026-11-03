"""Parity harness for the ipm_kernels module.

Each test builds the inputs of one reference kernel twice from the same seed
(a ``build(seed)`` callable returning a dict keyed by the reference kernel
parameter names), runs the reference Warp kernel on one copy and the native
launcher on the other, and compares every array argument afterwards.

Building twice (instead of cloning) keeps strided column slices of the
Variables buffers strided on both sides, so the native RowsView strides are
exercised exactly as the reference sees them.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from cupiqp.settings import SettingsFloatIdx, SettingsIntIdx
from cupiqp.utils import column_slice

DEV = "cuda:0"


# ---------------------------------------------------------------- arrays
def f64(a) -> wp.array:
    return wp.array(np.ascontiguousarray(a, dtype=np.float64), dtype=wp.float64, device=DEV)


def i32(a) -> wp.array:
    return wp.array(np.ascontiguousarray(a, dtype=np.int32), dtype=wp.int32, device=DEV)


def boolean(a) -> wp.array:
    return wp.array(np.ascontiguousarray(a, dtype=np.bool_), dtype=wp.bool, device=DEV)


def encode(v):
    """Native `_testing` encoding of one argument (see bind_ipm_kernels.cpp)."""
    if isinstance(v, wp.array):
        if v.size == 0 or v.ptr is None:
            return None
        if v.ndim == 1:
            assert v.strides[0] == np.dtype(wp.dtype_to_numpy(v.dtype)).itemsize, "1-D arrays must be contiguous"
            return int(v.ptr)
        assert v.ndim == 2
        item = np.dtype(wp.dtype_to_numpy(v.dtype)).itemsize
        assert v.strides[1] == item, "rows must be contiguous"
        return (int(v.ptr), int(v.strides[0] // item))
    return v


def host(v) -> np.ndarray:
    return v.numpy().copy()


# ---------------------------------------------------------------- settings
def settings_floats(**over) -> np.ndarray:
    F = SettingsFloatIdx
    vals = {
        F.rho_init: 1e-6, F.delta_init: 1e-4, F.eps_abs: 1e-8, F.eps_rel: 1e-9,
        F.eps_duality_gap_abs: 1e-8, F.eps_duality_gap_rel: 1e-9,
        F.infeasibility_threshold: 0.9, F.reg_lower_limit: 1e-10,
        F.reg_finetune_lower_limit: 1e-13, F.tau: 0.99,
        F.iterative_refinement_eps_abs: 1e-12, F.iterative_refinement_eps_rel: 1e-12,
        F.iterative_refinement_min_improvement_rate: 5.0,
    }
    for k, v in over.items():
        vals[getattr(F, k)] = v
    out = np.zeros(len(F))
    for k, v in vals.items():
        out[int(k)] = v
    return out


def settings_ints(**over) -> np.ndarray:
    I = SettingsIntIdx
    vals = {I.check_duality_gap: 1, I.reg_finetune_primal_update_threshold: 25,
            I.reg_finetune_dual_update_threshold: 25, I.max_iter: 250,
            I.max_factor_retires: 10, I.iterative_refinement_max_iter: 10}
    for k, v in over.items():
        vals[getattr(I, k)] = v
    out = np.zeros(len(I), dtype=np.int32)
    for k, v in vals.items():
        out[int(k)] = v
    return out


# ---------------------------------------------------------------- variables
class Vars:
    """Variables-like buffers ([x | s_l | s_u | s_bl | s_bu], [y | z_l | ...])
    with the reference column_slice views (strided when B > 1)."""

    def __init__(self, B, n, p, num_hl, num_hu, num_xl, num_xu, primal: np.ndarray, dual: np.ndarray):
        num_ineq = num_hl + num_hu + num_xl + num_xu
        assert primal.shape == (B, n + num_ineq) and dual.shape == (B, p + num_ineq)
        self.primals_all = f64(primal)
        self.duals_all = f64(dual)
        P, D = self.primals_all, self.duals_all
        self.x = column_slice(P, 0, n)
        self.s_all = column_slice(P, n, None)
        o = n
        self.s_l = column_slice(P, o, o + num_hl); o += num_hl
        self.s_u = column_slice(P, o, o + num_hu); o += num_hu
        self.s_bl = column_slice(P, o, o + num_xl); o += num_xl
        self.s_bu = column_slice(P, o, o + num_xu)
        self.y = column_slice(D, 0, p)
        self.z_all = column_slice(D, p, None)
        o = p
        self.z_l = column_slice(D, o, o + num_hl); o += num_hl
        self.z_u = column_slice(D, o, o + num_hu); o += num_hu
        self.z_bl = column_slice(D, o, o + num_xl); o += num_xl
        self.z_bu = column_slice(D, o, o + num_xu)


# ---------------------------------------------------------------- run + compare
def param_names(kernel) -> list[str]:
    return [a.label for a in kernel.adj.args]


def run_pair(build, kernel, native_fn, native_args=(), *, dim=None, tiled_dim=None, block_dim=None):
    """Run the reference kernel and the native launcher on two copies of the inputs.

    Returns (ref, nat): the two argument dicts after the launches.
    """
    ref = build()
    nat = build()
    names = param_names(kernel)
    missing = [k for k in names if k not in ref]
    assert not missing, f"build() lacks reference parameters {missing}"
    inputs = [ref[k] for k in names]
    if tiled_dim is not None:
        wp.launch_tiled(kernel, dim=tiled_dim, inputs=inputs, block_dim=block_dim, device=DEV)
    else:
        wp.launch(kernel, dim=dim, inputs=inputs, device=DEV)
    wp.synchronize()
    native_fn({k: encode(v) for k, v in nat.items()}, *native_args)
    wp.synchronize()
    return ref, nat


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ULP distance between float64 arrays (0 where both are NaN or equal)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.int64(0x7FFFFFFFFFFFFFFF)
    ia = a.view(np.int64)
    ib = b.view(np.int64)
    oa = np.where(ia >= 0, ia, -(ia & mask))  # monotone in the float value
    ob = np.where(ib >= 0, ib, -(ib & mask))
    same_side = (oa >= 0) == (ob >= 0)
    with np.errstate(over="ignore"):
        d = np.where(same_side, np.abs(oa - ob), 0).astype(np.float64)
    # opposite signs: |oa| + |ob| (computed in float, only large values matter)
    d = np.where(same_side, d, np.abs(oa).astype(np.float64) + np.abs(ob).astype(np.float64))
    both_nan = np.isnan(a) & np.isnan(b)
    d[both_nan] = 0
    d[np.isnan(a) ^ np.isnan(b)] = np.inf
    return d


class Report:
    """Collects per-array parity numbers (max ulp, max abs, max rel)."""

    def __init__(self):
        self.rows = []

    def add(self, name, key, ulp, abs_err, rel_err):
        self.rows.append((name, key, ulp, abs_err, rel_err))


REPORT = Report()


def compare(ref: dict, nat: dict, keys=None, *, ulp=4, rtol_keys=(), rtol=1e-14, atol_keys=(), atol=0.0,
            label=""):
    """Compare every array argument. Integer/bool arrays must match exactly;
    floats within `ulp` ulps, except `rtol_keys` (reduction outputs and their
    dependents) within relative `rtol` (absolute `atol` for `atol_keys`)."""
    keys = keys if keys is not None else [k for k, v in ref.items() if isinstance(v, wp.array)]
    for k in keys:
        r, n = ref[k], nat[k]
        if not isinstance(r, wp.array) or r.size == 0:
            continue
        hr, hn = host(r).ravel(), host(n).ravel()
        if hr.dtype.kind in "biu":
            assert np.array_equal(hr, hn), f"{label}{k}: integer/bool mismatch\nref={hr}\nnat={hn}"
            continue
        same_nan = np.array_equal(np.isnan(hr), np.isnan(hn))
        assert same_nan, f"{label}{k}: NaN pattern differs"
        fin = ~np.isnan(hr)
        # same infinities
        assert np.array_equal(hr[np.isinf(hr)], hn[np.isinf(hr)]), f"{label}{k}: inf mismatch"
        d_ulp = ulp_distance(hr, hn)
        diff = np.abs(np.where(fin, hr, 0) - np.where(fin, hn, 0))
        diff[np.isinf(hr)] = 0
        scale = np.maximum(np.abs(np.where(fin, hr, 0)), np.finfo(np.float64).tiny)
        rel = diff / scale
        max_ulp = float(d_ulp.max()) if d_ulp.size else 0.0
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float(rel[np.isfinite(rel)].max()) if diff.size else 0.0
        REPORT.add(label, k, max_ulp, max_abs, max_rel)
        if k in atol_keys:
            assert max_abs <= atol, f"{label}{k}: max abs diff {max_abs} > {atol}"
        elif k in rtol_keys:
            ok = (d_ulp <= ulp) | (rel <= rtol)
            assert ok.all(), f"{label}{k}: max rel diff {max_rel} > {rtol} (max ulp {max_ulp})"
        else:
            assert max_ulp <= ulp, f"{label}{k}: max ulp {max_ulp} > {ulp} (max abs {max_abs})"
