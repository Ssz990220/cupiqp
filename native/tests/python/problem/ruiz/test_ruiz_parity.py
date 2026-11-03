"""Parity of the native Ruiz equilibration (M3) with the pinned reference
``cupiqp.sparse.sparse_preconditioner.SparseRuizEquilibration``.

Each test builds two identical reference ``SparseData`` objects from the same
host problem. The reference preconditioner works on one, the native
``_testing.RuizTester`` on the other (through device addresses), and every
output is compared: scaled data, scaling state, unscale factors, the RHS norm,
the number of Ruiz passes, (un)scale_solution, and the solver's update
sequence (unscale_data -> overwrite blocks -> reuse_scaling or reset +
scale_data).

Run with ``-s`` to print the parity summary (max abs / rel differences).
"""
import zlib

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp.results import Variables
from cupiqp.sparse.sparse_data import SparseData
from cupiqp.sparse.sparse_preconditioner import SparseRuizEquilibration
from cupiqp_native._interop import dev_ptr
from tests.helpers import random_feasible_qp, random_sparse_qp

# Relative tolerance (DEPENDENCY_GRAPH M3 acceptance: 1e-13). Only the cost
# scaling mean (a block sum) can differ from Warp's tile_sum in the last bits.
RTOL = 1e-13

REPORT = {}  # case -> (max_abs, max_rel, n_bitwise, n_arrays, passes_ref, passes_nat)


@pytest.fixture(scope="module", autouse=True)
def _print_report():
    yield
    if REPORT:
        print("\nRuiz parity summary (case: max_abs max_rel bitwise/arrays passes ref/native)")
        for k, (a, r, nb, na, pr, pn) in sorted(REPORT.items()):
            print(f"  {k}: {a:.3e} {r:.3e} {nb}/{na} {pr}/{pn}")


# ---------------------------------------------------------------- problems
def _csr(M):
    """CSR triple keeping explicit zeros (the input pattern is preserved)."""
    M = sp.csr_matrix(M)
    M.sort_indices()
    return (wp.array(M.indptr.astype(np.int32), dtype=wp.int32, device="cuda:0"),
            wp.array(M.indices.astype(np.int32), dtype=wp.int32, device="cuda:0"),
            wp.array(M.data.astype(np.float64), dtype=wp.float64, device="cuda:0"))


def _vec(v):
    return None if v is None else wp.array(np.asarray(v, np.float64), dtype=wp.float64, device="cuda:0")


def _make_data(qp):
    d = SparseData()
    d.init(P=_csr(qp["P"]), c=_vec(qp["c"]),
           A=None if qp.get("A") is None else _csr(qp["A"]), b=_vec(qp.get("b")),
           G=None if qp.get("G") is None else _csr(qp["G"]),
           h_u=_vec(qp.get("h_u")), h_l=_vec(qp.get("h_l")),
           x_u=_vec(qp.get("x_u")), x_l=_vec(qp.get("x_l")))
    return d


def _drop(qp, *keys):
    return {k: (None if k in keys else v) for k, v in qp.items()}


def _with_explicit_zeros(qp, rng):
    """Sparse problem where some stored entries of P (symmetric pairs), A and
    G are explicit zeros."""
    out = dict(qp)
    for k in ("A", "G"):
        M = sp.csr_matrix(qp[k]).copy()
        M.data[rng.random(M.nnz) < 0.2] = 0.0
        out[k] = M
    P = sp.csr_matrix(qp["P"]).tocoo()
    off = (P.row != P.col) & (rng.random(P.nnz) < 0.2)
    Pd = P.toarray()
    for r, c in zip(P.row[off], P.col[off]):
        Pd[r, c] = Pd[c, r] = 0.0
    pattern = sp.csr_matrix(qp["P"]).copy()
    pattern.sort_indices()
    rows = np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr))
    pattern.data = Pd[rows, pattern.indices]
    out["P"] = pattern
    return out


def _badly_scaled(qp, rng):
    """Rows/columns spanning many decades so clamping and several passes occur."""
    out = dict(qp)
    n = qp["P"].shape[0]
    sx = 10.0 ** rng.uniform(-5, 5, n)
    P = sp.csr_matrix(qp["P"])
    out["P"] = sp.csr_matrix(sp.diags(sx) @ P @ sp.diags(sx))
    out["c"] = qp["c"] * sx * 1e3
    for k, vk in (("A", ("b",)), ("G", ("h_l", "h_u"))):
        if qp.get(k) is None:
            continue
        M = sp.csr_matrix(qp[k])
        sr = 10.0 ** rng.uniform(-6, 6, M.shape[0])
        out[k] = sp.csr_matrix(sp.diags(sr) @ M @ sp.diags(sx))
        for v in vk:
            if qp.get(v) is not None:
                out[v] = qp[v] * sr
    for v in ("x_l", "x_u"):
        if qp.get(v) is not None:
            out[v] = qp[v] / sx
    return out


def _problem(name):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    if name == "feasible":
        qp = random_feasible_qp(1, n=5, p=1, m=3)
    elif name == "sparse":
        qp = random_sparse_qp(2, n=30, p=8, m=12, density=0.2)
    elif name == "sparse_zeros":
        qp = _with_explicit_zeros(random_sparse_qp(3, n=30, p=8, m=12, density=0.3), rng)
    elif name == "sparse_inf":
        qp = random_sparse_qp(4, n=40, p=5, m=20, density=0.15, inf_fraction=0.3)
    elif name == "bad_scaling":
        qp = _badly_scaled(random_sparse_qp(5, n=25, p=6, m=10, density=0.3, inf_fraction=0.2), rng)
    elif name == "large":
        qp = random_sparse_qp(6, n=300, p=40, m=150, density=0.02, inf_fraction=0.1)
    else:
        raise KeyError(name)
    return {k: (sp.csr_matrix(v) if k in ("P", "A", "G") else np.asarray(v, np.float64)) for k, v in qp.items()}


# Block layouts: (name, keys dropped from the problem).
LAYOUTS = {
    "all": (),
    "no_A": ("A", "b"),
    "no_G": ("G", "h_l", "h_u"),
    "no_AG": ("A", "b", "G", "h_l", "h_u"),
    "only_hu": ("h_l",),
    "only_hl": ("h_u",),
    "no_box": ("x_l", "x_u"),
    "only_xl": ("x_u",),
    "only_xu": ("x_l",),
    "only_P": ("A", "b", "G", "h_l", "h_u", "x_l", "x_u"),
}


# ---------------------------------------------------------------- pairs
def _ptr(a, dtype=np.float64):
    if a is None or int(np.prod(a.shape)) == 0:
        return 0
    return dev_ptr(a, dtype)


def _native(testing, d):
    dims = dict(n=d.n, p=d.p, m=d.m, num_hl=d.num_hl, num_hu=d.num_hu, num_xl=d.num_xl, num_xu=d.num_xu,
                nnz_P=int(d.P.nnz), nnz_A=int(d.A.nnz), nnz_G=int(d.G.nnz))
    ptrs = {}
    for name, M in (("P", d.P), ("A", d.A), ("G", d.G)):
        ptrs[f"{name}_indptr"] = _ptr(M.indptr, np.int32)
        ptrs[f"{name}_indices"] = _ptr(M.indices, np.int32)
        ptrs[f"{name}_row_idx"] = _ptr(M.row_indices, np.int32) if M.nnz else 0
        ptrs[f"{name}_values"] = _ptr(M.data)
    for k in ("c", "b", "h_l", "h_u", "x_l", "x_u", "finite_mask_all", "active_G_row", "active_x_bound",
              "num_finite_bounds"):
        ptrs[k] = _ptr(getattr(d, k))
    wp.synchronize()
    return testing.RuizTester(dims, ptrs)


def _reference(d):
    pc = SparseRuizEquilibration(1, d.n, d.p, d.m, has_h_l=d.has_h_l, has_h_u=d.has_h_u,
                                 has_x_l=d.has_x_l, has_x_u=d.has_x_u, active_x_bound=d.active_x_bound)
    pc.passes = 0
    orig = pc.compute_kkt_norms

    def counting(*a, **k):  # one call per Ruiz pass
        pc.passes += 1
        return orig(*a, **k)

    pc.compute_kkt_norms = counting
    return pc


class Pair:
    def __init__(self, testing, qp):
        self.qp = qp
        self.rd = _make_data(qp)
        self.nd = _make_data(qp)
        self.ref = _reference(self.rd)
        self.nat = _native(testing, self.nd)
        wp.synchronize()

    def native_state(self):
        out = {}
        for k, (ptr, size) in self.nat.state().items():
            if size == 0:
                out[k] = np.zeros(0)
                continue
            a = wp.array(ptr=int(ptr), dtype=wp.float64, shape=(int(size),), device="cuda:0", copy=False)
            out[k] = a.numpy().copy()
        return out

    def ref_state(self):
        r = self.ref
        return dict(delta=r.delta, delta_inv=r.delta_inv, delta_b=r.delta_b, delta_b_inv=r.delta_b_inv,
                    cost_scaling=r.cost_scaling, cost_scaling_inv=r.cost_scaling_inv, x_b_scaling=r.x_b_scaling,
                    x_b_scaling_init=r._x_b_scaling_init, dual_res_unscale_factor=r.dual_res_unscale_factor,
                    primal_res_unscale_factor=r.primal_res_unscale_factor)

    @staticmethod
    def data_arrays(d):
        return dict(P=d.P.data, A=d.A.data, G=d.G.data, c=d.c, b=d.b, h_l=d.h_l, h_u=d.h_u, x_l=d.x_l, x_u=d.x_u)

    def compare(self, case, keys=None, rtol=RTOL, data=True, state=True):
        wp.synchronize()
        pairs = []
        if data:
            ra, na = self.data_arrays(self.rd), self.data_arrays(self.nd)
            pairs += [(f"data.{k}", ra[k].numpy().ravel(), na[k].numpy().ravel()) for k in ra
                      if keys is None or k in keys]
        if state:
            rs, ns = self.ref_state(), self.native_state()
            pairs += [(f"state.{k}", rs[k].numpy().ravel(), ns[k].ravel()) for k in rs]
        return _check(case, pairs, rtol, self.ref.passes, self.nat.passes_run())


def _check(case, pairs, rtol, passes_ref=-1, passes_nat=-1):
    max_abs = max_rel = 0.0
    n_bit = 0
    for name, r, n in pairs:
        assert r.shape == n.shape, f"{case} {name}: shape {r.shape} vs {n.shape}"
        if r.size == 0:
            n_bit += 1
            continue
        same_inf = np.isinf(r) & (r == n)
        fin = ~same_inf
        diff = np.abs(r[fin] - n[fin])
        a = float(diff.max()) if diff.size else 0.0
        rel = float((diff / np.maximum(np.abs(r[fin]), 1e-300)).max()) if diff.size else 0.0
        if np.array_equal(r, n):
            n_bit += 1
        max_abs, max_rel = max(max_abs, a), max(max_rel, rel)
        np.testing.assert_allclose(n, r, rtol=rtol, atol=0.0, err_msg=f"{case}: {name}")
    REPORT[case] = (max_abs, max_rel, n_bit, len(pairs), passes_ref, passes_nat)
    return n_bit, len(pairs)


# ---------------------------------------------------------------- tests
PROBLEMS = ["feasible", "sparse", "sparse_zeros", "sparse_inf", "bad_scaling"]


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("scale_cost", [False, True])
@pytest.mark.parametrize("max_iter", [0, 1, 10])
def test_scale_data(testing, problem, layout, scale_cost, max_iter):
    pr = Pair(testing, _drop(_problem(problem), *LAYOUTS[layout]))
    pr.ref.scale_data(pr.rd, scale_cost, max_iter)
    pr.nat.scale_data(scale_cost, max_iter)
    case = f"scale/{problem}/{layout}/cost{int(scale_cost)}/it{max_iter}"
    pr.compare(case)
    assert pr.nat.passes_run() == pr.ref.passes, case
    if not scale_cost:
        # Without the cost-scaling sum every operation is order-independent.
        n_bit, n_arr = REPORT[case][2], REPORT[case][3]
        assert n_bit == n_arr, f"{case}: expected bitwise parity ({n_bit}/{n_arr})"


def test_large_problem(testing):
    pr = Pair(testing, _problem("large"))
    pr.ref.scale_data(pr.rd, True, 25)
    pr.nat.scale_data(True, 25)
    pr.compare("scale/large/all/cost1/it25")
    assert pr.nat.passes_run() == pr.ref.passes


def test_x_b_scaling_init_alias(testing):
    """Without box bounds the reference SparseSolver passes the all-zero
    active_x_bound, so x_b_scaling is zero (not one)."""
    pr = Pair(testing, _drop(_problem("sparse"), "x_l", "x_u"))
    st = pr.native_state()
    assert np.all(st["x_b_scaling_init"] == 0.0) and np.all(st["x_b_scaling"] == 0.0)
    pr.compare("init/no_box", data=False)


@pytest.mark.parametrize("problem", ["sparse_inf", "bad_scaling"])
@pytest.mark.parametrize("layout", ["all", "only_hu", "only_xl", "no_A"])
def test_rhs_inf_norm(testing, problem, layout):
    pr = Pair(testing, _drop(_problem(problem), *LAYOUTS[layout]))
    pr.ref.scale_data(pr.rd, True, 10)
    pr.nat.scale_data(True, 10)
    out_r = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    out_n = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    pr.ref.compute_constraints_rhs_inf_norm_unscaled(pr.rd, out_r)
    wp.synchronize()
    pr.nat.rhs_inf_norm(dev_ptr(out_n))
    _check(f"rhs/{problem}/{layout}", [("rhs", out_r.numpy(), out_n.numpy())], RTOL)


@pytest.mark.parametrize("layout", ["all", "only_hl", "only_xu", "no_AG", "no_box"])
def test_unscale_and_scale_solution(testing, layout):
    pr = Pair(testing, _drop(_problem("bad_scaling"), *LAYOUTS[layout]))
    pr.ref.scale_data(pr.rd, True, 10)
    pr.nat.scale_data(True, 10)
    rng = np.random.default_rng(7)
    vr, vn = Variables(), Variables()
    vr.init(pr.rd)
    vn.init(pr.nd)
    prim = rng.standard_normal(vr._primal_buffer.shape)
    dual = rng.standard_normal(vr._dual_buffer.shape)
    for v in (vr, vn):
        v._primal_buffer.assign(prim)
        v._dual_buffer.assign(dual)
    wp.synchronize()
    pr.ref.unscale_solution(vr, pr.rd)
    pr.nat.unscale_solution(dev_ptr(vn._primal_buffer), dev_ptr(vn._dual_buffer))
    _check(f"unscale_solution/{layout}", [("primal", vr._primal_buffer.numpy(), vn._primal_buffer.numpy()),
                                          ("dual", vr._dual_buffer.numpy(), vn._dual_buffer.numpy())], RTOL)
    pr.ref.scale_solution(vr, pr.rd)
    pr.nat.scale_solution(dev_ptr(vn._primal_buffer), dev_ptr(vn._dual_buffer))
    _check(f"scale_solution/{layout}", [("primal", vr._primal_buffer.numpy(), vn._primal_buffer.numpy()),
                                        ("dual", vr._dual_buffer.numpy(), vn._dual_buffer.numpy())], RTOL)
    # Round trip back to the original point.
    np.testing.assert_allclose(vn._primal_buffer.numpy(), prim, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(vn._dual_buffer.numpy(), dual, rtol=1e-14, atol=1e-14)


def _new_values(qp, rng, keys):
    """Fresh values for the replaced blocks (same patterns); bound updates
    flip some entries between finite and infinite."""
    new = {}
    for k in keys:
        v = qp[k]
        if k in ("P", "A", "G"):
            M = sp.csr_matrix(v).copy()
            M.sort_indices()
            if k == "P":  # keep symmetry: scale by a symmetric positive factor
                Md = M.toarray()
                Md = Md * (1.0 + 0.1 * rng.random()) + np.diag(rng.random(M.shape[0]))
                rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
                M.data = Md[rows, M.indices]
            else:
                M.data = M.data * rng.uniform(0.5, 2.0, M.nnz)
            new[k] = wp.array(M.data, dtype=wp.float64, device="cuda:0")
        else:
            x = np.array(v, np.float64) * rng.uniform(0.5, 1.5, v.shape)
            if k in ("h_l", "x_l"):
                x[rng.random(x.shape) < 0.3] = -np.inf
            elif k in ("h_u", "x_u"):
                x[rng.random(x.shape) < 0.3] = np.inf
            new[k] = wp.array(x, dtype=wp.float64, device="cuda:0")
    return new


UPDATES = {
    "b_only": ("b",),
    "bounds": ("b", "h_l", "h_u", "x_l", "x_u"),
    "box_only": ("x_l", "x_u"),
    "c_P": ("P", "c"),
    "A_G": ("A", "G"),
    "all": ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u"),
}


@pytest.mark.parametrize("problem", ["sparse_inf", "bad_scaling"])
@pytest.mark.parametrize("update", list(UPDATES))
@pytest.mark.parametrize("reuse_on_update", [False, True])
def test_update_sequence(testing, problem, update, reuse_on_update):
    """Reference SparseSolver._update_impl sequence, twice in a row."""
    pr = Pair(testing, _problem(problem))
    scale_cost, max_iter = True, 10
    pr.ref.scale_data(pr.rd, scale_cost, max_iter)
    pr.nat.scale_data(scale_cost, max_iter)
    pr.compare(f"update/{problem}/{update}/reuse{int(reuse_on_update)}/setup")
    rng = np.random.default_rng(11)
    keys = UPDATES[update]
    for rep in range(2):
        tag = f"update/{problem}/{update}/reuse{int(reuse_on_update)}/rep{rep}"
        pr.ref.unscale_data(pr.rd)
        pr.nat.unscale_data({k: True for k in keys})
        # Retained blocks and the scaling state match after unscale_data.
        retained = [k for k in ("P", "A", "G", "c", "b", "h_l", "h_u", "x_l", "x_u") if k not in keys]
        pr.compare(tag + "/unscaled", keys=retained)
        new = _new_values(pr.qp, rng, keys)
        for d in (pr.rd, pr.nd):
            for k in keys:
                getattr(d, f"set_{k}")(new[k], check=False)
        matrix_changed = any(k in ("P", "A", "G") for k in keys)
        if reuse_on_update or not matrix_changed:
            pr.ref.reuse_scaling(pr.rd)
            pr.nat.reuse_scaling()
        else:
            pr.ref.passes = 0
            pr.ref.reset()
            pr.ref.scale_data(pr.rd, scale_cost, max_iter)
            pr.nat.reset()
            pr.nat.scale_data(scale_cost, max_iter)
        if "x_l" in keys or "x_u" in keys:
            pr.ref.refresh_x_b_scaling()
            pr.nat.refresh_x_b_scaling()
        pr.compare(tag + "/rescaled")
        if not (reuse_on_update or not matrix_changed):
            assert pr.nat.passes_run() == pr.ref.passes
        out_r = wp.zeros(1, dtype=wp.float64, device="cuda:0")
        out_n = wp.zeros(1, dtype=wp.float64, device="cuda:0")
        pr.ref.compute_constraints_rhs_inf_norm_unscaled(pr.rd, out_r)
        wp.synchronize()
        pr.nat.rhs_inf_norm(dev_ptr(out_n))
        _check(tag + "/rhs", [("rhs", out_r.numpy(), out_n.numpy())], RTOL)
