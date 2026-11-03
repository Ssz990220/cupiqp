"""Parity of the native KKT assembly with the reference SparseKKTSolver.

The reference solver is built from a reference SparseData; its ``_kkt_mats``
holds the KKT indptr / indices / values. The native KktAssembly runs on
copies of the same device inputs. Patterns, slot maps and values must match
bitwise after init, after update_data (new values and bound switches) and
after the diagonal update.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp.sparse.sparse_data import SparseData
from cupiqp.sparse.sparse_kkt_solver import SparseKKTSolver
from cupiqp.sparse.csr_helpers import csr_diag_indices, csr_subblock_indices
from cupiqp_native._interop import dev_ptr

DEV = "cuda:0"


def wp64(a):
    return wp.array(np.ascontiguousarray(a, dtype=np.float64), dtype=wp.float64, device=DEV)


def wp32(a):
    return wp.array(np.ascontiguousarray(a, dtype=np.int32), dtype=wp.int32, device=DEV)


def random_csr(rng, rows, cols, density, symmetric=False, zero_frac=0.2, drop_diag=0.0):
    """CSR with sorted indices and explicit zeros kept as stored entries."""
    keep = rng.random((rows, cols)) < density
    if symmetric:
        keep = np.triu(keep) | np.triu(keep).T
        drop = rng.random(rows) < drop_diag
        np.fill_diagonal(keep, ~drop)
    r, c = np.nonzero(keep)
    vals = rng.standard_normal(r.size)
    vals[rng.random(r.size) < zero_frac] = 0.0
    if symmetric:
        dense = np.zeros((rows, cols))
        dense[r, c] = vals
        dense = np.triu(dense) + np.triu(dense, 1).T
        vals = dense[r, c]
    M = sp.csr_matrix((vals, (r, c)), shape=(rows, cols))
    M.sort_indices()
    assert M.nnz == r.size  # explicit zeros stay stored
    return M


def triple(M):
    return (wp32(M.indptr), wp32(M.indices), wp64(M.data))


def build_problem(rng, n, p, m, inf_frac=0.3, drop_diag=0.3):
    P = random_csr(rng, n, n, 0.3, symmetric=True, drop_diag=drop_diag)
    kw = dict(P=triple(P), c=wp64(rng.standard_normal(n)))
    A = G = None
    if p:
        A = random_csr(rng, p, n, 0.4)
        kw.update(A=triple(A), b=wp64(rng.standard_normal(p)))
    if m:
        G = random_csr(rng, m, n, 0.4)
        h_l = -1.0 - rng.random(m)
        h_u = 1.0 + rng.random(m)
        both = rng.random(m) < inf_frac  # inactive rows: both bounds infinite
        h_l[both] = -np.inf
        h_u[both] = np.inf
        kw.update(G=triple(G), h_l=wp64(h_l), h_u=wp64(h_u))
    data = SparseData(dtype=wp.float64, device=DEV)
    data.init(**kw)
    return data, P, A, G


def stream():
    return wp.get_stream(DEV).cuda_stream


def native_inputs(data):
    """Copies of the reference device inputs (kept alive by the caller)."""
    arrs = dict(
        P=wp.clone(data.P.data),
        A=wp.clone(data.A.data) if data.p else None,
        G=wp.clone(data.G.data) if data.m else None,
        G_rows=wp.clone(data.G.row_indices) if data.m else None,
        active=wp.clone(data.active_G_row) if data.m else None,
    )
    ptrs = dict(
        P_val=dev_ptr(arrs["P"], np.float64, data.P.nnz),
        A_val=dev_ptr(arrs["A"], np.float64, data.A.nnz) if data.p else 0,
        G_val=dev_ptr(arrs["G"], np.float64, data.G.nnz) if data.m else 0,
        G_rows=dev_ptr(arrs["G_rows"], np.int32, data.G.nnz) if data.m else 0,
        active_G_row=dev_ptr(arrs["active"], np.float64, data.m) if data.m else 0,
    )
    return arrs, ptrs


def make_native(testing, data):
    n, p, m = data.n, data.p, data.m
    P0, A0, G0 = data.P.pattern(), data.A.pattern(), data.G.pattern()
    arrs, ptrs = native_inputs(data)
    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)  # noqa: E731
    kkt = testing.KktAssembly(
        n, p, m, i32(P0.indptr), i32(P0.indices), i32(A0.indptr), i32(A0.indices),
        i32(G0.indptr), i32(G0.indices), stream=stream(), **ptrs)
    return kkt, arrs


def ref_values(ref):
    wp.synchronize_device(DEV)
    return ref._kkt_mats.data.numpy().reshape(-1)


def assert_bitwise(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, what
    bad = np.flatnonzero(a.view(np.uint64) != b.view(np.uint64))
    assert bad.size == 0, f"{what}: {bad.size} mismatches, first {bad[:5]}: {a[bad[:5]]} vs {b[bad[:5]]}"


SHAPES = [(12, 4, 9), (15, 0, 7), (10, 3, 0), (8, 0, 0), (30, 10, 40), (1, 1, 1)]


@pytest.mark.parametrize("n,p,m", SHAPES)
def test_pattern_maps_and_init_values(testing, n, p, m):
    rng = np.random.default_rng(100 + n + 7 * p + 13 * m)
    data, P, A, G = build_problem(rng, n, p, m)
    ref = SparseKKTSolver(data)
    kkt, _keep = make_native(testing, data)

    K = ref._kkt_mats
    ref_indptr = K.indptr.numpy()
    ref_indices = K._host_indices() if hasattr(K, "_host_indices") else K.indices.numpy()
    np.testing.assert_array_equal(kkt.indptr(), ref_indptr)
    np.testing.assert_array_equal(kkt.indices(), ref_indices)
    assert kkt.dim == n + p + m and kkt.nnz == ref_indices.size

    # Template values: the reference bmat of all-ones patterns.
    kkt0 = SparseKKTSolver._initialize_kkt_csr(data.P.pattern(), data.A.pattern(), data.G.pattern())
    assert_bitwise(kkt.template_values(), kkt0.data, "template")

    # Slot maps against the reference helpers.
    diag = csr_diag_indices(kkt0)
    np.testing.assert_array_equal(kkt.host_map("diag_x"), diag[:n])
    np.testing.assert_array_equal(kkt.host_map("diag_y"), diag[n:n + p])
    np.testing.assert_array_equal(kkt.host_map("diag_z"), diag[n + p:])
    np.testing.assert_array_equal(kkt.host_map("P_slots"), ref._P_indices.numpy())
    if p:
        np.testing.assert_array_equal(kkt.host_map("A_slots"), ref._A_indices.numpy())
    if m:
        np.testing.assert_array_equal(kkt.host_map("G_slots"), ref._G_indices.numpy())
        np.testing.assert_array_equal(kkt.host_map("G_slots"), csr_subblock_indices(data.G.pattern(), kkt0, n + p, 0))
    np.testing.assert_array_equal(kkt.host_map("P_diag_src"), ref._P_diag_src_cols.numpy())
    np.testing.assert_array_equal(kkt.host_map("P_diag_dst"), ref._P_diag_dst_cols.numpy())

    assert_bitwise(kkt.values(), ref_values(ref), "init values")
    assert_bitwise(kkt.P_diag(), ref._P_diag.numpy().reshape(-1), "P_diag")


@pytest.mark.parametrize("n,p,m", [(12, 4, 9), (15, 0, 7), (10, 3, 0), (30, 10, 40)])
def test_update_data_and_diagonal(testing, n, p, m):
    rng = np.random.default_rng(7 + n + 7 * p + 13 * m)
    data, P, A, G = build_problem(rng, n, p, m)
    ref = SparseKKTSolver(data)
    kkt, keep = make_native(testing, data)
    addr = kkt.values_ptr

    def run_update(flags):
        ref.update_data(data, update_P=flags[0], update_A=flags[1], update_G=flags[2])
        arrs, ptrs = native_inputs(data)
        keep.update(arrs)
        kkt.scatter(**ptrs, P=flags[0], A=flags[1], G=flags[2], stream=stream())
        assert kkt.values_ptr == addr
        assert_bitwise(kkt.values(), ref_values(ref), f"values after update {flags}")
        assert_bitwise(kkt.P_diag(), ref._P_diag.numpy().reshape(-1), f"P_diag after update {flags}")

    # Explicit zeros become nonzero (all values redrawn without zeros).
    data.set_P(wp64(P.data + np.where(P.data == 0.0, 1.5, 0.0)))
    if p:
        data.set_A(wp64(np.where(A.data == 0.0, -2.5, 2.0 * A.data)))
    if m:
        data.set_G(wp64(np.where(G.data == 0.0, 0.75, -G.data)))
        # Bound switches: some inactive rows become active and vice versa.
        h_l, h_u = data.h_l.numpy().reshape(-1).copy(), data.h_u.numpy().reshape(-1).copy()
        flip = rng.random(m) < 0.5
        inactive = np.isinf(h_l) & np.isinf(h_u)
        h_l = np.where(flip & inactive, -1.0, np.where(flip & ~inactive, -np.inf, h_l))
        h_u = np.where(flip & inactive, 1.0, np.where(flip & ~inactive, np.inf, h_u))
        data.set_h_l(wp64(h_l))
        data.set_h_u(wp64(h_u))
    run_update((True, p > 0, m > 0))
    # Partial updates.
    data.set_P(wp64(rng.standard_normal(P.nnz)))
    run_update((True, False, False))
    if m:
        data.set_G(wp64(rng.standard_normal(G.nnz)))
        run_update((False, False, True))

    # Diagonal update with random x_reg / delta / z_reg (z_reg including zeros).
    for trial in range(2):
        x_reg = wp64(np.abs(rng.standard_normal((1, n))) * 1e-3)
        delta = wp64(np.abs(rng.standard_normal(1)) * 1e-2)
        z = np.abs(rng.standard_normal((1, m)))
        z[rng.random((1, m)) < 0.3] = 0.0
        z_reg = wp64(z)
        z_reg_inv = wp64(np.zeros((1, m)))
        ref.update_and_factor_kkt(data, delta, x_reg, z_reg, z_reg_inv)
        xr_c, dl_c, zr_c = wp.clone(x_reg), wp.clone(delta), wp.clone(z_reg)  # alive across the call
        kkt.update_diagonal(dev_ptr(xr_c, np.float64, n), dev_ptr(dl_c, np.float64, 1),
                            dev_ptr(zr_c, np.float64, m) if m else 0, stream=stream())
        assert kkt.values_ptr == addr
        assert_bitwise(kkt.values(), ref_values(ref), f"values after diagonal update {trial}")
