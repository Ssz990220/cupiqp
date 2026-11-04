"""Parity of the native spmv module with the reference SparseMatVecProduct.

Both operators run on the same device CSR arrays and the same inputs. The
native kernels sum in their own fixed order (bitwise reproducible), the
reference library product in another (and for transposed products or
beta != 0 it varies from run to run), so every case is checked against the
reference and a float64 host product within 1e-14 of the magnitude sum
|alpha| |op(M)| |x| + |beta| |y|.
"""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_matvec import SparseMatVecProduct
from cupiqp_native._interop import dev_ptr

RTOL = 1e-14

SHAPES = [(1, 1, 1.0), (1, 8, 0.5), (8, 1, 0.5), (200, 32, 0.3), (32, 200, 0.3), (37, 53, 0.05),
          (37, 53, 0.9), (1024, 256, 0.3), (2521 // 4, 169, 0.59)]


def _random_csr(rows, cols, density, seed, zero_frac=0.3, empty_row_frac=0.1):
    """Host CSR with explicit zeros and empty rows, sorted indices."""
    rng = np.random.default_rng(seed)
    mask = rng.random((rows, cols)) < density
    mask[rng.random(rows) < empty_row_frac, :] = False
    r, c = np.nonzero(mask)
    vals = rng.standard_normal(r.size)
    vals[rng.random(r.size) < zero_frac] = 0.0  # explicit zeros stay stored
    M = sp.csr_matrix((vals, (r, c)), shape=(rows, cols))
    M.sort_indices()
    assert M.nnz == r.size
    return M


def _wp(a, dtype=wp.float64):
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device="cuda:0")


def _mat(M):
    return UniformBatchedCsrMatrix(1, M.indptr.astype(np.int32), M.indices.astype(np.int32),
                                   M.data.astype(np.float64), shape=M.shape, dtype=wp.float64)


def _native_op(testing, mat, transpose):
    return testing.SpMVOp(mat.rows, mat.cols, mat.nnz,
                          dev_ptr(mat.indptr, np.int32), dev_ptr(mat.indices, np.int32),
                          dev_ptr(mat.data, np.float64, allow_2d_single_row=True), transpose)


def _stream():
    return wp.get_stream("cuda:0").cuda_stream


def _magnitude(M, transpose, alpha, x, beta, y0):
    Ma = abs(M).T if transpose else abs(M)
    return abs(alpha) * (Ma @ np.abs(x)) + abs(beta) * np.abs(y0)


def _report(name, got, want, mag):
    diff = np.abs(got - want)
    rel = np.max(diff / np.maximum(mag, 1e-300)) if diff.size else 0.0
    print(f"{name}: max abs {diff.max(initial=0.0):.3e}, max rel-to-magnitude {rel:.3e}, "
          f"bitwise {np.array_equal(got.view(np.int64), want.view(np.int64))}")
    return rel


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (2.0, 0.5), (-1.5, 2.0), (0.75, 1.0)])
@pytest.mark.parametrize("transpose", [False, True], ids=["N", "T"])
@pytest.mark.parametrize("rows,cols,density", SHAPES)
def test_spmv_matches_reference(testing, rows, cols, density, transpose, alpha, beta, graph):
    M = _random_csr(rows, cols, density, seed=rows * 1000 + cols)
    mat = _mat(M)
    nx, ny = (rows, cols) if transpose else (cols, rows)
    rng = np.random.default_rng(rows + 7 * cols)
    x = rng.standard_normal(nx)
    y0 = rng.standard_normal(ny)

    ref_op = SparseMatVecProduct(mat, transa=transpose)
    x_ref, y_ref = _wp(x), _wp(y0)
    ref_op(x_ref, y_ref, alpha=alpha, beta=beta)
    y_ref_h = y_ref.numpy()

    nat_op = _native_op(testing, mat, transpose)
    if M.nnz:
        assert nat_op.computes_product
    x_nat, y_nat = _wp(x), _wp(y0)
    nat_op(dev_ptr(x_nat, size=nx), dev_ptr(y_nat, size=ny), alpha, beta, _stream(), graph)
    y_nat_h = y_nat.numpy()

    host = alpha * ((M.T if transpose else M) @ x) + beta * y0
    mag = _magnitude(M, transpose, alpha, x, beta, y0)
    assert np.all(np.abs(y_nat_h - y_ref_h) <= RTOL * mag)
    assert np.all(np.abs(y_nat_h - host) <= RTOL * mag + 1e-300)


@pytest.mark.parametrize("rows,cols", [(0, 5), (5, 0), (4, 6)])
@pytest.mark.parametrize("transpose", [False, True], ids=["N", "T"])
def test_empty_matrix_matches_reference(testing, rows, cols, transpose):
    """Empty matrices (no stored entries): native computes no product."""
    M = sp.csr_matrix((rows, cols))
    mat = _mat(M)
    nx, ny = (rows, cols) if transpose else (cols, rows)
    nat_op = _native_op(testing, mat, transpose)
    assert not nat_op.computes_product
    rng = np.random.default_rng(0)
    x = rng.standard_normal(nx)
    for beta in (0.0, 1.0, -2.0):
        y0 = rng.standard_normal(ny)
        y_nat = _wp(y0)
        nat_op(dev_ptr(_wp(x), size=nx) if nx else 0, dev_ptr(y_nat, size=ny) if ny else 0, 3.0, beta, _stream())
        want = beta * y0
        if rows > 0 and cols > 0:
            # The reference handles nnz == 0 (it never builds A / G operators with
            # zero rows): compare against it too.
            y_ref = _wp(y0)
            SparseMatVecProduct(mat, transa=transpose)(_wp(x), y_ref, alpha=3.0, beta=beta)
            want = y_ref.numpy()
        np.testing.assert_array_equal(y_nat.numpy(), want)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("n,p,m", [(40, 10, 60), (40, 0, 60), (40, 10, 0), (40, 0, 0)])
def test_sparse_operators_match_reference(testing, n, p, m, graph):
    """SparseOperators vs the reference SparseKKTSolver.eval_* products
    (beta = 0). Absent blocks write zeros for the transposed products, which
    is what KKTSystem's zero-initialised work buffers hold when it skips them."""
    P = _random_csr(n, n, 0.2, 1, empty_row_frac=0.0)
    P = (P + P.T).tocsr()
    P.sort_indices()
    A = _random_csr(p, n, 0.3, 2) if p else sp.csr_matrix((0, n))
    G = _random_csr(m, n, 0.6, 3, zero_frac=0.8) if m else sp.csr_matrix((0, n))
    mats = {k: _mat(v) for k, v in dict(P=P, A=A, G=G).items()}

    def ptrs(k):
        mt = mats[k]
        if mt.nnz == 0:
            return mt.nnz, dev_ptr(mt.indptr, np.int32), 0, 0
        return mt.nnz, dev_ptr(mt.indptr, np.int32), dev_ptr(mt.indices, np.int32), dev_ptr(mt.data, np.float64)

    ops = testing.SparseOps(n, p, m, *ptrs("P"), *ptrs("A"), *ptrs("G"))
    rng = np.random.default_rng(5)
    alpha = -0.75
    for which, key, transpose in [("P", "P", False), ("A", "A", False), ("AT", "A", True),
                                  ("G", "G", False), ("GT", "G", True)]:
        M = dict(P=P, A=A, G=G)[key]
        nx, ny = (M.shape[0], M.shape[1]) if transpose else (M.shape[1], M.shape[0])
        x = rng.standard_normal(nx)
        y_nat = _wp(np.full(ny, np.nan))
        ops.apply(which, alpha, dev_ptr(_wp(x)) if nx else 0, dev_ptr(y_nat) if ny else 0, _stream(), graph)
        got = y_nat.numpy()
        if M.shape[0] == 0:
            np.testing.assert_array_equal(got, np.zeros(ny))
            continue
        y_ref = _wp(np.zeros(ny))
        SparseMatVecProduct(mats[key], transa=transpose)(_wp(x), y_ref, alpha=alpha, beta=0.0)
        ref = y_ref.numpy()
        mag = _magnitude(M, transpose, alpha, x, 0.0, np.zeros(ny))
        assert np.all(np.abs(got - ref) <= RTOL * mag), which


def test_parity_summary(testing, capsys):
    """Max differences native vs reference on the MPC-like G shape (report)."""
    M = _random_csr(2521, 169, 0.59, 9, zero_frac=0.82, empty_row_frac=0.0)
    mat = _mat(M)
    rng = np.random.default_rng(3)
    worst = 0.0
    with capsys.disabled():
        print()
        for transpose in (False, True):
            nx, ny = (M.shape[0], M.shape[1]) if transpose else (M.shape[1], M.shape[0])
            x = rng.standard_normal(nx)
            y_ref, y_nat = _wp(np.zeros(ny)), _wp(np.zeros(ny))
            SparseMatVecProduct(mat, transa=transpose)(_wp(x), y_ref)
            _native_op(testing, mat, transpose)(dev_ptr(_wp(x)), dev_ptr(y_nat), 1.0, 0.0, _stream())
            y_ref2 = _wp(np.zeros(ny))
            SparseMatVecProduct(mat, transa=transpose)(_wp(x), y_ref2)
            mag = _magnitude(M, transpose, 1.0, x, 0.0, np.zeros(ny))
            worst = max(worst, _report(f"G{'^T' if transpose else ''} native vs reference",
                                       y_nat.numpy(), y_ref.numpy(), mag))
            _report(f"G{'^T' if transpose else ''} reference vs reference", y_ref2.numpy(), y_ref.numpy(), mag)
    assert worst <= RTOL
