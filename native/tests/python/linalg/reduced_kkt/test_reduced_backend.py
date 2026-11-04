"""Python-level checks of the reduced KKT backend (Settings.kkt_backend)."""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp


def _qp(rng, n=30, p=5, m=40):
    L = rng.standard_normal((n, n)) * (rng.random((n, n)) < 0.3)
    P = sp.csr_matrix(L @ L.T + np.eye(n))
    A = sp.csr_matrix(rng.standard_normal((p, n)) * (rng.random((p, n)) < 0.5))
    G = sp.csr_matrix(rng.standard_normal((m, n)) * (rng.random((m, n)) < 0.5))
    x0 = rng.uniform(-1, 1, n)
    c = rng.standard_normal(n)
    return P, c, A, A @ x0, G, G @ x0 - 1.0, G @ x0 + 1.0


def _solve(native, backend, data):
    P, c, A, b, G, h_l, h_u = data
    trip = lambda M: (wp.array(M.indptr.astype(np.int32)), wp.array(M.indices.astype(np.int32)),
                      wp.array(M.data, dtype=wp.float64))
    s = native.SparseSolver()
    s.settings.kkt_backend = backend
    s.setup(P=trip(P), c=wp.array(c, dtype=wp.float64), A=trip(A), b=wp.array(b, dtype=wp.float64),
            G=trip(G), h_l=wp.array(h_l, dtype=wp.float64), h_u=wp.array(h_u, dtype=wp.float64))
    s.solve()
    info = s.result.info.to_host()
    return info, s.result.x.numpy()[0]


def test_setting_accepts_enum_and_name(native):
    s = native.SparseSolver()
    assert s.settings.kkt_backend == native.KktBackend.Cudss
    s.settings.kkt_backend = "reduced"
    assert s.settings.kkt_backend == native.KktBackend.Reduced
    s.settings.kkt_backend = native.KktBackend.Cudss
    assert s.settings.kkt_backend == native.KktBackend.Cudss
    with pytest.raises(ValueError):
        s.settings.kkt_backend = "dense"


def test_reduced_matches_cudss(native, rng):
    data = _qp(rng)
    info_c, x_c = _solve(native, "cudss", data)
    info_r, x_r = _solve(native, "reduced", data)
    assert info_c.status[0] == native.Status.CUPIQP_SOLVED
    assert info_r.status[0] == native.Status.CUPIQP_SOLVED
    assert np.max(np.abs(x_r - x_c)) < 1e-6
    assert abs(int(info_r.iter[0]) - int(info_c.iter[0])) <= 2


def test_reduced_rejects_large_n(native):
    n = 2049
    P = sp.identity(n, format="csr")
    s = native.SparseSolver()
    s.settings.kkt_backend = "reduced"
    with pytest.raises(RuntimeError, match="supports n <= 2048"):
        s.setup(P=(wp.array(P.indptr.astype(np.int32)), wp.array(P.indices.astype(np.int32)),
                   wp.array(P.data, dtype=wp.float64)), c=wp.array(np.ones(n), dtype=wp.float64))
