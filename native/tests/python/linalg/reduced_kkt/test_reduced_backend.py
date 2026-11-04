"""Python-level checks of the native solver's KKT solve (dense Schur
complement): agreement with the reference cupiqp.SparseSolver, the n <= 2048
limit, and the reference settings field that is accepted and ignored."""
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


def _trip(M):
    return (wp.array(M.indptr.astype(np.int32)), wp.array(M.indices.astype(np.int32)),
            wp.array(M.data, dtype=wp.float64))


def _solve(solver_cls, data, **settings):
    P, c, A, b, G, h_l, h_u = data
    s = solver_cls()
    for k, v in settings.items():
        setattr(s.settings, k, v)
    s.setup(P=_trip(P), c=wp.array(c, dtype=wp.float64), A=_trip(A), b=wp.array(b, dtype=wp.float64),
            G=_trip(G), h_l=wp.array(h_l, dtype=wp.float64), h_u=wp.array(h_u, dtype=wp.float64))
    s.solve()
    info = s.result.info.to_host()
    return info, s.result.x.numpy()[0]


def test_matches_reference_solver(native, rng):
    import cupiqp
    data = _qp(rng)
    info_r, x_r = _solve(cupiqp.SparseSolver, data)
    info_n, x_n = _solve(native.SparseSolver, data)
    assert int(info_r.status[0]) == int(cupiqp.Status.CUPIQP_SOLVED)
    assert info_n.status[0] == native.Status.CUPIQP_SOLVED
    assert np.max(np.abs(x_n - x_r)) < 1e-6
    assert abs(int(info_n.iter[0]) - int(info_r.iter[0])) <= 2


@pytest.mark.parametrize("value", [False, True])
def test_deterministic_mode_setting_is_accepted_and_ignored(native, rng, value):
    data = _qp(rng)
    _, x_default = _solve(native.SparseSolver, data)
    s = native.SparseSolver()
    assert s.settings.use_deterministic_mode_for_cudss is False
    s.settings.use_deterministic_mode_for_cudss = value
    assert s.settings.use_deterministic_mode_for_cudss is value
    info, x = _solve(native.SparseSolver, data, use_deterministic_mode_for_cudss=value)
    assert info.status[0] == native.Status.CUPIQP_SOLVED
    np.testing.assert_array_equal(x, x_default)


def test_native_only_backend_settings_are_gone(native):
    s = native.SparseSolver()
    for name in ("kkt_backend", "custom_spmv"):
        with pytest.raises(AttributeError):
            setattr(s.settings, name, True)
    assert not hasattr(native, "KktBackend")


def test_rejects_large_n(native):
    n = 2049
    P = sp.identity(n, format="csr")
    s = native.SparseSolver()
    with pytest.raises(RuntimeError, match="supports n <= 2048"):
        s.setup(P=_trip(P), c=wp.array(np.ones(n), dtype=wp.float64))
