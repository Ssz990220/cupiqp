"""Python-level checks of the experimental initial-point hook
(SparseSolver.warm_start_buffers / solve_warm)."""
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
    c = rng.standard_normal(n) * 3
    h_l = G @ x0 - 0.1
    h_u = G @ x0 + 0.1
    h_l[::4] = -1e20
    return P, c, A, A @ x0, G, h_l, h_u


def _trip(M):
    return (wp.array(M.indptr.astype(np.int32)), wp.array(M.indices.astype(np.int32)),
            wp.array(M.data, dtype=wp.float64))


def _setup(native, data):
    P, c, A, b, G, h_l, h_u = data
    s = native.SparseSolver()
    s.setup(P=_trip(P), c=wp.array(c, dtype=wp.float64), A=_trip(A), b=wp.array(b, dtype=wp.float64),
            G=_trip(G), h_l=wp.array(h_l, dtype=wp.float64), h_u=wp.array(h_u, dtype=wp.float64))
    return s


def _result(s):
    return {k: getattr(s.result, k).numpy()[0].copy() for k in ("x", "y", "s_l", "s_u", "z_l", "z_u")}


@pytest.mark.parametrize("host", [True, False])
def test_own_solution_terminates_at_once(native, rng, host):
    s = _setup(native, _qp(rng))
    s.solve()
    r = _result(s)
    buf = s.warm_start_buffers(host=host)
    assert set(buf) == {"x", "y", "s_l", "s_u", "z_l", "z_u"}
    for k, v in r.items():
        if host:
            buf[k][:] = v
        else:
            buf[k].assign(wp.array(v, dtype=wp.float64))
    s.solve_warm("xysz", from_host=host)
    info = s.result.info.to_host()
    assert info.status[0] == native.Status.CUPIQP_SOLVED
    assert int(info.iter[0]) <= 1
    assert np.max(np.abs(s.result.x.numpy()[0] - r["x"])) < 1e-9


def test_default_solve_unchanged(native, rng):
    data = _qp(rng)
    ref = _setup(native, data)
    ref.solve()
    x_ref = ref.result.x.numpy()[0].copy()
    s = _setup(native, data)
    s.solve()
    r = _result(s)
    buf = s.warm_start_buffers()
    for k, v in r.items():
        buf[k][:] = v
    s.solve_warm("z")
    assert s.result.info.to_host().status[0] == native.Status.CUPIQP_SOLVED
    s.solve()
    np.testing.assert_array_equal(s.result.x.numpy()[0], x_ref)
