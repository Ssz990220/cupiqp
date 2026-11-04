"""Python-level checks of the device-side warm start from the previous solve
(SparseSolver.set_warm_start / row_key_buffers / solve_warm_start)."""
import numpy as np
import pytest
import warp as wp

from test_warm_start import _qp, _setup


def _x(s):
    return s.result.x.numpy()[0].copy()


def _info(s):
    i = s.result.info.to_host()
    return int(i.status[0]), int(i.iter[0])


def _shift(data, x0):
    """The QP in the increment d = x - x0 (same duals, solution x* - x0)."""
    P, c, A, b, G, h_l, h_u = data
    Gx = G @ x0
    return (P, c + P @ x0, A, b - A @ x0, G, np.where(h_l > -1e20, h_l - Gx, h_l), np.where(h_u < 1e20, h_u - Gx, h_u))


def _update(s, data):
    P, c, A, b, G, h_l, h_u = data
    s.update(c=wp.array(c, dtype=wp.float64), b=wp.array(b, dtype=wp.float64),
             h_l=wp.array(h_l, dtype=wp.float64), h_u=wp.array(h_u, dtype=wp.float64))


def test_first_is_cold_then_warm(native, rng):
    data = _qp(rng)
    ref = _setup(native, data)
    ref.solve()
    s = _setup(native, data)
    s.set_warm_start(s_min=1e-9, z_min=1e-9)
    s.solve_warm_start()
    assert _info(s) == _info(ref)
    assert np.array_equal(_x(s), _x(ref))
    _update(s, _shift(data, _x(s)))
    s.solve_warm_start()
    st, it = _info(s)
    assert st == native.Status.CUPIQP_SOLVED and it <= 3
    assert np.max(np.abs(_x(s))) < 1e-6
    c0 = s.native.counters()
    s.solve_warm_start()
    c1 = s.native.counters()
    assert c1["graph_launches"] - c0["graph_launches"] == 1
    assert c1["graph_captures"] == c0["graph_captures"]


@pytest.mark.parametrize("host", [True, False])
def test_keys_from_host_and_device(native, rng, host):
    data = _qp(rng)
    s = _setup(native, data)
    s.set_warm_start(row_match=native.RowMatch.Keys, row_groups=True, key_row_begin=4, key_row_count=16)
    kb = s.row_key_buffers(host=host)
    assert kb["keys"].shape == (16,) and kb["groups"].shape == (16,)
    keys = np.arange(16, dtype=np.int64)
    groups = (np.arange(16) // 4).astype(np.int32)
    for it in range(2):
        if host:
            kb["keys"][:] = keys
            kb["groups"][:] = groups
        else:
            kb["keys"].assign(wp.array(keys, dtype=wp.int64))
            kb["groups"].assign(wp.array(groups, dtype=wp.int32))
            wp.synchronize()
        s.solve_warm_start(keys_from_host=host)
        assert _info(s)[0] == native.Status.CUPIQP_SOLVED
    match = next(wp.array(ptr=p, dtype=wp.int32, shape=(c,), device="cuda:0").numpy()
                 for name, p, _, c, _ in s.native.debug_buffers() if name == "warm.match")
    assert np.array_equal(match, np.arange(16))
