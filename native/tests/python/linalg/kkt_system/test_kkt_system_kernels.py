"""Parity of the native kkt_system kernels with the reference Warp kernels.

Every test builds random inputs with NumPy (finite and infinite bound masks,
G rows with both bounds infinite so that z_reg == 0), runs the reference
kernel from cupiqp/kkt_systems_kernels.py or cupiqp/solver_kernels.py on Warp
arrays, runs the native launcher (cupiqp_native._ext._testing.kkt_*) on
copies of the same inputs, and compares bitwise (at most 4 ulp allowed).
"""
import itertools
import os

import numpy as np
import pytest
import warp as wp

from cupiqp import kkt_systems_kernels as K
from cupiqp.solver_kernels import create_solution_finite_kernel
from cupiqp.settings import SettingsFloatIdx, SettingsIntIdx
from cupiqp.utils import column_slice, reduction_block_dim

from cupiqp_native._interop import dev_ptr

DEV = "cuda:0"
# Allowed ulp distance (CUPIQP_KKT_MAX_ULP=0 checks bitwise equality).
MAX_ULP = int(os.environ.get("CUPIQP_KKT_MAX_ULP", "4"))

SIDES = list(itertools.product([False, True], repeat=4))  # (h_l, h_u, x_l, x_u)
SIZES = [(7, 0, 5), (33, 4, 0), (300, 20, 260), (1, 1, 1)]  # (n, p, m)


# ---------------------------------------------------------------- helpers
def row(a):
    """(1, k) float64 Warp array of a 1-D host vector."""
    return wp.array(np.asarray(a, dtype=np.float64).reshape(1, -1), dtype=wp.float64, device=DEV)


def ptr(a, dtype=np.float64):
    if a is None or a.size == 0:
        return 0
    return dev_ptr(a, dtype)


def ulp_diff(a, b):
    """Max distance in units in the last place (NaN == NaN, +0 == -0)."""
    a = np.ascontiguousarray(a, dtype=np.float64).ravel()
    b = np.ascontiguousarray(b, dtype=np.float64).ravel()
    assert a.shape == b.shape
    if a.size == 0:
        return 0

    def line(v):
        i = v.view(np.int64)
        return np.where(i >= 0, i, np.int64(-0x8000000000000000) - i)  # monotonic integer line

    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if same.all():
        return 0
    if (np.isnan(a) != np.isnan(b)).any() or (np.isinf(a[~same]) | np.isinf(b[~same])).any():
        return 1 << 62
    d = np.abs(line(a[~same]).astype(np.float64) - line(b[~same]).astype(np.float64))
    return int(d.max())


def assert_close(ref, nat, name):
    r = ref.numpy() if hasattr(ref, "numpy") else np.asarray(ref)
    t = nat.numpy() if hasattr(nat, "numpy") else np.asarray(nat)
    u = ulp_diff(r, t)
    assert u <= MAX_ULP, f"{name}: {u} ulp"
    return u


def sync():
    wp.synchronize()


def make_masks(rng, n, m, sides):
    """[hl | hu | xl | xu] finite masks; a quarter of the G rows have both
    sides infinite, some box entries too."""
    hl, hu, xl, xu = sides
    mask_hl = (rng.random(m) > 0.3).astype(float) if hl else np.zeros(0)
    mask_hu = (rng.random(m) > 0.3).astype(float) if hu else np.zeros(0)
    if m and hl and hu:
        dead = rng.random(m) < 0.25
        dead[0] = True
        mask_hl[dead] = 0.0
        mask_hu[dead] = 0.0
    mask_xl = (rng.random(n) > 0.4).astype(float) if xl else np.zeros(0)
    mask_xu = (rng.random(n) > 0.4).astype(float) if xu else np.zeros(0)
    return np.concatenate([mask_hl, mask_hu, mask_xl, mask_xu]), (len(mask_hl), len(mask_hu), len(mask_xl), len(mask_xu))


def slices(buf, counts):
    """Reference column slices (hl, hu, xl, xu) of a (1, num_ineq) buffer."""
    hl, hu, xl, _ = counts
    return (column_slice(buf, 0, hl), column_slice(buf, hl, hl + hu),
            column_slice(buf, hl + hu, hl + hu + xl), column_slice(buf, hl + hu + xl, None))


def reg_state(rng, n, m, sides):
    """Inputs and reference outputs of update_regularizations."""
    mask, counts = make_masks(rng, n, m, sides)
    ni = mask.size
    s_all = rng.uniform(1e-3, 3.0, ni)
    z_all = rng.uniform(1e-3, 3.0, ni)
    xbs = rng.uniform(0.2, 2.0, n)
    rho = np.array([rng.uniform(1e-8, 1e-3)])
    delta = np.array([rng.uniform(1e-8, 1e-3)])
    return dict(mask=mask, counts=counts, s_all=s_all, z_all=z_all, xbs=xbs, rho=rho, delta=delta)


def run_reg_ref(st, n, m, sides):
    hl, hu, xl, xu = sides
    ni = st["mask"].size
    kern = K.create_update_regularizations_kernel(n, m, has_h_l=hl, has_h_u=hu, has_x_l=xl, has_x_u=xu,
                                                  dtype=wp.float64)
    out = dict(m_s=wp.zeros((1, ni), dtype=wp.float64, device=DEV),
               m_z_inv=wp.zeros((1, ni), dtype=wp.float64, device=DEV),
               w=wp.zeros((1, ni), dtype=wp.float64, device=DEV),
               rho=wp.zeros(1, dtype=wp.float64, device=DEV), delta=wp.zeros(1, dtype=wp.float64, device=DEV),
               x_reg=wp.zeros((1, n), dtype=wp.float64, device=DEV),
               z_reg=wp.zeros((1, m), dtype=wp.float64, device=DEV),
               z_reg_inv=wp.zeros((1, m), dtype=wp.float64, device=DEV))
    wp.launch(kern, dim=(1, n + m),
              inputs=[row(st["s_all"]), row(st["z_all"]), row(st["mask"]), row(st["xbs"]),
                      wp.array(st["rho"], dtype=wp.float64, device=DEV),
                      wp.array(st["delta"], dtype=wp.float64, device=DEV),
                      out["m_s"], out["m_z_inv"], out["w"], out["rho"], out["delta"],
                      out["x_reg"], out["z_reg"], out["z_reg_inv"]], device=DEV)
    return out


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize("sides", SIDES)
@pytest.mark.parametrize("n,p,m", SIZES)
def test_update_regularizations(testing, rng, sides, n, p, m):
    st = reg_state(rng, n, m, sides)
    ref = run_reg_ref(st, n, m, sides)
    ni = st["mask"].size
    nat = {k: wp.full(v.shape, 7.0, dtype=wp.float64, device=DEV) for k, v in ref.items()}
    ins = [row(st["s_all"]), row(st["z_all"]), row(st["mask"]), row(st["xbs"]),
           wp.array(st["rho"], dtype=wp.float64, device=DEV), wp.array(st["delta"], dtype=wp.float64, device=DEV)]
    sync()
    testing.kkt_update_regularizations(n, m, *sides, *[ptr(a) for a in ins],
                                       ptr(nat["m_s"]), ptr(nat["m_z_inv"]), ptr(nat["w"]), ptr(nat["rho"]),
                                       ptr(nat["delta"]), ptr(nat["x_reg"]), ptr(nat["z_reg"]),
                                       ptr(nat["z_reg_inv"]))
    testing.synchronize()
    for k in ref:
        assert_close(ref[k], nat[k], k)
    if m and sides[0] and sides[1]:
        assert (ref["z_reg"].numpy() == 0).any(), "expected inactive G rows"
    assert ni == 0 or np.isfinite(nat["w"].numpy()).all()


@pytest.mark.parametrize("n,p,m", SIZES)
def test_eliminate_slacks_and_recover_slacks(testing, rng, n, p, m):
    ni = 2 * m + 2 * n
    rz, rs, mzi, ms, lz = (rng.standard_normal(ni) for _ in range(5))
    mzi[rng.random(ni) < 0.3] = 0.0
    ms[mzi == 0] = 0.0
    k1 = K.create_eliminate_slacks_kernel(wp.float64)
    ref = wp.zeros((1, ni), dtype=wp.float64, device=DEV)
    wp.launch(k1, dim=(1, ni), inputs=[row(rz), row(rs), row(mzi), ref], device=DEV)
    nat = wp.zeros((1, ni), dtype=wp.float64, device=DEV)
    a = [row(rz), row(rs), row(mzi)]
    sync()
    testing.kkt_eliminate_slacks(ni, *[ptr(x) for x in a], ptr(nat))
    testing.synchronize()
    assert_close(ref, nat, "updated_rhs_z")

    k2 = K.create_recover_slacks_kernel(wp.float64)
    ref2 = wp.zeros((1, ni), dtype=wp.float64, device=DEV)
    wp.launch(k2, dim=(1, ni), inputs=[row(rs), row(lz), row(ms), row(mzi), ref2], device=DEV)
    nat2 = wp.zeros((1, ni), dtype=wp.float64, device=DEV)
    b = [row(rs), row(lz), row(ms), row(mzi)]
    sync()
    testing.kkt_recover_slacks(ni, *[ptr(x) for x in b], ptr(nat2))
    testing.synchronize()
    assert_close(ref2, nat2, "lhs_s")


@pytest.mark.parametrize("sides", SIDES)
@pytest.mark.parametrize("n,p,m", SIZES)
def test_eliminate_duals(testing, rng, sides, n, p, m):
    hl, hu, xl, xu = sides
    st = reg_state(rng, n, m, sides)
    reg = run_reg_ref(st, n, m, sides)
    ni = st["mask"].size
    counts = st["counts"]
    rhs_x = rng.standard_normal(n)
    upd = row(rng.standard_normal(ni))
    w = reg["w"]
    w_l, w_u, w_bl, w_bu = slices(w, counts)
    u_l, u_u, u_bl, u_bu = slices(upd, counts)
    kern = K.create_eliminate_duals_kernel(n, m, has_h_l=hl, has_h_u=hu, has_x_l=xl, has_x_u=xu, dtype=wp.float64)
    ref_x = wp.zeros((1, n), dtype=wp.float64, device=DEV)
    ref_z = wp.zeros((1, m), dtype=wp.float64, device=DEV)
    xbs = row(st["xbs"])
    wp.launch(kern, dim=(1, n + m),
              inputs=[row(rhs_x), w_bu, w_bl, xbs, u_bu, u_bl, ref_x, w_u, w_l, u_u, u_l, reg["z_reg"], ref_z],
              device=DEV)
    nat_x = wp.full((1, n), 3.0, dtype=wp.float64, device=DEV)
    nat_z = wp.full((1, m), 3.0, dtype=wp.float64, device=DEV)
    rx = row(rhs_x)
    sync()
    testing.kkt_eliminate_duals(n, m, *sides, ptr(rx), ptr(w) if ni else 0, ptr(xbs), ptr(upd) if ni else 0,
                                ptr(reg["z_reg"]), ptr(nat_x), ptr(nat_z))
    testing.synchronize()
    assert_close(ref_x, nat_x, "rhs_x_bar")
    assert_close(ref_z, nat_z, "rhs_z_bar")


@pytest.mark.parametrize("sides", SIDES)
@pytest.mark.parametrize("n,p,m", SIZES)
def test_recover_duals(testing, rng, sides, n, p, m):
    st = reg_state(rng, n, m, sides)
    reg = run_reg_ref(st, n, m, sides)
    ni = st["mask"].size
    counts = st["counts"]
    num_hl, num_hu, num_xl, num_xu = counts
    G_dx = row(rng.standard_normal(m))
    lhs_x = row(rng.standard_normal(n))
    upd = row(rng.standard_normal(ni))
    rhs_z = row(rng.standard_normal(ni))
    rhs_s = row(rng.standard_normal(ni))
    xbs = row(st["xbs"])
    w, mzi = reg["w"], reg["m_z_inv"]
    w_l, w_u, w_bl, w_bu = slices(w, counts)
    u_l, u_u, _, _ = slices(upd, counts)
    _, _, mzi_bl, mzi_bu = slices(mzi, counts)
    _, _, rz_bl, rz_bu = slices(rhs_z, counts)
    _, _, rs_bl, rs_bu = slices(rhs_s, counts)
    ref = wp.full((1, ni), 5.0, dtype=wp.float64, device=DEV)
    l_l, l_u, l_bl, l_bu = slices(ref, counts)
    kern = K.create_recover_duals_kernel(num_hu, num_hl, num_xu, num_xl, dtype=wp.float64)
    wp.launch(kern, dim=(1, ni),
              inputs=[G_dx, lhs_x, w_u, u_u, l_u, w_l, u_l, l_l, w_bu, mzi_bu, rz_bu, rs_bu, l_bu,
                      w_bl, mzi_bl, rz_bl, rs_bl, l_bl, xbs], device=DEV)
    nat = wp.full((1, ni), 5.0, dtype=wp.float64, device=DEV)
    sync()
    if ni:
        testing.kkt_recover_duals(n, m, *sides, ptr(G_dx), ptr(lhs_x), ptr(w), ptr(upd), ptr(mzi), ptr(rhs_z),
                                  ptr(rhs_s), ptr(nat), ptr(xbs))
    testing.synchronize()
    assert_close(ref, nat, "lhs_z_all")


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("n,p,m", SIZES)
def test_condensed_combine(testing, rng, residual, n, p, m):
    v = {k: row(rng.standard_normal(sz)) for k, sz in
         dict(P_x=n, x_reg=n, lhs_x=n, AT_y=n, GT_z=n, A_x=p, lhs_y=p, G_x=m, lhs_z=m,
              rhs_x=n, rhs_y=p, rhs_z=m).items()}
    zr = rng.uniform(0.1, 2.0, m)
    zr[rng.random(m) < 0.3] = 0.0
    v["z_reg"] = row(zr)
    v["active"] = row((zr > 0).astype(float))
    delta = wp.array([rng.uniform(1e-6, 1e-2)], dtype=wp.float64, device=DEV)
    order = ["P_x", "x_reg", "lhs_x", "AT_y", "GT_z", "A_x", "lhs_y"]
    order2 = ["G_x", "active", "z_reg", "lhs_z", "rhs_x", "rhs_y", "rhs_z"]

    def outs():
        return [wp.full((1, k), 9.0, dtype=wp.float64, device=DEV) for k in (n, p, m)] + \
               [wp.zeros(1, dtype=wp.float64, device=DEV)]

    ref = outs()
    kern = K.create_condensed_kkt_combine_kernel(wp.float64)
    wp.launch(kern, dim=(1, n + p + m),
              inputs=[v[k] for k in order] + [delta] + [v[k] for k in order2] + [wp.int32(1 if residual else 0)]
              + ref, device=DEV)
    nat = outs()
    sync()
    testing.kkt_condensed_combine(n, p, m, *[ptr(v[k]) for k in order], ptr(delta), *[ptr(v[k]) for k in order2],
                                  residual, *[ptr(a) for a in nat])
    testing.synchronize()
    for name, r, t in zip(["out_x", "out_y", "out_z", "norm"], ref, nat):
        assert_close(r, t, name)


def test_condensed_combine_norm_ignores_nan(testing):
    """Warp atomic_max (fmax) drops NaN; the native port must do the same."""
    n, p, m = 4, 0, 0
    rx = row([1.0, np.nan, -3.0, 2.0])
    z = row(np.zeros(n))
    e0 = wp.empty((1, 0), dtype=wp.float64, device=DEV)
    delta = wp.zeros(1, dtype=wp.float64, device=DEV)
    kern = K.create_condensed_kkt_combine_kernel(wp.float64)
    ref = [wp.zeros((1, n), dtype=wp.float64, device=DEV), wp.empty((1, 0), dtype=wp.float64, device=DEV),
           wp.empty((1, 0), dtype=wp.float64, device=DEV), wp.zeros(1, dtype=wp.float64, device=DEV)]
    wp.launch(kern, dim=(1, n), inputs=[z, z, z, z, z, e0, e0, delta, e0, e0, e0, e0, rx, e0, e0, wp.int32(1)]
              + ref, device=DEV)
    nat_x = wp.zeros((1, n), dtype=wp.float64, device=DEV)
    nat_n = wp.zeros(1, dtype=wp.float64, device=DEV)
    sync()
    testing.kkt_condensed_combine(n, p, m, ptr(z), ptr(z), ptr(z), ptr(z), ptr(z), 0, 0, ptr(delta), 0, 0, 0, 0,
                                  ptr(rx), 0, 0, True, ptr(nat_x), 0, 0, ptr(nat_n))
    testing.synchronize()
    assert ref[3].numpy()[0] == 3.0
    assert nat_n.numpy()[0] == 3.0
    assert_close(ref[0], nat_x, "out_x")


@pytest.mark.parametrize("k", [0, 1, 300])
def test_mul(testing, rng, k):
    a, b = row(rng.standard_normal(k)), row((rng.random(k) > 0.5).astype(float))
    ref = wp.zeros((1, k), dtype=wp.float64, device=DEV)
    wp.map(wp.mul, a, b, out=ref)
    nat = wp.zeros((1, k), dtype=wp.float64, device=DEV)
    sync()
    testing.kkt_mul(k, ptr(a), ptr(b), ptr(nat))
    testing.synchronize()
    assert_close(ref, nat, "mul")


def _settings(max_iter, eps_abs=1e-12, eps_rel=1e-12, rate=5.0):
    f = np.zeros(len(SettingsFloatIdx))
    i = np.zeros(len(SettingsIntIdx), dtype=np.int32)
    f[SettingsFloatIdx.iterative_refinement_eps_abs] = eps_abs
    f[SettingsFloatIdx.iterative_refinement_eps_rel] = eps_rel
    f[SettingsFloatIdx.iterative_refinement_min_improvement_rate] = rate
    i[SettingsIntIdx.iterative_refinement_max_iter] = max_iter
    return (wp.array(f, dtype=wp.float64, device=DEV), wp.array(i, dtype=wp.int32, device=DEV))


def test_settings_indices_match():
    import cupiqp_native  # noqa: F401
    assert int(SettingsFloatIdx.iterative_refinement_eps_abs) == 10
    assert int(SettingsFloatIdx.iterative_refinement_eps_rel) == 11
    assert int(SettingsFloatIdx.iterative_refinement_min_improvement_rate) == 12
    assert int(SettingsIntIdx.iterative_refinement_max_iter) == 5


@pytest.mark.parametrize("max_iter", [0, 1, 3])
def test_refinement_begin(testing, max_iter):
    _, ints = _settings(max_iter)

    def state():
        return [wp.full(1, 4.0, dtype=wp.float64, device=DEV) for _ in range(3)] + \
               [wp.full(1, 7, dtype=wp.int32, device=DEV) for _ in range(2)]

    ref = state()
    wp.launch(K.create_refinement_begin_kernel(wp.float64), dim=1, inputs=[ints] + ref, device=DEV)
    nat = state()
    sync()
    testing.kkt_refinement_begin(ptr(ints, np.int32), *[ptr(a) for a in nat[:3]],
                                 *[ptr(a, np.int32) for a in nat[3:]])
    testing.synchronize()
    for r, t in zip(ref, nat):
        np.testing.assert_array_equal(r.numpy(), t.numpy())


# (step, previous error, current error, rhs_norm): covers non-finite, within
# tolerance at step 0 / later, slow and fast improvement, worsening.
DECIDE_CASES = [
    (0, np.inf, np.nan, 1.0), (2, 1e-3, np.inf, 1.0), (0, np.inf, 1e-13, 1.0), (3, 1e-6, 1e-13, 1.0),
    (1, 1e-6, 5e-7, 1.0), (1, 1e-6, 2e-6, 1.0), (1, 1e-3, 1e-6, 1.0), (0, np.inf, 1e-3, 1.0),
    (2, 1e-3, 1e-6, 1e3), (9, 1e-3, 1e-6, 1.0), (4, 1e-3, 1e-6, 1.0),
]


@pytest.mark.parametrize("case", DECIDE_CASES)
def test_refinement_decide(testing, case):
    step, prev, cur, rn = case
    floats, ints = _settings(5)

    def state():
        return ([wp.array([rn], dtype=wp.float64, device=DEV), wp.array([cur], dtype=wp.float64, device=DEV),
                 wp.array([prev], dtype=wp.float64, device=DEV)],
                [wp.array([step], dtype=wp.int32, device=DEV)] + [wp.full(1, 7, dtype=wp.int32, device=DEV)
                                                                  for _ in range(3)])

    rf, ri = state()
    wp.launch(K.create_refinement_decide_kernel(wp.float64), dim=1, inputs=[floats, ints] + rf + ri, device=DEV)
    nf, ni_ = state()
    sync()
    testing.kkt_refinement_decide(ptr(floats), ptr(ints, np.int32), *[ptr(a) for a in nf],
                                  *[ptr(a, np.int32) for a in ni_])
    testing.synchronize()
    for r, t in zip(rf + ri, nf + ni_):
        np.testing.assert_array_equal(r.numpy(), t.numpy())


@pytest.mark.parametrize("accept", [0, 1])
@pytest.mark.parametrize("n,p,m", SIZES)
def test_refinement_accept_add_rhs_norm(testing, rng, accept, n, p, m):
    acc = wp.array([accept], dtype=wp.int32, device=DEV)
    cand = [row(rng.standard_normal(k)) for k in (n, p, m)]

    def lhs():
        return [row(np.arange(k, dtype=float)) for k in (n, p, m)]

    ref = lhs()
    wp.launch(K.create_refinement_accept_kernel(n, p, m, wp.float64), dim=(1, n + p + m),
              inputs=[acc] + cand + ref, device=DEV)
    nat = lhs()
    sync()
    testing.kkt_refinement_accept(n, p, m, ptr(acc, np.int32), *[ptr(a) for a in cand], *[ptr(a) for a in nat])
    testing.synchronize()
    for r, t in zip(ref, nat):
        assert_close(r, t, "accept")

    # add: cand += lhs
    lh = [row(rng.standard_normal(k)) for k in (n, p, m)]
    ref_c = [wp.clone(a) for a in cand]
    nat_c = [wp.clone(a) for a in cand]
    wp.launch(K.create_refinement_add_kernel(n, p, m, wp.float64), dim=(1, n + p + m), inputs=lh + ref_c,
              device=DEV)
    sync()
    testing.kkt_refinement_add(n, p, m, *[ptr(a) for a in lh], *[ptr(a) for a in nat_c])
    testing.synchronize()
    for r, t in zip(ref_c, nat_c):
        assert_close(r, t, "add")

    # rhs norm (accumulates into a nonzero start value)
    ref_n = wp.array([0.25], dtype=wp.float64, device=DEV)
    nat_n = wp.array([0.25], dtype=wp.float64, device=DEV)
    wp.launch(K.create_refinement_rhs_norm_kernel(n, p, m, wp.float64), dim=(1, n + p + m), inputs=lh + [ref_n],
              device=DEV)
    sync()
    testing.kkt_refinement_rhs_norm(n, p, m, *[ptr(a) for a in lh], ptr(nat_n))
    testing.synchronize()
    assert_close(ref_n, nat_n, "rhs_norm")


@pytest.mark.parametrize("poison", [None, "nan", "inf", "-inf"])
@pytest.mark.parametrize("where", ["x", "y", "z"])
@pytest.mark.parametrize("n,p,m", SIZES)
def test_solution_finite(testing, rng, poison, where, n, p, m):
    vecs = {k: rng.standard_normal(sz) for k, sz in dict(x=n, y=p, z=m).items()}
    if poison is not None and vecs[where].size:
        vecs[where][rng.integers(vecs[where].size)] = float(poison)
    xs = [row(vecs[k]) for k in "xyz"]
    kern = create_solution_finite_kernel(wp.float64)
    ref = wp.full(1, 7, dtype=wp.int32, device=DEV)
    wp.launch_tiled(kern, dim=[1], inputs=xs + [ref], block_dim=reduction_block_dim(n + p + m), device=DEV)
    nat = wp.full(1, 7, dtype=wp.int32, device=DEV)
    sync()
    testing.kkt_solution_finite(n, p, m, *[ptr(a) for a in xs], ptr(nat, np.int32))
    testing.synchronize()
    np.testing.assert_array_equal(ref.numpy(), nat.numpy())
    expect = int(poison is not None and vecs[where].size > 0)
    assert int(nat.numpy()[0]) == expect


@pytest.mark.parametrize("n,p,m", SIZES)
def test_assemble_and_split(testing, rng, n, p, m):
    xs = [row(rng.standard_normal(k)) for k in (n, p, m)]
    buf = wp.zeros((1, n + p + m), dtype=wp.float64, device=DEV)
    sync()
    testing.kkt_assemble_rhs(n, p, m, *[ptr(a) for a in xs], ptr(buf))
    testing.synchronize()
    np.testing.assert_array_equal(buf.numpy()[0], np.concatenate([a.numpy()[0] for a in xs]))
    outs = [wp.zeros((1, k), dtype=wp.float64, device=DEV) for k in (n, p, m)]
    testing.kkt_split_sol(n, p, m, ptr(buf), *[ptr(a) for a in outs])
    testing.synchronize()
    for a, b in zip(xs, outs):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
