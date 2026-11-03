"""Parity of the initial-guess rhs and residual kernels against the reference
Warp kernels of cupiqp/solver_kernels.py."""
import numpy as np
import pytest

from cupiqp import solver_kernels as sk
from cupiqp.utils import reduction_block_dim
from ipm_kernels_harness import Vars, compare, f64, run_pair

import warp as wp

# (B, n, p, m, has_h_l, has_h_u, has_x_l, has_x_u)
CASES = {
    "B1_all_sides": (1, 30, 10, 20, True, True, True, True),
    "B3_hl_xu_no_eq": (3, 50, 0, 40, True, False, False, True),
    "B2_no_G_xl": (2, 40, 15, 0, False, False, True, False),
    "B1_G_no_bounds": (1, 25, 5, 10, False, False, False, False),
    "B1_wide_block128": (1, 600, 300, 900, True, True, True, True),
    "B2_wide_block256": (2, 2100, 50, 30, False, True, True, True),
}


class Case:
    def __init__(self, name, seed):
        B, n, p, m, hl, hu, xl, xu = CASES[name]
        self.B, self.n, self.p, self.m = B, n, p, m
        self.num_hl, self.num_hu = (m if hl else 0), (m if hu else 0)
        self.num_xl, self.num_xu = (n if xl else 0), (n if xu else 0)
        self.sides = (self.num_hl, self.num_hu, self.num_xl, self.num_xu)
        self.num_ineq = sum(self.sides)
        self.num_duals = p + self.num_ineq
        self.rng = np.random.default_rng(seed)

    @property
    def dims(self):
        return (self.B, self.n, self.p, self.m, *self.sides)

    def mask(self, width, present):
        if not present:
            return np.zeros((self.B, 0))
        return (self.rng.random((self.B, width)) < 0.7).astype(np.float64)

    def bound(self, mask, sign):
        if mask.shape[1] == 0:
            return np.zeros_like(mask)
        v = sign * self.rng.uniform(0.5, 3.0, mask.shape)
        return np.where(mask > 0.5, v, sign * np.inf)  # infinite where the mask is 0

    def vars(self, lo=-2.0, hi=2.0):
        B, n, p = self.B, self.n, self.p
        primal = self.rng.uniform(lo, hi, (B, n + self.num_ineq))
        dual = self.rng.uniform(lo, hi, (B, p + self.num_ineq))
        return Vars(B, n, p, *self.sides, primal, dual)

    def data(self):
        c = self
        d = {}
        d["mhl"] = c.mask(c.m, c.num_hl > 0)
        d["mhu"] = c.mask(c.m, c.num_hu > 0)
        d["mxl"] = c.mask(c.n, c.num_xl > 0)
        d["mxu"] = c.mask(c.n, c.num_xu > 0)
        d["h_l"] = c.bound(d["mhl"], -1.0)
        d["h_u"] = c.bound(d["mhu"], 1.0)
        d["x_l"] = c.bound(d["mxl"], -1.0)
        d["x_u"] = c.bound(d["mxu"], 1.0)
        d["c"] = c.rng.normal(size=(c.B, c.n))
        d["b"] = c.rng.normal(size=(c.B, c.p))
        return d


def _w(a):
    return f64(a) if a.shape[1] > 0 else wp.empty((a.shape[0], 0), dtype=wp.float64, device="cuda:0")


@pytest.mark.parametrize("case", list(CASES))
def test_init_guess_rhs(testing, case):
    def build():
        c = Case(case, 21)
        d = c.data()
        res = c.vars()
        return dict(c=f64(d["c"]), b_eq=_w(d["b"]), h_l=_w(d["h_l"]), h_u=_w(d["h_u"]),
                    x_l=_w(d["x_l"]), x_u=_w(d["x_u"]),
                    finite_mask_hl=_w(d["mhl"]), finite_mask_hu=_w(d["mhu"]),
                    finite_mask_xl=_w(d["mxl"]), finite_mask_xu=_w(d["mxu"]),
                    res_x=res.x, res_y=res.y, res_zl=res.z_l, res_zu=res.z_u,
                    res_zbl=res.z_bl, res_zbu=res.z_bu,
                    res_primals=res.primals_all, res_duals=res.duals_all)

    c = Case(case, 21)
    kernel = sk.create_init_guess_rhs_kernel(c.n, c.p, *c.sides, dtype=wp.float64)
    ref, nat = run_pair(build, kernel, testing.ipm_init_guess_rhs, c.dims,
                        dim=(c.B, c.n + c.p + c.num_ineq))
    compare(ref, nat, ["res_primals", "res_duals"], ulp=0, label=f"init_guess_rhs[{case}]: ")


@pytest.mark.parametrize("case", list(CASES))
def test_prepare_zu_minus_zl_and_zbu_minus_zbl(testing, case):
    def build():
        c = Case(case, 22)
        v = c.vars()
        return dict(z_u=v.z_u, z_l=v.z_l, z_bl=v.z_bl, z_bu=v.z_bu,
                    x_b_scaling=f64(c.rng.uniform(0.5, 2.0, (c.B, c.n))),
                    zu_minus_zl=_w(np.full((c.B, c.m), np.nan)),
                    zbu_minus_zbl=f64(np.full((c.B, c.n), np.nan)))

    c = Case(case, 22)
    kernel = sk.create_prepare_zu_minus_zl_and_zbu_minus_zbl_kernel(
        c.m, c.n, c.num_hl > 0, c.num_hu > 0, c.num_xl > 0, c.num_xu > 0, dtype=wp.float64)
    ref, nat = run_pair(build, kernel, testing.ipm_prepare_zu_minus_zl_and_zbu_minus_zbl, c.dims,
                        dim=(c.B, c.m + c.n))
    compare(ref, nat, ["zu_minus_zl", "zbu_minus_zbl"], ulp=0, label=f"prepare_zu_minus_zl[{case}]: ")


@pytest.mark.parametrize("case", list(CASES))
def test_update_residual_nr(testing, case):
    def build():
        c = Case(case, 23)
        d = c.data()
        B, n, p, m = c.B, c.n, c.p, c.m
        result = c.vars(0.1, 2.0)
        res = c.vars()       # res.y = A x, res.x = A^T y (scratch slots of the reference)
        res_nr = c.vars()    # res_nr.x = -P x on entry, residual on exit
        step = c.vars()      # step.x = G^T (z_u - z_l)
        prev = c.rng.uniform(0.1, 1.0, B)
        return dict(
            minus_Px=res_nr.x, A_x=res.y, AT_y=res.x, G_x=_w(c.rng.normal(size=(B, m))),
            GT_zh_assembled=step.x, zb_assembled=f64(c.rng.normal(size=(B, n))),
            data_c=f64(d["c"]), data_b=_w(d["b"]), data_h_l=_w(d["h_l"]), data_h_u=_w(d["h_u"]),
            data_x_l=_w(d["x_l"]), data_x_u=_w(d["x_u"]),
            finite_mask_hl=_w(d["mhl"]), finite_mask_hu=_w(d["mhu"]),
            finite_mask_xl=_w(d["mxl"]), finite_mask_xu=_w(d["mxu"]),
            result_x=result.x, result_y=result.y,
            result_z_hl=result.z_l, result_z_hu=result.z_u, result_z_xl=result.z_bl, result_z_xu=result.z_bu,
            result_s_hl=result.s_l, result_s_hu=result.s_u, result_s_xl=result.s_bl, result_s_xu=result.s_bu,
            x_b_scaling=f64(c.rng.uniform(0.5, 2.0, (B, n))),
            cost_scaling_inv=f64(c.rng.uniform(0.5, 2.0, B)),
            delta_inv=f64(c.rng.uniform(0.5, 2.0, (B, n + p + m))),
            delta_b_inv=f64(c.rng.uniform(0.5, 2.0, (B, n))),
            constraints_rhs_inf_norm=f64(c.rng.uniform(0.0, 30.0, B)),
            res_nr_x=res_nr.x, res_nr_y=res_nr.y, res_nr_z_hl=res_nr.z_l, res_nr_z_hu=res_nr.z_u,
            res_nr_z_xl=res_nr.z_bl, res_nr_z_xu=res_nr.z_bu,
            info_primal_obj=f64(np.zeros(B)), info_dual_obj=f64(np.zeros(B)),
            info_duality_gap=f64(np.zeros(B)), info_duality_gap_rel=f64(np.zeros(B)),
            info_primal_res=f64(prev), info_primal_res_rel=f64(np.zeros(B)),
            info_dual_res=f64(prev * 2), info_dual_res_rel=f64(np.zeros(B)),
            info_prev_primal_res=f64(np.zeros(B)), info_prev_dual_res=f64(np.zeros(B)),
            res_nr_primals=res_nr.primals_all, res_nr_duals=res_nr.duals_all,
        )

    c = Case(case, 23)
    ref, nat = run_pair(build, sk.create_update_residual_nr_kernel(wp.float64), testing.ipm_update_residual_nr,
                        c.dims, tiled_dim=[c.B], block_dim=reduction_block_dim(max(c.n, c.p, c.m)))
    sums = ("info_primal_obj", "info_dual_obj", "info_duality_gap", "info_duality_gap_rel")
    exact = ("info_primal_res", "info_primal_res_rel", "info_dual_res", "info_dual_res_rel",
             "info_prev_primal_res", "info_prev_dual_res")
    compare(ref, nat, ["res_nr_primals", "res_nr_duals", *exact, *sums], rtol_keys=sums,
            label=f"update_residual_nr[{case}]: ")


@pytest.mark.parametrize("case", list(CASES))
def test_update_residuals_r(testing, case):
    def build():
        c = Case(case, 24)
        B = c.B
        res_nr, result, prox, res = c.vars(), c.vars(), c.vars(), c.vars()
        rel = c.rng.uniform(0.1, 1.0, (4, B))
        rel[:, 0] = 0.0  # problem 0: the "rel == 0" branch
        return dict(rho=f64(c.rng.uniform(1e-8, 1e-2, B)), delta=f64(c.rng.uniform(1e-8, 1e-2, B)),
                    res_nr_x=res_nr.x, res_nr_duals=res_nr.duals_all,
                    result_x=result.x, result_duals=result.duals_all,
                    prox_x=prox.x, prox_duals=prox.duals_all,
                    res_x=res.x, res_duals=res.duals_all,
                    dual_res_unscale_factor=f64(c.rng.uniform(0.5, 2.0, (B, c.n))),
                    primal_res_unscale_factor=_w(c.rng.uniform(0.5, 2.0, (B, c.num_duals))),
                    primal_res=f64(c.rng.uniform(0.1, 1.0, B)), primal_res_rel=f64(rel[0]),
                    dual_res=f64(c.rng.uniform(0.1, 1.0, B)), dual_res_rel=f64(rel[1]),
                    primal_res_reg=f64(np.zeros(B)), primal_res_reg_rel=f64(np.zeros(B)),
                    dual_res_reg=f64(np.zeros(B)), dual_res_reg_rel=f64(np.zeros(B)),
                    primal_prox_inf=f64(np.zeros(B)), dual_prox_inf=f64(np.zeros(B)),
                    res_primals=res.primals_all)

    c = Case(case, 24)
    ref, nat = run_pair(build, sk.create_update_residuals_r_kernel(wp.float64), testing.ipm_update_residuals_r,
                        (c.B, c.n, c.num_duals), tiled_dim=[c.B],
                        block_dim=reduction_block_dim(max(c.n, c.p + c.num_ineq)))
    compare(ref, nat, ["res_primals", "res_duals", "primal_res_reg", "primal_res_reg_rel", "dual_res_reg",
                       "dual_res_reg_rel", "primal_prox_inf", "dual_prox_inf"],
            label=f"update_residuals_r[{case}]: ")


@pytest.mark.parametrize("case", ["B1_all_sides", "B3_hl_xu_no_eq"])
def test_update_residuals_r_nonfinite(testing, case):
    """NaN in the iterate: the max-norms use fmax (NaN ignored) as Warp does."""
    def build():
        c = Case(case, 25)
        B = c.B
        res_nr, prox, res = c.vars(), c.vars(), c.vars()
        rp = c.rng.uniform(-2, 2, (B, c.n + c.num_ineq))
        rd = c.rng.uniform(-2, 2, (B, c.p + c.num_ineq))
        for a in (rp, rd):
            sel = c.rng.random(a.shape) < 0.1
            a[sel] = np.nan
        result = Vars(B, c.n, c.p, *c.sides, rp, rd)
        return dict(rho=f64(c.rng.uniform(1e-8, 1e-2, B)), delta=f64(c.rng.uniform(1e-8, 1e-2, B)),
                    res_nr_x=res_nr.x, res_nr_duals=res_nr.duals_all,
                    result_x=result.x, result_duals=result.duals_all,
                    prox_x=prox.x, prox_duals=prox.duals_all, res_x=res.x, res_duals=res.duals_all,
                    dual_res_unscale_factor=f64(c.rng.uniform(0.5, 2.0, (B, c.n))),
                    primal_res_unscale_factor=_w(c.rng.uniform(0.5, 2.0, (B, c.num_duals))),
                    primal_res=f64(c.rng.uniform(0.1, 1.0, B)), primal_res_rel=f64(c.rng.uniform(0.1, 1.0, B)),
                    dual_res=f64(c.rng.uniform(0.1, 1.0, B)), dual_res_rel=f64(c.rng.uniform(0.1, 1.0, B)),
                    primal_res_reg=f64(np.zeros(B)), primal_res_reg_rel=f64(np.zeros(B)),
                    dual_res_reg=f64(np.zeros(B)), dual_res_reg_rel=f64(np.zeros(B)),
                    primal_prox_inf=f64(np.zeros(B)), dual_prox_inf=f64(np.zeros(B)),
                    res_primals=res.primals_all)

    c = Case(case, 25)
    ref, nat = run_pair(build, sk.create_update_residuals_r_kernel(wp.float64), testing.ipm_update_residuals_r,
                        (c.B, c.n, c.num_duals), tiled_dim=[c.B],
                        block_dim=reduction_block_dim(max(c.n, c.p + c.num_ineq)))
    compare(ref, nat, ["res_primals", "res_duals", "primal_res_reg", "primal_res_reg_rel", "dual_res_reg",
                       "dual_res_reg_rel", "primal_prox_inf", "dual_prox_inf"],
            label=f"update_residuals_r_nonfinite[{case}]: ")
