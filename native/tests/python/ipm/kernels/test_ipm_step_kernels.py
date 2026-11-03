"""Parity of the step / mu / centering / predictor-corrector / variable-update
kernels against the reference Warp kernels of cupiqp/solver_kernels.py."""
import numpy as np
import pytest

from cupiqp import solver_kernels as sk
from cupiqp.utils import reduction_block_dim
from ipm_kernels_harness import Vars, boolean, compare, f64, i32, run_pair, settings_floats

import warp as wp

# (B, n, p, num_hl, num_hu, num_xl, num_xu): num_ineq 5 / 400 / 600 / 3000
# -> reduction_block_dim 32 / 64 / 128 / 256. B > 1 makes the s/z slices strided.
CASES = {
    "B1_ineq5": (1, 4, 2, 5, 0, 0, 0),
    "B3_ineq400": (3, 20, 7, 150, 150, 50, 50),
    "B1_ineq600_box": (1, 300, 0, 0, 0, 300, 300),
    "B2_ineq3000": (2, 500, 10, 1000, 1000, 500, 500),
}


def _case(name, seed):
    B, n, p, hl, hu, xl, xu = CASES[name]
    num_ineq = hl + hu + xl + xu
    rng = np.random.default_rng(seed)
    mask = (rng.random((B, num_ineq)) < 0.7).astype(np.float64)
    if B > 1:
        mask[1] = 0.0  # a problem without finite bounds (num_finite_bounds == 0)
    return B, n, p, (hl, hu, xl, xu), num_ineq, rng, mask


def _vars(B, n, p, sides, rng, lo=0.1, hi=2.0, garbage=True, mask=None):
    hl, hu, xl, xu = sides
    num_ineq = sum(sides)
    primal = rng.uniform(lo, hi, (B, n + num_ineq))
    dual = rng.uniform(lo, hi, (B, p + num_ineq))
    if garbage and mask is not None:
        # entries of infinite bounds hold arbitrary values (incl. negative)
        g = rng.normal(0, 5, (B, num_ineq))
        primal[:, n:] = np.where(mask > 0.5, primal[:, n:], g)
        dual[:, p:] = np.where(mask > 0.5, dual[:, p:], -g)
    return Vars(B, n, p, hl, hu, xl, xu, primal, dual)


@pytest.mark.parametrize("case", list(CASES))
def test_calculate_step(testing, case, blocks_per_problem):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 11)
        v = _vars(B, n, p, sides, rng, mask=mask)
        st = _vars(B, n, p, sides, rng, lo=-3.0, hi=1.0, garbage=False)
        return dict(s_all=v.s_all, z_all=v.z_all, finite_mask_all=f64(mask),
                    step_s_all=st.s_all, step_z_all=st.z_all, settings=f64(settings_floats()),
                    alpha_s=f64(np.zeros(B)), alpha_z=f64(np.zeros(B)))

    B, _, _, _, num_ineq, _, _ = _case(case, 11)
    ref, nat = run_pair(build, sk.create_calculate_step_kernel(wp.float64), testing.ipm_calculate_step,
                        (B, num_ineq), tiled_dim=[B], block_dim=reduction_block_dim(num_ineq))
    compare(ref, nat, ["alpha_s", "alpha_z", "s_all", "z_all"], ulp=0, label=f"calculate_step[{case}]: ")
    assert (ref["alpha_s"].numpy() < 0.99).any()  # some step was limited


@pytest.mark.parametrize("case", list(CASES))
def test_calculate_mu(testing, case, blocks_per_problem):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 12)
        v = _vars(B, n, p, sides, rng, mask=mask)
        return dict(s_all=v.s_all, z_all=v.z_all, finite_mask_all=f64(mask),
                    num_finite_bounds=f64(mask.sum(axis=1)), mu=f64(np.full(B, -1.0)))

    B, _, _, _, num_ineq, _, _ = _case(case, 12)
    ref, nat = run_pair(build, sk.create_calculate_mu_kernel(wp.float64), testing.ipm_calculate_mu,
                        (B, num_ineq), tiled_dim=[B], block_dim=reduction_block_dim(num_ineq))
    compare(ref, nat, ["mu"], rtol_keys=("mu",), label=f"calculate_mu[{case}]: ")


@pytest.mark.parametrize("case", list(CASES))
def test_init_guess_center(testing, case, blocks_per_problem):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 13)
        # initial solve output: arbitrary signs
        v = _vars(B, n, p, sides, rng, lo=-2.0, hi=3.0, garbage=False)
        return dict(finite_mask_all=f64(mask), num_finite_bounds=f64(mask.sum(axis=1)),
                    s_all=v.s_all, z_all=v.z_all, mu=f64(np.zeros(B)),
                    primals_all=v.primals_all, duals_all=v.duals_all)

    B, _, _, _, num_ineq, _, _ = _case(case, 13)
    ref, nat = run_pair(build, sk.create_init_guess_center_kernel(wp.float64), testing.ipm_init_guess_center,
                        (B, num_ineq), tiled_dim=[B], block_dim=reduction_block_dim(num_ineq))
    # s and z depend on the summed mu.
    compare(ref, nat, ["mu", "primals_all", "duals_all"], rtol_keys=("mu", "primals_all", "duals_all"),
            label=f"init_guess_center[{case}]: ")
    if B > 1:
        assert ref["mu"].numpy()[1] == 1e-10  # clipped mu of the problem without finite bounds


@pytest.mark.parametrize("case", list(CASES))
def test_prepare_factor_and_predictor_step(testing, case, blocks_per_problem):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 14)
        v = _vars(B, n, p, sides, rng, mask=mask)
        primal, dual = v.primals_all.numpy(), v.duals_all.numpy()
        z = dual[:, p:]
        # duals at / below the boundary: shifted by eps
        tiny = rng.choice([0.0, 1e-17, -1e-20, 1e-300, 3e-16], size=z.shape)
        dual[:, p:] = np.where(rng.random(z.shape) < 0.2, tiny, z)
        v = Vars(B, n, p, *sides, primal, dual)
        active = np.array([True, False, True][:B] if B <= 3 else rng.random(B) < 0.7)
        finetune = np.array([True, True, False][:B] if B <= 3 else rng.random(B) < 0.5)
        return dict(active_mask=boolean(active), finetune_mask=boolean(finetune),
                    settings=f64(settings_floats()), finite_mask_all=f64(mask),
                    num_finite_bounds=f64(mask.sum(axis=1)), s_all=v.s_all, z_all=v.z_all,
                    reg_limit=f64(np.full(B, 1e-10)), no_primal_update=i32(rng.integers(0, 9, B)),
                    no_dual_update=i32(rng.integers(0, 9, B)), mu=f64(np.zeros(B)),
                    res_s_all=_vars(B, n, p, sides, rng).s_all,
                    duals_all=v.duals_all)

    B, _, _, _, num_ineq, _, _ = _case(case, 14)
    ref, nat = run_pair(build, sk.create_prepare_factor_and_predictor_step_kernel(wp.float64),
                        testing.ipm_prepare_factor_and_predictor_step, (B, num_ineq),
                        tiled_dim=[B], block_dim=reduction_block_dim(num_ineq))
    compare(ref, nat, ["duals_all", "reg_limit", "no_primal_update", "no_dual_update", "mu", "res_s_all"],
            rtol_keys=("mu",), label=f"prepare_factor_and_predictor_step[{case}]: ")


@pytest.mark.parametrize("case", list(CASES))
def test_prepare_corrector_step(testing, case, blocks_per_problem):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 15)
        v = _vars(B, n, p, sides, rng, mask=mask)
        st = _vars(B, n, p, sides, rng, lo=-3.0, hi=1.0, garbage=False)
        res = _vars(B, n, p, sides, rng, lo=-1.0, hi=0.0, garbage=False)
        mu = rng.uniform(1.5, 4.0, B)
        if B > 2:
            mu[2] = 0.0  # denominator branch
        return dict(s_all=v.s_all, z_all=v.z_all, step_s_all=st.s_all, step_z_all=st.z_all,
                    finite_mask_all=f64(mask), num_finite_bounds=f64(mask.sum(axis=1)),
                    mu=f64(mu), settings=f64(settings_floats()),
                    primal_step=f64(np.zeros(B)), dual_step=f64(np.zeros(B)), sigma=f64(np.full(B, -1.0)),
                    res_s_all=res.s_all, res_primals=res.primals_all)

    B, _, _, _, num_ineq, _, _ = _case(case, 15)
    ref, nat = run_pair(build, sk.create_prepare_corrector_step_kernel(wp.float64),
                        testing.ipm_prepare_corrector_step, (B, num_ineq),
                        tiled_dim=[B], block_dim=reduction_block_dim(num_ineq))
    compare(ref, nat, ["primal_step", "dual_step", "sigma", "res_primals"],
            rtol_keys=("sigma", "res_primals"), label=f"prepare_corrector_step[{case}]: ")
    sig = ref["sigma"].numpy()
    assert ((sig > 0) & (sig < 1)).any()


@pytest.mark.parametrize("case", list(CASES))
def test_update_vars_after_corrector_step(testing, case):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 16)
        v = _vars(B, n, p, sides, rng, mask=mask)
        st = _vars(B, n, p, sides, rng, lo=-3.0, hi=3.0, garbage=False)
        active = np.array([True, False, True][:B]) if B <= 3 else rng.random(B) < 0.7
        return dict(active_mask=boolean(active), finite_mask_all=f64(mask),
                    primal_step=f64(rng.uniform(0.1, 1, B)), dual_step=f64(rng.uniform(0.1, 1, B)),
                    step_primals_all=st.primals_all, step_duals_all=st.duals_all,
                    primals_all=v.primals_all, duals_all=v.duals_all)

    B, n, p, _, num_ineq, _, _ = _case(case, 16)
    kernel = sk.create_update_vars_after_corrector_step_kernel(n, p, num_ineq, wp.float64)
    ref, nat = run_pair(build, kernel, testing.ipm_update_vars_after_corrector_step, (B, n, p, num_ineq),
                        dim=(B, n + num_ineq + p + num_ineq))
    compare(ref, nat, ["primals_all", "duals_all"], label=f"update_vars_after_corrector_step[{case}]: ")


@pytest.mark.parametrize("case", list(CASES))
def test_run_full_newton_step(testing, case):
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 17)
        v = _vars(B, n, p, sides, rng, mask=mask)
        st = _vars(B, n, p, sides, rng, lo=-3.0, hi=3.0, garbage=False)
        active = np.array([True, False, True][:B]) if B <= 3 else rng.random(B) < 0.7
        return dict(active_mask=boolean(active), step_x=st.x, step_y=st.y, result_x=v.x, result_y=v.y,
                    primal_step=f64(np.zeros(B)), dual_step=f64(np.zeros(B)),
                    primals_all=v.primals_all, duals_all=v.duals_all)

    B, n, p, _, _, _, _ = _case(case, 17)
    kernel = sk.create_run_full_newton_step_kernel(n, p, wp.float64)
    ref, nat = run_pair(build, kernel, testing.ipm_run_full_newton_step, (B, n, p), dim=(B, n + p))
    compare(ref, nat, ["primals_all", "duals_all", "primal_step", "dual_step"],
            label=f"run_full_newton_step[{case}]: ")


def _poison(a: np.ndarray, rng, frac=0.05):
    a = a.copy()
    sel = rng.random(a.shape) < frac
    a[sel] = np.nan
    return a


@pytest.mark.parametrize("case", ["B3_ineq400", "B2_ineq3000"])
def test_reductions_with_nonfinite_inputs(testing, case, blocks_per_problem):
    """NaN entries (a poisoned KKT solve): Warp min/max ignore a NaN
    operand (fmin/fmax), sums propagate it; the native kernels must agree."""
    def build():
        B, n, p, sides, num_ineq, rng, mask = _case(case, 18)
        v = _vars(B, n, p, sides, rng, mask=mask)
        v = Vars(B, n, p, *sides, _poison(v.primals_all.numpy(), rng), _poison(v.duals_all.numpy(), rng))
        st = _vars(B, n, p, sides, rng, lo=-3.0, hi=1.0, garbage=False)
        sp = _poison(st.primals_all.numpy(), rng)
        sd = _poison(st.duals_all.numpy(), rng)
        if B > 2:
            sp[2] = np.nan  # a whole row of NaN directions
        st = Vars(B, n, p, *sides, sp, sd)
        return dict(s_all=v.s_all, z_all=v.z_all, finite_mask_all=f64(mask),
                    step_s_all=st.s_all, step_z_all=st.z_all, settings=f64(settings_floats()),
                    alpha_s=f64(np.zeros(B)), alpha_z=f64(np.zeros(B)),
                    num_finite_bounds=f64(mask.sum(axis=1)), mu=f64(np.full(B, 2.0)),
                    primal_step=f64(np.zeros(B)), dual_step=f64(np.zeros(B)), sigma=f64(np.zeros(B)),
                    res_s_all=_vars(B, n, p, sides, rng).s_all)

    B, _, _, _, num_ineq, _, _ = _case(case, 18)
    bd = reduction_block_dim(num_ineq)
    ref, nat = run_pair(build, sk.create_calculate_step_kernel(wp.float64), testing.ipm_calculate_step,
                        (B, num_ineq), tiled_dim=[B], block_dim=bd)
    compare(ref, nat, ["alpha_s", "alpha_z"], ulp=0, label=f"calculate_step_nonfinite[{case}]: ")
    ref, nat = run_pair(build, sk.create_prepare_corrector_step_kernel(wp.float64),
                        testing.ipm_prepare_corrector_step, (B, num_ineq), tiled_dim=[B], block_dim=bd)
    compare(ref, nat, ["primal_step", "dual_step", "sigma", "res_s_all"], rtol_keys=("sigma", "res_s_all"),
            label=f"prepare_corrector_step_nonfinite[{case}]: ")
    # mu over NaN slacks: the sum propagates NaN
    def build_mu():
        d = build()
        d["s_all"], d["step_s_all"] = d["step_s_all"], d["s_all"]
        return d
    ref, nat = run_pair(build_mu, sk.create_calculate_mu_kernel(wp.float64), testing.ipm_calculate_mu,
                        (B, num_ineq), tiled_dim=[B], block_dim=bd)
    compare(ref, nat, ["mu"], rtol_keys=("mu",), label=f"calculate_mu_nonfinite[{case}]: ")
