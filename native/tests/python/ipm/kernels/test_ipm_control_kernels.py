"""Parity of the solve-state, regularization and loop-control kernels against
the reference Warp kernels of cupiqp/solver_kernels.py.

Per-problem kernels run on a large batch whose entries are drawn from small
discrete sets around the thresholds, so every branch is taken by some
problem; each test asserts that the reference output covers the branches."""
import numpy as np
import pytest

from cupiqp import solver_kernels as sk
from cupiqp.typedef import (STATUS_DUAL_INFEASIBLE, STATUS_MAX_ITER_REACHED, STATUS_NUMERICAL_ISSUES,
                            STATUS_PRIMAL_INFEASIBLE, STATUS_SOLVED, STATUS_UNSOLVED)
from cupiqp.utils import reduction_block_dim
from ipm_kernels_harness import Vars, boolean, compare, f64, i32, run_pair, settings_floats, settings_ints

import warp as wp

BIG_B = 4096


def _pick(rng, values, size):
    return rng.choice(np.asarray(values), size=size)


# ---------------------------------------------------------------- reset / finetune
def test_reset_solve_state(testing):
    B = 37

    def build():
        rng = np.random.default_rng(31)
        return dict(settings=f64(settings_floats(rho_init=3e-6, delta_init=7e-4, reg_lower_limit=2e-10)),
                    rho=f64(rng.random(B)), delta=f64(rng.random(B)), reg_limit=f64(rng.random(B)),
                    mu=f64(rng.random(B)), primal_step=f64(rng.random(B)), dual_step=f64(rng.random(B)),
                    no_primal_update=i32(rng.integers(0, 9, B)), no_dual_update=i32(rng.integers(0, 9, B)),
                    status=i32(rng.integers(0, 5, B)), iteration=i32(rng.integers(0, 50, B)),
                    retries=i32(rng.integers(0, 5, B)), active=boolean(rng.random(B) < 0.5))

    ref, nat = run_pair(build, sk.create_reset_solve_state_kernel(wp.float64), testing.ipm_reset_solve_state,
                        (B,), dim=(B,))
    compare(ref, nat, ulp=0, label="reset_solve_state: ")
    assert (ref["status"].numpy() == STATUS_UNSOLVED).all()


def test_apply_finetune(testing):
    B = 300

    def build():
        rng = np.random.default_rng(32)
        return dict(mask=boolean(rng.random(B) < 0.5), reg_limit=f64(rng.random(B)),
                    no_primal_update=i32(rng.integers(0, 9, B)), no_dual_update=i32(rng.integers(0, 9, B)),
                    settings=f64(settings_floats()))

    ref, nat = run_pair(build, sk.create_apply_finetune_kernel(wp.float64), testing.ipm_apply_finetune,
                        (B,), dim=(B,))
    compare(ref, nat, ulp=0, label="apply_finetune: ")


# ---------------------------------------------------------------- termination
@pytest.mark.parametrize("check_gap", [0, 1])
def test_update_termination(testing, check_gap):
    B = BIG_B
    lower = 1e-13

    def build():
        rng = np.random.default_rng(33 + check_gap)
        small_or_big = [1e-10, 1e-7, 1.0]  # vs eps_abs 1e-8 / eps_rel 1e-9 / gap eps 1e-8, 1e-9
        regs = [lower, 1e-10, 1e-6]
        return dict(
            primal_res=f64(_pick(rng, small_or_big, B)), primal_res_rel=f64(_pick(rng, [1e-12, 1.0], B)),
            dual_res=f64(_pick(rng, small_or_big, B)), dual_res_rel=f64(_pick(rng, [1e-12, 1.0], B)),
            duality_gap=f64(_pick(rng, small_or_big, B)), duality_gap_rel=f64(_pick(rng, [1e-12, 1.0], B)),
            primal_res_reg=f64(_pick(rng, small_or_big, B)), primal_res_reg_rel=f64(_pick(rng, [1e-12, 1.0], B)),
            dual_res_reg=f64(_pick(rng, small_or_big, B)), dual_res_reg_rel=f64(_pick(rng, [1e-12, 1.0], B)),
            primal_prox_inf=f64(_pick(rng, [0.5, 0.9, 2.0], B)), dual_prox_inf=f64(_pick(rng, [0.5, 0.9, 2.0], B)),
            rho=f64(_pick(rng, regs, B)), delta=f64(_pick(rng, regs, B)), reg_limit=f64(_pick(rng, regs, B)),
            no_primal_update=i32(rng.integers(0, 12, B)), no_dual_update=i32(rng.integers(0, 12, B)),
            current_iter=i32([17]),
            settings_f=f64(settings_floats(reg_finetune_lower_limit=lower)),
            settings_i=i32(settings_ints(check_duality_gap=check_gap,
                                         reg_finetune_primal_update_threshold=3,
                                         reg_finetune_dual_update_threshold=8)),
            active=boolean(rng.random(B) < 0.8), status=i32(np.full(B, STATUS_UNSOLVED)),
            iteration=i32(np.full(B, 3)), finetune=boolean(rng.random(B) < 0.5))

    ref, nat = run_pair(build, sk.create_update_termination_kernel(wp.float64), testing.ipm_update_termination,
                        (B,), dim=(B,))
    compare(ref, nat, ["active", "status", "iteration", "finetune"], label="update_termination: ")
    st = set(ref["status"].numpy().tolist())
    assert {STATUS_UNSOLVED, STATUS_SOLVED, STATUS_PRIMAL_INFEASIBLE, STATUS_DUAL_INFEASIBLE} <= st
    assert ref["finetune"].numpy().any()


@pytest.mark.parametrize("B", [1, 5, 300, 3000])
@pytest.mark.parametrize("mode", ["none_active", "active_below_max", "active_at_max"])
def test_update_continue_flag(testing, B, mode):
    def build():
        rng = np.random.default_rng(34)
        active = np.zeros(B, dtype=bool)
        if mode != "none_active":
            active[rng.integers(0, B)] = True
        it = 250 if mode == "active_at_max" else 249
        return dict(active=boolean(active), current_iter=i32([it]), settings_i=i32(settings_ints(max_iter=250)),
                    continue_flag=i32([7]))

    ref, nat = run_pair(build, sk.update_continue_flag_kernel, testing.ipm_update_continue_flag, (B,),
                        tiled_dim=[1], block_dim=reduction_block_dim(B))
    compare(ref, nat, label="update_continue_flag: ")
    assert ref["continue_flag"].numpy()[0] == (1 if mode == "active_below_max" else 0)


# ---------------------------------------------------------------- factorization retry / failures
def test_factor_retry(testing):
    B = BIG_B

    def build():
        rng = np.random.default_rng(35)
        return dict(factor_status=i32(_pick(rng, [0, 0, 1, -3], B)), current_iter=i32([9]),
                    settings_f=f64(settings_floats(eps_abs=1e-8)),
                    settings_i=i32(settings_ints(max_factor_retires=4)),
                    active=boolean(rng.random(B) < 0.8), status=i32(np.full(B, STATUS_UNSOLVED)),
                    iteration=i32(np.full(B, 2)), retries=i32(rng.integers(0, 5, B)),
                    rho=f64(rng.uniform(1e-10, 1e-4, B)), delta=f64(rng.uniform(1e-10, 1e-4, B)),
                    reg_limit=f64(_pick(rng, [1e-10, 1e-9, 5e-9, 1e-7], B)),
                    needs_retry=i32(np.full(B, 9)))

    ref, nat = run_pair(build, sk.create_factor_retry_kernel(wp.float64), testing.ipm_factor_retry,
                        (B,), dim=(B,))
    compare(ref, nat, ulp=0, label="factor_retry: ")
    assert (ref["status"].numpy() == STATUS_NUMERICAL_ISSUES).any()
    assert ref["needs_retry"].numpy().any()


def test_mark_failed(testing):
    B = 500

    def build():
        rng = np.random.default_rng(36)
        return dict(failed=i32(_pick(rng, [0, 1, 0, 5], B)), current_iter=i32([4]),
                    active=boolean(rng.random(B) < 0.7), status=i32(np.full(B, STATUS_UNSOLVED)),
                    iteration=i32(np.zeros(B)))

    ref, nat = run_pair(build, sk.mark_failed_kernel, testing.ipm_mark_failed, (B,), dim=(B,))
    compare(ref, nat, label="mark_failed: ")
    assert (ref["status"].numpy() == STATUS_NUMERICAL_ISSUES).any()


@pytest.mark.parametrize("B", [1, 5, 300, 3000])
@pytest.mark.parametrize("any_set", [False, True])
def test_any_flag(testing, B, any_set):
    def build():
        rng = np.random.default_rng(37)
        flags = np.zeros(B, dtype=np.int32)
        if any_set:
            flags[rng.integers(0, B)] = -2
        return dict(flags=i32(flags), out=i32([5]))

    ref, nat = run_pair(build, sk.any_flag_kernel, testing.ipm_any_flag, (B,), tiled_dim=[1],
                        block_dim=reduction_block_dim(B))
    compare(ref, nat, label="any_flag: ")
    assert ref["out"].numpy()[0] == int(any_set)


def test_terminate_active(testing):
    B = 500

    def build():
        rng = np.random.default_rng(38)
        return dict(active=boolean(rng.random(B) < 0.5), current_iter=i32([42]),
                    new_status=wp.int32(STATUS_MAX_ITER_REACHED),
                    status=i32(rng.integers(0, 4, B)), iteration=i32(rng.integers(0, 40, B)))

    ref, nat = run_pair(build, sk.terminate_active_kernel,
                        lambda a, B: testing.ipm_terminate_active({**a, "new_status": STATUS_MAX_ITER_REACHED}, B),
                        (B,), dim=(B,))
    compare(ref, nat, ["active", "status", "iteration"], label="terminate_active: ")


def test_advance_iteration(testing):
    def build():
        return dict(current_iter=i32([41]))

    ref, nat = run_pair(build, sk.advance_iteration_kernel, testing.ipm_advance_iteration, (), dim=1)
    compare(ref, nat, label="advance_iteration: ")
    assert ref["current_iter"].numpy()[0] == 42


# ---------------------------------------------------------------- rho / delta / prox
def _rho_delta_inputs(rng, B, lower):
    regs = [lower, 1e-10, 1e-6, 3e-3]
    prev = _pick(rng, [1e-3, 1.0], B)
    return dict(
        active_mask=boolean(rng.random(B) < 0.8),
        info_dual_res=f64(prev * _pick(rng, [0.5, 0.95, 0.99, 2.0], B)),
        info_prev_dual_res=f64(prev),
        info_dual_res_rel=f64(_pick(rng, [1e-12, 1.0], B)),
        info_dual_prox_inf=f64(_pick(rng, [0.5, 2.0], B)),
        info_primal_res=f64(_pick(rng, [1e-9, 1e-3, 1.0], B)),
        info_prev_primal_res=f64(_pick(rng, [1e-3, 1.0], B)),
        info_primal_res_rel=f64(_pick(rng, [1e-12, 1.0], B)),
        info_primal_prox_inf=f64(_pick(rng, [0.5, 2.0], B)),
        info_reg_limit=f64(_pick(rng, [lower, 1e-10, 1e-7], B)),
        info_rho=f64(_pick(rng, regs, B)), info_delta=f64(_pick(rng, regs, B)),
        info_no_primal_update=i32(rng.integers(0, 9, B)), info_no_dual_update=i32(rng.integers(0, 9, B)),
    )


@pytest.mark.parametrize("it", [3, 5, 12])
def test_update_rho_delta_with_ineq(testing, it):
    B = BIG_B
    lower = 1e-13

    def build():
        rng = np.random.default_rng(39 + it)
        d = _rho_delta_inputs(rng, B, lower)
        d.update(dual_improved_out=boolean(rng.random(B) < 0.5), primal_improved_out=boolean(rng.random(B) < 0.5),
                 settings=f64(settings_floats(reg_finetune_lower_limit=lower)), current_iter=i32([it]))
        return d

    ref, nat = run_pair(build, sk.create_update_rho_delta_with_ineq_kernel(wp.float64),
                        testing.ipm_update_rho_delta_with_ineq, (B,), dim=(B,))
    compare(ref, nat, ulp=0, label="update_rho_delta_with_ineq: ")
    act = ref["active_mask"].numpy()
    assert ref["dual_improved_out"].numpy()[act].any() and not ref["dual_improved_out"].numpy()[act].all()


@pytest.mark.parametrize("dims", [(1, 30, 40), (3, 20, 0), (5, 300, 700)])
def test_update_prox_vars(testing, dims):
    B, n, num_duals = dims
    p, num_ineq = num_duals // 2, num_duals - num_duals // 2

    def build():
        rng = np.random.default_rng(40)
        result = Vars(B, n, p, num_ineq, 0, 0, 0, rng.normal(size=(B, n + num_ineq)),
                      rng.normal(size=(B, p + num_ineq)))
        prox = Vars(B, n, p, num_ineq, 0, 0, 0, rng.normal(size=(B, n + num_ineq)),
                    rng.normal(size=(B, p + num_ineq)))
        return dict(active_mask=boolean(rng.random(B) < 0.7), dual_improved=boolean(rng.random(B) < 0.5),
                    primal_improved=boolean(rng.random(B) < 0.5), result_x=result.x, prox_x=prox.x,
                    result_duals=result.duals_all, prox_duals=prox.duals_all, prox_primals=prox.primals_all)

    ref, nat = run_pair(build, sk.create_update_prox_vars_kernel(n, num_duals, wp.float64),
                        testing.ipm_update_prox_vars, (B, n, num_duals), dim=(B, n + num_duals))
    compare(ref, nat, ["prox_primals", "prox_duals"], ulp=0, label="update_prox_vars: ")


@pytest.mark.parametrize("it", [3, 12])
@pytest.mark.parametrize("dims", [(64, 7, 3), (16, 40, 0), (3, 300, 200)])
def test_update_rho_delta_without_ineq(testing, it, dims):
    B, n, p = dims
    lower = 1e-13

    def build():
        rng = np.random.default_rng(41 + it)
        d = _rho_delta_inputs(rng, B, lower)
        result = Vars(B, n, p, 0, 0, 0, 0, rng.normal(size=(B, n)), rng.normal(size=(B, p)))
        prox = Vars(B, n, p, 0, 0, 0, 0, rng.normal(size=(B, n)), rng.normal(size=(B, p)))
        d.update(result_x=result.x, prox_x=prox.x, result_y=result.y, prox_y=prox.y,
                 settings=f64(settings_floats(reg_finetune_lower_limit=lower)), current_iter=i32([it]),
                 prox_primals=prox.primals_all, prox_duals=prox.duals_all)
        return d

    ref, nat = run_pair(build, sk.create_update_rho_delta_without_ineq_kernel(n, p, wp.float64),
                        testing.ipm_update_rho_delta_without_ineq, (B, n, p), dim=(B, n + p))
    compare(ref, nat, ["info_rho", "info_delta", "info_no_primal_update", "info_no_dual_update",
                       "prox_primals", "prox_duals"], ulp=0, label="update_rho_delta_without_ineq: ")
