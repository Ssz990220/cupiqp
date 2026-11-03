"""Parity of the native cuDSS wrapper (M5) with the reference CudssStackedSolver (B = 1).

The same KKT-shaped matrix (lower view, full P block stored, laid out like the
reference ``SparseKKTSolver._init_kkt``) and right-hand side go through the
reference solver and, on separate device copies, through
``cupiqp_native._ext._testing.CudssSolver``. Both use the cuDSS default
(non-deterministic) mode: cuDSS 0.8.0.10 is not bitwise reproducible run to run
there, so solutions agree to ~1e-13 relative, not bitwise. Deterministic mode
is not exercised: its SOLVE faults intermittently inside cuDSS (see
native/docs/modules/cudss.md).
"""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp.sparse.batched_csr import UniformBatchedCsrMatrix
from cupiqp.sparse.sparse_direct_solver import CudssStackedSolver
from cupiqp_native._interop import dev_ptr, wrap

# Native vs reference: both are cuDSS with the same configuration, but cuDSS's
# default mode is not run-to-run reproducible: two runs of the REFERENCE on the
# same inputs differ by up to ~1e-11 relative on the worse-conditioned systems
# here (printed as "ref vs ref"). The bound is DESIGN 8's 1e-10 for quantities
# downstream of cuDSS; the measured numbers are printed.
REL_TOL = 1e-10
DENSE_TOL = 1e-10  # vs a dense/direct host solve


def _assemble(top, A, Ap, G, Gm):
    """[[top], [A, Ap], [G, 0, Gm]] with empty blocks dropped."""
    p, m = A.shape[0], G.shape[0]
    rows = [[top] + [None] * (p > 0) + [None] * (m > 0)]
    if p:
        rows.append([A, Ap] + [None] * (m > 0))
    if m:
        rows.append([G] + [None] * (p > 0) + [Gm])
    return sp.csr_matrix(sp.bmat(rows, format="csr"))


def kkt_pattern(P, A, G):
    """Reference KKT pattern (SparseKKTSolver._init_kkt): [[P + I], [A, I], [G, 0, I]]
    with every stored entry of P (full), A and G kept, sorted columns."""
    ones = lambda M: sp.csr_matrix((np.ones(M.nnz), M.indices, M.indptr), shape=M.shape)
    n, p, m = P.shape[0], A.shape[0], G.shape[0]
    K = _assemble(ones(P) + sp.eye(n, format="csr"), ones(A), sp.eye(p, format="csr"),
                  ones(G), sp.eye(m, format="csr"))
    K.sum_duplicates()
    K.sort_indices()
    return K


def kkt_values(K, P, A, G, rho, delta, z):
    """Quasi-definite values at the slots of K: P + rho I, -delta I, -diag(z).
    Slots whose block value is zero hold explicit zeros."""
    n, p, m = P.shape[0], A.shape[0], G.shape[0]
    full = _assemble(sp.csr_matrix(P) + rho * sp.eye(n, format="csr"), sp.csr_matrix(A),
                     -delta * sp.eye(p, format="csr"), sp.csr_matrix(G), -sp.diags(z, format="csr"))
    coo = K.tocoo()
    return np.asarray(full[coo.row, coo.col]).ravel()


def random_blocks(rng, n, p, m, dens_p=0.1, dens_a=0.2, dens_g=0.2):
    B = sp.random(n, n, density=dens_p, random_state=rng, format="csr")
    P = B @ B.T + sp.diags(rng.uniform(0.5, 2.0, n))  # SPD, symmetric pattern
    A = sp.random(p, n, density=dens_a, random_state=rng, format="csr")
    G = sp.random(m, n, density=dens_g, random_state=rng, format="csr")
    return sp.csr_matrix(P), sp.csr_matrix(A), sp.csr_matrix(G)


def lower_sym_solve(K_vals, K, b):
    """Host solve of the symmetric matrix defined by the lower triangle."""
    M = sp.csr_matrix((K_vals, K.indices, K.indptr), shape=K.shape)
    L = sp.tril(M, k=0)
    S = (L + sp.tril(M, k=-1).T).tocsc()
    return sp.linalg.spsolve(S, b)


class Pair:
    """Reference and native solvers over separate copies of one KKT."""

    def __init__(self, testing, K, vals, rhs):
        self.dim, self.nnz = K.shape[0], K.nnz
        self.mat = UniformBatchedCsrMatrix(1, K.indptr, K.indices, vals, K.shape, device="cuda:0")
        self.ref = CudssStackedSolver(self.mat)
        self.indptr = wp.array(K.indptr.astype(np.int32), dtype=wp.int32, device="cuda:0")
        self.indices = wp.array(K.indices.astype(np.int32), dtype=wp.int32, device="cuda:0")
        self.values = wp.array(vals, dtype=wp.float64, device="cuda:0")
        wp.synchronize()
        self.nat = testing.CudssSolver(self.dim, self.nnz, dev_ptr(self.indptr, np.int32, self.dim + 1),
                                       dev_ptr(self.indices, np.int32, self.nnz),
                                       dev_ptr(self.values, np.float64, self.nnz), stream=self.stream())
        self.nat_rhs = wrap(self.nat.rhs_ptr, (self.dim,), owner=self.nat)
        self.nat_sol = wrap(self.nat.sol_ptr, (self.dim,), owner=self.nat)
        self.set_rhs(rhs)

    def stream(self):
        return wp.get_stream("cuda:0").cuda_stream

    def set_rhs(self, rhs):
        self.ref.rhs.assign(wp.array(rhs[None, :], dtype=wp.float64, device="cuda:0"))
        self.nat_rhs.assign(wp.array(rhs, dtype=wp.float64, device="cuda:0"))
        wp.synchronize()

    def set_values(self, vals):
        self.mat.data.assign(wp.array(vals[None, :], dtype=wp.float64, device="cuda:0"))
        self.values.assign(wp.array(vals, dtype=wp.float64, device="cuda:0"))
        wp.synchronize()

    def plan(self):
        assert self.ref.plan(self.stream())
        wp.synchronize()
        self.nat.analyze(self.stream())

    def ref_factor_solve(self):
        self.ref.factor(self.stream())
        self.ref.solve(self.stream())
        wp.synchronize()
        return self.ref.sol.numpy()[0].copy()

    def factor_solve(self, graph=False):
        self.ref_factor_solve()
        if graph:
            self.nat.launch_graph(self.stream())
        else:
            self.nat.factor(self.stream())
            self.nat.solve(self.stream())
        return self.ref.sol.numpy()[0].copy(), self.nat_sol.numpy().copy()


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


CASES = [
    pytest.param(30, 0, 0, id="P-only"),
    pytest.param(40, 10, 0, id="P-A"),
    pytest.param(40, 0, 60, id="P-G"),
    pytest.param(80, 20, 150, id="P-A-G"),
]


@pytest.mark.parametrize("n,p,m", CASES)
def test_solution_matches_reference(testing, rng, n, p, m):
    P, A, G = random_blocks(rng, n, p, m)
    K = kkt_pattern(P, A, G)
    vals = kkt_values(K, P, A, G, rho=1e-6, delta=1e-4, z=rng.uniform(0.5, 2.0, m))
    rhs = rng.standard_normal(K.shape[0])
    pair = Pair(testing, K, vals, rhs)
    pair.plan()
    x_ref, x_nat = pair.factor_solve()
    x_host = lower_sym_solve(vals, K, rhs)
    r_nr, r_nh, r_rh = rel(x_nat, x_ref), rel(x_nat, x_host), rel(x_ref, x_host)
    print(f"\n[{n},{p},{m}] nnz={K.nnz}: native vs ref {r_nr:.3e}, native vs host {r_nh:.3e}, "
          f"ref vs host {r_rh:.3e}, ref vs ref {rel(pair.ref_factor_solve(), x_ref):.3e}, "
          f"bitwise equal {np.array_equal(x_nat, x_ref)}")
    assert r_nr < REL_TOL
    assert r_nh < DENSE_TOL


def test_repeated_updates_explicit_zeros_and_graph(testing, rng):
    """One analysis; values updated in place (incl. explicit zeros that become
    nonzero); eager and captured-graph native solves vs the reference."""
    n, p, m = 60, 15, 100
    P, A, G = random_blocks(rng, n, p, m)
    K = kkt_pattern(P, A, G)
    z = rng.uniform(0.5, 2.0, m)
    # First values: A and G zero (stored explicit zeros), P diagonal only.
    vals0 = kkt_values(K, sp.diags(P.diagonal()).tocsr(), A * 0.0, G * 0.0, 1e-6, 1e-4, z)
    rhs = rng.standard_normal(K.shape[0])
    pair = Pair(testing, K, vals0, rhs)
    pair.plan()
    worst = 0.0
    for it, graph in enumerate([False, False, True, True, False, True]):
        if it == 0:
            vals = vals0
        else:
            # same pattern, new values: rescaled blocks, new regularization
            vals = kkt_values(K, P * rng.uniform(0.5, 2.0), A * rng.uniform(0.5, 2.0),
                              G * rng.uniform(0.5, 2.0), 1e-6 * it, 1e-4 / it, rng.uniform(0.5, 2.0, m))
            pair.set_values(vals)
        x_ref, x_nat = pair.factor_solve(graph=graph)
        x_ref2 = pair.ref_factor_solve()
        x_host = lower_sym_solve(vals, K, rhs)
        worst = max(worst, rel(x_nat, x_ref))
        print(f"\n  update {it} ({'graph' if graph else 'eager'}): native vs ref {rel(x_nat, x_ref):.3e}, "
              f"ref vs ref {rel(x_ref2, x_ref):.3e}, native vs host {rel(x_nat, x_host):.3e}")
        assert rel(x_nat, x_ref) < REL_TOL, it
        assert rel(x_nat, x_host) < DENSE_TOL, it
    c = pair.nat.counters()
    print(f"\nworst native vs ref {worst:.3e}; counters {c}")
    assert c["cudss_analysis_calls"] == 1
    assert c["graph_captures"] == 1


def test_mpc_shape(testing):
    """MPC KKT shape: n=169, P 8977 stored, A 64x169 (920), G 2521x169 (249801)."""
    rng = np.random.default_rng(7)
    n, p, m = 169, 64, 2521
    S = sp.random(n, n, density=0.155, random_state=rng, format="csr")
    S = S + S.T
    P = sp.csr_matrix(S + sp.diags(np.abs(S).sum(axis=1).A1 + 1.0))  # ~8977 stored entries
    A = sp.random(p, n, density=920 / (p * n), random_state=rng, format="csr")
    G = sp.random(m, n, density=249801 / (m * n), random_state=rng, format="csr")
    K = kkt_pattern(P, A, G)
    vals = kkt_values(K, P, A, G, 1e-6, 1e-4, rng.uniform(0.5, 2.0, m))
    rhs = rng.standard_normal(K.shape[0])
    pair = Pair(testing, K, vals, rhs)
    pair.plan()
    x_ref, x_nat = pair.factor_solve()
    _, x_nat_g = pair.factor_solve(graph=True)
    x_ref2 = pair.ref_factor_solve()
    print(f"\nMPC shape ref vs ref {rel(x_ref2, x_ref):.3e}")
    x_host = lower_sym_solve(vals, K, rhs)
    print(f"\nMPC shape dim={K.shape[0]} nnz={K.nnz} (P {P.nnz}, A {A.nnz}, G {G.nnz}): native vs ref "
          f"{rel(x_nat, x_ref):.3e}, graph vs ref {rel(x_nat_g, x_ref):.3e}, native vs host {rel(x_nat, x_host):.3e}")
    assert rel(x_nat, x_ref) < 1e-10
    assert rel(x_nat_g, x_ref) < 1e-10
    assert rel(x_nat, x_host) < 1e-8
