"""Parity of the native data module (cupiqp::SparseData) with the reference
cupiqp SparseData, read from a reference SparseSolver with
preconditioner_iter = 0 (so solver.data holds the unscaled values).

Values, patterns, COO row indices and finite-bound masks must be bitwise
equal, at setup and after updates (bound switches finite <-> infinite,
explicit zeros becoming nonzero)."""
import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from cupiqp import SparseSolver
from cupiqp.sparse.sparse_solver import _check_csr
from cupiqp_native._interop import dev_ptr, wrap
from tests.helpers import quiet, random_sparse_qp, random_sparse_spd

VEC = ("c", "b", "h_l", "h_u", "x_l", "x_u")
MASKS = ("finite_mask_all", "active_G_row", "active_x_bound", "num_finite_bounds")


def _wp(a, dtype=wp.float64):
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device="cuda:0")


def _bits_equal(a, b, what):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64)).ravel()
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float64)).ravel()
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), f"{what}: {a} vs {b}"


class Pair:
    """The same problem in the reference solver and in the native SparseData."""

    def __init__(self, testing, qp):
        self.qp = qp
        P, A, G = qp["P"], qp.get("A"), qp.get("G")
        # Device value arrays (kept alive for the native side).
        self.dev = {"P": _wp(P.data), "c": _wp(qp["c"])}
        if A is not None:
            self.dev["A"] = _wp(A.data)
            self.dev["b"] = _wp(qp["b"])
        if G is not None:
            self.dev["G"] = _wp(G.data)
        for k in ("h_l", "h_u", "x_l", "x_u"):
            if qp.get(k) is not None:
                self.dev[k] = _wp(qp[k])

        self.ref = quiet(SparseSolver(), preconditioner_iter=0)
        trip = lambda M, k: (M.indptr.astype(np.int32), M.indices.astype(np.int32), _wp(M.data))  # noqa: E731
        self.ref.setup(
            P=trip(P, "P"), c=_wp(qp["c"]),
            A=trip(A, "A") if A is not None else None, b=_wp(qp["b"]) if A is not None else None,
            G=trip(G, "G") if G is not None else None,
            h_u=_wp(qp["h_u"]) if qp.get("h_u") is not None else None,
            h_l=_wp(qp["h_l"]) if qp.get("h_l") is not None else None,
            x_u=_wp(qp["x_u"]) if qp.get("x_u") is not None else None,
            x_l=_wp(qp["x_l"]) if qp.get("x_l") is not None else None,
        )

        self.nat = testing.SparseData()
        wp.synchronize()
        self.nat.init(
            P.indptr.tolist(), P.indices.tolist(),
            A.indptr.tolist() if A is not None else None, A.indices.tolist() if A is not None else None,
            G.indptr.tolist() if G is not None else None, G.indices.tolist() if G is not None else None,
            {k: dev_ptr(v) for k, v in self.dev.items()},
        )
        testing.synchronize()
        self.testing = testing

    def update(self, **vals):
        """Apply the same update to both sides (host arrays in, copied to GPU)."""
        dev = {k: _wp(v) for k, v in vals.items()}
        self.ref.update(**dev)
        wp.synchronize()
        mask = self.nat.set_values({k: dev_ptr(v) for k, v in dev.items()})
        self.testing.synchronize()
        assert {k for k, on in mask.items() if on} == set(vals)
        self._keep = dev

    def native_arrays(self):
        d, v = self.nat.dims(), self.nat.view()
        shapes = {
            "P": d["nnz_P"], "A": d["nnz_A"], "G": d["nnz_G"], "c": d["n"], "b": d["p"],
            "h_l": d["num_hl"], "h_u": d["num_hu"], "x_l": d["num_xl"], "x_u": d["num_xu"],
            "finite_mask_all": d["num_ineq"], "active_G_row": d["m"], "active_x_bound": d["n"],
            "num_finite_bounds": 1,
        }
        out = {k: wrap(v[k], (s,)).numpy() for k, s in shapes.items()}
        for M in ("P", "A", "G"):
            rows = {"P": d["n"], "A": d["p"], "G": d["m"]}[M]
            nnz = d[f"nnz_{M}"]
            out[f"{M}_indptr"] = wrap(v[f"{M}_indptr"], (rows + 1,), np.int32).numpy()
            out[f"{M}_indices"] = wrap(v[f"{M}_indices"], (nnz,), np.int32).numpy()
            out[f"{M}_row_idx"] = wrap(v[f"{M}_row_idx"], (nnz,), np.int32).numpy()
        return out

    def compare(self):
        wp.synchronize()
        self.testing.synchronize()
        rd = self.ref.data
        nat = self.native_arrays()
        d = self.nat.dims()
        assert (d["n"], d["p"], d["m"]) == (rd.n, rd.p, rd.m)
        assert (d["num_hl"], d["num_hu"], d["num_xl"], d["num_xu"]) == (rd.num_hl, rd.num_hu, rd.num_xl, rd.num_xu)
        for M in ("P", "A", "G"):
            R = getattr(rd, M)
            assert d[f"nnz_{M}"] == R.nnz
            _bits_equal(nat[M], R.data.numpy()[0], M)
            np.testing.assert_array_equal(nat[f"{M}_indptr"], R.indptr.numpy())
            if R.nnz:
                np.testing.assert_array_equal(nat[f"{M}_indices"], R.indices.numpy())
                np.testing.assert_array_equal(nat[f"{M}_row_idx"], R.row_indices.numpy())
        for k in VEC:
            _bits_equal(nat[k], getattr(rd, k).numpy()[0], k)
        for k in MASKS:
            ref = getattr(rd, k).numpy()
            _bits_equal(nat[k], ref[0] if ref.ndim == 2 else ref, k)


def _qp(seed, n=8, p=3, m=5, density=0.5, inf_fraction=0.0, drop=()):
    qp = random_sparse_qp(seed, n=n, p=p, m=m, density=density, inf_fraction=inf_fraction)
    for k in drop:
        qp[k] = None
    if qp.get("A") is None:
        qp["b"] = None
    if qp.get("G") is None:
        qp["h_l"] = qp["h_u"] = None
    return qp


CASES = {
    "all_finite": dict(seed=1),
    "mixed_inf": dict(seed=2, inf_fraction=0.4),
    "no_A": dict(seed=3, drop=("A",)),
    "no_G": dict(seed=4, drop=("G",)),
    "no_A_no_G_box": dict(seed=5, drop=("A", "G")),
    "only_h_u_x_l": dict(seed=6, inf_fraction=0.3, drop=("h_l", "x_u")),
    "only_h_l_x_u": dict(seed=7, inf_fraction=0.3, drop=("h_u", "x_l")),
    "no_box": dict(seed=8, drop=("x_l", "x_u")),
    "larger": dict(seed=9, n=300, p=40, m=200, density=0.05, inf_fraction=0.25),
}


@pytest.mark.parametrize("case", list(CASES))
def test_setup_parity(testing, case):
    Pair(testing, _qp(**CASES[case])).compare()


def test_edge_values_parity(testing):
    """+-inf vs +-1e20, NaN, lower +inf and upper -inf: the reference
    comparisons (lower > -1e20, upper < 1e20) are matched bitwise."""
    qp = _qp(10, n=6, m=6)
    qp["h_l"] = np.array([-np.inf, -1e20, -9.999999999999999e19, np.nan, np.inf, 0.0])
    qp["h_u"] = np.array([np.inf, 1e20, 9.999999999999999e19, np.nan, -np.inf, 0.0])
    qp["x_l"] = np.array([np.nan, -1e20, -np.inf, 1.0, np.inf, -1e300])
    qp["x_u"] = np.array([1e20, np.nan, np.inf, -np.inf, 2.0, 1e300])
    Pair(testing, qp).compare()


def test_bound_switches_parity(testing):
    qp = _qp(11, n=10, p=2, m=8)
    pair = Pair(testing, qp)
    rng = np.random.default_rng(0)
    for it in range(6):
        upd = {}
        for k, size in (("h_l", 8), ("h_u", 8), ("x_l", 10), ("x_u", 10)):
            v = qp[k].copy()
            mask = rng.random(size) < (0.0 if it == 0 else 0.5 if it < 5 else 1.0)
            v[mask] = -np.inf if k.endswith("l") else np.inf
            upd[k] = v
        if it == 3:
            upd = {"h_u": upd["h_u"], "x_l": upd["x_l"]}  # partial update
        pair.update(**upd)
        pair.compare()


def test_explicit_zeros_become_nonzero(testing):
    """Stored zeros keep their slot; updating them to nonzero values lands in
    the same stored-entry positions on both sides."""
    qp = _qp(12, n=7, p=2, m=4)
    for k in ("P", "A", "G"):
        M = qp[k].copy()
        M.data[::2] = 0.0  # explicit zeros, still stored
        qp[k] = M
    pair = Pair(testing, qp)
    assert pair.nat.dims()["nnz_P"] == qp["P"].nnz
    pair.compare()
    pair.update(P=qp["P"].data + 1.5, A=qp["A"].data - 2.0, G=qp["G"].data * 0 + 3.0)
    pair.compare()
    pair.update(c=np.arange(7.0), b=np.array([1.0, -1.0]))
    pair.compare()


def test_set_values_from_host(testing):
    qp = _qp(13, n=6, p=2, m=3)
    pair = Pair(testing, qp)
    new = {"G": qp["G"].data * 2.0, "h_u": np.array([np.inf, 1.0, 1e20]), "x_l": -np.ones(6)}
    host = {k: np.ascontiguousarray(v) for k, v in new.items()}
    pair.ref.update(**{k: _wp(v) for k, v in new.items()})
    wp.synchronize()
    mask = pair.nat.set_values_from_host({k: v.ctypes.data for k, v in host.items()})
    testing.synchronize()
    assert {k for k, on in mask.items() if on} == set(new)
    pair.compare()


def test_reference_kernel_matches(testing):
    """Version 1 (one thread per problem) and the block kernel agree."""
    pair = Pair(testing, _qp(**CASES["larger"]))
    pair.compare()
    pair.nat.update_finite_masks_reference()
    pair.compare()


def test_update_rejects_absent_side(testing):
    from cupiqp_native._ext import Error
    qp = _qp(14, drop=("h_l", "x_u"))
    pair = Pair(testing, qp)
    v = _wp(np.zeros(qp["P"].shape[0]))
    with pytest.raises(Error, match=r"Cannot set x_u: no x_u block was provided at setup\(\)"):
        pair.nat.set_values({"x_u": dev_ptr(v)})
    with pytest.raises(ValueError, match=r"Cannot set x_u: no x_u block was provided at setup\(\)"):
        pair.ref.update(x_u=v)


def test_init_consistency_messages(testing):
    from cupiqp_native._ext import Error
    from cupiqp.sparse.sparse_data import SparseData as RefSparseData
    P = random_sparse_spd(15, 4)
    G = sp.csr_matrix(np.eye(2, 4))
    Pv, c, Gv, h = _wp(P.data), _wp(np.zeros(4)), _wp(G.data), _wp(np.zeros(2))
    ref_trip = (P.indptr.astype(np.int32), P.indices.astype(np.int32), Pv)
    cases = [
        (dict(b=h), dict(b=h), "A and b must either both"),
        (dict(G=(G.indptr.astype(np.int32), G.indices.astype(np.int32), Gv)), dict(G=Gv, G_pat=G),
         "Either h_l or h_u must be provided when G is given."),
        (dict(h_l=h), dict(h_l=h), "h_l and h_u must be None when G is None."),
    ]
    for ref_kw, nat_kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            RefSparseData().init(P=ref_trip, c=c, **ref_kw)
        Gp = nat_kw.pop("G_pat", None)
        vals = {"P": dev_ptr(Pv), "c": dev_ptr(c), **{k: dev_ptr(v) for k, v in nat_kw.items()}}
        with pytest.raises(Error, match=msg):
            testing.SparseData().init(P.indptr.tolist(), P.indices.tolist(), None, None,
                                      Gp.indptr.tolist() if Gp is not None else None,
                                      Gp.indices.tolist() if Gp is not None else None, vals)


BAD_PATTERNS = {
    "indptr_start": ([1, 2, 3], [0, 1]),
    "indptr_decrease": ([0, 2, 1], [0, 1]),
    "indptr_end": ([0, 1, 3], [0, 1]),
    "col_too_large": ([0, 1, 2], [0, 2]),
    "col_negative": ([0, 1, 2], [-1, 0]),
    "unsorted": ([0, 2, 3], [1, 0, 1]),
    "duplicate": ([0, 2, 2], [1, 1]),
}


@pytest.mark.parametrize("name", list(BAD_PATTERNS))
def test_validation_messages_match_reference(testing, name):
    from cupiqp_native._ext import Error
    indptr, indices = BAD_PATTERNS[name]
    values = _wp(np.ones(len(indices)))
    with pytest.raises(ValueError) as ref_err:
        _check_csr("A", (np.array(indptr, np.int32), np.array(indices, np.int32), values), wp.float64, cols=2)
    with pytest.raises(Error) as nat_err:
        testing.validate_csr(indptr, indices, 2, 2, "A")
    assert str(nat_err.value) == str(ref_err.value)


def test_validation_accepts_reference_valid(testing):
    P = random_sparse_spd(16, 20)
    testing.validate_csr(P.indptr.tolist(), P.indices.tolist(), 20, -1, "P")
    _check_csr("P", (P.indptr, P.indices, _wp(P.data)), wp.float64)
