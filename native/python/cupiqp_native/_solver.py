"""Reference-compatible Python front end of the native solver.

``cupiqp_native.SparseSolver`` mirrors ``cupiqp.SparseSolver`` (sparse,
float64, batch size 1): ``setup`` takes CSR triples ``(indptr, indices,
values)`` with GPU values, ``update`` takes GPU value arrays, ``solve`` runs
the native C++/CUDA solver, and ``result`` exposes ``(1, k)`` Warp arrays that
alias the native buffers plus ``result.info`` with ``to_host()``. The native
library does all the work; this module only converts arrays and orders the
native stream against the caller's current Warp stream.
"""
from __future__ import annotations

from enum import IntEnum
from types import SimpleNamespace

import numpy as np

from . import _ext
from ._interop import dev_ptr, is_cuda_array, wrap

PIQP_INF = _ext.PIQP_INF


class Status(IntEnum):
    """Same names and codes as ``cupiqp.Status``."""
    CUPIQP_UNSOLVED = -1
    CUPIQP_SOLVED = 0
    CUPIQP_MAX_ITER_REACHED = 1
    CUPIQP_PRIMAL_INFEASIBLE = 2
    CUPIQP_DUAL_INFEASIBLE = 3
    CUPIQP_NUMERICAL_ISSUES = 4


GraphMode = _ext.GraphMode

# Reference Settings fields that only exist for unsupported features; setting
# them to a non-default value is rejected at setup().
_UNSUPPORTED_DEFAULTS = {"enable_grad": False, "gradient_smoothing": False}
_IGNORED = {"gradient_smoothing_mu", "gradient_smoothing_tol", "gradient_smoothing_max_iter",
            "gradient_on_failure"}


class Settings:
    """Attribute proxy over the native settings (reference field names)."""

    def __init__(self, native_settings):
        object.__setattr__(self, "_native", native_settings)
        object.__setattr__(self, "_extra", dict(_UNSUPPORTED_DEFAULTS))

    @property
    def dtype(self):
        import warp as wp
        return wp.float64

    def __getattr__(self, name):
        native = object.__getattribute__(self, "_native")
        if hasattr(native, name):
            return getattr(native, name)
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        native = object.__getattribute__(self, "_native")
        if name == "dtype":
            raise AttributeError("Settings.dtype is fixed (float64 in the native milestone 1)")
        if hasattr(native, name):
            setattr(native, name, value)
        elif name in _UNSUPPORTED_DEFAULTS or name in _IGNORED:
            object.__getattribute__(self, "_extra")[name] = value
        else:
            raise AttributeError(f"unknown setting {name!r}")

    def verify_settings(self) -> bool:
        return self._native.verify()


class _HostArray:
    """``.numpy()`` adapter for per-problem info fields (host snapshot)."""

    def __init__(self, solver, key, dtype):
        self._solver, self._key, self._dtype = solver, key, dtype

    def numpy(self):
        return np.array([self._solver._native.info()[self._key]], dtype=self._dtype)


class Info:
    def __init__(self, solver):
        self._solver = solver

    def to_host(self):
        self._solver._sync_caller()
        d = self._solver._native.info()
        snap = SimpleNamespace()
        for k, v in d.items():
            if k == "status":
                snap.status = [Status(v)]
                snap.status_value = np.array([v], dtype=np.int32)
            elif k == "iter_total":
                snap.iter_total = int(v)
            else:
                setattr(snap, k, np.array([v], dtype=np.float64 if isinstance(v, float) else np.int32))
        return snap

    @property
    def status(self):
        return _HostArray(self._solver, "status", np.int32)

    @property
    def iter(self):
        return _HostArray(self._solver, "iter", np.int32)

    def reset_status(self):  # pragma: no cover - compatibility
        pass


class Result:
    def __init__(self, solver):
        self._solver = solver
        self.info = Info(solver)
        self._views = {}

    def __getattr__(self, name):
        views = object.__getattribute__(self, "_views")
        if name in views:
            return views[name]
        raise AttributeError(name)

    def _bind(self, ptrs: dict):
        for k, (ptr, length) in ptrs.items():
            self._views[k] = wrap(ptr, (1, length), owner=self._solver)


def _host_index(a) -> np.ndarray:
    if is_cuda_array(a):
        a = a.numpy() if hasattr(a, "numpy") else np.asarray(a.get())
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype not in (np.int32, np.int64):
        raise TypeError(f"CSR index arrays must be int32 or int64; got {a.dtype}")
    if a.ndim != 1:
        raise ValueError(f"CSR index arrays must be 1-D; got shape {a.shape}")
    return a.astype(np.int64, copy=False)


def _values_ptr(name, arr, size):
    if arr is None:
        return None
    if not is_cuda_array(arr):
        raise TypeError(f"{name} values must be a GPU array (exposing __cuda_array_interface__); "
                        f"got {type(arr).__name__}")
    return dev_ptr(arr, np.float64, size=size, name=name)


class SparseSolver:
    """Native-backed equivalent of ``cupiqp.SparseSolver`` (float64, B = 1)."""

    def __init__(self, dtype=None, stream=None):
        import warp as wp
        if dtype is not None and dtype is not wp.float64:
            raise TypeError("cupiqp_native milestone 1 supports float64 only")
        external = 0 if stream is None else int(stream.cuda_stream)
        self._native = _ext.NativeSparseSolver(external)
        self._settings = Settings(self._native.settings)
        self._wp_stream = stream if stream is not None else wp.Stream(
            "cuda:0", cuda_stream=self._native.stream)
        self._result = Result(self)
        self._sizes = None

    # ------------------------------------------------------------- properties
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def stream(self):
        return self._wp_stream

    @property
    def result(self) -> Result:
        return self._result

    @property
    def native(self):
        """The underlying ``_ext.NativeSparseSolver`` (counters, debug buffers)."""
        return self._native

    # ------------------------------------------------------------- stream order
    def _enter(self):
        import warp as wp
        cur = wp.get_stream("cuda:0")
        if cur.cuda_stream != self._wp_stream.cuda_stream:
            self._wp_stream.wait_stream(cur)

    def _exit(self):
        import warp as wp
        cur = wp.get_stream("cuda:0")
        if cur.cuda_stream != self._wp_stream.cuda_stream:
            cur.wait_stream(self._wp_stream)

    def _sync_caller(self):
        self._exit()

    # ------------------------------------------------------------- API
    def setup(self, P, c, A=None, b=None, G=None, h_u=None, h_l=None, x_u=None, x_l=None):
        for k, v in _UNSUPPORTED_DEFAULTS.items():
            if getattr(self._settings, k) != v:
                raise NotImplementedError(f"{k} is not supported by cupiqp_native milestone 1")
        if P is None:
            raise TypeError("SparseSolver.setup requires P.")

        def triple(name, M):
            if M is None:
                return None
            if not (isinstance(M, (tuple, list)) and len(M) == 3):
                raise TypeError(f"SparseSolver requires {name} as a CSR triple (indptr, indices, values)")
            return _host_index(M[0]), _host_index(M[1]), M[2]

        Pt, At, Gt = triple("P", P), triple("A", A), triple("G", G)
        n = len(Pt[0]) - 1
        p = len(At[0]) - 1 if At else 0
        m = len(Gt[0]) - 1 if Gt else 0
        nnz = lambda t: int(t[0][-1]) if t is not None and len(t[0]) else 0  # noqa: E731
        self._sizes = dict(P=nnz(Pt), A=nnz(At), G=nnz(Gt), c=n, b=p, h_l=m, h_u=m, x_l=n, x_u=n)
        vals = {"P": _values_ptr("P", Pt[2], self._sizes["P"]), "c": _values_ptr("c", c, n)}
        if At is not None:
            vals["A"] = _values_ptr("A", At[2], self._sizes["A"])
        if Gt is not None:
            vals["G"] = _values_ptr("G", Gt[2], self._sizes["G"])
        for name, arr in (("b", b), ("h_l", h_l), ("h_u", h_u), ("x_l", x_l), ("x_u", x_u)):
            vals[name] = _values_ptr(name, arr, self._sizes[name])
        self._enter()
        self._native.setup(n, Pt[0], Pt[1], At[0] if At else None, At[1] if At else None, p,
                           Gt[0] if Gt else None, Gt[1] if Gt else None, m, vals)
        self._result._bind(self._native.result_ptrs())
        self._exit()

    def update(self, P=None, c=None, A=None, b=None, G=None, h_u=None, h_l=None, x_u=None, x_l=None,
               check_validity=False):
        if self._sizes is None:
            raise RuntimeError("Solver not setup yet. Call setup() first.")
        vals = {}
        for name, arr in (("P", P), ("c", c), ("A", A), ("b", b), ("G", G), ("h_l", h_l),
                          ("h_u", h_u), ("x_l", x_l), ("x_u", x_u)):
            if arr is not None:
                vals[name] = _values_ptr(name, arr, self._sizes[name])
        self._enter()
        self._native.update(vals)
        self._exit()

    def update_from_host(self, **arrays):
        """Native extension: values in host NumPy arrays (pinned recommended)."""
        vals = {}
        keep = []
        for name, arr in arrays.items():
            if arr is None:
                continue
            if hasattr(arr, "__array_interface__") and not is_cuda_array(arr):
                a = np.ascontiguousarray(arr, dtype=np.float64)
                if a.size != self._sizes[name]:
                    raise ValueError(f"{name}: expected {self._sizes[name]} values, got {a.size}")
                keep.append(a)
                vals[name] = a.__array_interface__["data"][0]
            else:  # pinned warp host array
                a = arr.numpy() if hasattr(arr, "numpy") else arr
                keep.append(arr)
                vals[name] = int(arr.ptr)
        self._enter()
        self._native.update(vals, True)
        self._exit()
        return keep

    def solve(self):
        if self._sizes is None:
            raise RuntimeError("Solver not setup yet. Call setup() first.")
        self._enter()
        self._native.solve()
        self._exit()
