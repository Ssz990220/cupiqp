"""cuPIQP native: Python binding of the native CUDA C++ solver.

Mirrors the reference ``cupiqp`` sparse API (``SparseSolver``, ``Settings``,
``Status``, ``PIQP_INF``) for milestone 1: sparse, float64, batch size 1.
The solver itself runs in C++; this package only converts arrays.
"""
from . import _ext
from ._ext import PIQP_INF, version  # noqa: F401

try:  # populated by the M8/M9 integration
    from ._solver import GraphMode, SparseSolver, Settings, Status  # noqa: F401
except ImportError:  # pragma: no cover - before integration
    pass

__all__ = ["SparseSolver", "Settings", "Status", "GraphMode", "PIQP_INF", "version"]
