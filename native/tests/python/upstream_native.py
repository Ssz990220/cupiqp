"""pytest plugin: run the upstream cuPIQP tests against the native solver.

Load with ``-p upstream_native`` (this directory and ``<build>/python`` on
PYTHONPATH). Before the upstream test modules are imported, it replaces
``cupiqp.SparseSolver`` with ``cupiqp_native.SparseSolver`` so that every
``from cupiqp import SparseSolver`` in the upstream tests gets the native
solver, while the rest of the reference package (helpers, Status, dense and
multistage backends, PIQP comparisons) stays the reference. Tests that need
features outside milestone 1 (batch size > 1, float32, gradients, solver
internals) are skipped with the reason, never silently passed.

``CUPIQP_NATIVE_KKT_BACKEND=reduced`` (default ``cudss``) makes every native
solver created by the upstream tests use that KKT backend. With ``reduced``,
a problem whose n exceeds the reduced backend's limit is skipped at setup()
with the reason (the backend rejects it), never silently passed.
"""
import os
import re
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_BUILD = Path(os.environ.get("CUPIQP_NATIVE_BUILD", _REPO / "build" / "native"))
for p in (_BUILD / "python", _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cupiqp  # noqa: E402
import cupiqp_native  # noqa: E402

KKT_BACKEND = os.environ.get("CUPIQP_NATIVE_KKT_BACKEND", "cudss").strip().lower()
if KKT_BACKEND not in ("cudss", "reduced"):
    raise ValueError(f"CUPIQP_NATIVE_KKT_BACKEND must be 'cudss' or 'reduced'; got {KKT_BACKEND!r}")

if KKT_BACKEND == "cudss":
    cupiqp.SparseSolver = cupiqp_native.SparseSolver
else:
    class _BackendSparseSolver(cupiqp_native.SparseSolver):
        """Native solver with the KKT backend selected by the environment."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.settings.kkt_backend = KKT_BACKEND

        def setup(self, *args, **kwargs):
            try:
                return super().setup(*args, **kwargs)
            except cupiqp_native._ext.Error as e:
                if "kkt_backend = Reduced supports n <=" in str(e):
                    pytest.skip(f"reduced KKT backend: {e}")
                raise

    _BackendSparseSolver.__name__ = "SparseSolver"
    cupiqp.SparseSolver = _BackendSparseSolver
NATIVE_ACTIVE = True

# Name patterns of upstream tests that exercise features outside milestone 1.
_OUT_OF_SCOPE = [
    (re.compile(r"batch", re.I), "batch size > 1 (milestone 1 is B = 1)"),
    (re.compile(r"float32|fp32", re.I), "float32 (milestone 1 is float64)"),
    (re.compile(r"grad|backward|smoothing", re.I), "gradients (out of milestone 1)"),
]


def pytest_collection_modifyitems(config, items):
    for item in items:
        for pat, reason in _OUT_OF_SCOPE:
            if pat.search(item.nodeid):
                item.add_marker(pytest.mark.skip(reason=f"native milestone 1: {reason}"))
                break
