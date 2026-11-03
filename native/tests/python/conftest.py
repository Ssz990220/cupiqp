"""pytest configuration for native parity tests.

Puts the built binding (``<build>/python``) and the upstream repository root on
``sys.path`` so tests can import ``cupiqp_native`` (port), ``cupiqp``
(pinned reference) and ``tests.helpers`` (upstream problem builders).
Set CUPIQP_NATIVE_BUILD to the CMake build directory (default: <repo>/build/native).
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get("CUPIQP_NATIVE_BUILD", REPO / "build" / "native"))
for p in (BUILD / "python", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("WARP_CACHE_PATH", str(Path.home() / ".cache" / "warp-cupiqp-native"))

import warp as wp  # noqa: E402

wp.config.quiet = True
wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
wp.init()


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def native():
    import cupiqp_native
    return cupiqp_native


@pytest.fixture(scope="session")
def testing(native):
    return native._ext._testing
