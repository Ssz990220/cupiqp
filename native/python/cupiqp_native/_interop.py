"""Array interop for the nanobind layer (integration owner).

Device arrays cross into C++ as integer addresses. ``dev_ptr`` accepts any
object exposing ``__cuda_array_interface__`` (Warp, CuPy, Torch, Numba) and
checks dtype, contiguity and size. ``wrap`` exposes a native device buffer as a
zero-copy ``warp.array`` (the buffer stays owned by the native solver).
"""
from __future__ import annotations

import numpy as np

_TYPESTR = {np.dtype(np.float64): "<f8", np.dtype(np.float32): "<f4",
            np.dtype(np.int32): "<i4", np.dtype(np.int64): "<i8", np.dtype(np.uint8): "|u1",
            np.dtype(np.bool_): "|b1"}


def is_cuda_array(obj) -> bool:
    return hasattr(obj, "__cuda_array_interface__")


def cai(obj) -> dict:
    if not is_cuda_array(obj):
        raise TypeError(f"expected a GPU array exposing __cuda_array_interface__; got {type(obj).__name__}")
    return obj.__cuda_array_interface__


def dev_ptr(obj, dtype=np.float64, size: int | None = None, name: str = "array",
            allow_2d_single_row: bool = True) -> int:
    """Device address of a C-contiguous GPU array of ``dtype``.

    ``size`` (optional) is the required element count. ``(1, k)`` arrays are
    accepted for ``k``-vectors (batch size 1).
    """
    if obj is None:
        return 0
    iface = cai(obj)
    want = _TYPESTR[np.dtype(dtype)]
    typestr = iface["typestr"]
    if typestr != want and not (want == "|b1" and typestr == "|u1"):
        raise TypeError(f"{name}: expected dtype {np.dtype(dtype)} ({want}), got {typestr}")
    shape = tuple(iface["shape"])
    strides = iface.get("strides")
    itemsize = np.dtype(dtype).itemsize
    if strides is not None:
        expect = []
        acc = itemsize
        for dim in reversed(shape):
            expect.append(acc)
            acc *= dim
        if tuple(reversed(expect)) != tuple(strides) and int(np.prod(shape)) > 1:
            raise ValueError(f"{name}: array must be C-contiguous (shape {shape}, strides {strides})")
    count = int(np.prod(shape)) if shape else 1
    if len(shape) == 2 and not (allow_2d_single_row and shape[0] == 1):
        raise ValueError(f"{name}: batch size must be 1 in milestone 1; got shape {shape}")
    if len(shape) > 2:
        raise ValueError(f"{name}: expected a 1-D (or (1, k)) array; got shape {shape}")
    if size is not None and count != size:
        raise ValueError(f"{name}: expected {size} elements; got shape {shape}")
    return int(iface["data"][0])


def wrap(ptr: int, shape, dtype=np.float64, owner=None):
    """Zero-copy ``warp.array`` view of native device memory."""
    import warp as wp
    wp_dtype = {np.dtype(np.float64): wp.float64, np.dtype(np.float32): wp.float32,
                np.dtype(np.int32): wp.int32, np.dtype(np.uint8): wp.uint8}[np.dtype(dtype)]
    shape = tuple(int(s) for s in shape)
    if ptr == 0 or int(np.prod(shape)) == 0:
        return wp.empty(shape, dtype=wp_dtype, device="cuda:0")
    arr = wp.array(ptr=int(ptr), dtype=wp_dtype, shape=shape, device="cuda:0", copy=False)
    if owner is not None:
        arr._cupiqp_native_owner = owner  # keep the native solver alive while the view lives
    return arr
