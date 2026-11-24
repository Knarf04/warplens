"""CUDA tensor-map conversion, pitched layouts, and driver argument validation."""

import ctypes
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

import warplens
from warplens.host import tensor_map as tm
from warplens.host.shim_server import ShimServer, TENSOR_MAP

from conftest import cuda_include, kernel


def driver_map(**changes):
    options = dict(data_type=7, rank=2, address=16, dims=(4, 3), strides=(32,),
                   box=(4, 3), steps=(1, 1), swizzle=0, nan_fill=0)
    options.update(changes)
    return tm.from_driver(**options)


@pytest.mark.parametrize("dims,strides", [((4, 3), (32,)), ((4, 2, 2), (32, 96))])
def test_pitched_driver_map_round_trips_and_loads_the_requested_elements(sim, dims, strides):
    source = sim.from_numpy(np.arange(64, dtype=np.float32))
    info = driver_map(address=source.addr, rank=len(dims), dims=dims, strides=strides,
                      box=dims, steps=(1,) * len(dims))
    raw = tm.encode(info)
    assert tm.decode(raw) == info
    descriptor = sim.from_numpy(np.frombuffer(raw, dtype=np.uint8))
    out = sim.empty(int(np.prod(dims)), "f32")
    coords = ", ".join("0" for _ in dims)
    ptx = kernel(f"""
        .reg .u64 %map, %out;
        .reg .pred %ready;
        .shared .align 16 .b8 buffer[{info.box_bytes}];
        .shared .align 8 .b64 barrier;
        ld.param.u64 %map, [map];
        ld.param.u64 %out, [out];
        mbarrier.init.shared::cta.b64 [barrier], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [barrier], {info.box_bytes};
        cp.async.bulk.tensor.{len(dims)}d.shared::cluster.global.mbarrier::complete_tx::bytes
            [buffer], [%map, {{{coords}}}], [barrier];
        mbarrier.try_wait.parity.shared::cta.b64 %ready, [barrier], 0;
        cp.async.bulk.global.shared::cta.bulk_group [%out], [buffer], {info.box_bytes};
        cp.async.bulk.commit_group;
        cp.async.bulk.wait_group 0;
        ret;
    """, ".param .u64 map, .param .u64 out")
    stats = sim.launch(ptx, "k", grid=1, block=1, args=[descriptor, out])
    expected = ([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19] if len(dims) == 2 else
                [0, 1, 2, 3, 8, 9, 10, 11, 24, 25, 26, 27, 32, 33, 34, 35])
    np.testing.assert_array_equal(out.numpy(), expected)
    assert not stats.findings


@pytest.mark.parametrize("dimension", [2**31 + 1, 2**32])
def test_driver_map_max_dimension_is_encodable(dimension):
    info = driver_map(rank=1, dims=(dimension,), strides=(), box=(4,), steps=(1,))
    assert len(tm.encode(info)) == 128
    assert tm.decode(tm.encode(info)) == info


@pytest.mark.parametrize("changes", [
    {"dims": (4,)}, {"strides": ()}, {"box": (4,)}, {"steps": (1,)},
    {"dims": (4.0, 3)}, {"strides": (32.0,)}, {"address": -16},
    {"strides": (0,)}, {"strides": (-16,)}, {"strides": (2**40,)},
    {"dims": (8, 3), "strides": (16,)},
    {"rank": 3, "dims": (4, 3, 2), "strides": (32, 64), "box": (4, 3, 2), "steps": (1, 1, 1)},
    {"nan_fill": 2}, {"dims": (2**32 + 1, 3)},
])
def test_driver_map_rejects_invalid_geometry_and_enums(changes):
    with pytest.raises(warplens.LaunchError):
        driver_map(**changes)


def test_bf16_nan_fill_is_rejected_instead_of_emitting_finite_half_nan_bits():
    with pytest.raises(warplens.LaunchError, match="BF16.*NaN.*not supported"):
        driver_map(data_type=9, box=(8, 3), dims=(8, 3), nan_fill=1)
    # Ordinary BF16 transfers are bit copies and remain supported.
    assert driver_map(data_type=9, box=(8, 3), dims=(8, 3)).elem_size == 2


def test_tensor_map_request_preserves_pitch_and_upper_dimension_bound():
    request = struct.pack("<5IQ5Q4Q5I5I", 7, 2, 0, 0, 0, 16,
                          4, 2**32, 0, 0, 0, 32, 0, 0, 0,
                          4, 2, 0, 0, 0, 1, 1, 0, 0, 0)
    raw = ShimServer().handle(TENSOR_MAP, request)
    info = tm.decode(raw)
    assert info.dims == (4, 2**32) and info.strides == (32,)


@pytest.fixture(scope="module")
def driver_shim(tmp_path_factory):
    """Test the real C entry point with a bounded protocol stub, without a GPU/server."""
    root = Path(__file__).resolve().parents[1]
    include = cuda_include("cuda.h")
    cc = shutil.which("cc")
    if not cc or include is None:
        pytest.skip("C compiler and CUDA headers required")
    tmp = tmp_path_factory.mktemp("driver_shim")
    stub = tmp / "stub.c"
    stub.write_text("""
        #include <string.h>
        #include "protocol.h"
        int calls = 0;
        int warplens_call(uint32_t op, const struct wl_part *parts, int count,
                          void *reply, size_t capacity, size_t *length) {
            (void)op; (void)parts; (void)count;
            ++calls;
            memset(reply, 0xa5, capacity);
            if (length) *length = capacity;
            return 0;
        }
        /* The module and launch calls of the driver shim are not exercised here. */
        int warplens_register_function(uint32_t module, const char *name, const void *key) {
            (void)module; (void)name; (void)key; return 1;
        }
        int warplens_launch(const void *key, const uint32_t grid[3], const uint32_t block[3],
                            const uint32_t cluster[3], uint64_t shared_memory, void **arguments) {
            (void)key; (void)grid; (void)block; (void)cluster; (void)shared_memory; (void)arguments;
            return 1;
        }
        int warplens_device_attribute(int attribute) { (void)attribute; return 0; }
    """)
    library = tmp / "libdriver_test.so"
    subprocess.run([cc, "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(include), "-I" + str(root / "warplens/shim"),
                    str(root / "warplens/shim/cuda_shim.c"), str(stub), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    lib = ctypes.CDLL(str(library))
    lib.cuTensorMapEncodeTiled.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.cuTensorMapEncodeTiled.restype = ctypes.c_int
    return lib


def invoke_driver(lib, **changes):
    storage = ctypes.create_string_buffer(191)
    pointer = (ctypes.addressof(storage) + 63) & ~63
    dims = (ctypes.c_uint64 * 2)(4, 3)
    strides = (ctypes.c_uint64 * 1)(32)
    box = (ctypes.c_uint32 * 2)(4, 3)
    steps = (ctypes.c_uint32 * 2)(1, 1)
    options = dict(output=pointer, data_type=7, rank=2, address=16,
                   dims=ctypes.addressof(dims), strides=ctypes.addressof(strides),
                   box=ctypes.addressof(box), steps=ctypes.addressof(steps),
                   interleave=0, swizzle=0, promotion=0, fill=0)
    options.update(changes)
    return lib.cuTensorMapEncodeTiled(*options.values())


@pytest.mark.parametrize("changes", [
    {"output": None}, {"output": 1}, {"dims": None}, {"strides": None},
    {"box": None}, {"steps": None}, {"rank": 0}, {"rank": 6},
    {"promotion": -1}, {"promotion": 4}, {"fill": -1}, {"fill": 2},
])
def test_driver_returns_invalid_value_before_dereferencing_invalid_arguments(driver_shim, changes):
    before = ctypes.c_int.in_dll(driver_shim, "calls").value
    assert invoke_driver(driver_shim, **changes) == 1
    assert ctypes.c_int.in_dll(driver_shim, "calls").value == before


def test_rank_one_driver_map_does_not_need_global_strides(driver_shim):
    assert invoke_driver(driver_shim, rank=1, strides=None) == 0


def test_driver_init_rejects_nonzero_flags(driver_shim):
    assert driver_shim.cuInit(0) == 0
    assert driver_shim.cuInit(1) == 1
