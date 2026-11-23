"""Host validation and corrupted descriptor regressions for M2."""

from dataclasses import replace

import numpy as np
import pytest

import warplens
from warplens.host import tensor_map as tm

from conftest import kernel


@pytest.mark.parametrize("kwargs", [
    {"box": (4.0, 8)}, {"box": (1.5, 8)}, {"box": (True, 8)},
    {"box": 4.0}, {"box": (4, "8")},
    {"box": (4, 8), "element_strides": (1.5, 1)},
    {"box": (4, 8), "element_strides": (True, 1)},
])
def test_tensor_map_rejects_noninteger_geometry(sim, kwargs):
    array = sim.empty((8, 16), "u32")
    with pytest.raises(warplens.LaunchError, match="integers"):
        sim.tensor_map_tiled(array, **kwargs)


def test_tensor_map_accepts_numpy_integer_geometry(sim):
    array = sim.empty((8, 16), "u32")
    descriptor = sim.tensor_map_tiled(array, np.array([4, 8], dtype=np.int64),
                                     element_strides=np.array([2, 3], dtype=np.uint32))
    assert descriptor.info.counts == (8, 2)
    assert tm.decode(descriptor.numpy().tobytes()) == descriptor.info


def test_tensor_map_dimension_limit_round_trips():
    # Exercise descriptor encoding without allocating a multi-gigabyte array.
    info = tm.make_tiled(address=16, shape=(2**31,), dtype=np.dtype("u1"), box=(16,))
    assert tm.decode(tm.encode(info)) == info
    for dimension in (2**31 + 1, 2**32):
        with pytest.raises(warplens.LaunchError, match=r"between 1 and 2\^31"):
            tm.make_tiled(address=16, shape=(dimension,), dtype=np.dtype("u1"), box=(16,))


@pytest.mark.parametrize("changes", [
    {"elem_size": 3}, {"elem_size": 1, "is_float": True}, {"swizzle": 4},
    {"is_float": 2}, {"nan_fill": 2}, {"nan_fill": True}, {"address": 17},
    {"dims": (0, 8)}, {"box": (0, 4)}, {"box": (8, 257)},
    {"steps": (1, 0)}, {"steps": (2, 1)}, {"steps": (1, 9)}, {"strides": (48,)},
])
def test_corrupt_descriptor_fields_raise_execution_error(changes):
    info = tm.make_tiled(address=16, shape=(8, 16), dtype=np.dtype("u4"), box=(4, 8))
    with pytest.raises(warplens.ExecutionError, match="invalid tensor"):
        tm.decode(tm.encode(replace(info, **changes)))


def test_corrupt_descriptor_error_has_kernel_source_location(sim):
    descriptor = sim.tensor_map_tiled(sim.empty((8, 16), "u32"), (4, 8))
    # A kernel/user can overwrite the device allocation backing a TensorMap.
    sim.memory.buffer(descriptor.addr)[:] = np.frombuffer(
        tm.encode(replace(descriptor.info, steps=(1, 0))), dtype=np.uint8)
    mod = kernel("""
        .reg .b64 %map;
        .shared .align 16 .b8 buf[128];
        .shared .align 8 .b64 bar;
        ld.param.u64 %map, [map];
        mbarrier.init.shared::cta.b64 [bar], 1;
        cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [buf], [%map, {0, 0}], [bar];
        ret;
    """, ".param .u64 map")
    with pytest.raises(warplens.ExecutionError) as exc:
        sim.launch(mod, "k", grid=1, block=1, args=[descriptor])
    assert "invalid tensor map" in str(exc.value)
    assert "cp.async.bulk.tensor.2d" in str(exc.value)


@pytest.mark.parametrize("length", [0, 92, 127, 129])
def test_truncated_or_oversized_descriptor_raises_execution_error(length):
    with pytest.raises(warplens.ExecutionError, match="must contain 128 bytes"):
        tm.decode(bytes(length))


@pytest.mark.parametrize("delay", [-1, 0.5, float("nan"), float("inf"), "4", None, True])
def test_invalid_async_delay_fails_before_kernel_execution(sim, delay):
    output = sim.empty(1, "u32")
    mod = kernel("""
        .reg .b64 %out;
        ld.param.u64 %out, [out];
        st.global.u32 [%out], 123;
        ret;
    """, ".param .u64 out")
    with pytest.raises(warplens.LaunchError, match="async_delay must be a non-negative integer"):
        sim.launch(mod, "k", grid=1, block=1, args=[output], async_delay=delay)
    assert output.numpy()[0] == 0


@pytest.mark.parametrize("delay", [0, np.int64(4)])
def test_async_delay_accepts_nonnegative_integer(sim, delay):
    sim.launch(kernel("ret;"), "k", grid=1, block=1, async_delay=delay)
