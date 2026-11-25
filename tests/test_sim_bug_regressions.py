"""Regressions for three simulator bugs found while building the Blackwell probe suite
(`validation/blackwell/README.md`, "Found in the simulator"):

* arrivals beyond an mbarrier's count left its pending count negative for good (they end
  the kernel since a B200 was seen to);
* an M = 64 MMA accepted any destination lane offset, where Layout F has two positions;
* a launch accepted 228 KB of shared memory per CTA while the shim reported 227 KB.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import warplens
from warplens.frontend.ptx_parser import parse_ptx
from warplens.units.smem import DEFAULT_SMEM_BYTES, MAX_SMEM_BYTES

from conftest import HEADER
from test_tcgen05 import run as run_mma
from test_tma import mb_kernel, record

SHIM = Path(warplens.__file__).resolve().parent / "shim"


# -- mbarrier over-arrival ---------------------------------------------------------------

def phases_after(sim, expected, counts):
    """Run `init expected` and then one `arrive` per entry of `counts`. Returns the phase
    the barrier is in after each arrival, and the findings."""
    body = f"mbarrier.init.shared::cta.b64 [%bar], {expected};\n"
    for k, count in enumerate(counts):
        body += (f"mbarrier.arrive.shared::cta.b64 _, [%bar], {count};\n"
                 "mbarrier.arrive.expect_tx.shared::cta.b64 %tok, [%bar], 8;\n"  # low half: phase now
                 "mbarrier.complete_tx.shared::cta.b64 [%bar], 8;\n"
                 f"st.global.u64 [%o+{8 * k}], %tok;\n")
    out = sim.empty(len(counts), "u64")
    stats = sim.launch(mb_kernel(body), "k", grid=1, block=1, args=[out])
    tokens = out.numpy().tolist()
    assert all(t >> 63 == (t & 1) for t in tokens)  # bit 63 is the phase parity, as on a GPU
    return [t & 0xFFFFFFFF for t in tokens], stats.findings


def test_the_phase_token_helper_counts_phases(sim):
    # Each step is `arrive n` plus one more arrival that reads the phase, so with 3 expected
    # and n = 2 every step completes exactly one phase and nothing is in excess.
    phases, findings = phases_after(sim, 3, [2, 2, 2])
    assert phases == [0, 1, 2] and not findings


def test_an_over_arrival_ends_the_kernel(sim):
    """Two expected, five arrive. The first model consumed the surplus over the following
    phases (and before that left the pending count negative for good); a B200 ends the
    kernel (validation/blackwell/measured/b200/mbarrier/over_arrival.txt), and so does the
    simulator now, at the arrival."""
    with pytest.raises(warplens.ExecutionError, match=r"5 arrival\(s\).*needed only 2 more"):
        phases_after(sim, 2, [5, 1, 1])
    # An arrival while the phase still waits for its bytes is one too many as well.
    from warplens.units.mbarrier import Mbarrier

    class Cta:
        completed = 0

        def phase_completed(self):
            self.completed += 1

    class Ins:
        mods = ("arrive", "relaxed")

    cta = Cta()
    bar = Mbarrier(cta, 0, 3, warp_id=0)
    bar.expect_tx(64)
    bar.arrive(3, None, Ins, [0])  # held back by the bytes: nothing completes yet
    assert (bar.phase, bar.pending, cta.completed) == (0, 0, 0)
    with pytest.raises(warplens.ExecutionError, match="needed only 0 more"):
        bar.arrive(1, None, Ins, [0])
    bar.complete_tx(64, None, Ins, [0])
    assert (bar.phase, bar.pending, cta.completed) == (1, 3, 1)


# -- M = 64 destination lane offset ---------------------------------------------------------

@pytest.mark.parametrize("lane", [1, 8, 15])
def test_m64_mma_at_a_lane_offset_between_the_two_positions_is_refused(sim, lane):
    with pytest.raises(warplens.ExecutionError, match=rf"lane {lane}.*must be 0 or 16"):
        run_mma(sim, m=64, n=8, dlane=lane)


@pytest.mark.parametrize("lane", [32, 48, 64])
def test_m64_mma_naming_another_window_is_refused_for_its_lane(sim, lane):
    with pytest.raises(warplens.ExecutionError, match="must be 0 or 16"):
        run_mma(sim, m=64, n=8, dlane=lane)


def test_m128_mma_at_a_nonzero_lane_is_refused_for_its_lane(sim):
    with pytest.raises(warplens.ExecutionError, match=r"lane 16.*must be 0"):
        run_mma(sim, m=128, n=8, dlane=16)


def test_m64_mma_at_both_positions_still_runs(sim):
    for lane in (0, 16):
        got, expect, _ = run_mma(sim, m=64, n=8, dlane=lane)
        rows = 32 * (np.arange(64) // 16) + np.arange(64) % 16 + lane
        np.testing.assert_allclose(got[rows], expect, rtol=1e-5, atol=1e-5)


# -- shared memory per CTA ----------------------------------------------------------------

SMEM_KERNEL = HEADER + """
    .extern .shared .align 16 .b8 buf[];
    .visible .entry k() {
        .reg .b32 %a;
        mov.u32 %a, buf;
        st.shared.u8 [%a], 1;
        ret;
    }
"""


def test_the_launch_limit_is_the_opt_in_maximum_per_block(sim):
    """227 KB: CUDA C++ Programming Guide, "Technical Specifications per Compute Capability"
    (maximum shared memory per thread block; 228 KB is per SM), and CUTLASS
    `cutlass/arch/arch.h`: `sm100_smem_capacity_bytes = 232448`."""
    assert MAX_SMEM_BYTES == 232448 == 227 * 1024
    mod = warplens.parse_ptx(SMEM_KERNEL)
    stats = sim.launch(mod, "k", grid=1, block=1, smem=227 * 1024, args=[])
    assert stats.smem_bytes == 227 * 1024
    for too_much in (227 * 1024 + 16, 228 * 1024):
        with pytest.raises(warplens.LaunchError, match="the limit is 232448"):
            sim.launch(mod, "k", grid=1, block=1, smem=too_much, args=[])


def test_the_shim_reports_the_limit_the_launch_enforces():
    header = (SHIM / "protocol.h").read_text()
    (expr,) = re.findall(r"^#define WL_MAX_SMEM_BYTES \((\d+ \* \d+)\)", header, re.M)
    a, b = map(int, expr.split(" * "))
    assert a * b == MAX_SMEM_BYTES
    (expr,) = re.findall(r"^#define WL_DEFAULT_SMEM_BYTES \((\d+ \* \d+)\)", header, re.M)
    a, b = map(int, expr.split(" * "))
    assert a * b == DEFAULT_SMEM_BYTES == 48 * 1024
    # One place for each number: the libraries name the limits and do not repeat them.
    for name in ("cudart_shim.c", "cuda_shim.c"):
        source = (SHIM / name).read_text()
        assert not re.search(r"(22[78]|48) \* 1024", source.replace("return 228 * 1024;", "")), name
    runtime = (SHIM / "cudart_shim.c").read_text()
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin: return WL_MAX_SMEM_BYTES;" in runtime
    assert "cudaDevAttrMaxSharedMemoryPerBlock: return WL_DEFAULT_SMEM_BYTES;" in runtime
    # Per SM, not per block: stays at 228 KB.
    assert "cudaDevAttrMaxSharedMemoryPerMultiprocessor: return 228 * 1024;" in (SHIM / "cudart_shim.c").read_text()


def test_the_triton_driver_reports_the_same_limit():
    # Read, not imported: importing the driver needs the optional `triton` package, and what
    # is checked here is a line of its source.
    source = (SHIM.parent / "triton_driver.py").read_text()
    assert '"max_shared_mem": MAX_SMEM_BYTES' in source
    assert "from .units.smem import MAX_SMEM_BYTES" in source


# -- [reg+offset] with a 32-bit register ---------------------------------------------------

def reversed_shared_read(offset):
    """What nvcc 13.5 makes of `fixed[tid] = 100 + tid; __syncthreads(); out[tid] = fixed[31 - tid];`
    for a shared `int fixed[32]`: the address register is `fixed - 4 * tid`, below zero in 32
    bits for every lane but lane 0, and the constant 124 is the instruction's offset."""
    return parse_ptx(HEADER + f"""
.visible .entry k(.param .u64 out)
{{
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<5>;
	.shared .align 4 .b8 fixed[128];
	ld.param.u64 	%rd1, [out];
	mov.u32 	%r1, %tid.x;
	add.s32 	%r2, %r1, 100;
	shl.b32 	%r3, %r1, 2;
	mov.b32 	%r4, fixed;
	add.s32 	%r5, %r4, %r3;
	st.shared.b32 	[%r5], %r2;
	bar.sync 	0;
	sub.s32 	%r6, %r4, %r3;
	ld.shared.b32 	%r7, [%r6+{offset}];
	mul.wide.u32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd1, %rd3;
	st.global.b32 	[%rd4], %r7;
	ret;
}}
""")


def test_an_offset_is_added_to_a_32_bit_address_register_in_32_bits(sim):
    """Measured on an RTX 3080 Ti (driver 580.173.02, 2026-11-24) with this kernel under
    `.version 7.8`, `.target sm_86`: out = 131, 130, ..., 100. The simulator added the offset
    in 64 bits and reported a read at offset 2^32 + 120."""
    out = sim.empty(32, np.int32)
    stats = sim.launch(reversed_shared_read(124), "k", grid=1, block=32, args=[out])
    assert out.numpy().tolist() == list(range(131, 99, -1))
    assert not stats.findings


def test_a_wrapped_address_outside_shared_memory_is_still_a_fault(sim):
    out = sim.empty(32, np.int32)
    with pytest.raises(warplens.MemoryFault, match="outside the CTA's"):
        sim.launch(reversed_shared_read(120), "k", grid=1, block=32, args=[out])
