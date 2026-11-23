"""Dense tcgen05 MMA regression cases from the PTX instruction contract.

PTX ISA sections: tcgen05-mma-instructions-mma,
tcgen05-shared-memory-descriptor, tcgen05-instruction-descriptor,
and tcgen05-memory-consistency-model-shared-memory-accesses.
"""

import numpy as np
import pytest

import warplens
from warplens.frontend.ir import Imm, Vec
from test_tcgen05 import mma_kernel, place


def _run(sim, *, scale=None, kind="f16", first=True, mask=None, modifier=None,
         descriptor_bit=None, idesc_bit=None, source="tma", wait_source=True,
         async_delay=4):
    dtype = np.float16 if kind == "f16" else np.float32
    k = 32 // np.dtype(dtype).itemsize
    image = np.zeros(32768, dtype=np.uint8)
    da = place(np.ones((128, k), dtype=dtype), "128B", 0, image)
    db = place(np.ones((8, k), dtype=dtype), "128B", 16384, image)
    if descriptor_bit is not None:
        da ^= 1 << descriptor_bit
    fmt = 0 if kind == "f16" else 2
    idesc = (1 << 4) | (fmt << 7) | (fmt << 10) | (1 << 17) | (8 << 24)
    if idesc_bit is not None:
        idesc |= 1 << idesc_bit
    mod = mma_kernel(kind)
    fn = mod.functions["k"]
    mmas = [ins for ins in fn.instrs if ins.name.startswith("tcgen05.mma.")]
    if scale is not None:
        mmas[1].operands += (Imm("int", scale),)
    if mask is not None:
        for ins in mmas:
            ins.operands = (*ins.operands[:4], Vec(tuple(Imm("int", x) for x in mask)),
                            *ins.operands[5:])
    if modifier is not None:
        for ins in mmas:
            ins.mods += (modifier,)
    if not first:
        # Predicate the initialization MMA with the always-false %no register.
        mmas[0].pred = mmas[0].operands[-1]
    original_instrs = list(fn.instrs)
    if source == "uninitialized":
        fn.instrs = [ins for ins in fn.instrs
                     if not ins.name.startswith(("cp.async.bulk.", "mbarrier.arrive.expect_tx"))
                     and not (ins.name.startswith("mbarrier.try_wait")
                              and ins.operands[-1] == Imm("int", 0))]
        # The MMA commit then completes phase zero, rather than phase one.
        for ins in fn.instrs:
            if ins.name.startswith("mbarrier.try_wait"):
                ins.operands = (*ins.operands[:-1], Imm("int", 0))
    elif not wait_source:
        fn.instrs = [ins for ins in fn.instrs
                     if not (ins.name.startswith("mbarrier.try_wait")
                             and ins.operands[-1] == Imm("int", 0))]
        # The completion barrier still has a pending TMA transaction at high delay.
        # No consumer wait is needed to observe the issue-time diagnostic.
        fn.instrs = [ins for ins in fn.instrs if not ins.name.startswith("mbarrier.try_wait")]
    kept = {id(ins) for ins in fn.instrs}
    fn.labels = {label: sum(id(ins) in kept for ins in original_instrs[:pc])
                 for label, pc in fn.labels.items()}
    out = sim.empty((128, 8), "f32")
    stats = sim.launch(mod, "k", grid=1, block=128,
                       args=[sim.from_numpy(image), out, da, db, idesc, len(image),
                             32, 8, 1, 1, 0, 1, 0], async_delay=async_delay)
    return out.numpy(), stats, k


@pytest.mark.parametrize("kind", ["f16", "tf32"])
@pytest.mark.parametrize("scale", [0, 1, 4, 15])
def test_scale_input_d_scales_only_prior_accumulator(sim, kind, scale):
    got, stats, k = _run(sim, kind=kind, scale=scale)
    np.testing.assert_array_equal(got, np.float32(k * (1 + 2 ** -scale)))
    assert not stats.findings


@pytest.mark.parametrize("scale", [-1, 16])
def test_scale_input_d_must_be_within_isa_range(sim, scale):
    with pytest.raises(warplens.ExecutionError, match="scale-input-d"):
        _run(sim, scale=scale)


@pytest.mark.parametrize("modifier", ["ws", "collector::a::use", "ashift", "sp"])
def test_unimplemented_mma_modes_are_rejected(sim, modifier):
    with pytest.raises(warplens.UnsupportedError, match=modifier):
        _run(sim, modifier=modifier)


@pytest.mark.parametrize("bit", [46, 47, 48, 52, 54])
def test_unsupported_or_malformed_shared_descriptor_is_not_ignored(sim, bit):
    with pytest.raises((warplens.ExecutionError, warplens.UnsupportedError), match="descriptor"):
        _run(sim, descriptor_bit=bit)


@pytest.mark.parametrize("bit", [6, 23, 29, 30])
def test_malformed_instruction_descriptor_is_not_ignored(sim, bit):
    with pytest.raises(warplens.ExecutionError, match="descriptor"):
        _run(sim, idesc_bit=bit)


@pytest.mark.parametrize("words", [[], [0], [0, 0, 0, 0, 0]])
def test_disable_output_mask_has_exactly_four_words(sim, words):
    with pytest.raises(warplens.ExecutionError, match="four|4"):
        _run(sim, mask=words)


def test_accumulating_uninitialized_tmem_is_reported(sim):
    _, stats, _ = _run(sim, first=False)
    assert "tmem-uninitialized-read" in {f.kind for f in stats.findings}


def test_mma_reads_of_never_written_shared_operands_are_reported(sim):
    _, stats, _ = _run(sim, source="uninitialized")
    assert "smem-uninitialized-read" in {f.kind for f in stats.findings}


@pytest.mark.parametrize("delay", [0, 1000])
def test_mma_requires_synchronization_with_tma_operand_producer(sim, delay):
    _, stats, _ = _run(sim, wait_source=False, async_delay=delay)
    assert "tma-load-unsynchronized" in {f.kind for f in stats.findings}
