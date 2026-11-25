"""The driver's own 128-byte tensor map, read back.

validation/blackwell/measured/b200/tma/tensormap_encode_sweep.txt is what a B200's driver
(610.57.04) wrote for 1380 parameter sets of cuTensorMapEncodeTiled, 1087 of which it
accepted. The layout is not documented; warplens/host/tensor_map.py reads it from this
file, and every accepted line must come back as the parameters that made it."""

import re
import struct
from pathlib import Path

import numpy as np
import pytest

import warplens
from warplens.host import tensor_map as tm

SWEEP = Path(__file__).parents[1] / "validation/blackwell/measured/b200/tma/tensormap_encode_sweep.txt"
LINE = re.compile(r"(\S+) type=(\d+) rank=(\d+) address=(\S+) base\+\S+ dims=(\S+) strides=(\S*) box=(\S+) "
                  r"estrides=(\S+) interleave=(\d+) swizzle=(\d+) l2=(\d+) fill=(\d+) -> (\d+)(.*)")


def sweep():
    rows = []
    for line in SWEEP.read_text().splitlines():
        if line.startswith("#"):
            continue
        g = LINE.match(line).groups()
        ints = lambda text: tuple(int(v) for v in text.split(",")) if text else ()
        rank = int(g[2])
        rows.append({"group": g[0], "data_type": int(g[1]), "rank": rank, "address": int(g[3], 16),
                     "dims": ints(g[4])[:rank], "strides": ints(g[5]), "box": ints(g[6])[:rank],
                     "steps": ints(g[7])[:rank], "interleave": int(g[8]), "swizzle": int(g[9]),
                     "l2_promotion": int(g[10]), "nan_fill": bool(int(g[11])), "status": int(g[12]),
                     "raw": struct.pack("<16Q", *(int(v, 16) for v in g[13].split())) if g[13].split() else None})
    return rows


ROWS = sweep()
ACCEPTED = [row for row in ROWS if row["status"] == 0]


def test_the_sweep_is_the_one_that_was_recorded():
    assert len(ROWS) == 1380 and len(ACCEPTED) == 1087
    assert {row["rank"] for row in ACCEPTED} == {1, 2, 3, 4, 5}
    assert {row["data_type"] for row in ACCEPTED} == set(range(13))  # every type the driver takes
    assert {row["swizzle"] for row in ACCEPTED} == {0, 1, 2, 3, 4, 5, 6}


def test_every_map_the_driver_encoded_reads_back_as_its_parameters():
    for row in ACCEPTED:
        got = tm.driver_fields(row["raw"])
        # Swizzle modes 4 to 6 are the 128-byte swizzle (3) with an atom of 32, 16 or 8 bytes.
        swizzle, atom = (row["swizzle"], 0) if row["swizzle"] <= 3 else (3, row["swizzle"] - 3)
        want = {key: row[key] for key in ("data_type", "rank", "address", "dims", "strides", "box", "steps",
                                          "interleave", "l2_promotion", "nan_fill")}
        want.update(swizzle=swizzle, swizzle_atom=atom)
        assert got == want, row["group"]


def test_the_two_words_that_are_not_read_follow_from_the_rest():
    """Word 8 is the bytes of one box in shared memory and word 9 the swizzle's period, so
    nothing a kernel needs is lost by not reading them.

    Word 8 counts box // element stride elements per dimension, rounded DOWN, dimension 0
    included. The simulator's TensorMapInfo.counts rounds up and takes dimension 0's stride
    as 1; the two agree whenever the strides divide the box, which is all that was run on a
    GPU (tma/element_strides). Which is right for the other cases is not measured."""
    elem = {key: size for key, (size, _) in tm.DRIVER_TYPES.items()}
    for row in ACCEPTED:
        words = struct.unpack("<16Q", row["raw"])
        if row["interleave"]:
            continue  # interleaved layouts are not modelled, and their box bytes follow another rule
        counts = [b // s for b, s in zip(row["box"], row["steps"])]
        assert words[8] == int(np.prod(counts)) * elem[row["data_type"]], row
        assert words[9] == (16 if row["swizzle"] == 0 else 128 << min(row["swizzle"], 3)), row


def test_a_driver_map_runs_like_the_map_the_python_api_makes(sim):
    """One of the recorded maps, pointed at a simulated tensor, loads the same box as
    Device.tensor_map_tiled's map of that tensor."""
    row = next(r for r in ACCEPTED if r["group"] == "gemm" and r["dims"] == (128, 128) and r["box"] == (64, 32))
    assert (row["data_type"], row["swizzle"], row["strides"]) == (6, 3, (256,))
    data = np.arange(128 * 128, dtype=np.float16).reshape(128, 128)
    tensor = sim.from_numpy(data)
    ours = sim.tensor_map_tiled(tensor, (32, 64), swizzle="128B")
    theirs = bytearray(row["raw"])
    theirs[:8] = struct.pack("<Q", tensor.addr)
    assert tm.decode(bytes(theirs)) == ours.info
    assert tm.decode(tm.encode(ours.info)) == ours.info  # and our own encoding still reads as before


def test_what_is_not_modelled_is_refused_by_name_and_noise_is_not_a_map():
    atom = next(r for r in ACCEPTED if r["swizzle"] == 4)
    with pytest.raises(warplens.ExecutionError, match="swizzle of a smaller atom"):
        tm.decode(atom["raw"])
    interleaved = next(r for r in ACCEPTED if r["interleave"])
    with pytest.raises(warplens.ExecutionError, match="interleaved tensor maps are not supported"):
        tm.decode(interleaved["raw"])
    for raw in (bytes(128), b"\xa5" * 128, bytes(range(128))):
        with pytest.raises(warplens.ExecutionError, match="neither a tensor map created by|invalid tensor map"):
            tm.decode(raw)
    with pytest.raises(warplens.ExecutionError, match="must contain 128 bytes"):
        tm.driver_fields(bytes(64))


# The first 64 bytes of the four tensor maps CUTLASS's fp16_gemm_0.py handed its kernel on a
# B200 (matrix row RT-11, 2026-11-25): operands A and B at --mnk 256,512,512 and at the
# tutorial's own 8192 x 8192 x 8192. The CuTe DSL writes these bytes itself and does not call
# cuTensorMapEncodeTiled; what followed them in the captured 128 bytes was whatever the host's
# stack held (pointers), which is why only the first half is kept here.
DSL_MAPS = {
    "A, 256 x 512": ("0000005dfc7f0000126324004000000000000000000000000000000000000000"
                     "ff010000ff0000000000000000000000000000000000003f7f00000000000000"),
    "B, 512 x 512": ("0000045dfc7f0000126324004000000000000000000000000000000000000000"
                     "ff010000ff0100000000000000000000000000000000003fff00000000000000"),
    "A, 8192 x 8192": ("00000020fc7f0000126324000004000000000000000000000000000000000000"
                       "ff1f0000ff1f00000000000000000000000000000000003f7f00000000000000"),
    "B, 8192 x 8192": ("000000d0fb7f0000126324000004000000000000000000000000000000000000"
                       "ff1f0000ff1f00000000000000000000000000000000003fff00000000000000"),
}


@pytest.mark.parametrize("name, dims, box, address", [
    ("A, 256 x 512", (512, 256), (64, 128), 0x7FFC5D000000),
    ("B, 512 x 512", (512, 512), (64, 256), 0x7FFC5D040000),
    ("A, 8192 x 8192", (8192, 8192), (64, 128), 0x7FFC20000000),
    ("B, 8192 x 8192", (8192, 8192), (64, 256), 0x7FFBD0000000),
])
def test_the_maps_the_cute_dsl_writes_itself_are_read(name, dims, box, address):
    """K is dimension 0: 512 (or 8192) half-precision elements a row, so 1024 (16384) bytes
    between rows; boxes of 64 x 128 for A and 64 x 256 for B; the 128-byte swizzle."""
    head = bytes.fromhex(DSL_MAPS[name])
    for tail in (bytes(64), b"\x97\xd0\x68\x5f\x55\x55\x00\x00" * 8):  # zeros, or a host's leftovers
        info = tm.decode(head + tail)
        assert info == tm.TensorMapInfo(rank=2, elem_size=2, is_float=True, swizzle=3, nan_fill=False,
                                        address=address, dims=dims, strides=(dims[0] * 2,), box=box, steps=(1, 1))
    fields = tm.driver_fields(head + bytes(64))
    assert fields["data_type"] == 6 and fields["l2_promotion"] == 2
    # The driver's own map of the same tensor differs from the DSL's in bit 1 of the flag
    # word only (and in the two derived words the DSL leaves alone).
    flags = struct.unpack_from("<I", head, 8)[0]
    assert flags & 0xF == 2 and all(struct.unpack_from("<I", r["raw"], 8)[0] & 0xF == 0 for r in ACCEPTED)
