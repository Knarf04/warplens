"""TMA and mbarrier kernels: the copy of examples/02_tma_copy with one of its seeded bugs, a box
loaded and stored at any rank, a kernel around an mbarrier, and one buffer that a store reads."""

import numpy as np

import warplens

from tests.support.paths import EXAMPLES
from tests.support.ptx import kernel

TMA_COPY = EXAMPLES / "02_tma_copy" / "tma_copy.ptx"
NONE, EXPECT_TOO_FEW, EXPECT_TOO_MANY, NO_WAIT, NO_FENCE, EARLY_REUSE = range(6)


def tma_copy(swizzle, box, bug=NONE, shape=(48, 64), **launch):
    sim = warplens.Device()
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    src, dst = sim.from_numpy(data), sim.from_numpy(np.full(shape, np.nan, dtype=np.float32))
    maps = [sim.tensor_map_tiled(a, box, swizzle=swizzle) for a in (src, dst)]
    grid = (-(-shape[1] // box[1]), -(-shape[0] // box[0]))
    stats = sim.launch(TMA_COPY, "tma_copy", grid=grid, block=128,
                       args=[*maps, box[1], box[0], bug], **launch)
    return np.array_equal(dst.numpy(), 2 * data), stats


def box_kernel(rank: int, raw_wait: str = "cp.async.bulk.wait_group 0;"):
    """One thread: TMA-load the box at (c0..), then TMA-store it at the origin of `dst`."""
    coords = ", ".join(f"%c{d}" for d in range(rank))
    zeros = ", ".join("0" for _ in range(rank))
    loads = "\n".join(f"ld.param.s32 %c{d}, [c{d}];" for d in range(rank))
    params = ".param .u64 src, .param .u64 dst, .param .u32 bytes, " + \
        ", ".join(f".param .s32 c{d}" for d in range(rank))
    return kernel(f"""
        .reg .b64 %s, %d; .reg .b32 %n, %c<5>, %buf, %bar; .reg .pred %p;
        .shared .align 1024 .b8 buf[16384];
        .shared .align 8 .u64 bar;
        ld.param.u64 %s, [src]; ld.param.u64 %d, [dst]; ld.param.u32 %n, [bytes];
        {loads}
        mov.b32 %buf, buf; mov.b32 %bar, bar;
        mbarrier.init.shared::cta.b64 [%bar], 1;
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], %n;
        cp.async.bulk.tensor.{rank}d.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%s, {{{coords}}}], [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        cp.async.bulk.tensor.{rank}d.global.shared::cta.bulk_group [%d, {{{zeros}}}], [%buf];
        cp.async.bulk.commit_group;
        {raw_wait}
        ret;
    """, params)


MB = """
    .reg .b32 %bar, %v; .reg .b64 %o, %tok; .reg .pred %p;
    .shared .align 8 .u64 bar;
    mov.b32 %bar, bar;
    ld.param.u64 %o, [out];
"""


def mb_kernel(body):
    return kernel(MB + body + "\nret;", ".param .u64 out")


def record(k):
    return f"selp.u32 %v, 1, 0, %p;\nst.global.u32 [%o+{4 * k}], %v;"


# -- one 16-byte buffer, written by thread 0 and fenced, for kernels that copy from or into it ----

ONE_BUFFER = """
    .reg .b64 %s, %d;
    .reg .b32 %t;
    .reg .pred %p, %one;
    .shared .align 16 .b8 buf[16];
    .shared .align 8 .u64 bar;
    ld.param.u64 %s, [src]; ld.param.u64 %d, [dst];
    mov.u32 %t, %tid.x;
    setp.eq.u32 %one, %t, 0;
    @%one st.shared.u32 [buf], 7;
    @%one st.shared.u32 [buf+4], 7;
    @%one st.shared.u32 [buf+8], 7;
    @%one st.shared.u32 [buf+12], 7;
    fence.proxy.async.shared::cta;
"""


def launch_one_buffer(sim, body, *, block=1, delay=1000):
    src = sim.from_numpy(np.arange(8, dtype=np.uint32))
    dst = sim.empty(4, "u32")
    stats = sim.launch(kernel(ONE_BUFFER + body + "\nret;",
                              ".param .u64 src, .param .u64 dst"),
                       "k", grid=1, block=block, args=[src, dst], async_delay=delay)
    return dst.numpy(), stats
