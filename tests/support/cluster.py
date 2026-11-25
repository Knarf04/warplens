"""Clusters: the kernel of the two-CTA MMA."""

from tests.support.ptx import kernel

PAIR_PARAMS = (".param .u64 img, .param .u64 out, .param .u64 da, .param .u64 db, .param .u32 idesc, "
               ".param .u32 n")


def pair_kernel():
    """Two CTAs. Each copies its own operand image into shared memory. The even one issues
    one cta_group::2 MMA and commits to both; each then dumps its 128 lanes of TMEM."""
    return kernel("""
        .reg .b64 %img, %out, %da, %db, %off;
        .reg .b32 %idesc, %n, %t, %w, %rank, %buf, %bar, %fin, %tp, %base, %taddr, %i, %v, %x;
        .reg .b16 %both; .reg .pred %p, %one, %lead, %q, %no;
        .shared .align 1024 .b8 buf[32768];
        .shared .align 8 .u64 bar;
        .shared .align 8 .u64 fin;
        .shared .align 4 .u32 tptr;
        ld.param.u64 %img, [img]; ld.param.u64 %out, [out]; ld.param.u64 %da, [da];
        ld.param.u64 %db, [db]; ld.param.u32 %idesc, [idesc]; ld.param.u32 %n, [n];
        mov.u32 %t, %tid.x; mov.u32 %w, %warpid; mov.u32 %rank, %cluster_ctarank;
        setp.eq.u32 %one, %t, 0; setp.eq.u32 %q, %w, 0; setp.ne.u32 %no, %t, %t;
        setp.eq.u32 %lead, %rank, 0; and.pred %lead, %lead, %one;
        mov.b32 %buf, buf; mov.b32 %bar, bar; mov.b32 %fin, fin; mov.b32 %tp, tptr;
        @%one mbarrier.init.shared::cta.b64 [%bar], 1;
        @%one mbarrier.init.shared::cta.b64 [%fin], 1;
        @%q tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%tp], 32;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        mul.wide.u32 %off, %rank, 32768;
        add.u64 %img, %img, %off;
        @%one mbarrier.arrive.expect_tx.shared::cta.b64 _, [%bar], 32768;
        @%one cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%buf], [%img], 32768, [%bar];
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%bar], 0;
        barrier.cluster.arrive.aligned;
        barrier.cluster.wait.aligned;
        ld.shared.u32 %base, [%tp];
        @%lead tcgen05.mma.cta_group::2.kind::f16 [%base], %da, %db, %idesc, {0, 0, 0, 0, 0, 0, 0, 0}, %no;
        mov.b16 %both, 3;
        @%lead tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%fin], %both;
        mbarrier.try_wait.parity.shared::cta.b64 %p, [%fin], 0;
        shl.b32 %x, %w, 21;
        add.u32 %taddr, %base, %x;
        mov.u32 %i, 0;
      LOOP:
        add.u32 %x, %taddr, %i;
        tcgen05.ld.sync.aligned.32x32b.x1.b32 {%v}, [%x];
        mad.lo.u32 %x, %rank, 128, %t;
        mad.lo.u32 %x, %x, %n, %i;
        mul.wide.u32 %off, %x, 4;
        add.u64 %off, %out, %off;
        st.global.u32 [%off], %v;
        add.u32 %i, %i, 1;
        setp.lt.u32 %p, %i, %n;
        @%p bra LOOP;
        bar.sync 0;
        @%q tcgen05.dealloc.cta_group::2.sync.aligned.b32 %base, 32;
        ret;
    """, PAIR_PARAMS)
