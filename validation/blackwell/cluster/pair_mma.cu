// The two-CTA MMA (tcgen05.mma.cta_group::2): where operands are read and where results land.
//
// A cluster of two CTAs. A is M x 16 with A[m] = (m, 1, 0, ...) and B is 16 x 16 with
// B[n] = (256, n, 0, ...), half precision, so D[m][n] = 256 m + n exactly and every
// accumulator names its row and column. The even-ranked CTA holds the first half of A's rows
// and of B's rows in its shared memory, the odd one the second half, at the same offsets.
// Both fill 16 columns of tensor memory with a sentinel. One CTA issues ONE MMA and a
// tcgen05.commit multicast to both; every thread of both CTAs waits on it, and each CTA
// dumps its 128 lanes x 16 columns.
//
// Claims tested (docs/m5.md, "Two-CTA MMA"):
//   * M = 256: the even CTA gets rows 0-127 on lanes 0-127, the odd CTA rows 128-255;
//     all N = 16 columns in TMEM columns 0-15;
//   * M = 128 (Layout B): each CTA gets 64 rows; columns 0-7 on lanes 0-63 and columns 8-15
//     on lanes 64-127, in TMEM columns 0-7 (the course notes' example: C[10][11] is on the
//     even CTA's lane 74);
//   * B is the concatenation of the two CTAs' halves: every row sees all 16 columns;
//   * either CTA may issue; placement follows rank parity, not the issuer;
//   * tcgen05.alloc.cta_group::2 in each CTA returns the same columns, and
//     tcgen05.commit ... .multicast::cluster arrives on the mbarrier of every CTA in the mask.
// MAY HANG: if the multicast commit does not reach a CTA, that CTA waits forever.
//
//   pair_mma <M: 256|128> <issuer rank: 0|1>
#include "../tcgen05_common.h"

#define NCOLS 16

extern "C" __global__ void pair_mma(const uint32_t* image, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                    uint32_t issuer, uint32_t* out, uint32_t* bases) {
    __shared__ alignas(1024) uint32_t operands[OPERAND_BYTES / 4];
    __shared__ uint64_t fin;
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32, rank = cluster_ctarank();

    const uint32_t* mine = image + rank * (OPERAND_BYTES / 4);
    for (uint32_t i = tid; i < OPERAND_BYTES / 4; i += blockDim.x) operands[i] = mine[i];
    fence_proxy_async();
    if (tid == 0) {
        mbar_init(&fin, 1);
        mbar_init_fence();
    }
    __syncthreads();  // see mma_dump in ../tcgen05_common.h
    if (warp == 0)
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(smem_addr(&tmem_base)), "r"(32) : "memory");
    __syncthreads();
    uint32_t base = tmem_base;
    uint32_t own = base + ((warp * 32) << 16);
    if (tid == 0) bases[rank] = base;
    for (uint32_t c = 0; c < NCOLS; ++c) tmem_st_32x32b(own + c, MMA_TMEM_SENTINEL);
    tmem_wait_st();
    cluster_sync();  // both CTAs have their operands, their allocation and their mbarrier

    if (tid == 0 && rank == issuer) {
        uint64_t where = (uint64_t)((smem_addr(operands) >> 4) & 0x3fff);
        uint16_t both = 3;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, %4;\n"  // false: overwrite
                     "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n"
                     "}" :: "r"(base), "l"(desc_a + where), "l"(desc_b + where), "r"(idesc), "r"(0) : "memory");
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                     " [%0], %1;" :: "r"(smem_addr(&fin)), "h"(both) : "memory");
    }
    mbar_wait(&fin, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    for (uint32_t c = 0; c < NCOLS; ++c) {
        uint32_t v = tmem_ld_32x32b(own + c);
        tmem_wait_ld();
        out[(rank * 128 + tid) * NCOLS + c] = v;
    }
    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(base), "r"(32) : "memory");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: pair_mma <M: 256|128> <issuer rank: 0|1>\n");
        return 1;
    }
    int m_dim = atoi(argv[1]), issuer = atoi(argv[2]), n_dim = 16;
    static uint8_t image[2][OPERAND_BYTES];
    static uint16_t a[256 * 16], b[16 * 16];
    for (int m = 0; m < m_dim; ++m) {
        a[m * 16] = half_bits((float)m);
        a[m * 16 + 1] = half_bits(1.0f);
    }
    for (int n = 0; n < n_dim; ++n) {
        b[n * 16] = half_bits(256.0f);
        b[n * 16 + 1] = half_bits((float)n);
    }
    uint64_t desc_a = 0, desc_b = 0;
    for (int cta = 0; cta < 2; ++cta) {
        desc_a = place_k_major(image[cta], 0, a + cta * (m_dim / 2) * 16, m_dim / 2, 2, LAYOUTS[3]);
        desc_b = place_k_major(image[cta], B_OFFSET, b + cta * (n_dim / 2) * 16, n_dim / 2, 2, LAYOUTS[3]);
    }
    uint32_t* d_image = dev_alloc<uint32_t>(2 * OPERAND_BYTES / 4);
    HOST_CHECK(cudaMemcpy(d_image, image, 2 * OPERAND_BYTES, cudaMemcpyHostToDevice));
    uint32_t* d_out = dev_alloc<uint32_t>(2 * 128 * NCOLS);
    uint32_t* d_bases = dev_alloc<uint32_t>(2);
    dev_fill(d_out, 2 * 128 * NCOLS, 0);
    dev_fill(d_bases, 2, 0xff);
    printf("cta_group::2 MMA, M = %d, N = %d, issued by the CTA of rank %d; D[m][n] = 256 m + n\n", m_dim, n_dim,
           issuer);
    fflush(stdout);
    launch_cluster(pair_mma, 2, 2, 128, (const uint32_t*)d_image, desc_a, desc_b,
                   mma_idesc(m_dim, n_dim, 0, 0, 0), (uint32_t)issuer, d_out, d_bases);
    probe_sync("pair_mma");
    static uint32_t out[2 * 128 * NCOLS];
    uint32_t bases[2];
    dev_read(out, d_out, 2 * 128 * NCOLS);
    dev_read(bases, d_bases, 2);
    printf("tcgen05.alloc.cta_group::2 returned %08x in rank 0 and %08x in rank 1\n", bases[0], bases[1]);
    for (int cta = 0; cta < 2; ++cta) {
        printf("CTA rank %d: for each lane, the row and the result columns found in TMEM columns 0-15\n", cta);
        for (int lane = 0; lane < 128; ++lane) {
            const uint32_t* cells = out + (cta * 128 + lane) * NCOLS;
            int written = 0;
            while (written < NCOLS && cells[written] != MMA_TMEM_SENTINEL) ++written;
            bool tail_clean = true, regular = written > 0;
            for (int c = written; c < NCOLS; ++c) tail_clean = tail_clean && cells[c] == MMA_TMEM_SENTINEL;
            float first = bits_float(cells[0]);
            regular = regular && first >= 0 && first < 65536 && first == floorf(first);
            for (int c = 0; c < written; ++c) regular = regular && bits_float(cells[c]) == first + c;
            if (written == 0 && tail_clean) {
                printf("  lane %3d: not written\n", lane);
            } else if (regular && tail_clean && ((int)first % 256) + written <= 256) {
                printf("  lane %3d: row %3d, columns %2d-%2d in TMEM columns 0-%d\n", lane, (int)first / 256,
                       (int)first % 256, (int)first % 256 + written - 1, written - 1);
            } else {
                printf("  lane %3d: raw", lane);
                for (int c = 0; c < NCOLS; ++c) printf(" %08x", cells[c]);
                printf("\n");
            }
        }
    }
    return 0;
}
