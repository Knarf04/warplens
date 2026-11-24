// Shared by the tensor-core probes (tmem/, mma/): placing MMA operands in a shared-memory
// image on the host, and one kernel that runs an MMA and dumps tensor memory.
#pragma once
#include "common.h"

#include <math.h>

// -------------------------------------------------------------------------------------
// Host: operand images
// -------------------------------------------------------------------------------------
// A probe prepares the bytes of the kernel's operand buffer on the host. The kernel copies
// them into shared memory with ordinary stores, so no TMA is involved and the MMA's reading
// of its descriptors is tested alone. The buffer is aligned to 1024 bytes, the period of the
// widest swizzle; the swizzle below is computed on offsets into it, which is the same as on
// addresses for such a buffer.

#define OPERAND_BYTES 32768  // A at offset 0, B at offset 16384
#define B_OFFSET 16384

struct Layout {
    const char* name;
    int code;       // layout type field of the descriptor
    int bits;       // swizzled bits: Swizzle<bits, 4, 3>
    int row_bytes;  // bytes of one operand row (K-major) or one swizzle atom (MN-major)
};
static const Layout LAYOUTS[4] = {{"none", 0, 0, 16}, {"32B", 6, 1, 32}, {"64B", 4, 2, 64}, {"128B", 2, 3, 128}};

static inline uint32_t swizzle_offset(uint32_t offset, int bits) {
    return offset ^ (((offset >> 7) & ((1u << bits) - 1)) << 4);
}

// Place a rows x k operand (elements of `size` bytes, row-major in `data`) in `image` at
// `start`, K-major: the elements of one row along K are contiguous. With T = 16 / size
// elements per 16-byte chunk and K = 2T, element (r, k) is at
//   start + (r % 8) * row_bytes + (r / 8) * SBO + (k % T) * size + (k / T) * step
// before swizzling, where SBO = 8 * row_bytes and step = 16 (swizzled) or LBO (not).
// Returns the descriptor, relative to the start of the buffer.
static uint64_t place_k_major(uint8_t* image, uint32_t start, const void* data, int rows, int size,
                              const Layout& layout) {
    int per_chunk = 16 / size, k = 2 * per_chunk;
    uint32_t sbo = 8 * layout.row_bytes;
    uint32_t lbo = layout.bits ? 16 : (uint32_t)((rows + 7) / 8) * sbo;
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) {
            uint32_t offset = start + (r % 8) * layout.row_bytes + (r / 8) * sbo + (j % per_chunk) * size
                              + (j / per_chunk) * (layout.bits ? 16 : lbo);
            memcpy(image + swizzle_offset(offset, layout.bits), (const uint8_t*)data + (r * k + j) * size, size);
        }
    return mma_smem_desc(start, lbo, sbo, layout.code);
}

// The same operand MN-major: the elements of one column (along M or N) are contiguous.
// This is the canonical layout of PTX 9.7.18.3 as warplens reads it: each group of eight K
// columns is stored column by column, in tiles of row_bytes / size rows. Element (r, k) is at
//   swizzled:    start + (r % T) * size + (r / T % A) * 16 + (r / (T * A)) * LBO + (k % 8) * row_bytes + (k / 8) * SBO
//   unswizzled:  start + (r % T) * size + (r / T) * SBO + (k % 8) * 16 + (k / 8) * LBO
// with A = row_bytes / 16, LBO = 8 * row_bytes and SBO = 16 * rows when swizzled,
// LBO = 16 * rows and SBO = 128 when not.
static uint64_t place_mn_major(uint8_t* image, uint32_t start, const void* data, int rows, int size,
                               const Layout& layout) {
    int per_chunk = 16 / size, k = 2 * per_chunk, atom = layout.row_bytes / 16;
    uint32_t lbo = layout.bits ? 8 * layout.row_bytes : 16 * rows;
    uint32_t sbo = layout.bits ? 16 * rows : 128;
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) {
            uint32_t offset = start + (r % per_chunk) * size;
            if (layout.bits)
                offset += (r / per_chunk % atom) * 16 + (r / (per_chunk * atom)) * lbo + (j % 8) * layout.row_bytes
                          + (j / 8) * sbo;
            else
                offset += (r / per_chunk) * sbo + (j % 8) * 16 + (j / 8) * lbo;
            memcpy(image + swizzle_offset(offset, layout.bits), (const uint8_t*)data + (r * k + j) * size, size);
        }
    return mma_smem_desc(start, lbo, sbo, layout.code);
}

// float -> IEEE half bits, round to nearest even. Exact for the values the probes use.
static uint16_t half_bits(float value) {
    uint32_t x;
    memcpy(&x, &value, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exponent = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffffu;
    if ((x & 0x7fffffffu) == 0) return (uint16_t)sign;
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        int shift = 14 - exponent;
        uint32_t half = mantissa >> shift, rest = mantissa & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rest > mid || (rest == mid && (half & 1))) ++half;
        return (uint16_t)(sign | half);
    }
    uint32_t half = (uint32_t)(exponent << 10) | (mantissa >> 13), rest = mantissa & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (half & 1))) ++half;
    return (uint16_t)(sign | half);
}

static inline uint32_t float_bits(float value) {
    uint32_t x;
    memcpy(&x, &value, 4);
    return x;
}
static inline float bits_float(uint32_t x) {
    float value;
    memcpy(&value, &x, 4);
    return value;
}

// -------------------------------------------------------------------------------------
// Device: one MMA (or several accumulating ones), then a dump of tensor memory
// -------------------------------------------------------------------------------------
#define MMA_TMEM_SENTINEL 0xEEEEEEEEu
#define READ_CANONICAL 0xffffffffu

// 128 threads = 4 warps. Steps:
//   1. every thread copies part of `image` (OPERAND_BYTES) into shared memory; fence; barrier
//   2. warp 0 allocates `alloc_cols` columns of tensor memory
//   3. if `prefill`: every warp stores MMA_TMEM_SENTINEL to its own 32 lanes of the first
//      `ncols` columns (32x32b at lane 32 * warp, the addressing CUTLASS uses)
//   4. thread 0 issues `repeat` MMAs to [base + (d_lane << 16)]: the first overwrites, the
//      rest accumulate. `kind` is 0 for kind::f16 and 1 for kind::tf32. Then tcgen05.commit
//      to an mbarrier, on which every thread waits.
//   5. every warp reads `ncols` columns with tcgen05.ld.32x32b at lane field 32 * warp
//      (READ_CANONICAL) or at the lane field `read_lane`, the same for all warps, and writes
//      them to out[thread][column].
//   6. warp 0 deallocates.
extern "C" __global__ void mma_dump(const uint32_t* image, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                    uint32_t kind, uint32_t d_lane, uint32_t repeat, uint32_t prefill,
                                    uint32_t read_lane, uint32_t ncols, uint32_t alloc_cols, uint32_t* out) {
    __shared__ alignas(1024) uint32_t operands[OPERAND_BYTES / 4];
    __shared__ uint64_t bar;
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;

    for (uint32_t i = tid; i < OPERAND_BYTES / 4; i += blockDim.x) operands[i] = image[i];
    fence_proxy_async();
    if (tid == 0) {
        mbar_init(&bar, 1);
        mbar_init_fence();
    }
    if (warp == 0) tmem_alloc(&tmem_base, alloc_cols);
    __syncthreads();
    uint32_t base = tmem_base;
    uint32_t own = base + ((warp * 32) << 16);

    if (prefill) {
        for (uint32_t c = 0; c < ncols; ++c) tmem_st_32x32b(own + c, MMA_TMEM_SENTINEL);
        tmem_wait_st();
        __syncthreads();
    }
    if (tid == 0) {
        uint64_t where = (uint64_t)(smem_addr(operands) >> 4);  // descriptors are buffer-relative
        uint32_t d = base + (d_lane << 16);
        for (uint32_t i = 0; i < repeat; ++i) {
            if (kind == 0) mma_f16(d, desc_a + where, desc_b + where, idesc, i);
            else mma_tf32(d, desc_a + where, desc_b + where, idesc, i);
        }
        mma_commit(&bar);
    }
    mbar_wait(&bar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    uint32_t from = read_lane == READ_CANONICAL ? own : base + (read_lane << 16);
    for (uint32_t c = 0; c < ncols; ++c) {
        uint32_t v = tmem_ld_32x32b(from + c);
        tmem_wait_ld();
        out[tid * ncols + c] = v;
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(base, alloc_cols);
}

struct MmaJob {
    uint64_t desc_a = 0, desc_b = 0;
    int m = 128, n = 8;
    int fmt = 0;  // 0 = F16, 2 = TF32
    int a_mn_major = 0, b_mn_major = 0;
    uint32_t d_lane = 0, repeat = 1, prefill = 0, read_lane = READ_CANONICAL;
};

// Runs the kernel; out[thread][column] for 128 threads and job.n columns.
static void run_mma(const uint8_t* image, const MmaJob& job, uint32_t* out, const char* what) {
    static uint32_t* d_image = dev_alloc<uint32_t>(OPERAND_BYTES / 4);
    static uint32_t* d_out = dev_alloc<uint32_t>(128 * 256);
    HOST_CHECK(cudaMemcpy(d_image, image, OPERAND_BYTES, cudaMemcpyHostToDevice));
    dev_fill(d_out, 128 * 256, 0);
    uint32_t alloc = 32;
    while (alloc < (uint32_t)job.n) alloc *= 2;
    uint32_t idesc = mma_idesc(job.m, job.n, job.fmt, job.a_mn_major, job.b_mn_major);
    mma_dump<<<1, 128>>>(d_image, job.desc_a, job.desc_b, idesc, job.fmt == 0 ? 0u : 1u, job.d_lane, job.repeat,
                         job.prefill, job.read_lane, (uint32_t)job.n, alloc, d_out);
    probe_sync(what);
    dev_read(out, d_out, 128 * (size_t)job.n);
}
