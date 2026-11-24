// Preflight: two kernels that only a datacenter Blackwell (sm_100a) can run.
//
//   pf_tmem  allocates 32 columns of tensor memory, reports the address, frees them.
//   pf_tma   loads one box of a 2D float32 tensor into shared memory with TMA and
//            copies it out with ordinary stores.
//
// Neither waits without a bound: a kernel that hangs would hang preflight. The
// TMA wait gives up after WAIT_LIMIT polls and says so in out_status.
#include <stdint.h>

#define WAIT_LIMIT (1u << 24)

// Same size and alignment as CUtensorMap in cuda.h; declared here so this file
// needs no toolkit header.
struct alignas(64) TensorMap { uint64_t opaque[16]; };

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// Launch with exactly one warp (block = 32): tcgen05.alloc is warp-wide.
// out[0] = 0xB200B200, out[1] = the TMEM address the allocation returned.
extern "C" __global__ void pf_tmem(uint32_t* out) {
    __shared__ uint32_t address;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr(&address)), "r"(32u) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    __syncthreads();
    uint32_t a = address;
    if (threadIdx.x == 0) {
        out[0] = 0xB200B200u;
        out[1] = a;
    }
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(a), "r"(32u)
                 : "memory");
}

#define BOX_W 16
#define BOX_H 8

// One CTA. The tensor map must describe float32 with a BOX_W x BOX_H box and no
// swizzle. The box at element coordinates (c0, c1) lands in `out` row by row.
// out_status[0] = 1 when the load completed, 2 when the wait ran out.
extern "C" __global__ void pf_tma(const __grid_constant__ TensorMap map, float* out,
                                  uint32_t* out_status, int c0, int c1) {
    __shared__ alignas(128) float tile[BOX_W * BOX_H];
    __shared__ uint64_t bar;
    const uint32_t bytes = BOX_W * BOX_H * sizeof(float);
    int t = threadIdx.x;

    if (t == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(&bar)), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (t == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(smem_addr(&bar)), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1, {%3, %4}], [%2];"
                     :: "r"(smem_addr(tile)), "l"(&map), "r"(smem_addr(&bar)), "r"(c0), "r"(c1)
                     : "memory");
    }
    uint32_t done = 0;
    for (uint32_t tries = 0; tries < WAIT_LIMIT && !done; ++tries) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, p;\n"
                     "}" : "=r"(done) : "r"(smem_addr(&bar)), "r"(0u) : "memory");
    }
    if (t == 0) out_status[0] = done ? 1u : 2u;
    if (done)
        for (int i = t; i < BOX_W * BOX_H; i += blockDim.x) out[i] = tile[i];
}
