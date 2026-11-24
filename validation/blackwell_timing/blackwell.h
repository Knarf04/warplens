// PTX wrappers for the Blackwell-only benchmarks (mbarrier, TMA, tcgen05, TMEM), and the
// host call that builds a tensor map. Same forms as examples/02_tma_copy and
// examples/04_pipelined_gemm, which the simulator runs.
#pragma once
#include <cuda.h>
#include "common.h"

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// -- mbarrier ---------------------------------------------------------------------------
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(arrivals));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase_parity) {
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra DONE;\n"
                 "bra LAB_WAIT;\n"
                 "DONE:\n"
                 "}" :: "r"(smem_addr(bar)), "r"(phase_parity) : "memory");
}

// -- TMA --------------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_2d(void* smem, const void* tensor_map, uint64_t* bar,
                                            int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(tensor_map), "r"(smem_addr(bar)), "r"(c0), "r"(c1)
                 : "memory");
}
__device__ __forceinline__ void tma_store_2d(const void* tensor_map, const void* smem, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                 :: "l"(tensor_map), "r"(smem_addr(smem)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// -- tensor memory ------------------------------------------------------------------------
__device__ __forceinline__ void tmem_alloc(uint32_t* where, uint32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr(where)), "r"(columns) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(address), "r"(columns)
                 : "memory");
}
__device__ __forceinline__ uint32_t tmem_ld1(uint32_t address) {
    uint32_t value;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(value) : "r"(address));
    return value;
}
__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void mma_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_addr(bar)) : "memory");
}

// -- host: tensor maps --------------------------------------------------------------------
// A rank-2 map of bytes over `base`: dims (width, rows), box (box_w, box_h), no swizzle.
// The driver function is fetched by name, as CUTLASS does, so the program does not link
// libcuda and `warplens run` can stand in for it.
inline CUtensorMap make_byte_map(void* base, uint64_t width, uint64_t rows, uint32_t box_w,
                                 uint32_t box_h) {
    typedef CUresult (*Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                               const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                               CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
                               CUtensorMapFloatOOBfill);
    static Encode encode = nullptr;
    if (!encode) {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult status;
        CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                               cudaEnableDefault, &status));
        if (!fn || status != cudaDriverEntryPointSuccess) {
            printf("BENCH_ERROR where=cuTensorMapEncodeTiled error=entry_point_not_found\n");
            exit(1);
        }
        encode = (Encode)fn;
    }
    CUtensorMap map;
    cuuint64_t dims[2] = {width, rows}, strides[1] = {width};
    cuuint32_t box[2] = {box_w, box_h}, steps[2] = {1, 1};
    CUresult r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, steps,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
        printf("BENCH_ERROR where=cuTensorMapEncodeTiled error=%d\n", (int)r);
        exit(1);
    }
    return map;
}
