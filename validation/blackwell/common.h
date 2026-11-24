// Shared by every Blackwell probe: thin wrappers over the PTX each probe exercises, and
// the host-side helpers that keep the printed text identical on a GPU and in the simulator.
//
// Rules the probes follow:
//   * Results go to global memory and are printed by the host program. No device printf.
//   * Bit patterns are printed in hex. No addresses, times or other machine-dependent values
//     are printed, except on lines that start with "info:" (compare.py ignores those).
//   * A wait whose outcome is the thing under test is a bounded poll (mbar_poll), so the
//     kernel returns either way and the probe prints what happened.
//   * After a failed launch the program prints "KERNEL FAILED" and exits with status 3:
//     a real GPU's context is unusable after a fault, the simulator's is not, and the two
//     must print the same thing.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -------------------------------------------------------------------------------------
// Device side
// -------------------------------------------------------------------------------------

// Polls a kernel makes before it reports "did not complete". Each poll is a few
// instructions; on a GPU the whole budget takes on the order of a millisecond, far longer
// than any copy or MMA a probe issues. The simulator spends it only when the phase really
// never completes.
#define POLL_BUDGET 20000

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// Read a shared word the compiler cannot assume unchanged (TMA writes behind its back).
__device__ __forceinline__ uint32_t smem_peek(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_addr(p)) : "memory");
    return v;
}

// ---- mbarrier ----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void mbar_init_fence() {
#ifndef PROBE_AMPERE  // sm_90 and newer; the Ampere build of mbarrier/script.cu does without
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_addr(bar)) : "memory");
}
#ifndef PROBE_AMPERE  // everything from here to the matching #endif needs sm_90 or sm_100a
__device__ __forceinline__ void mbar_arrive_n(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;" :: "r"(smem_addr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_complete_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.complete_tx.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(bar)), "r"(bytes) : "memory");
}
#endif  // PROBE_AMPERE
// Has the phase with this parity completed? Never blocks.
__device__ __forceinline__ uint32_t mbar_test(uint64_t* bar, uint32_t parity) {
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "}" : "=r"(done) : "r"(smem_addr(bar)), "r"(parity) : "memory");
    return done;
}
// Poll up to `budget` times; 1 if the phase completed.
__device__ __forceinline__ uint32_t mbar_poll(uint64_t* bar, uint32_t parity, int budget = POLL_BUDGET) {
    for (int i = 0; i < budget; ++i)
        if (mbar_test(bar, parity)) return 1;
    return 0;
}
// The loop real kernels use. Never returns if the phase never completes.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra DONE;\n"
                 "bra LAB_WAIT;\n"
                 "DONE:\n"
                 "}" :: "r"(smem_addr(bar)), "r"(parity) : "memory");
}

#ifndef PROBE_AMPERE
// ---- TMA ----
#define TMA_LOAD_HEAD "cp.async.bulk.tensor."
#define TMA_LOAD_TAIL "d.shared::cluster.global.mbarrier::complete_tx::bytes"
__device__ __forceinline__ void tma_load_1d(void* smem, const void* map, uint64_t* bar, int c0) {
    asm volatile(TMA_LOAD_HEAD "1" TMA_LOAD_TAIL " [%0], [%1, {%3}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(map), "r"(smem_addr(bar)), "r"(c0) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* smem, const void* map, uint64_t* bar, int c0, int c1) {
    asm volatile(TMA_LOAD_HEAD "2" TMA_LOAD_TAIL " [%0], [%1, {%3, %4}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_load_3d(void* smem, const void* map, uint64_t* bar, int c0, int c1,
                                            int c2) {
    asm volatile(TMA_LOAD_HEAD "3" TMA_LOAD_TAIL " [%0], [%1, {%3, %4, %5}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1), "r"(c2)
                 : "memory");
}
__device__ __forceinline__ void tma_load_4d(void* smem, const void* map, uint64_t* bar, int c0, int c1,
                                            int c2, int c3) {
    asm volatile(TMA_LOAD_HEAD "4" TMA_LOAD_TAIL " [%0], [%1, {%3, %4, %5, %6}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
                 : "memory");
}
__device__ __forceinline__ void tma_load_5d(void* smem, const void* map, uint64_t* bar, int c0, int c1,
                                            int c2, int c3, int c4) {
    asm volatile(TMA_LOAD_HEAD "5" TMA_LOAD_TAIL " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                 :: "r"(smem_addr(smem)), "l"(map), "r"(smem_addr(bar)), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                    "r"(c4) : "memory");
}
__device__ __forceinline__ void tma_store_2d(const void* map, const void* smem, int c0, int c1) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
                 :: "l"(map), "r"(smem_addr(smem)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void tma_store_commit_and_wait() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---- cluster ----
__device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ uint32_t cluster_nctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
// The cluster-wide address of this CTA's shared `p`, as seen in CTA `rank`.
__device__ __forceinline__ uint32_t mapa(const void* p, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(smem_addr(p)), "r"(rank));
    return r;
}

// ---- tcgen05 ----
// Every thread of the warp must execute these together (.sync.aligned).
__device__ __forceinline__ void tmem_alloc(uint32_t* where, uint32_t columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr(where)), "r"(columns) : "memory");
}
__device__ __forceinline__ void tmem_relinquish() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t address, uint32_t columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(address), "r"(columns)
                 : "memory");
}
__device__ __forceinline__ uint32_t tmem_ld_32x32b(uint32_t address) {
    uint32_t v;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(v) : "r"(address));
    return v;
}
__device__ __forceinline__ void tmem_st_32x32b(uint32_t address, uint32_t v) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(address), "r"(v) : "memory");
}
__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
// D (+)= A * B^T, issued by one thread. `accumulate` = 0 overwrites D.
__device__ __forceinline__ void mma_f16(uint32_t tmem, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                        uint32_t accumulate) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                 "}" :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void mma_tf32(uint32_t tmem, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                         uint32_t accumulate) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "setp.ne.b32 p, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n"
                 "}" :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate) : "memory");
}
// Arrive on `bar` once every MMA this thread has issued is done.
__device__ __forceinline__ void mma_commit(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(smem_addr(bar)) : "memory");
}

#endif  // PROBE_AMPERE

// Instruction descriptor for float32 accumulators (cute/arch/mma_sm100_desc.hpp):
// c_format = F32 (1) at bits 4-5, a/b format at 7-9 / 10-12 (0 = F16, 2 = TF32),
// a/b MN-major at 15 / 16, N / 8 at 17-22, M / 16 at 24-28.
__host__ __device__ inline uint32_t mma_idesc(int m, int n, int fmt, int a_mn_major, int b_mn_major) {
    return (1u << 4) | ((uint32_t)fmt << 7) | ((uint32_t)fmt << 10) | ((uint32_t)a_mn_major << 15)
           | ((uint32_t)b_mn_major << 16) | ((uint32_t)(n >> 3) << 17) | ((uint32_t)(m >> 4) << 24);
}
// Shared-memory matrix descriptor: start / 16 at bits 0-13, leading byte offset / 16 at 16-29,
// stride byte offset / 16 at 32-45, the constant 1 at bit 46, layout type at 61-63
// (0 no swizzle, 6 / 4 / 2 for 32B / 64B / 128B).
__host__ __device__ inline uint64_t mma_smem_desc(uint32_t start, uint32_t lbo, uint32_t sbo, int layout) {
    return (uint64_t)(start >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32)
           | (1ull << 46) | ((uint64_t)layout << 61);
}

// -------------------------------------------------------------------------------------
// Host side
// -------------------------------------------------------------------------------------
#define HOST_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t probe_error_ = (call);                                                 \
        if (probe_error_ != cudaSuccess) {                                                 \
            printf("HOST ERROR: %s returned %s\n", #call, cudaGetErrorName(probe_error_)); \
            fflush(stdout);                                                                \
            exit(2);                                                                       \
        }                                                                                  \
    } while (0)

// Errors that say the driver never had a kernel to run: it could not compile or load what
// this program carries (PTX newer than the driver, no code for this GPU, no JIT compiler),
// or there is no usable device. They say nothing about the behaviour a probe tests, so they
// must never be read as "the GPU refused the kernel". The first launch of a program is
// where they appear, because the runtime loads the module then.
static inline bool probe_is_load_error(cudaError_t error) {
    switch (error) {
        case cudaErrorUnsupportedPtxVersion:
        case cudaErrorInvalidPtx:
        case cudaErrorJitCompilerNotFound:
        case cudaErrorJitCompilationDisabled:
        case cudaErrorInvalidSource:
        case cudaErrorInvalidKernelImage:
        case cudaErrorNoKernelImageForDevice:
        case cudaErrorInvalidDeviceFunction:
        case cudaErrorSymbolNotFound:
        case cudaErrorFileNotFound:
        case cudaErrorSharedObjectSymbolNotFound:
        case cudaErrorSharedObjectInitFailed:
        case cudaErrorInsufficientDriver:
        case cudaErrorCallRequiresNewerDriver:
        case cudaErrorSystemDriverMismatch:
        case cudaErrorCompatNotSupportedOnDevice:
        case cudaErrorStubLibrary:
        case cudaErrorInitializationError:
        case cudaErrorNoDevice:
        case cudaErrorInvalidDevice:
        case cudaErrorDevicesUnavailable:
            return true;
        default:
            return false;
    }
}

// Exit statuses of a probe: 0 ran to the end; 2 HOST ERROR (a host-side CUDA call failed);
// 3 KERNEL FAILED (the kernel started, or was refused for what it asked: the outcome of the
// test); 4 LAUNCH FAILED (the program's kernels could not be loaded at all: no outcome).
// run_all.py and compare.py tell these apart by the first word and by the status.
static inline void probe_report_failure(const char* what, cudaError_t error, const char* where) {
    if (probe_is_load_error(error)) {
        printf("LAUNCH FAILED: %s\n", what);
        printf("info: error %s %s: %s\n", cudaGetErrorName(error), where, cudaGetErrorString(error));
        printf("info: the driver could not load this program's kernels, so nothing ran. With "
               "cudaErrorUnsupportedPtxVersion the driver is older than the toolkit that wrote the PTX: "
               "run the machine-code build (build.sh --native, run_all.py --binaries native)\n");
        fflush(stdout);
        fprintf(stderr, "LAUNCH FAILED: %s: %s (%s)\n", what, cudaGetErrorName(error), cudaGetErrorString(error));
        exit(4);
    }
    printf("KERNEL FAILED: %s\n", what);
    printf("info: error %s\n", cudaGetErrorName(error));
    fflush(stdout);
    // Where it was reported goes to stderr only, so that the committed predictions (stdout)
    // do not depend on it: the simulator reports a refusal at the launch, a GPU a fault at
    // the synchronize.
    fprintf(stderr, "KERNEL FAILED: %s: %s %s (%s)\n", what, cudaGetErrorName(error), where,
            cudaGetErrorString(error));
    exit(3);
}

// Wait for the launch just made. On failure, say so on stdout and stop: nothing after a
// fault can be trusted on a real GPU. The error name goes on an "info:" line because the
// simulator reports every fault as cudaErrorLaunchFailure and a GPU distinguishes them.
// Both the launch's own error (cudaGetLastError) and the one the kernel ended with
// (cudaDeviceSynchronize, then cudaGetLastError again) are checked.
static inline void probe_sync(const char* what) {
    cudaError_t at_launch = cudaGetLastError();
    if (at_launch != cudaSuccess) probe_report_failure(what, at_launch, "at launch");
    cudaError_t at_sync = cudaDeviceSynchronize();
    if (at_sync != cudaSuccess) probe_report_failure(what, at_sync, "at synchronize");
    cudaError_t after = cudaGetLastError();
    if (after != cudaSuccess) probe_report_failure(what, after, "after synchronize");
}

template <class T> static T* dev_alloc(size_t count) {
    T* p = nullptr;
    HOST_CHECK(cudaMalloc(&p, count * sizeof(T) > 0 ? count * sizeof(T) : 1));
    return p;
}
template <class T> static T* dev_copy(const T* host, size_t count) {
    T* p = dev_alloc<T>(count);
    HOST_CHECK(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> static void dev_read(T* host, const T* dev, size_t count) {
    HOST_CHECK(cudaMemcpy(host, dev, count * sizeof(T), cudaMemcpyDeviceToHost));
}
template <class T> static void dev_fill(T* dev, size_t count, int byte) {
    HOST_CHECK(cudaMemset(dev, byte, count * sizeof(T)));
}

#ifndef PROBE_AMPERE  // tensor maps: nothing for a GPU without TMA
typedef CUresult (*probe_encode_fn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                                    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
                                    CUtensorMapFloatOOBfill);

// The driver's cuTensorMapEncodeTiled, found through the runtime as CUTLASS does, so the
// program does not link libcuda.
static inline probe_encode_fn probe_encoder() {
    static probe_encode_fn fn = nullptr;
    if (!fn) {
        void* found = nullptr;
        cudaDriverEntryPointQueryResult status;
        HOST_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &found, 12000, cudaEnableDefault,
                                                    &status));
        if (!found || status != cudaDriverEntryPointSuccess) {
            printf("HOST ERROR: the driver has no cuTensorMapEncodeTiled\n");
            exit(2);
        }
        fn = (probe_encode_fn)found;
    }
    return fn;
}

// Tiled tensor map over `rank` dimensions, dimension 0 innermost, strides derived from the
// dimensions (no padding). Exits on an encoder error.
static inline CUtensorMap tensor_map(CUtensorMapDataType type, int elem_bytes, int rank, void* data,
                                     const uint64_t* dims, const uint32_t* box, const uint32_t* elem_strides,
                                     CUtensorMapSwizzle swizzle, CUtensorMapFloatOOBfill fill) {
    CUtensorMap map;
    memset(&map, 0, sizeof map);
    cuuint64_t d[5], strides[5];
    cuuint32_t b[5], e[5];
    uint64_t stride = elem_bytes;
    for (int i = 0; i < rank; ++i) {
        d[i] = dims[i];
        b[i] = box[i];
        e[i] = elem_strides ? elem_strides[i] : 1;
        stride *= dims[i];
        strides[i] = stride;  // entry i is the byte stride of dimension i + 1
    }
    CUresult r = probe_encoder()(&map, type, rank, data, d, strides, b, e, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, fill);
    if (r != CUDA_SUCCESS) {
        printf("HOST ERROR: cuTensorMapEncodeTiled returned %d\n", (int)r);
        fflush(stdout);
        exit(2);
    }
    return map;
}

#endif  // PROBE_AMPERE

static inline const char* swizzle_name(int s) {
    static const char* names[] = {"none", "32B", "64B", "128B"};
    return s >= 0 && s < 4 ? names[s] : "?";
}

// Launch `kernel` on a grid of `ctas` x 1 x 1 CTAs grouped in clusters of `cluster` CTAs.
template <class... Args>
static void launch_cluster(void (*kernel)(Args...), unsigned ctas, unsigned cluster, unsigned threads,
                           Args... args) {
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof config);
    config.gridDim = dim3(ctas, 1, 1);
    config.blockDim = dim3(threads, 1, 1);
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = cluster;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    cudaError_t error = cudaLaunchKernelEx(&config, kernel, args...);
    (void)error;  // probe_sync() reports it
}
