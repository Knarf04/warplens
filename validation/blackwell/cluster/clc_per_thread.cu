// Cluster launch control: does every thread that executes try_cancel make its own request?
// (matrix CL-16)
//
// A 64 x 32 grid of one-CTA clusters, 32 threads each, two CTAs per SM (clc_common.h). Every
// CTA marks itself started; all but the first return at once. In the first CTA thread 0 arms
// ONE mbarrier with one arrival and expect_tx 64, then lanes 0 to 3 each execute one
// try_cancel into a response slot of their own. Thread 0 polls the phase (bounded); if it
// completed, each of the four lanes decodes its own response.
//
// Claim (the book's; the manual does not say): each executing thread makes a separate
// request, with its own response and its own 16 bytes of complete_tx, and successful
// requests name different clusters.
//
// Compared lines: the phase completes with exactly 64 bytes expected; the coordinates of the
// successful requests are pairwise distinct and name clusters that did not start.
// "info:" lines: how many of the four succeeded and what they returned (on a GPU that
// depends on how many clusters had started), and the raw response words.
// MAY HANG or leave the phase open: if the instruction makes one request per warp, only 16
// bytes arrive and the probe prints "phase completed 0".
//
// In the simulator all four succeed and name the next four clusters, (1, 0) to (4, 0).
//
//   clc_per_thread
#include "clc_common.h"

#define NX 64
#define NY 32

extern "C" __global__ void clc_per_thread(uint32_t* out, uint32_t* started) {
    __shared__ alignas(16) uint32_t response[4][4];
    __shared__ uint64_t bar;
    __shared__ uint32_t completed;
    uint32_t tid = threadIdx.x, me = blockIdx.x + NX * blockIdx.y;
    if (tid == 0) started[me] = 1;
    // Every thread meets here, so that a compiler cannot merge the test above into the one
    // below and send thread 0 down a path of its own (see mma_dump in ../tcgen05_common.h).
    __syncthreads();
    if (me != 0) return;
    if (tid == 0) {
        for (int i = 0; i < 16; ++i) response[i / 4][i % 4] = CLC_SENTINEL;
        mbar_init(&bar, 1);
        mbar_init_fence();
        fence_proxy_async();
        mbar_arrive_expect_tx(&bar, 64);
    }
    __syncthreads();
    if (tid < 4) clc_try_cancel(response[tid], &bar);
    if (tid == 0) completed = mbar_poll(&bar, 0, CLC_POLL_BUDGET);
    __syncthreads();
    if (tid < 4) {
        uint32_t canceled = 0xFFFFFFFFu, x = 0, y = 0, z = 0;
        if (completed) canceled = clc_query(response[tid], x, y, z);
        uint32_t* mine = out + tid * 8;
        mine[0] = canceled; mine[1] = x; mine[2] = y; mine[3] = z;
        for (int i = 0; i < 4; ++i) mine[4 + i] = smem_peek(&response[tid][i]);
    }
    if (tid == 0) out[32] = completed;
}

int main() {
    const uint32_t total = NX * NY;
    uint32_t out[33];
    static uint32_t started[NX * NY];
    uint32_t* d_out = dev_alloc<uint32_t>(33);
    uint32_t* d_started = dev_alloc<uint32_t>(total);
    dev_fill(d_out, 33, 0);
    dev_fill(d_started, total, 0);
    HOST_CHECK(cudaFuncSetAttribute(clc_per_thread, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    CLC_OCCUPANCY_BYTES));
    cudaLaunchConfig_t config;
    memset(&config, 0, sizeof config);
    config.gridDim = dim3(NX, NY, 1);
    config.blockDim = dim3(32, 1, 1);
    config.dynamicSmemBytes = CLC_OCCUPANCY_BYTES;
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    printf("lanes 0 to 3 of the first CTA each execute one try_cancel; one mbarrier expects 64 bytes:\n");
    fflush(stdout);
    cudaError_t error = cudaLaunchKernelEx(&config, clc_per_thread, d_out, d_started);
    (void)error;  // probe_sync() reports it
    probe_sync("clc_per_thread");
    dev_read(out, d_out, 33);
    dev_read(started, d_started, total);

    printf("  phase completed %u\n", out[32]);
    uint32_t coords[4], successes = 0, clusters_started = 0;
    bool distinct = true, not_started = true, in_range = true;
    for (uint32_t c = 0; c < total; ++c) clusters_started += started[c];
    for (int lane = 0; lane < 4; ++lane) {
        const uint32_t* m = out + lane * 8;
        if (m[0] == 1) {
            uint32_t c = m[1] + NX * m[2];
            if (m[1] >= NX || m[2] >= NY || m[3] != 0) in_range = false;
            else if (started[c]) not_started = false;
            for (uint32_t i = 0; i < successes; ++i) distinct = distinct && coords[i] != c;
            coords[successes++] = c;
        }
    }
    if (out[32]) {
        printf("  coordinates of the successful requests are pairwise distinct: %s\n", distinct ? "yes" : "no");
        printf("  they are inside the grid: %s; they name clusters that did not start: %s\n", in_range ? "yes" : "no",
               not_started ? "yes" : "no");
    }
    printf("info: %u of 4 requests succeeded; %u of %u clusters started\n", successes, clusters_started, total);
    for (int lane = 0; lane < 4; ++lane) {
        const uint32_t* m = out + lane * 8;
        printf("info: lane %d: is_canceled %d, coordinate (%u, %u, %u), response words %08x %08x %08x %08x\n", lane,
               (int)m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
    }
    return 0;
}
