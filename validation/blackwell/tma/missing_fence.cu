// A TMA store of bytes that ordinary stores have just written, with and without
// fence.proxy.async in between.
//
// The tensor is uint32, N rows of 4 words (16 bytes), filled with 0x11111111. Iteration i
// (1 .. N) writes the counter i to the four words of a shared buffer with ordinary stores
// and at once TMA-stores the buffer to row i - 1, then commit_group and wait_group 0, so the
// store has finished before the next iteration writes again. Three variants, two warps:
//   no fence        thread 0 writes and stores
//   fence           thread 0 writes, fence.proxy.async.shared::cta, stores
//   fence elsewhere thread 32 (the other warp) writes; __syncthreads(); thread 0 fences and
//                   stores; __syncthreads()
// The host counts rows that do not hold their own counter: "stale" rows hold the counter of
// the iteration before (or the initial sentinel, for row 0), anything else is "other".
//
// Claims tested (docs/history/m2.md, "Proxy fences"; warplens/units/tma.py `_store`):
//   * the simulator stores the current bytes in every variant: 0 stale, 0 other. Without the
//     fence it reports `tma-store-missing-fence` once per store site;
//   * a fence issued after a barrier also covers another warp's earlier writes (no finding
//     in the third variant).
// On hardware, stale rows without the fence confirm the finding; stale rows in the third
// variant would mean the scope assumption is wrong. Informational: a GPU is free to show no
// staleness at all.
//
//   missing_fence [N]       default 2000; the iteration count is printed on an info: line
//                           only, so one prediction serves every N
#include "../common.h"

extern "C" __global__ void missing_fence(const __grid_constant__ CUtensorMap map, int variant, int n) {
    __shared__ alignas(128) uint32_t buffer[4];
    for (int i = 1; i <= n; ++i) {
        if (variant < 2) {
            if (threadIdx.x == 0) {
                for (int k = 0; k < 4; ++k) buffer[k] = (uint32_t)i;
                if (variant == 1) fence_proxy_async();
                tma_store_2d(&map, buffer, 0, i - 1);
                tma_store_commit_and_wait();
            }
        } else {
            if (threadIdx.x == 32)
                for (int k = 0; k < 4; ++k) buffer[k] = (uint32_t)i;
            __syncthreads();
            if (threadIdx.x == 0) {
                fence_proxy_async();
                tma_store_2d(&map, buffer, 0, i - 1);
                tma_store_commit_and_wait();
            }
            __syncthreads();
        }
    }
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2000;
    if (n < 1 || n > 1000000) {
        printf("HOST ERROR: iteration count %d out of range\n", n);
        return 2;
    }
    static const char* names[3] = {"no fence", "fence", "fence in another warp, after a barrier"};
    uint32_t* host = (uint32_t*)malloc((size_t)n * 16);
    uint32_t* d = dev_alloc<uint32_t>((size_t)n * 4);
    const uint64_t dims[2] = {4, (uint64_t)n};
    const uint32_t box[2] = {4, 1};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 2, d, dims, box, nullptr,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    printf("info: %d iterations per variant\n", n);
    for (int variant = 0; variant < 3; ++variant) {
        dev_fill(d, (size_t)n * 4, 0x11);
        missing_fence<<<1, 64>>>(map, variant, n);
        probe_sync(names[variant]);
        dev_read(host, d, (size_t)n * 4);
        int right = 0, stale = 0, other = 0;
        for (int i = 1; i <= n; ++i) {
            const uint32_t* row = host + (size_t)(i - 1) * 4;
            uint32_t before = i == 1 ? 0x11111111u : (uint32_t)(i - 1);
            bool all_own = true, all_before = true;
            for (int k = 0; k < 4; ++k) {
                all_own &= row[k] == (uint32_t)i;
                all_before &= row[k] == before;
            }
            if (all_own) ++right;
            else if (all_before) ++stale;
            else ++other;
        }
        printf("%-40s rows holding a stale counter: %d; rows holding something else: %d; all rows right: %s\n",
               names[variant], stale, other, right == n ? "yes" : "no");
    }
    free(host);
    return 0;
}
