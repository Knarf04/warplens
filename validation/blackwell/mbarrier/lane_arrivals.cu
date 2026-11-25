// How many arrivals one mbarrier.arrive makes when several threads of a warp execute it.
//
// 64 threads (two warps). Thread 0 initializes an mbarrier that expects `count` arrivals;
// after a barrier, the threads with index below `arrivers` each execute one
// mbarrier.arrive; after another barrier thread 0 tests both parities.
//
// Claim tested (docs/history/m2.md "mbarrier": "Every lane that executes an arrive arrives"): the
// phase completes exactly when arrivers == count, including when only part of a warp
// arrives; one short leaves it open.
//
//   lane_arrivals
#include "../common.h"

extern "C" __global__ void lane_arrivals(uint32_t count, uint32_t arrivers, uint32_t* out) {
    __shared__ uint64_t bar;
    if (threadIdx.x == 0) {
        mbar_init(&bar, count);
        mbar_init_fence();
    }
    __syncthreads();
    if (threadIdx.x < arrivers) mbar_arrive(&bar);
    __syncthreads();
    if (threadIdx.x == 0) {
        out[0] = mbar_test(&bar, 0);
        out[1] = mbar_test(&bar, 1);
    }
}

int main() {
    static const uint32_t cases[][2] = {{1, 1},   {32, 32}, {32, 31}, {64, 64}, {64, 63},
                                        {48, 48}, {48, 47}, {5, 5},   {33, 33}, {33, 32}};
    uint32_t* d_out = dev_alloc<uint32_t>(2);
    printf("expected arrivals  threads arriving  test_wait(parity 0)  test_wait(parity 1)\n");
    for (const auto& c : cases) {
        uint32_t out[2];
        dev_fill(d_out, 2, 0xff);
        lane_arrivals<<<1, 64>>>(c[0], c[1], d_out);
        probe_sync("lane_arrivals");
        dev_read(out, d_out, 2);
        printf("%17u  %16u  %19u  %19u\n", c[0], c[1], out[0], out[1]);
    }
    return 0;
}
