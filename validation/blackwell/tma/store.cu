// TMA tensor stores: what reaches global memory, with and without swizzle, and at the edges.
//
// The tensor is uint32, 6 rows x 32 columns (128 bytes per row), filled with 0x11111111.
// The kernel writes 0x20000000 + i to word i of a shared buffer with ordinary stores, runs
// fence.proxy.async, and stores one box of 4 rows x 32 columns from the buffer with
// cp.async.bulk.tensor.2d.global.shared::cta.bulk_group, then commit_group and wait_group 0.
// The host reads the tensor back and prints, for every element, the buffer word it holds
// ("." where the tensor was not written).
//
// Claims tested (docs/history/m2.md; warplens/units/tma.py `_cp`, `_store`):
//   * after commit_group and wait_group 0 the data is in global memory;
//   * an unswizzled store reads the box densely, dimension 0 fastest;
//   * a 128B-swizzled store reads through the same permutation a load writes with;
//   * elements of the box outside the tensor are dropped, and nothing outside is written
//     (cases `edge` and `outside`, own processes: hardware could fault instead).
//
//   store                the box inside the tensor, swizzle none and 128B
//   store edge           boxes across the last row, the first row, and the right edge
//   store outside        a box wholly outside the tensor
#include "../common.h"

#define ROWS 6
#define COLS 32
#define GUARD 64  // words before and after the tensor, to show that nothing else is written

extern "C" __global__ void tma_store(const __grid_constant__ CUtensorMap map, int c0, int c1) {
    __shared__ alignas(1024) uint32_t buffer[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) buffer[i] = 0x20000000u + i;
    fence_proxy_async();
    __syncthreads();
    if (threadIdx.x == 0) {
        tma_store_2d(&map, buffer, c0, c1);
        tma_store_commit_and_wait();
    }
}

static void run(int swizzle, int c0, int c1) {
    static uint32_t host[GUARD + ROWS * COLS + GUARD];
    for (uint32_t& v : host) v = 0x11111111u;
    uint32_t* d = dev_copy(host, GUARD + ROWS * COLS + GUARD);
    const uint64_t dims[2] = {COLS, ROWS};
    const uint32_t box[2] = {COLS, 4};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 2, d + GUARD, dims, box, nullptr,
                                 (CUtensorMapSwizzle)swizzle, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    printf("store of a 4 x 32 box at column %d, row %d, swizzle %s: buffer word found in each tensor element\n", c0,
           c1, swizzle_name(swizzle));
    fflush(stdout);
    tma_store<<<1, 128>>>(map, c0, c1);
    probe_sync("tma_store");
    dev_read(host, d, GUARD + ROWS * COLS + GUARD);
    int outside = 0;
    for (int i = 0; i < GUARD; ++i) outside += (host[i] != 0x11111111u) + (host[GUARD + ROWS * COLS + i] != 0x11111111u);
    for (int r = 0; r < ROWS; ++r) {
        printf("  row %d:", r);
        for (int c = 0; c < COLS; ++c) {
            uint32_t v = host[GUARD + r * COLS + c];
            if (v == 0x11111111u) printf("   .");
            else if (v >= 0x20000000u && v < 0x20000400u) printf(" %3u", v - 0x20000000u);
            else printf("   ?");
        }
        printf("\n");
    }
    printf("  words changed in the %d words before and after the tensor: %d\n", GUARD, outside);
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    if (!strcmp(which, "edge")) {
        run(0, 0, 4);    // rows 4-7: two inside
        run(0, 0, -2);   // rows -2..1: two inside
        run(0, 16, 1);   // columns 16-47: half inside
        run(0, -16, 1);  // columns -16..15: half inside
    } else if (!strcmp(which, "outside")) {
        run(0, 0, 6);
        run(0, 32, 0);
    } else {
        run(0, 0, 1);
        run(3, 0, 1);
    }
    return 0;
}
