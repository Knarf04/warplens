// One kernel for the TMA load probes: load one box into shared memory and copy the whole
// shared buffer out, so the host can see which bytes the load wrote and what it wrote.
//
// A single thread does everything:
//   1. fills the 4096-byte buffer with the sentinel 0xEEEEEEEE (ordinary stores)
//   2. arrives on a fresh mbarrier with expect_tx = `expect` bytes and issues the load to
//      buffer offset `offset`
//   3. polls the mbarrier (bounded; see POLL_BUDGET) and records whether phase 0 completed
//   4. copies the buffer to `out`
// It also records the shared-memory address of the buffer, which the simulator and a GPU
// need not agree on; probes print it on an "info:" line only.
//
// If the phase did not complete in time the buffer is copied anyway: the copy has had far
// longer than any load needs.
#pragma once
#include "../common.h"

#define DUMP_BYTES 4096
#define DUMP_WORDS (DUMP_BYTES / 4)
#define SENTINEL 0xEEEEEEEEu

extern "C" __global__ void tma_dump(const __grid_constant__ CUtensorMap map, int rank, int c0, int c1, int c2,
                                    int c3, int c4, int offset, uint32_t expect, uint32_t* out,
                                    uint32_t* completed) {
    __shared__ alignas(1024) uint32_t buffer[DUMP_WORDS];
    __shared__ uint64_t bar;
    int t = threadIdx.x;
    for (int i = t; i < DUMP_WORDS; i += blockDim.x) buffer[i] = SENTINEL;
    if (t == 0) {
        mbar_init(&bar, 1);
        mbar_init_fence();
    }
    __syncthreads();
    if (t == 0) {
        void* dst = (char*)buffer + offset;
        mbar_arrive_expect_tx(&bar, expect);
        switch (rank) {
            case 1: tma_load_1d(dst, &map, &bar, c0); break;
            case 2: tma_load_2d(dst, &map, &bar, c0, c1); break;
            case 3: tma_load_3d(dst, &map, &bar, c0, c1, c2); break;
            case 4: tma_load_4d(dst, &map, &bar, c0, c1, c2, c3); break;
            case 5: tma_load_5d(dst, &map, &bar, c0, c1, c2, c3, c4); break;
        }
        completed[0] = mbar_poll(&bar, 0);
        completed[1] = smem_addr(buffer);
    }
    __syncthreads();
    for (int i = t; i < DUMP_WORDS; i += blockDim.x) out[i] = buffer[i];
}

struct Dump {
    uint32_t words[DUMP_WORDS];
    uint32_t completed;
    uint32_t shared_address;  // of the start of the buffer
    const uint8_t* bytes() const { return (const uint8_t*)words; }
    // Number of bytes that no longer hold the sentinel. A loaded byte that happens to equal
    // 0xEE is not counted, so probes use data that avoids it.
    int written() const {
        int n = 0;
        for (int i = 0; i < DUMP_BYTES; ++i) n += bytes()[i] != 0xEE;
        return n;
    }
    // First and one-past-last byte that changed, or (-1, -1).
    void extent(int* first, int* last) const {
        *first = *last = -1;
        for (int i = 0; i < DUMP_BYTES; ++i)
            if (bytes()[i] != 0xEE) {
                if (*first < 0) *first = i;
                *last = i + 1;
            }
    }
};

// Run the kernel. `coords` has `rank` entries, dimension 0 (innermost) first.
static Dump load_and_dump(const CUtensorMap& map, int rank, const int* coords, int offset, uint32_t expect,
                          const char* what) {
    static uint32_t* d_out = dev_alloc<uint32_t>(DUMP_WORDS);
    static uint32_t* d_completed = dev_alloc<uint32_t>(2);
    int c[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < rank; ++i) c[i] = coords[i];
    dev_fill(d_out, DUMP_WORDS, 0);
    dev_fill(d_completed, 2, 0);
    tma_dump<<<1, 128>>>(map, rank, c[0], c[1], c[2], c[3], c[4], offset, expect, d_out, d_completed);
    probe_sync(what);
    Dump dump;
    dev_read(dump.words, d_out, DUMP_WORDS);
    uint32_t status[2];
    dev_read(status, d_completed, 2);
    dump.completed = status[0];
    dump.shared_address = status[1];
    return dump;
}

static void print_tuple(const char* name, const long long* v, int n) {
    printf("%s (", name);
    for (int i = 0; i < n; ++i) printf("%s%lld", i ? ", " : "", v[i]);
    printf(")");
}
