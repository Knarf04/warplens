// A second tcgen05.dealloc of the same columns: which ones does a GPU end the kernel for?
//
// Measured on a B200 (2026-11-25): one warp that releases an allocation twice in a row ends
// with cudaErrorLaunchFailure (tmem/alloc dealloc-twice), and the kernels hgemm_v7 to v9 of
// the MLSys book run to the end, in which the first warp of each warpgroup executes
//     tcgen05.relinquish_alloc_permit; tcgen05.dealloc
// after the last barrier, so the same columns are released once by each of two warps. The
// two differ in three ways at once, and each case below changes one of them:
//
//   dealloc_who two-warps        two warps release the allocation once each (no relinquish)
//   dealloc_who two-warps-order  the same, the second after a barrier that follows the first
//   dealloc_who relinquish-two   each of two warps relinquishes the permit, then releases:
//                                what the book's kernels do
//   dealloc_who relinquish-once  one warp relinquishes, then releases twice in a row
//   dealloc_who once             one warp releases once and the other does nothing (runs)
//
// Two warps of 32 threads; warp 0 allocates 32 columns. Each case is a process of its own
// and is followed by a fresh launch that allocates and releases 512 columns, which shows
// whether anything was left behind. The simulator runs all of them and reports
// `tmem-dealloc-repeated` for the second release; a GPU may end the kernel.
#include "../common.h"

enum Mode { TWO_WARPS = 0, TWO_WARPS_ORDER = 1, RELINQUISH_TWO = 2, RELINQUISH_ONCE = 3, ONCE = 4 };

extern "C" __global__ void tmem_dealloc_who(int mode, uint32_t* out) {
    __shared__ uint32_t slot;
    const int warp = threadIdx.x / 32;
    if (warp == 0) tmem_alloc(&slot, 32);
    __syncthreads();
    const uint32_t address = slot;
    if (mode == TWO_WARPS) {
        tmem_dealloc(address, 32);
    } else if (mode == TWO_WARPS_ORDER) {
        if (warp == 0) tmem_dealloc(address, 32);
        __syncthreads();
        if (warp == 1) tmem_dealloc(address, 32);
    } else if (mode == RELINQUISH_TWO) {
        tmem_relinquish();
        tmem_dealloc(address, 32);
    } else if (mode == RELINQUISH_ONCE) {
        if (warp == 0) {
            tmem_relinquish();
            tmem_dealloc(address, 32);
            tmem_dealloc(address, 32);
        }
    } else {
        if (warp == 0) tmem_dealloc(address, 32);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) out[warp] = 0x600D0000u + warp;  // this warp got to the end
    if (threadIdx.x == 0) out[2] = address;
}

extern "C" __global__ void tmem_fresh(uint32_t* out) {
    __shared__ uint32_t slot;
    tmem_alloc(&slot, 512);
    __syncthreads();
    if (threadIdx.x == 0) out[0] = slot;
    tmem_dealloc(slot, 512);
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    const char* names[] = {"two-warps", "two-warps-order", "relinquish-two", "relinquish-once", "once"};
    const char* titles[] = {"two warps release the same allocation once each",
                            "two warps release the same allocation once each, the second after a barrier",
                            "two warps each relinquish the permit, then release the same allocation",
                            "one warp relinquishes the permit, then releases the same allocation twice",
                            "one warp releases the allocation once"};
    int mode = -1;
    for (int i = 0; i < 5; ++i) if (!strcmp(which, names[i])) mode = i;
    if (mode < 0) { fprintf(stderr, "usage: dealloc_who two-warps|two-warps-order|relinquish-two|relinquish-once|once\n"); return 2; }
    uint32_t out[3];
    uint32_t* d_out = dev_alloc<uint32_t>(3);
    dev_fill(d_out, 3, 0xff);
    printf("script: %s\n", titles[mode]);
    fflush(stdout);
    tmem_dealloc_who<<<1, 64>>>(mode, d_out);
    probe_sync(titles[mode]);
    dev_read(out, d_out, 3);
    printf("  allocation at address %08x; warp 0 ended with %08x, warp 1 with %08x\n", out[2], out[0], out[1]);
    dev_fill(d_out, 3, 0xff);
    printf("script: a fresh launch afterwards\n");
    fflush(stdout);
    tmem_fresh<<<1, 32>>>(d_out);
    probe_sync("a fresh launch afterwards");
    dev_read(out, d_out, 1);
    printf("  512 columns at address %08x\n", out[0]);
    return 0;
}
