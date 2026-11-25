// tcgen05.alloc / dealloc / relinquish_alloc_permit, driven by a script.
//
// One warp (32 threads, as .sync.aligned requires) executes a list of operations and records
// the address every allocation returned. Nothing else touches tensor memory.
//
// Claims tested (docs/history/m3.md, "TMEM"; warplens/units/tcgen05.py `Tmem.alloc`):
//   * a column count must be a power of two from 32 to 512;
//   * the address is (lane 0, first column), and an allocation takes the first free
//     naturally aligned slot of its size (addresses are compared exactly; a different
//     placement policy on hardware will show here);
//   * later allocations cannot be larger than an earlier one (case `increasing`);
//   * no allocation after relinquish_alloc_permit (case `after-relinquish`);
//   * a full tensor memory makes alloc wait (case `exhaust`, hangs);
//   * dealloc must name an allocation's address and size (case `dealloc-wrong-size`);
//   * a second dealloc of the same columns is ignored (case `dealloc-twice`): kernels of
//     the MLSys book do this and run on hardware, but what the hardware does is not known;
//   * columns still allocated at exit are reported, and do not affect the next launch
//     (case `leak`).
// Every case other than the default runs in its own process: the simulator refuses most of
// them, and hardware may fault or hang.
//
//   alloc [increasing|size-48|size-16|size-0|size-1024|after-relinquish|exhaust|
//               dealloc-twice|dealloc-wrong-size|leak]
#include "../common.h"

enum Op { ALLOC = 1, DEALLOC = 2, RELINQUISH = 3, DEALLOC_SIZE = 4 };

// DEALLOC k releases the k-th allocation of the script with its own size.
// DEALLOC_SIZE (k << 16 | n) releases it with n columns instead.
extern "C" __global__ void tmem_alloc_script(const int* ops, const int* args, int n, uint32_t* addresses) {
    __shared__ uint32_t slots[32];
    __shared__ uint32_t sizes[32];
    int allocations = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t a = (uint32_t)args[i];
        switch (ops[i]) {
            case ALLOC:
                tmem_alloc(&slots[allocations], a);
                __syncthreads();  // one warp; __syncwarp (bar.warp.sync) is not in the simulator
                if (threadIdx.x == 0) {
                    sizes[allocations] = a;
                    addresses[i] = slots[allocations];
                }
                __syncthreads();  // one warp; __syncwarp (bar.warp.sync) is not in the simulator
                ++allocations;
                break;
            case DEALLOC: tmem_dealloc(slots[a], sizes[a]); break;
            case DEALLOC_SIZE: tmem_dealloc(slots[a >> 16], a & 0xffff); break;
            case RELINQUISH: tmem_relinquish(); break;
        }
        if (threadIdx.x == 0 && ops[i] != ALLOC) addresses[i] = 0x600D0000u;  // the step was executed
    }
}

struct Step { int op, arg; };

static void run(const char* title, const Step* steps, int n) {
    int ops[64], args[64];
    uint32_t addresses[64];
    for (int i = 0; i < n; ++i) { ops[i] = steps[i].op; args[i] = steps[i].arg; }
    int* d_ops = dev_copy(ops, n);
    int* d_args = dev_copy(args, n);
    uint32_t* d_addresses = dev_alloc<uint32_t>(n);
    dev_fill(d_addresses, n, 0xff);
    printf("script: %s\n", title);
    fflush(stdout);
    tmem_alloc_script<<<1, 32>>>(d_ops, d_args, n, d_addresses);
    probe_sync(title);
    dev_read(addresses, d_addresses, n);
    int allocation = 0;
    for (int i = 0; i < n; ++i) {
        if (ops[i] == ALLOC)
            printf("  step %2d  alloc %4d columns -> allocation %d at address %08x (lane %u, column %u)\n", i,
                   args[i], allocation++, addresses[i], addresses[i] >> 16, addresses[i] & 0xffff);
        else if (ops[i] == DEALLOC)
            printf("  step %2d  dealloc allocation %d: %s\n", i, args[i],
                   addresses[i] == 0x600D0000u ? "executed" : "NOT executed");
        else if (ops[i] == DEALLOC_SIZE)
            printf("  step %2d  dealloc allocation %d as %d columns: %s\n", i, args[i] >> 16, args[i] & 0xffff,
                   addresses[i] == 0x600D0000u ? "executed" : "NOT executed");
        else
            printf("  step %2d  relinquish_alloc_permit: %s\n", i,
                   addresses[i] == 0x600D0000u ? "executed" : "NOT executed");
    }
}

#define RUN(title, ...)                               \
    do {                                              \
        const Step s[] = {__VA_ARGS__};               \
        run(title, s, (int)(sizeof s / sizeof s[0])); \
    } while (0)

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "default";
    if (!strcmp(which, "default")) {
        RUN("every size alone, largest first",
            {ALLOC, 512}, {DEALLOC, 0}, {ALLOC, 256}, {DEALLOC, 1}, {ALLOC, 128}, {DEALLOC, 2}, {ALLOC, 64},
            {DEALLOC, 3}, {ALLOC, 32}, {DEALLOC, 4});
        RUN("sizes that do not increase, filling all 512 columns",
            {ALLOC, 256}, {ALLOC, 128}, {ALLOC, 64}, {ALLOC, 32}, {ALLOC, 32}, {DEALLOC, 4}, {DEALLOC, 3},
            {DEALLOC, 2}, {DEALLOC, 1}, {DEALLOC, 0});
        RUN("equal sizes, a hole in the middle is reused",
            {ALLOC, 64}, {ALLOC, 64}, {ALLOC, 64}, {DEALLOC, 1}, {ALLOC, 64}, {ALLOC, 32}, {DEALLOC, 0},
            {DEALLOC, 2}, {DEALLOC, 3}, {DEALLOC, 4});
        RUN("allocate, relinquish the permit, then use and release as usual",
            {ALLOC, 32}, {RELINQUISH, 0}, {DEALLOC, 0});
    } else if (!strcmp(which, "increasing")) {
        RUN("32 columns, then 64 (a larger allocation after a smaller one)",
            {ALLOC, 32}, {ALLOC, 64}, {DEALLOC, 1}, {DEALLOC, 0});
    } else if (!strncmp(which, "size-", 5)) {
        int columns = atoi(which + 5);
        char title[64];
        snprintf(title, sizeof title, "an allocation of %d columns", columns);
        RUN(title, {ALLOC, columns}, {DEALLOC, 0});
    } else if (!strcmp(which, "after-relinquish")) {
        RUN("relinquish the permit, then allocate", {RELINQUISH, 0}, {ALLOC, 32}, {DEALLOC, 0});
    } else if (!strcmp(which, "exhaust")) {
        RUN("512 columns, then 32 more with nothing free (expected to wait forever)",
            {ALLOC, 512}, {ALLOC, 32}, {DEALLOC, 1}, {DEALLOC, 0});
    } else if (!strcmp(which, "dealloc-twice")) {
        RUN("release the same allocation twice", {ALLOC, 32}, {DEALLOC, 0}, {DEALLOC, 0});
        RUN("a fresh launch afterwards", {ALLOC, 512}, {DEALLOC, 0});
    } else if (!strcmp(which, "dealloc-wrong-size")) {
        RUN("allocate 64 columns, release them as 32", {ALLOC, 64}, {DEALLOC_SIZE, (0 << 16) | 32});
        RUN("a fresh launch afterwards", {ALLOC, 512}, {DEALLOC, 0});
    } else if (!strcmp(which, "leak")) {
        RUN("allocate 256 columns and return without releasing them", {ALLOC, 256});
        RUN("a fresh launch afterwards", {ALLOC, 512}, {DEALLOC, 0});
    } else {
        printf("unknown case %s\n", which);
        return 1;
    }
    return 0;
}
