// expect_tx against the bytes TMA really delivers: exact, too small, too large, missing.
//
// One thread runs a script on one mbarrier (one arrival per phase). `load` issues a TMA load
// of 256 bytes that reports to the mbarrier. `poll p` tests the phase with parity p up to
// POLL_BUDGET times and stops early when it has completed; `settle` always runs the whole
// budget, so that every load issued so far has landed before the next line is printed.
// test_wait for both parities is printed after every step except `load`, where it would
// depend on timing.
//
// Claims tested (docs/history/m2.md, "Undersized expect_tx"):
//   * the transaction count is signed, and a phase completes only when pending arrivals
//     and the count are both exactly zero;
//   * a TMA load reports its bytes in pieces, so with one load and an undersized expect_tx
//     the count reaches zero part of the way through the load and the phase completes
//     early; the rest of the load is charged to the next phase. (Measured on a B200,
//     2026-11-25: validation/blackwell/measured/b200/mbarrier/expect_tx_small.txt. The first
//     prediction was that the phase never completes, as it does not for one
//     mbarrier.complete_tx instruction: mbarrier/complete_before_expect.);
//   * with two loads and an expect_tx for one, the first completion ends the phase early
//     and the second is charged to the next phase;
//   * an oversized expect_tx waits for more data.
// The data itself arrives in every case; the probe prints whether it did.
//
//   expect_tx               the polled scripts (never hangs)
//   expect_tx wait-small    expect_tx 128 for a 256-byte load, then the unbounded
//                                try_wait loop real kernels use. Returns on a B200.
//   expect_tx wait-large    expect_tx 512 for a 256-byte load, same loop. Expected to hang.
//   expect_tx pieces        how large the pieces are: one 256-byte load against expect_tx
//                                of 4, 8, 16, 32, 48, 64, 96, 192 and 255 bytes, each polled
//                                and then settled (never hangs). On a B200 the answer changes
//                                from run to run (2026-11-25, second rental): the phase
//                                completes only at a multiple of 32, and there in some runs
//                                and not in others. No prediction is held against this run.
//   expect_tx sweep FROM TO STEP   the same for every count in a range. Not in the manifest:
//                                run by hand, 80 times, to count how often each completes
//                                (measured/b200/mbarrier/expect_tx_sweep.txt).
#include "../common.h"

enum Op { ARRIVE_EXPECT = 1, EXPECT = 2, ARRIVE = 3, LOAD = 4, POLL = 5, SETTLE = 6, WAIT = 7 };
static const char* op_names[] = {"?", "arrive.expect_tx", "expect_tx", "arrive", "load 256 bytes", "poll parity",
                                 "settle", "try_wait loop, parity"};

#ifndef LOAD_BYTES  // the sweep was also built with -DLOAD_BYTES=128 and 512 (measured/b200)
#define LOAD_BYTES 256
#endif
#define MAX_LOADS 4

extern "C" __global__ void mbar_expect_tx(const __grid_constant__ CUtensorMap map, const int* ops, const int* args,
                                          int n, int* results, uint32_t* data) {
    __shared__ alignas(128) uint32_t buffer[MAX_LOADS * LOAD_BYTES / 4];
    __shared__ uint64_t bar;
    for (int i = 0; i < MAX_LOADS * LOAD_BYTES / 4; ++i) buffer[i] = 0xEEEEEEEEu;
    mbar_init(&bar, 1);
    mbar_init_fence();
    int loads = 0;
    for (int i = 0; i < n; ++i) {
        uint32_t a = (uint32_t)args[i];
        switch (ops[i]) {
            case ARRIVE_EXPECT: mbar_arrive_expect_tx(&bar, a); break;
            case EXPECT: mbar_expect_tx(&bar, a); break;
            case ARRIVE: mbar_arrive(&bar); break;
            case LOAD:  // load k reads tensor elements 64 k .. 64 k + 63 into its own slot
                tma_load_1d(buffer + loads * (LOAD_BYTES / 4), &map, &bar, loads * (LOAD_BYTES / 4));
                ++loads;
                break;
            case POLL: mbar_poll(&bar, a); break;
            case SETTLE:
                for (int j = 0; j < POLL_BUDGET; ++j) (void)mbar_test(&bar, 0);
                break;
            case WAIT: mbar_wait(&bar, a); break;
        }
        results[2 * i] = (int)mbar_test(&bar, 0);
        results[2 * i + 1] = (int)mbar_test(&bar, 1);
    }
    for (int i = 0; i < MAX_LOADS; ++i) {  // first and last word of each slot
        data[2 * i] = buffer[i * (LOAD_BYTES / 4)];
        data[2 * i + 1] = buffer[(i + 1) * (LOAD_BYTES / 4) - 1];
    }
}

struct Step { int op, arg; };

static void run(const char* title, const Step* steps, int n, uint32_t* d_tensor) {
    const uint64_t dims[1] = {MAX_LOADS * LOAD_BYTES / 4};
    const uint32_t box[1] = {LOAD_BYTES / 4};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 1, d_tensor, dims, box, nullptr,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    int ops[32], args[32], results[64], loads = 0;
    uint32_t data[2 * MAX_LOADS];
    for (int i = 0; i < n; ++i) {
        ops[i] = steps[i].op;
        args[i] = steps[i].arg;
        loads += ops[i] == LOAD;
    }
    int* d_ops = dev_copy(ops, n);
    int* d_args = dev_copy(args, n);
    int* d_results = dev_alloc<int>(2 * n);
    uint32_t* d_data = dev_alloc<uint32_t>(2 * MAX_LOADS);
    dev_fill(d_results, 2 * n, 0xff);
    dev_fill(d_data, 2 * MAX_LOADS, 0);
    printf("script: %s\n", title);
    fflush(stdout);
    mbar_expect_tx<<<1, 1>>>(map, d_ops, d_args, n, d_results, d_data);
    probe_sync(title);
    dev_read(results, d_results, 2 * n);
    dev_read(data, d_data, 2 * MAX_LOADS);
    printf("  step  operation                    test_wait(parity 0)  test_wait(parity 1)\n");
    for (int i = 0; i < n; ++i) {
        printf("  %4d  %-22s", i, op_names[ops[i]]);
        if (ops[i] == LOAD || ops[i] == SETTLE || ops[i] == ARRIVE) printf("    ");
        else printf(" %3d", args[i]);
        if (ops[i] == LOAD) printf("  %19s  %19s\n", "-", "-");
        else printf("  %19d  %19d\n", results[2 * i], results[2 * i + 1]);
    }
    for (int k = 0; k < loads; ++k)
        printf("  load %d: first word %08x, last word %08x (expected %08x, %08x)\n", k, data[2 * k], data[2 * k + 1],
               0x10000000u + 64 * k, 0x10000000u + 64 * k + 63);
}

#define RUN(title, ...)                                       \
    do {                                                      \
        const Step s[] = {__VA_ARGS__};                       \
        run(title, s, (int)(sizeof s / sizeof s[0]), d_tensor); \
    } while (0)

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "default";
    uint32_t tensor[MAX_LOADS * LOAD_BYTES / 4];
    for (uint32_t i = 0; i < MAX_LOADS * LOAD_BYTES / 4; ++i) tensor[i] = 0x10000000u + i;
    uint32_t* d_tensor = dev_copy(tensor, MAX_LOADS * LOAD_BYTES / 4);
    if (!strcmp(which, "wait-small")) {
        RUN("expect_tx 128 for a 256-byte load, then wait without a bound",
            {ARRIVE_EXPECT, 128}, {LOAD, 0}, {WAIT, 0});
        printf("the wait returned\n");
    } else if (!strcmp(which, "wait-large")) {
        RUN("expect_tx 512 for a 256-byte load, then wait without a bound",
            {ARRIVE_EXPECT, 512}, {LOAD, 0}, {WAIT, 0});
        printf("the wait returned\n");
    } else if (!strcmp(which, "sweep") && argc > 4) {  // sweep FROM TO STEP: not in the manifest
        for (int bytes = atoi(argv[2]); bytes <= atoi(argv[3]); bytes += atoi(argv[4])) {
            char title[96];
            snprintf(title, sizeof title, "sweep: expect_tx %d, one load of %d bytes", bytes, LOAD_BYTES);
            RUN(title, {ARRIVE_EXPECT, bytes}, {LOAD, 0}, {POLL, 0}, {SETTLE, 0});
        }
    } else if (!strcmp(which, "pieces")) {
        for (int bytes : {4, 8, 16, 32, 48, 64, 96, 192, 255}) {
            char title[96];
            snprintf(title, sizeof title, "pieces: expect_tx %d, one load of 256 bytes", bytes);
            RUN(title, {ARRIVE_EXPECT, bytes}, {LOAD, 0}, {POLL, 0}, {SETTLE, 0});
        }
    } else {
        RUN("exact: expect_tx 256, one load", {ARRIVE_EXPECT, 256}, {LOAD, 0}, {POLL, 0});
        RUN("too small: expect_tx 128, one load; then expect_tx for the other 128",
            {ARRIVE_EXPECT, 128}, {LOAD, 0}, {POLL, 0}, {EXPECT, 128});
        RUN("too large: expect_tx 512, one load; then a second load",
            {ARRIVE_EXPECT, 512}, {LOAD, 0}, {POLL, 0}, {LOAD, 0}, {POLL, 0});
        RUN("expect_tx 256, two loads: the first completion ends the phase",
            {ARRIVE_EXPECT, 256}, {LOAD, 0}, {LOAD, 0}, {POLL, 0}, {SETTLE, 0}, {ARRIVE_EXPECT, 256},
            {ARRIVE, 0});
        RUN("no expect_tx at all: arrive, one load; then expect_tx 256 and an arrival",
            {ARRIVE, 0}, {LOAD, 0}, {SETTLE, 0}, {EXPECT, 256}, {ARRIVE, 0});
        RUN("expect_tx after the data has landed: load, settle, arrive.expect_tx 256",
            {LOAD, 0}, {SETTLE, 0}, {ARRIVE_EXPECT, 256});
    }
    return 0;
}
