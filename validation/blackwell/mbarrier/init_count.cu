// The range of arrival counts mbarrier.init accepts.
//
// One thread; a script of (init n, arrive k, ...) steps, with test_wait.parity 0 and 1
// recorded after every step, as in mbarrier/script.cu. Nothing waits, so nothing can hang
// in the simulator; the risky cases still run in their own processes.
//
//   init_count           init 2^20 - 1; arrive 2^20 - 2 (phase stays open); arrive 1 (completes);
//                        then one arrive with the whole count 2^20 - 1 (completes again)
//   init_count over      init 2^20, then arrive 2^20 - 1 and arrive 1
//   init_count zero      init 0, then arrive 1 twice
//
// Claim tested (docs/history/m2.md; warplens/units/mbarrier.py `mbarrier_init`): the count must be
// 1 to 2^20 - 1. The simulator refuses `over` and `zero`; what hardware does with them is
// not known (the manual gives the range and nothing else), so both are informational.
#include "../common.h"

enum Op { INIT = 1, ARRIVE = 2 };

extern "C" __global__ void mbar_init_count(const int* ops, const uint32_t* args, int n, int* results) {
    __shared__ uint64_t bar;
    for (int i = 0; i < n; ++i) {
        if (ops[i] == INIT) {
            mbar_init(&bar, args[i]);
            mbar_init_fence();
        } else {
            mbar_arrive_n(&bar, args[i]);
        }
        results[2 * i] = (int)mbar_test(&bar, 0);
        results[2 * i + 1] = (int)mbar_test(&bar, 1);
    }
}

struct Step { int op; uint32_t arg; };

static void run(const char* title, const Step* steps, int n) {
    int ops[16], results[32];
    uint32_t args[16];
    for (int i = 0; i < n; ++i) { ops[i] = steps[i].op; args[i] = steps[i].arg; }
    int* d_ops = dev_copy(ops, n);
    uint32_t* d_args = dev_copy(args, n);
    int* d_results = dev_alloc<int>(2 * n);
    dev_fill(d_results, 2 * n, 0xff);
    printf("script: %s\n", title);
    fflush(stdout);
    mbar_init_count<<<1, 1>>>(d_ops, d_args, n, d_results);
    probe_sync(title);
    dev_read(results, d_results, 2 * n);
    printf("  step  operation        test_wait(parity 0)  test_wait(parity 1)\n");
    for (int i = 0; i < n; ++i)
        printf("  %4d  %-6s %8u  %19d  %19d\n", i, ops[i] == INIT ? "init" : "arrive", args[i], results[2 * i],
               results[2 * i + 1]);
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    const uint32_t top = (1u << 20) - 1;
    if (!strcmp(which, "over")) {
        const Step s[] = {{INIT, 1u << 20}, {ARRIVE, top}, {ARRIVE, 1}};
        run("init 2^20 (one more than the documented maximum)", s, 3);
    } else if (!strcmp(which, "zero")) {
        const Step s[] = {{INIT, 0}, {ARRIVE, 1}, {ARRIVE, 1}};
        run("init 0", s, 3);
    } else {
        const Step s[] = {{INIT, top}, {ARRIVE, top - 1}, {ARRIVE, 1}, {ARRIVE, top}};
        run("init 2^20 - 1 (the documented maximum)", s, 4);
    }
    return 0;
}
