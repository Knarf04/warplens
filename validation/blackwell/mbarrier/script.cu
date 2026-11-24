// mbarrier state machine, driven by a script.
//
// One thread executes a list of mbarrier operations on one mbarrier. After every operation
// it records mbarrier.test_wait.parity for parity 0 and parity 1. test_wait never blocks, and
// nothing here is asynchronous, so the output is deterministic.
//
// Claims tested (docs/m2.md, "mbarrier"):
//   * a phase completes when its pending arrivals and its transaction count are both zero;
//   * test_wait.parity p is true exactly when the current phase number has parity != p,
//     so a new mbarrier answers (parity 0: no, parity 1: yes);
//   * the arrival count is reloaded when a phase completes;
//   * the transaction count is signed: complete_tx before expect_tx leaves the phase open
//     until expect_tx balances it (case `negative`);
//   * arrivals beyond the pending count carry into the next phase (case `over`).
//
//   script [basic|count|tx|negative|over]      default: basic, count and tx
#include "../common.h"

enum Op { INIT = 1, ARRIVE = 2, EXPECT = 3, COMPLETE = 4, ARRIVE_EXPECT = 5 };
static const char* op_names[] = {"?", "init", "arrive", "expect_tx", "complete_tx", "arrive.expect_tx"};

extern "C" __global__ void mbar_script(const int* ops, const int* args, int n, int* results) {
    __shared__ uint64_t bar;
    for (int i = 0; i < n; ++i) {
        uint32_t a = (uint32_t)args[i];
        switch (ops[i]) {
            case INIT: mbar_init(&bar, a); mbar_init_fence(); break;
            case ARRIVE: mbar_arrive_n(&bar, a); break;
            case EXPECT: mbar_expect_tx(&bar, a); break;
            case COMPLETE: mbar_complete_tx(&bar, a); break;
            case ARRIVE_EXPECT: mbar_arrive_expect_tx(&bar, a); break;
        }
        results[2 * i] = (int)mbar_test(&bar, 0);
        results[2 * i + 1] = (int)mbar_test(&bar, 1);
    }
}

struct Step { int op, arg; };

static void run(const char* title, const Step* steps, int n) {
    int ops[64], args[64], results[128];
    for (int i = 0; i < n; ++i) { ops[i] = steps[i].op; args[i] = steps[i].arg; }
    int* d_ops = dev_copy(ops, n);
    int* d_args = dev_copy(args, n);
    int* d_results = dev_alloc<int>(2 * n);
    dev_fill(d_results, 2 * n, 0xff);
    mbar_script<<<1, 1>>>(d_ops, d_args, n, d_results);
    probe_sync(title);
    dev_read(results, d_results, 2 * n);
    printf("script: %s\n", title);
    printf("  step  operation              test_wait(parity 0)  test_wait(parity 1)\n");
    for (int i = 0; i < n; ++i)
        printf("  %4d  %-16s %4d  %19d  %19d\n", i, op_names[ops[i]], args[i], results[2 * i],
               results[2 * i + 1]);
}

#define RUN(title, ...)                                    \
    do {                                                   \
        const Step s[] = {__VA_ARGS__};                    \
        run(title, s, (int)(sizeof s / sizeof s[0]));      \
    } while (0)

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "default";
    bool all = !strcmp(which, "default");
    if (all || !strcmp(which, "basic"))
        RUN("one arrival per phase, six phases",
            {INIT, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1});
    if (all || !strcmp(which, "count")) {
        RUN("three arrivals per phase, one at a time",
            {INIT, 3}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1});
        RUN("four arrivals per phase, arriving with counts 3 and 1",
            {INIT, 4}, {ARRIVE, 3}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 3});
        RUN("init again restarts at phase 0",
            {INIT, 1}, {ARRIVE, 1}, {INIT, 2}, {ARRIVE, 1}, {ARRIVE, 1});
    }
    if (all || !strcmp(which, "tx")) {
        RUN("expect_tx 64, arrive, complete_tx 32 + 32",
            {INIT, 1}, {EXPECT, 64}, {ARRIVE, 1}, {COMPLETE, 32}, {COMPLETE, 32});
        RUN("arrive.expect_tx 64, complete_tx 64, then a second phase the same way",
            {INIT, 1}, {ARRIVE_EXPECT, 64}, {COMPLETE, 64}, {ARRIVE_EXPECT, 64}, {COMPLETE, 64});
        RUN("bytes complete before the arrival",
            {INIT, 1}, {EXPECT, 64}, {COMPLETE, 64}, {ARRIVE, 1});
        RUN("expect_tx in two parts",
            {INIT, 1}, {EXPECT, 16}, {EXPECT, 48}, {ARRIVE, 1}, {COMPLETE, 64});
    }
    if (!strcmp(which, "negative")) {
        RUN("complete_tx 64 before expect_tx 64 (count goes negative first)",
            {INIT, 1}, {COMPLETE, 64}, {ARRIVE, 1}, {EXPECT, 64});
        RUN("complete_tx 64 against expect_tx 32, balanced later by expect_tx 32",
            {INIT, 1}, {ARRIVE_EXPECT, 32}, {COMPLETE, 64}, {EXPECT, 32}, {ARRIVE, 1});
    }
    if (!strcmp(which, "over")) {
        RUN("two arrivals expected, three arrive at once, then one at a time",
            {INIT, 2}, {ARRIVE, 3}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1}, {ARRIVE, 1});
        RUN("one arrival expected, an arrival with count 5",
            {INIT, 1}, {ARRIVE, 5}, {ARRIVE, 1}, {ARRIVE, 1});
    }
    return 0;
}
