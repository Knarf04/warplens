// What a single mbarrier.try_wait returns, and how long it takes to return.
//
// Real kernels use try_wait only in a loop (`mbar_wait` in ../common.h), which hides what
// one try_wait does on a phase that has not completed. PTX says it may suspend the thread
// and returns false when a system-dependent time limit passes first. The simulator does not
// model that: its try_wait blocks the warp until the phase completes and never returns
// false (warplens/units/mbarrier.py), so that a wait that can never succeed is reported as
// a deadlock. A kernel with a non-blocking fallback, one that does other work when
// try_wait returns false, is outside the v1 scope (PLAN.md, "v1 instruction scope").
//
// One thread, one mbarrier, nothing asynchronous.
//
//   try_wait              completed phases only, so nothing blocks and the simulator
//                         predicts every line: try_wait.parity and try_wait with an arrive
//                         token are true once the phase they name has completed, in three
//                         successive phases.
//   try_wait incomplete   NO PREDICTION (the simulator reports a deadlock at the first
//                         try_wait). On an mbarrier whose phase 0 never completes:
//                           one try_wait.parity 0                  result, clock ticks
//                           16 more of them                        how many false, ticks
//                           one test_wait.parity 0                 ticks, for comparison
//                         then with 2 arrivals per phase, after the first arrival:
//                           one try_wait with that arrival's token result, ticks
//                         then the second arrival completes the phase:
//                           try_wait.parity 0 and try_wait(token)  results, ticks
//                         Expected on hardware: false, false 16 times, false, true, true,
//                         each after a bounded time. The ticks (%clock64) say how long a
//                         failed try_wait suspends the thread; they go on info: lines.
//                         If try_wait never returns, the run ends in the harness timeout.
//
// try_wait needs sm_90 (ptxas for sm_86: "Feature 'mbarrier.try_wait.parity' requires
// .target sm_90 or higher"), so neither case runs on an Ampere GPU.
#include "../common.h"

__device__ __forceinline__ uint64_t mbar_arrive_token(uint64_t* bar) {
    uint64_t token;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(token) : "r"(smem_addr(bar)) : "memory");
    return token;
}
// ONE try_wait, no loop.
__device__ __forceinline__ uint32_t mbar_try_parity(uint64_t* bar, uint32_t parity) {
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "}" : "=r"(done) : "r"(smem_addr(bar)), "r"(parity) : "memory");
    return done;
}
__device__ __forceinline__ uint32_t mbar_try_token(uint64_t* bar, uint64_t token) {
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.try_wait.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "}" : "=r"(done) : "r"(smem_addr(bar)), "l"(token) : "memory");
    return done;
}
__device__ __forceinline__ long long clock_ticks() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

#define PHASES 3
#define REPEATS 16

// results[2 * phase], [2 * phase + 1]: try_wait.parity and try_wait(token) after the phase
// completed. ticks: the same, in clock ticks.
extern "C" __global__ void try_wait_completed(uint32_t* results, long long* ticks) {
    __shared__ uint64_t bar;
    mbar_init(&bar, 1);
    mbar_init_fence();
    for (int phase = 0; phase < PHASES; ++phase) {
        uint64_t token = mbar_arrive_token(&bar);  // completes the phase
        long long t0 = clock_ticks();
        results[2 * phase] = mbar_try_parity(&bar, phase & 1);
        long long t1 = clock_ticks();
        results[2 * phase + 1] = mbar_try_token(&bar, token);
        long long t2 = clock_ticks();
        ticks[2 * phase] = t1 - t0;
        ticks[2 * phase + 1] = t2 - t1;
    }
}

// results / ticks: 0 one try_wait.parity, incomplete; 1 false results of REPEATS more / their
// ticks in total; 2 test_wait.parity; 3 try_wait(token), incomplete; 4 try_wait.parity and
// 5 try_wait(token) after completion.
extern "C" __global__ void try_wait_incomplete(uint32_t* results, long long* ticks) {
    __shared__ uint64_t never, bar;
    mbar_init(&never, 1);
    mbar_init(&bar, 2);
    mbar_init_fence();

    long long t0 = clock_ticks();
    results[0] = mbar_try_parity(&never, 0);
    long long t1 = clock_ticks();
    ticks[0] = t1 - t0;

    uint32_t failed = 0;
    t0 = clock_ticks();
    for (int i = 0; i < REPEATS; ++i) failed += mbar_try_parity(&never, 0) == 0;
    t1 = clock_ticks();
    results[1] = failed;
    ticks[1] = t1 - t0;

    t0 = clock_ticks();
    results[2] = mbar_test(&never, 0);
    t1 = clock_ticks();
    ticks[2] = t1 - t0;

    uint64_t token = mbar_arrive_token(&bar);  // the first of two arrivals: the phase stays open
    t0 = clock_ticks();
    results[3] = mbar_try_token(&bar, token);
    t1 = clock_ticks();
    ticks[3] = t1 - t0;

    mbar_arrive(&bar);  // the second: phase 0 completes
    t0 = clock_ticks();
    results[4] = mbar_try_parity(&bar, 0);
    t1 = clock_ticks();
    results[5] = mbar_try_token(&bar, token);
    long long t2 = clock_ticks();
    ticks[4] = t1 - t0;
    ticks[5] = t2 - t1;
}

int main(int argc, char** argv) {
    int incomplete = argc > 1 && !strcmp(argv[1], "incomplete");
    uint32_t results[8];
    long long ticks[8];
    uint32_t* d_results = dev_alloc<uint32_t>(8);
    long long* d_ticks = dev_alloc<long long>(8);
    dev_fill(d_results, 8, 0xff);
    dev_fill(d_ticks, 8, 0xff);
    if (!incomplete) {
        try_wait_completed<<<1, 1>>>(d_results, d_ticks);
        probe_sync("try_wait_completed");
        dev_read(results, d_results, 8);
        dev_read(ticks, d_ticks, 8);
        printf("mbarrier with 1 arrival per phase; one try_wait of each form after the phase completed\n");
        for (int phase = 0; phase < PHASES; ++phase) {
            printf("phase %d completed: try_wait.parity %d %u, try_wait(token of its arrival) %u\n", phase, phase & 1,
                   results[2 * phase], results[2 * phase + 1]);
            printf("info: phase %d: clock ticks %lld (parity), %lld (token)\n", phase, ticks[2 * phase],
                   ticks[2 * phase + 1]);
        }
        return 0;
    }
    try_wait_incomplete<<<1, 1>>>(d_results, d_ticks);
    probe_sync("try_wait_incomplete");
    dev_read(results, d_results, 8);
    dev_read(ticks, d_ticks, 8);
    printf("phase 0 never completes: one try_wait.parity 0 returned %u\n", results[0]);
    printf("info: it took %lld clock ticks\n", ticks[0]);
    printf("phase 0 never completes: %d more try_wait.parity 0, of which false: %u\n", REPEATS, results[1]);
    printf("info: they took %lld clock ticks in total\n", ticks[1]);
    printf("phase 0 never completes: one test_wait.parity 0 returned %u\n", results[2]);
    printf("info: it took %lld clock ticks\n", ticks[2]);
    printf("2 arrivals per phase, after the first: try_wait(token of that arrival) returned %u\n", results[3]);
    printf("info: it took %lld clock ticks\n", ticks[3]);
    printf("after the second arrival: try_wait.parity 0 returned %u, try_wait(token) returned %u\n", results[4],
           results[5]);
    printf("info: they took %lld and %lld clock ticks\n", ticks[4], ticks[5]);
    return 0;
}
