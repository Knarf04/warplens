// The token mbarrier.arrive returns, and mbarrier.test_wait with a token.
//
// One thread, one mbarrier initialised with two arrivals per phase. For three phases:
//   a = arrive            keep the token; test_wait(a)          (the phase is still open)
//                         test_wait(a of the phase before)      (that phase has completed)
//   b = arrive            the phase completes; test_wait(a), test_wait(b)
// test_wait never blocks and nothing is asynchronous, so the output is deterministic.
//
// Claims tested (docs/m2.md, "mbarrier"; warplens/units/mbarrier.py `_mbarrier`):
//   * arrive returns a token for the phase it arrived in;
//   * test_wait with that token is false while that phase is open and true once it has
//     completed, also when asked during the next phase.
// The token VALUES are the simulator's own (the phase number: 0, 0, 1, 1, 2, 2); hardware
// returns something opaque, so they are printed on info: lines and never compared.
//
//   arrive_token          the above
//   arrive_token stale    also test_wait(a of the phase before) AFTER the current phase has
//                         completed, i.e. with a token that is two phases old. The simulator
//                         reports invalid use: PTX only permits a token from the current or
//                         immediately preceding phase. The RTX 3080 Ti's measured false
//                         result and the simulator's deterministic true fallback are both
//                         informational, never a portable hardware prediction.
//
// The forms used exist since sm_80, so the Ampere build (build.sh --ampere) runs this on
// the development machine's GPU.
#include "../common.h"

__device__ __forceinline__ uint64_t mbar_arrive_token(uint64_t* bar) {
    uint64_t token;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(token) : "r"(smem_addr(bar)) : "memory");
    return token;
}
__device__ __forceinline__ uint32_t mbar_test_token(uint64_t* bar, uint64_t token) {
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.test_wait.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "}" : "=r"(done) : "r"(smem_addr(bar)), "l"(token) : "memory");
    return done;
}

#define PHASES 3

// results, per phase: test(a) after a; test(a) after b; test(b) after b;
//                     test(previous a) after a, or 9; test(previous a) after b, or 9
extern "C" __global__ void arrive_token(uint64_t* tokens, uint32_t* results, int stale) {
    __shared__ uint64_t bar;
    mbar_init(&bar, 2);
    mbar_init_fence();
    uint64_t previous = 0;
    for (int phase = 0; phase < PHASES; ++phase) {
        uint64_t a = mbar_arrive_token(&bar);
        results[phase * 5] = mbar_test_token(&bar, a);
        results[phase * 5 + 3] = phase ? mbar_test_token(&bar, previous) : 9u;
        uint64_t b = mbar_arrive_token(&bar);
        results[phase * 5 + 1] = mbar_test_token(&bar, a);
        results[phase * 5 + 2] = mbar_test_token(&bar, b);
        results[phase * 5 + 4] = phase && stale ? mbar_test_token(&bar, previous) : 9u;
        tokens[phase * 2] = a;
        tokens[phase * 2 + 1] = b;
        previous = a;
    }
}

int main(int argc, char** argv) {
    int stale = argc > 1 && !strcmp(argv[1], "stale");
    uint64_t tokens[PHASES * 2];
    uint32_t results[PHASES * 5];
    uint64_t* d_tokens = dev_alloc<uint64_t>(PHASES * 2);
    uint32_t* d_results = dev_alloc<uint32_t>(PHASES * 5);
    dev_fill(d_tokens, PHASES * 2, 0xff);
    dev_fill(d_results, PHASES * 5, 0xff);
    arrive_token<<<1, 1>>>(d_tokens, d_results, stale);
    probe_sync("arrive_token");
    dev_read(tokens, d_tokens, PHASES * 2);
    dev_read(results, d_results, PHASES * 5);
    printf("mbarrier with 2 arrivals per phase; a and b are the tokens of its two arrivals\n");
    for (int phase = 0; phase < PHASES; ++phase) {
        const uint32_t* r = results + phase * 5;
        printf("phase %d: after the first arrival: test_wait(a) %u", phase, r[0]);
        if (phase) printf(", test_wait(a of phase %d) %u", phase - 1, r[3]);
        printf("; after the second arrival: test_wait(a) %u, test_wait(b) %u", r[1], r[2]);
        printf("\n");
        if (phase && stale)
            printf("info: phase %d: test_wait(a of phase %d, two phases old) %u (outside PTX token lifetime)\n",
                   phase, phase - 1, r[4]);
        printf("info: phase %d tokens a = %016llx, b = %016llx\n", phase, (unsigned long long)tokens[phase * 2],
               (unsigned long long)tokens[phase * 2 + 1]);
    }
    return 0;
}
