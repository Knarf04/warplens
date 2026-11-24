// Host program: record mbarrier_token_probe from the GPU for 1, 2, 3 and 5 arrivals per
// phase, eight phases each. One line per arrival:
//   count step token-bit-63 parity-answers own-token-answers made-up-token-answers
// the last three in hex (see mbarrier_token_probe.cu for the bits).
#include <cstdio>
#include <cuda_runtime.h>
#include "host_check.h"
#include "mbarrier_token_probe.cu"

int main() {
    unsigned long long h_out[4 * 64], *d_out;
    CHECK(cudaMalloc(&d_out, sizeof h_out));
    for (unsigned count : {1u, 2u, 3u, 5u}) {
        int steps = 8 * count;
        CHECK(cudaMemset(d_out, 0xff, sizeof h_out));
        mbarrier_token_probe<<<1, 1>>>(count, steps, d_out);
        CHECK_LAUNCH();
        CHECK(cudaMemcpy(h_out, d_out, sizeof h_out, cudaMemcpyDeviceToHost));
        for (int s = 0; s < steps; ++s)
            printf("%u %d %llu %llx %llx %llx\n", count, s, h_out[4 * s], h_out[4 * s + 1], h_out[4 * s + 2],
                   h_out[4 * s + 3]);
    }
    return 0;
}
