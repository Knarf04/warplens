// Host program: run tf32_probe on the GPU over rounding boundaries and special
// values, and print "input_bits output_bits" in hex, one pair per line.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "tf32_probe.cu"

int main() {
    std::vector<unsigned> in;
    const unsigned exponents[] = {0x00000000u, 0x00800000u, 0x3F800000u, 0x40000000u, 0x7F000000u};
    const unsigned low[] = {0x0000, 0x0001, 0x0FFF, 0x1000, 0x1001, 0x1FFF};  // the 13 discarded bits
    const unsigned kept[] = {0x000, 0x001, 0x155, 0x3FE, 0x3FF};              // the 10 kept bits
    for (unsigned sign = 0; sign < 2; ++sign)
        for (unsigned e : exponents)
            for (unsigned k : kept)
                for (unsigned l : low) in.push_back((sign << 31) | e | (k << 13) | l);
    const unsigned special[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u,
                                0x7FA12345u, 0xFFFFFFFFu, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F7FF000u,
                                0x7F7FEFFFu, 0x00000FFFu, 0x00001000u, 0x007FFFFFu, 0x807FF000u};
    for (unsigned s : special) in.push_back(s);
    unsigned state = 12345u;  // deterministic pseudo-random bit patterns
    for (int i = 0; i < 400; ++i) { state = state * 1664525u + 1013904223u; in.push_back(state); }

    int n = (int)in.size();
    std::vector<unsigned> out(n);
    unsigned *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, n * 4)); CHECK(cudaMalloc(&d_out, n * 4));
    CHECK(cudaMemcpy(d_in, in.data(), n * 4, cudaMemcpyHostToDevice));
    tf32_probe<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    CHECK_LAUNCH();
    CHECK(cudaMemcpy(out.data(), d_out, n * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) printf("%08x %08x\n", in[i], out[i]);
    return 0;
}
