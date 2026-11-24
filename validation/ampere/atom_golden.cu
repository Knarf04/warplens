// Host program: run atom_probe on the GPU (4 CTAs of 128 threads) and print what it left in
// memory, one array per line in hexadecimal (floating-point values as their bit patterns).
//
// The first four lines are the values the words started with, which the test feeds to the
// simulator. Two things depend on which thread came last and are printed in a form that
// does not: an exchanged word is added to the sum of everything the exchanges returned (and
// that sum printed as 0), and the old values a counter handed out are sorted.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "host_check.h"
#include "atom_probe.cu"

template <typename T> static void line(const char* label, const T* v, size_t n) {
    printf("%s", label);
    for (size_t k = 0; k < n; ++k) {
        unsigned long long bits = 0;
        memcpy(&bits, v + k, sizeof(T));
        printf(sizeof(T) == 8 ? " %016llx" : " %08llx", bits);
    }
    printf("\n");
}

template <typename T> struct Buffer {
    std::vector<T> host;
    T* device = nullptr;
    explicit Buffer(size_t n) : host(n) { CHECK(cudaMalloc(&device, n * sizeof(T))); }
    void upload() { CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    void download() { CHECK(cudaMemcpy(host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost)); }
};

// An exchanged word `at` and the sum of the exchanges' results at `at + 1`.
template <typename T> static void fold(T* words, int at) {
    words[at] += words[at + 1];
    words[at + 1] = 0;
}

int main() {
    const int ctas = 4, block = 128, n = ctas * block;
    Buffer<u32> g32(2 * N32 + 2 * ctas), ticket(2 * n), s32(ctas * N32);
    Buffer<u64> g64(2 * N64), s64(ctas * N64);
    Buffer<float> gf(2 * NF), sf(ctas * NF);
    Buffer<double> gd(2 * NF), sd(ctas * NF);

    u32 init32[N32] = {0, 1000, 0x80000000u, 5, 0xffffffffu, 0, 0x7fffffffu, 0x80000000u,
                       0xffffffffu, 0, 0x12345678u, 0, 0, 11, 0xabc, 0,
                       0, 0, 0, 0, 0, 0, 1, 0xffffffffu};
    for (int k = 32; k < N32; ++k) init32[k] = k;
    u64 init64[N64] = {0, ~0ull, 0, 0x7fffffffffffffffull, 0x8000000000000000ull, ~0ull, 0,
                       0x0123456789abcdefull, 0, 0x55, 0, ~0ull, 0, 0, 0, 0, 0, ~0ull};
    float initf[NF] = {1.5f, 2.0f, 0.0f, 0.25f};
    double initd[NF] = {1.5, 2.0, 0.0, 0.25};
    for (int k = 0; k < N32; ++k) g32.host[k] = g32.host[N32 + k] = init32[k];
    for (int k = 0; k < N64; ++k) g64.host[k] = g64.host[N64 + k] = init64[k];
    for (int k = 0; k < NF; ++k) {
        gf.host[k] = gf.host[NF + k] = initf[k];
        gd.host[k] = gd.host[NF + k] = initd[k];
    }
    g32.upload(); g64.upload(); gf.upload(); gd.upload();
    CHECK(cudaMemset(ticket.device, 0xff, 2 * n * sizeof(u32)));
    CHECK(cudaMemset(s32.device, 0xff, ctas * N32 * sizeof(u32)));
    CHECK(cudaMemset(s64.device, 0xff, ctas * N64 * sizeof(u64)));
    CHECK(cudaMemset(sf.device, 0xff, ctas * NF * sizeof(float)));
    CHECK(cudaMemset(sd.device, 0xff, ctas * NF * sizeof(double)));

    atom_probe<<<ctas, block>>>(g32.device, g64.device, gf.device, gd.device, ticket.device,
                                s32.device, s64.device, sf.device, sd.device, 1);
    CHECK_LAUNCH();
    g32.download(); g64.download(); gf.download(); gd.download(); ticket.download();
    s32.download(); s64.download(); sf.download(); sd.download();

    line("init32", init32, N32);
    line("init64", init64, N64);
    line("initf", initf, NF);
    line("initd", initd, NF);

    fold(g32.host.data(), 14);
    fold(g64.host.data(), 9);
    fold(gf.host.data(), 1);
    line("g32", g32.host.data(), N32);
    line("g32_cta", g32.host.data() + 2 * N32, 2 * ctas);
    line("g64", g64.host.data(), N64);
    line("gf", gf.host.data(), NF);
    line("gd", gd.host.data(), NF);
    std::vector<u32> tickets(n);
    for (int i = 0; i < n; ++i) tickets[i] = ticket.host[2 * i];
    std::sort(tickets.begin(), tickets.end());
    line("ticket_g", tickets.data(), n);
    for (int b = 0; b < ctas; ++b) {
        fold(s32.host.data() + b * N32, 14);
        fold(s64.host.data() + b * N64, 9);
        fold(s64.host.data() + b * N64, 14);
        fold(sf.host.data() + b * NF, 1);
        line("s32", s32.host.data() + b * N32, N32);
        line("s64", s64.host.data() + b * N64, N64);
        line("sf", sf.host.data() + b * NF, NF);
        line("sd", sd.host.data() + b * NF, NF);
        for (int t = 0; t < block; ++t) tickets[t] = ticket.host[2 * (b * block + t) + 1];
        std::sort(tickets.begin(), tickets.begin() + block);
        line("ticket_s", tickets.data(), block);
    }
    return 0;
}
