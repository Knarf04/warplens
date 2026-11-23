// Host program: record prmt.b32 from the GPU as "a b c result" in hex.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void prmt_probe(const unsigned* a, const unsigned* b, const unsigned* c, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned r;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a[i]), "r"(b[i]), "r"(c[i]));
    out[i] = r;
}

int main() {
    std::vector<unsigned> a, b, c;
    const unsigned controls[] = {0x3210, 0x7654, 0x0123, 0x4567, 0x7773, 0xba98, 0xfedc, 0x8888,
                                 0xffff, 0x0000, 0x1c0b, 0xabcd3210, 0x00013210};
    unsigned state = 99u;
    for (unsigned ctl : controls)
        for (int k = 0; k < 8; ++k) {
            state = state * 1664525u + 1013904223u; a.push_back(state);
            state = state * 1664525u + 1013904223u; b.push_back(state);
            c.push_back(ctl);
        }
    for (int k = 0; k < 200; ++k) {
        state = state * 1664525u + 1013904223u; a.push_back(state);
        state = state * 1664525u + 1013904223u; b.push_back(state);
        state = state * 1664525u + 1013904223u; c.push_back(state);
    }
    int n = (int)a.size();
    std::vector<unsigned> out(n);
    unsigned *da, *db, *dc, *dout;
    cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dc, n * 4); cudaMalloc(&dout, n * 4);
    cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice);
    prmt_probe<<<(n + 255) / 256, 256>>>(da, db, dc, dout, n);
    if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
    cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) printf("%x %x %x %x\n", a[i], b[i], c[i], out[i]);
    return 0;
}
