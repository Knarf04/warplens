// Host program: record f32 arithmetic with the .ftz modifier from the GPU as
// "op a b c result" in hex. Ops: 0 add, 1 sub, 2 mul, 3 div, 4 fma (a*b+c), 5 sqrt(a).
// The operands sit around the subnormal range, including products and sums whose exact value
// is subnormal but rounds to the smallest normal number, with and without an exponent limit.
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

__global__ void ftz_probe(const int* op, const float* a, const float* b, const float* c, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], y = b[i], z = c[i], r = 0.0f;
    switch (op[i]) {
        case 0: asm volatile("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y)); break;
        case 1: asm volatile("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y)); break;
        case 2: asm volatile("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y)); break;
        case 3: asm volatile("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y)); break;
        case 4: asm volatile("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(x), "f"(y), "f"(z)); break;
        default: asm volatile("sqrt.rn.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); break;
    }
    out[i] = r;
}

static float from_bits(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
static unsigned to_bits(float f) { unsigned u; memcpy(&u, &f, 4); return u; }

int main() {
    const unsigned values[] = {
        0x00000000, 0x80000000,              // zeros
        0x00000001, 0x80000001, 0x00400000,  // subnormals
        0x007fffff, 0x807fffff,              // the largest subnormals
        0x00800000, 0x80800000, 0x00800001,  // the smallest normals
        0x00ffffff, 0x01000000, 0x01800000,  // just above
        0x3f7ffffe, 0x3f7fffff, 0x3f800000, 0x3f800001,  // 1 - ulp, 1 - ulp/2, 1, 1 + ulp
        0xbf800000, 0x3f000000, 0x3fc00000,  // -1, 0.5, 1.5
        0x1e3ce508, 0x1a800000, 0x0c000000,  // 1e-20, 2^-74, 2^-103
        0x7f800000, 0x7fc00000,              // infinity, NaN
    };
    const unsigned addends[] = {0x00000000, 0x80000000, 0x007fffff, 0x00800000, 0x80800000, 0x00000001};
    std::vector<int> op;
    std::vector<float> a, b, c;
    auto push = [&](int o, unsigned x, unsigned y, unsigned z) {
        op.push_back(o); a.push_back(from_bits(x)); b.push_back(from_bits(y)); c.push_back(from_bits(z));
    };
    for (unsigned x : values)
        for (unsigned y : values) {
            for (int o = 0; o < 4; ++o) push(o, x, y, 0);
            for (unsigned z : addends) push(4, x, y, z);
        }
    for (unsigned x : values) push(5, x, 0, 0);
    unsigned state = 7u;
    for (int k = 0; k < 400; ++k) {  // random operands with small exponents
        state = state * 1664525u + 1013904223u; unsigned x = state & 0x81ffffffu;
        state = state * 1664525u + 1013904223u; unsigned y = (state & 0x80ffffffu) | 0x3f000000u;
        state = state * 1664525u + 1013904223u; unsigned z = state & 0x80ffffffu;
        push(k % 5, x, y, z);
        push(5, x & 0x7fffffffu, 0, 0);
    }
    int n = (int)op.size();
    std::vector<float> out(n);
    int* dop; float *da, *db, *dc, *dout;
    cudaMalloc(&dop, n * 4); cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dc, n * 4);
    cudaMalloc(&dout, n * 4);
    cudaMemcpy(dop, op.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice);
    ftz_probe<<<(n + 255) / 256, 256>>>(dop, da, db, dc, dout, n);
    if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "kernel failed\n"); return 1; }
    cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        printf("%d %08x %08x %08x %08x\n", op[i], to_bits(a[i]), to_bits(b[i]), to_bits(c[i]), to_bits(out[i]));
    return 0;
}
