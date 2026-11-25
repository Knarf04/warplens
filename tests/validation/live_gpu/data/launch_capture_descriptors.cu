// A stand-in, for a GPU without TMA, for a kernel that is handed tensor maps by value
// (tests/validation/live_gpu/test_launch_capture_on_gpu.py; matrix row RT-11). Each `Map` is 128 bytes, 64-byte
// aligned, as a CUtensorMap is, and holds the address of device memory the kernel reads:
//
//   plain      the address in its first 8 bytes
//   unaligned  the address 3 bytes in, at no 8-byte boundary
//   packed     the address shifted right by 4 bits, as a field of a packed layout might be:
//              no 8 bytes of it are the address
//
// The kernel adds the three arrays into `out`. The capture must find the first two addresses
// by scanning, and must still hold the third array's memory, because every live allocation
// is read when a launch has a parameter this large.
//
//   nvcc -arch=sm_86 -o launch_capture_descriptors launch_capture_descriptors.cu
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct alignas(64) Map {
    unsigned char bytes[128];
};

__device__ const float* address(const Map& map, int at) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(map.bytes[at + i]) << (8 * i);
    return reinterpret_cast<const float*>(value);
}

extern "C" __global__ void add_three(int n, Map plain, Map unaligned, Map packed, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* a = address(plain, 0);
    const float* b = address(unaligned, 3);
    const float* c = reinterpret_cast<const float*>(reinterpret_cast<uint64_t>(address(packed, 8)) << 4);
    out[i] = a[i] + b[i] + c[i];
}

#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            printf("HOST ERROR: %s: %s\n", #call, cudaGetErrorString(status));  \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static Map with_address(uint64_t value, int at) {
    Map map;
    memset(map.bytes, 0xA5, sizeof map.bytes);   // no other 8 bytes of it are an address
    memcpy(map.bytes + at, &value, 8);
    return map;
}

int main() {
    const int n = 1000;
    float *a, *b, *c, *out;
    std::vector<float> host(n), result(n);
    CHECK(cudaMalloc(&a, n * sizeof(float)));
    CHECK(cudaMalloc(&b, n * sizeof(float)));
    CHECK(cudaMalloc(&c, n * sizeof(float)));
    CHECK(cudaMalloc(&out, n * sizeof(float)));
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < n; ++i) host[i] = float((k + 1) * 1000 + i);    // 1000+i, 2000+i, 3000+i
        CHECK(cudaMemcpy(k == 0 ? a : k == 1 ? b : c, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemset(out, 0, n * sizeof(float)));
    add_three<<<(n + 255) / 256, 256>>>(n, with_address(reinterpret_cast<uint64_t>(a), 0),
                                        with_address(reinterpret_cast<uint64_t>(b), 3),
                                        with_address(reinterpret_cast<uint64_t>(c) >> 4, 8), out);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(result.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (result[i] != 6000.0f + 3 * i) {
            printf("FAILED: out[%d] = %g\n", i, result[i]);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
