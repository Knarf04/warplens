// Host program: what the CUDA runtime returns for requests it refuses before any kernel
// runs, and where the error shows up afterwards. For each case one line: what the call
// returned, cudaPeekAtLastError, cudaDeviceSynchronize, cudaGetLastError, and
// cudaGetLastError again. Recorded in launch_errors_3080ti.txt (RTX 3080 Ti, CUDA 12.4):
//   bin/launch_errors_hw > launch_errors_3080ti.txt
// The limits are this GPU's (48 KB of shared memory by default, 99 KB with opt-in, no
// clusters); warplens/shim returns the same codes at the simulated device's limits.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void touch(uint32_t bytes, uint32_t* out) {
    extern __shared__ uint8_t dynamic[];
    if (threadIdx.x == 0 && bytes) { dynamic[0] = 0x5A; dynamic[bytes - 1] = 0xA5; out[0] = dynamic[0] + dynamic[bytes - 1]; }
}
__global__ void touch2(uint32_t bytes, uint32_t* out) {
    extern __shared__ uint8_t dynamic[];
    if (threadIdx.x == 0 && bytes) { dynamic[0] = 0x5A; dynamic[bytes - 1] = 0xA5; out[0] = dynamic[0] + dynamic[bytes - 1]; }
}
__global__ void with_static(uint32_t bytes, uint32_t* out) {
    __shared__ uint8_t fixed[1024];
    extern __shared__ uint8_t dynamic[];
    if (threadIdx.x == 0) { fixed[5] = 1; if (bytes) dynamic[bytes - 1] = 0xA5; out[0] = fixed[5]; }
}

static void report(const char* what, cudaError_t ret) {
    cudaError_t peek = cudaPeekAtLastError();
    cudaError_t sync = cudaDeviceSynchronize();
    cudaError_t last = cudaGetLastError();
    cudaError_t again = cudaGetLastError();
    printf("%-58s returned %-30s peek %-30s sync %-14s getLast %-30s again %s\n", what, cudaGetErrorName(ret),
           cudaGetErrorName(peek), cudaGetErrorName(sync), cudaGetErrorName(last), cudaGetErrorName(again));
}

template <class K> static void launch(const char* what, K kernel, dim3 grid, dim3 block, size_t smem, uint32_t* out) {
    uint32_t bytes = (uint32_t)smem;
    void* args[] = {&bytes, &out};
    report(what, cudaLaunchKernel((const void*)kernel, grid, block, args, smem, 0));
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int optin = (int)prop.sharedMemPerBlockOptin, plain = (int)prop.sharedMemPerBlock;
    printf("%s: sharedMemPerBlock %d, sharedMemPerBlockOptin %d, sharedMemPerMultiprocessor %zu\n", prop.name, plain, optin,
           prop.sharedMemPerMultiprocessor);
    uint32_t* out; cudaMalloc(&out, 64);
    char what[128];

    launch("launch, no opt-in, default maximum", touch2, 1, 32, plain, out);
    launch("launch, no opt-in, default maximum + 16", touch2, 1, 32, plain + 16, out);
    launch("launch, no opt-in, opt-in maximum", touch2, 1, 32, optin, out);

    for (int value : {plain, optin - 16, optin, optin + 1, optin + 16, optin + 1024, 1 << 30, -1, 0}) {
        snprintf(what, sizeof what, "cudaFuncSetAttribute(MaxDynamicSharedMemorySize, %d)", value);
        report(what, cudaFuncSetAttribute(touch, cudaFuncAttributeMaxDynamicSharedMemorySize, value));
    }
    for (int value : {50, 100, 101, -1, -2}) {
        snprintf(what, sizeof what, "cudaFuncSetAttribute(PreferredSharedMemoryCarveout, %d)", value);
        report(what, cudaFuncSetAttribute(touch, cudaFuncAttributePreferredSharedMemoryCarveout, value));
    }
    report("cudaFuncSetAttribute(attribute 99, 0)", cudaFuncSetAttribute(touch, (cudaFuncAttribute)99, 0));

    report("cudaFuncSetAttribute(opt-in maximum)", cudaFuncSetAttribute(touch, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
    launch("launch, opted in, opt-in maximum", touch, 1, 32, optin, out);
    launch("launch, opted in, opt-in maximum + 16", touch, 1, 32, optin + 16, out);
    launch("launch, opted in, opt-in maximum + 1024", touch, 1, 32, optin + 1024, out);
    report("cudaFuncSetAttribute(with_static, opt-in maximum)", cudaFuncSetAttribute(with_static, cudaFuncAttributeMaxDynamicSharedMemorySize, optin));
    report("cudaFuncSetAttribute(with_static, opt-in maximum - 1024)", cudaFuncSetAttribute(with_static, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 1024));
    launch("launch, 1024 static + opt-in maximum - 1024", with_static, 1, 32, optin - 1024, out);
    launch("launch, 1024 static + opt-in maximum - 1008", with_static, 1, 32, optin - 1008, out);

    launch("launch, block of 1025 threads", touch, 1, 1025, 0, out);
    launch("launch, block (33, 32, 1)", touch, 1, dim3(33, 32, 1), 0, out);
    launch("launch, block (1, 1, 65)", touch, 1, dim3(1, 1, 65), 0, out);
    launch("launch, block (0, 1, 1)", touch, 1, dim3(0, 1, 1), 0, out);
    launch("launch, grid (0, 1, 1)", touch, dim3(0, 1, 1), 32, 0, out);
    launch("launch, grid (1, 65536, 1)", touch, dim3(1, 65536, 1), 32, 0, out);
    launch("launch, ordinary", touch, 4, 32, 64, out);

    for (int k : {1, 2, 3}) {
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(k == 3 ? 3 : 4, 1, 1); config.blockDim = dim3(32, 1, 1); config.dynamicSmemBytes = 64;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = k == 1 ? 1 : 2; attribute[0].val.clusterDim.y = 1; attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute; config.numAttrs = 1;
        snprintf(what, sizeof what, "cudaLaunchKernelEx, grid %d, cluster (%d, 1, 1)", config.gridDim.x, attribute[0].val.clusterDim.x);
        report(what, cudaLaunchKernelEx(&config, touch, 64u, out));
    }
    launch("launch, ordinary, afterwards", touch, 4, 32, 64, out);
    return 0;
}
