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
// With static shared memory. Every thread stores what another one loads after a barrier:
// an array that only one thread uses is removed by the compiler (the kernel then has no
// static shared memory at all, which `sharedSizeBytes` of cudaFuncGetAttributes shows).
#define WITH_STATIC(name, size) \
    __global__ void name(uint32_t bytes, uint32_t* out) { \
        __shared__ uint8_t fixed[size]; \
        extern __shared__ uint8_t dynamic[]; \
        fixed[threadIdx.x] = (uint8_t)(bytes + threadIdx.x); \
        fixed[size - 1 - threadIdx.x] = 1; \
        __syncthreads(); \
        if (threadIdx.x == 0) { if (bytes) dynamic[bytes - 1] = 0xA5; out[0] = fixed[1] + fixed[size - 2]; } \
    }
WITH_STATIC(with_static, 1024)

// For the opt-in rule (the cases after "launch, ordinary, afterwards"): kernels that
// nothing above has touched.
#define DYNAMIC_ONLY(name) \
    __global__ void name(uint32_t bytes, uint32_t* out) { \
        extern __shared__ uint8_t dynamic[]; \
        if (threadIdx.x == 0 && bytes) { dynamic[0] = 0x5A; dynamic[bytes - 1] = 0xA5; out[0] = dynamic[0] + dynamic[bytes - 1]; } \
    }
DYNAMIC_ONLY(fresh_a)
DYNAMIC_ONLY(fresh_b)
DYNAMIC_ONLY(fresh_c)
// 40960 static bytes are under the default of 49152 by themselves and over it with 8208
// dynamic ones; 61440 are over it by themselves.
WITH_STATIC(static_40k, 40960)
WITH_STATIC(static_40k_b, 40960)
WITH_STATIC(static_60k, 61440)
WITH_STATIC(static_60k_b, 61440)
WITH_STATIC(static_60k_c, 61440)
WITH_STATIC(static_48k, 49152)     // exactly the default
WITH_STATIC(static_48k_16, 49168)  // the default + 16
WITH_STATIC(static_99k, 101376)    // exactly the opt-in maximum

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

static void not_a_kernel() {}

template <class K> static void attributes(const char* what, K kernel) {
    cudaFuncAttributes a = {};
    cudaError_t ret = cudaFuncGetAttributes(&a, (const void*)kernel);
    printf("%-58s %s: sharedSizeBytes %zu, maxDynamicSharedSizeBytes %d\n", what, cudaGetErrorName(ret), a.sharedSizeBytes,
           a.maxDynamicSharedSizeBytes);
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

    // The opt-in rule: what a kernel reports before and after, whether the attribute of one
    // kernel reaches another, whether a small attribute lowers the default, and how static
    // shared memory counts.
    attributes("attributes, fresh kernel", fresh_a);
    attributes("attributes, kernel opted in to the maximum above", touch);
    launch("launch, fresh kernel, default maximum", fresh_a, 1, 32, plain, out);
    launch("launch, fresh kernel, default maximum + 1", fresh_a, 1, 32, plain + 1, out);
    launch("launch, fresh kernel, default maximum + 16", fresh_a, 1, 32, plain + 16, out);
    report("cudaFuncSetAttribute(fresh, 60000)", cudaFuncSetAttribute(fresh_a, cudaFuncAttributeMaxDynamicSharedMemorySize, 60000));
    attributes("attributes, opted in to 60000", fresh_a);
    launch("launch, opted in to 60000, 60000", fresh_a, 1, 32, 60000, out);
    launch("launch, opted in to 60000, 60001", fresh_a, 1, 32, 60001, out);
    launch("launch, opted in to 60000, 60016", fresh_a, 1, 32, 60016, out);
    launch("launch, another fresh kernel, default maximum + 16", fresh_b, 1, 32, plain + 16, out);
    report("cudaFuncSetAttribute(fresh, 1024)", cudaFuncSetAttribute(fresh_c, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024));
    attributes("attributes, set to 1024", fresh_c);
    launch("launch, set to 1024, 1024", fresh_c, 1, 32, 1024, out);
    launch("launch, set to 1024, 1040", fresh_c, 1, 32, 1040, out);
    launch("launch, set to 1024, default maximum", fresh_c, 1, 32, plain, out);
    launch("launch, set to 1024, default maximum + 16", fresh_c, 1, 32, plain + 16, out);
    report("cudaFuncSetAttribute(the same, 60000)", cudaFuncSetAttribute(fresh_c, cudaFuncAttributeMaxDynamicSharedMemorySize, 60000));
    report("cudaFuncSetAttribute(the same, 0)", cudaFuncSetAttribute(fresh_c, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    attributes("attributes, set to 60000 and then to 0", fresh_c);
    launch("launch, set to 60000 and then to 0, default maximum", fresh_c, 1, 32, plain, out);
    launch("launch, set to 60000 and then to 0, default maximum + 16", fresh_c, 1, 32, plain + 16, out);

    attributes("attributes, 40960 static", static_40k);
    launch("launch, 40960 static + 0, no opt-in", static_40k, 1, 32, 0, out);
    launch("launch, 40960 static + 8192 (default maximum), no opt-in", static_40k, 1, 32, plain - 40960, out);
    launch("launch, 40960 static + 8208, no opt-in", static_40k, 1, 32, plain - 40960 + 16, out);
    report("cudaFuncSetAttribute(40960 static, 8208)", cudaFuncSetAttribute(static_40k, cudaFuncAttributeMaxDynamicSharedMemorySize, 8208));
    attributes("attributes, 40960 static, set to 8208", static_40k);
    launch("launch, 40960 static + 8208, set to 8208", static_40k, 1, 32, 8208, out);
    launch("launch, 40960 static + 8224, set to 8208", static_40k, 1, 32, 8224, out);
    report("cudaFuncSetAttribute(40960 static (another), 1024)", cudaFuncSetAttribute(static_40k_b, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024));
    launch("launch, 40960 static + 1024, set to 1024", static_40k_b, 1, 32, 1024, out);
    launch("launch, 40960 static + 8192, set to 1024", static_40k_b, 1, 32, 8192, out);
    launch("launch, 40960 static + 8208, set to 1024", static_40k_b, 1, 32, 8208, out);

    attributes("attributes, 61440 static", static_60k);
    launch("launch, 61440 static + 0, no opt-in", static_60k, 1, 32, 0, out);
    launch("launch, 61440 static + 16, no opt-in", static_60k, 1, 32, 16, out);
    launch("launch, 61440 static + 39936 (opt-in maximum), no opt-in", static_60k, 1, 32, optin - 61440, out);
    launch("launch, 61440 static + 39952, no opt-in", static_60k, 1, 32, optin - 61440 + 16, out);
    report("cudaFuncSetAttribute(61440 static (another), 0)", cudaFuncSetAttribute(static_60k_b, cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    attributes("attributes, 61440 static, set to 0", static_60k_b);
    launch("launch, 61440 static + 0, set to 0", static_60k_b, 1, 32, 0, out);
    launch("launch, 61440 static + 16, set to 0", static_60k_b, 1, 32, 16, out);
    report("cudaFuncSetAttribute(61440 static (a third), 1024)", cudaFuncSetAttribute(static_60k_c, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024));
    attributes("attributes, 61440 static, set to 1024", static_60k_c);
    launch("launch, 61440 static + 1024, set to 1024", static_60k_c, 1, 32, 1024, out);
    launch("launch, 61440 static + 1040, set to 1024", static_60k_c, 1, 32, 1040, out);
    report("cudaFuncSetAttribute(61440 static (a third), 39936)", cudaFuncSetAttribute(static_60k_c, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 61440));
    report("cudaFuncSetAttribute(61440 static (a third), 39937)", cudaFuncSetAttribute(static_60k_c, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 61440 + 1));
    attributes("attributes, 61440 static, after the refused 39937", static_60k_c);

    // Where the default changes: static shared memory of exactly the default, and 16 more.
    attributes("attributes, 49152 static", static_48k);
    launch("launch, 49152 static + 0, no opt-in", static_48k, 1, 32, 0, out);
    launch("launch, 49152 static + 16, no opt-in", static_48k, 1, 32, 16, out);
    attributes("attributes, 49168 static", static_48k_16);
    launch("launch, 49168 static + 0, no opt-in", static_48k_16, 1, 32, 0, out);
    launch("launch, 49168 static + 16, no opt-in", static_48k_16, 1, 32, 16, out);
    launch("launch, 49168 static + 52208 (opt-in maximum), no opt-in", static_48k_16, 1, 32, optin - 49168, out);
    launch("launch, 49168 static + 52224, no opt-in", static_48k_16, 1, 32, optin - 49168 + 16, out);
    attributes("attributes, 101376 static", static_99k);
    launch("launch, 101376 static + 0, no opt-in", static_99k, 1, 32, 0, out);
    launch("launch, 101376 static + 16, no opt-in", static_99k, 1, 32, 16, out);
    // A function that is no kernel.
    report("cudaFuncSetAttribute(a host function, 1024)", cudaFuncSetAttribute((const void*)not_a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024));
    attributes("attributes, a host function", not_a_kernel);
    (void)cudaGetLastError();
    launch("launch, ordinary, at the end", touch, 4, 32, 64, out);
    return 0;
}
