// Static shared memory above the 48 KiB default: does a kernel that has it launch without
// the opt-in, and how much dynamic shared memory may it then use?
//
// The kernels declare 48 KiB + 16, 98 KiB and 227 KiB of static shared memory (98 KiB is what
// validation/blackwell_timing/pipelined_gemm.cu has, which never calls cudaFuncSetAttribute).
// The program launches each without any opt-in, with no dynamic shared memory and with
// sizes on both sides of what is left of 227 KiB, and prints whether each launch ran and
// what cudaFuncGetAttributes reports. e2e/smem_optin has the kernels under the default.
//
// Claim tested (warplens/units/smem.py `default_max_dynamic`): the rule an RTX 3080 Ti
// follows at its own limits (validation/README.md, "The shared-memory opt-in"). A kernel
// whose static shared memory is more than the default needs no opt-in, and without one may
// use all the dynamic shared memory up to the opt-in maximum less its static shared memory;
// setting the attribute still replaces that number.
//
// Not a simulator claim, and the reason these kernels have a program of their own: whether
// the driver compiles them. ptxas 13.5 accepts up to 227 KiB of static shared memory for
// sm_100a and 48 KiB for sm_100; a driver that refuses shows as LAUNCH FAILED, the load
// error on an info: line, and nothing here is then known about launching.
//
// A refused request is an error RETURNED by a call (the context stays usable); its name goes
// on an info: line. A launch that is accepted and then fails ends the program (KERNEL
// FAILED).
//
//   smem_static
#include "../common.h"

// Every thread stores what another one loads after a barrier: the compiler removes a static
// array that only one thread uses, and the kernel would have no static shared memory.
#define WITH_STATIC(name, size)                                                    \
    extern "C" __global__ void name(uint32_t bytes, uint32_t* out) {              \
        __shared__ uint8_t fixed[size];                                            \
        extern __shared__ uint8_t dynamic[];                                       \
        fixed[threadIdx.x] = (uint8_t)(bytes + threadIdx.x);                       \
        fixed[size - 32 + threadIdx.x] = 0xA5; /* the block has 32 threads */       \
        __syncthreads();                                                           \
        if (threadIdx.x == 0) {                                                    \
            if (bytes) dynamic[bytes - 1] = fixed[size - 2];                       \
            out[0] = bytes ? dynamic[bytes - 1] : fixed[size - 2];                 \
        }                                                                          \
    }

WITH_STATIC(static_just_over, 48 * 1024 + 16)
WITH_STATIC(static_98k, 98 * 1024)
WITH_STATIC(static_98k_set, 98 * 1024)
WITH_STATIC(static_all, 227 * 1024)

static uint32_t* d_out;

template <class K> static void attributes(const char* label, K kernel) {
    cudaFuncAttributes a = {};
    cudaError_t status = cudaFuncGetAttributes(&a, (const void*)kernel);
    (void)cudaGetLastError();
    if (status != cudaSuccess) {
        if (probe_is_load_error(status)) probe_report_failure(label, status, "at cudaFuncGetAttributes");
        printf("%s: no attributes\n", label);
        printf("info: %s: cudaFuncGetAttributes returned %s\n", label, cudaGetErrorName(status));
        return;
    }
    printf("%s: %zu static bytes, at most %d dynamic\n", label, a.sharedSizeBytes, a.maxDynamicSharedSizeBytes);
}

template <class K> static void set_maximum(const char* label, K kernel, int bytes) {
    cudaError_t status = cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    (void)cudaGetLastError();
    if (probe_is_load_error(status)) probe_report_failure(label, status, "at cudaFuncSetAttribute");
    printf("%s: maximum of %d %s\n", label, bytes, status == cudaSuccess ? "accepted" : "refused");
    if (status != cudaSuccess) printf("info: %s: cudaFuncSetAttribute returned %s\n", label, cudaGetErrorName(status));
}

template <class K> static void launch(const char* label, K kernel, uint32_t bytes) {
    char what[160];
    snprintf(what, sizeof what, "%s, %u dynamic bytes", label, bytes);
    dev_fill(d_out, 1, 0);
    kernel<<<1, 32, bytes>>>(bytes, d_out);
    cudaError_t at_launch = cudaGetLastError();
    if (at_launch != cudaSuccess) {
        if (probe_is_load_error(at_launch)) probe_report_failure(what, at_launch, "at launch");
        cudaError_t at_sync = cudaDeviceSynchronize();
        printf("%s: launch refused\n", what);
        printf("info: %s: the launch returned %s; cudaDeviceSynchronize then %s\n", what, cudaGetErrorName(at_launch),
               cudaGetErrorName(at_sync));
        (void)cudaGetLastError();
        return;
    }
    probe_sync(what);
    uint32_t out = 0;
    dev_read(&out, d_out, 1);
    printf("%s: launched; last byte %02x\n", what, out);
    fflush(stdout);
}

int main() {
    cudaDeviceProp prop;
    HOST_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("info: sharedMemPerBlock %zu, sharedMemPerBlockOptin %zu\n", prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
    d_out = dev_alloc<uint32_t>(1);
    const uint32_t most = 227u * 1024u;

    attributes("48 KiB + 16 static, not opted in", static_just_over);
    launch("48 KiB + 16 static, not opted in", static_just_over, 0u);
    launch("48 KiB + 16 static, not opted in", static_just_over, 16u);
    launch("48 KiB + 16 static, not opted in", static_just_over, most - (48u * 1024u + 16u));
    launch("48 KiB + 16 static, not opted in", static_just_over, most - (48u * 1024u + 16u) + 16u);

    attributes("98 KiB static, not opted in", static_98k);
    launch("98 KiB static, not opted in", static_98k, 0u);
    launch("98 KiB static, not opted in", static_98k, 16u);
    launch("98 KiB static, not opted in", static_98k, most - 98u * 1024u);
    launch("98 KiB static, not opted in", static_98k, most - 98u * 1024u + 16u);

    // The attribute still replaces the number, and is refused above the rest of the maximum.
    set_maximum("98 KiB static", static_98k_set, (int)(most - 98u * 1024u) + 16);
    set_maximum("98 KiB static", static_98k_set, 0);
    attributes("98 KiB static, maximum set to 0", static_98k_set);
    launch("98 KiB static, maximum set to 0", static_98k_set, 0u);
    launch("98 KiB static, maximum set to 0", static_98k_set, 16u);

    attributes("227 KiB static, not opted in", static_all);
    launch("227 KiB static, not opted in", static_all, 0u);
    launch("227 KiB static, not opted in", static_all, 16u);
    return 0;
}
