// The shared-memory opt-in: how much dynamic shared memory a launch may use before and after
// cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n).
//
// Each kernel writes the last byte of its dynamic shared memory and reads it back. The
// program launches kernels with sizes on both sides of what they may use and prints, for
// each launch, whether it ran; and what cudaFuncGetAttributes reports as the kernel's
// maximum. Kernels differ only in name (each has its own maximum) and in their static shared
// memory, which here stays under the default; e2e/smem_static has the kernels over it, which
// a driver may refuse to compile, in a program of their own.
//
// Claim tested (warplens/units/smem.py `default_max_dynamic`, applied by
// warplens/host/shim_server.py): the rule an RTX 3080 Ti follows at its own limits
// (validation/README.md, "The shared-memory opt-in"), at the limits the simulated device
// reports. A kernel may use the default (48 KiB) less its static shared memory in dynamic
// shared memory; setting the attribute replaces that number, also by a smaller one, and is
// refused above the opt-in maximum (227 KiB) less the static shared memory; the number
// belongs to one kernel; a launch above it is refused by the launch call.
//
// A refused request is an error RETURNED by a call (the context stays usable); its name goes
// on an info: line. A launch that is accepted and then fails ends the program (KERNEL
// FAILED), and an error that says the kernels could not be loaded is LAUNCH FAILED, as in
// every probe.
//
//   smem_optin
#include "../common.h"

#define DYNAMIC_ONLY(name)                                                         \
    extern "C" __global__ void name(uint32_t bytes, uint32_t* out) {              \
        extern __shared__ uint8_t dynamic[];                                       \
        if (threadIdx.x == 0) {                                                    \
            if (bytes) dynamic[bytes - 1] = 0xA5;                                  \
            out[0] = bytes ? dynamic[bytes - 1] : 0xA5;                            \
        }                                                                          \
    }
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

DYNAMIC_ONLY(optin_plain)
DYNAMIC_ONLY(optin_raised)
DYNAMIC_ONLY(optin_other)
DYNAMIC_ONLY(optin_lowered)
WITH_STATIC(optin_static, 40960)

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
    const int plain = (int)prop.sharedMemPerBlock, most = (int)prop.sharedMemPerBlockOptin;
    printf("info: sharedMemPerBlock %d, sharedMemPerBlockOptin %d\n", plain, most);
    d_out = dev_alloc<uint32_t>(1);

    // Without opt-in: the default, and no more.
    attributes("not opted in", optin_plain);
    launch("not opted in", optin_plain, 48u * 1024u);
    launch("not opted in", optin_plain, 48u * 1024u + 16u);
    launch("not opted in", optin_plain, 227u * 1024u);

    // Opted in to 64 KiB: exactly that much, and for this kernel only.
    set_maximum("opted in to 64 KiB", optin_raised, 64 * 1024);
    attributes("opted in to 64 KiB", optin_raised);
    launch("opted in to 64 KiB", optin_raised, 64u * 1024u);
    launch("opted in to 64 KiB", optin_raised, 64u * 1024u + 16u);
    launch("another kernel, not opted in", optin_other, 64u * 1024u);
    set_maximum("opted in to the maximum", optin_raised, 227 * 1024);
    launch("opted in to the maximum", optin_raised, 227u * 1024u);

    // A maximum under the default lowers what a launch may use.
    set_maximum("maximum set to 1 KiB", optin_lowered, 1024);
    attributes("maximum set to 1 KiB", optin_lowered);
    launch("maximum set to 1 KiB", optin_lowered, 1024u);
    launch("maximum set to 1 KiB", optin_lowered, 1040u);
    launch("maximum set to 1 KiB", optin_lowered, 48u * 1024u);

    // Static shared memory counts against the default, and against what may be opted in to.
    attributes("40 KiB static, not opted in", optin_static);
    launch("40 KiB static, not opted in", optin_static, 0u);
    launch("40 KiB static, not opted in", optin_static, 8u * 1024u);
    launch("40 KiB static, not opted in", optin_static, 8u * 1024u + 16u);
    set_maximum("40 KiB static", optin_static, 227 * 1024 - 40960 + 16);
    set_maximum("40 KiB static", optin_static, 227 * 1024 - 40960);
    attributes("40 KiB static, opted in to the rest of the maximum", optin_static);
    launch("40 KiB static, opted in to the rest of the maximum", optin_static, 227u * 1024u - 40960u);
    return 0;
}
