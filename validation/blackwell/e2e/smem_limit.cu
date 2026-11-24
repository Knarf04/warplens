// The most shared memory one CTA can be launched with.
//
// A kernel with no static shared memory writes the first and the last byte of its dynamic
// shared memory and reads them back. For 227 x 1024, 227 x 1024 + 16 and 228 x 1024 bytes
// the program asks for that maximum (cudaFuncSetAttribute, cudaFuncAttributeMaxDynamic-
// SharedMemorySize) and launches with it, and prints what each call said.
//
// Claim tested (warplens/units/smem.py `MAX_SMEM_BYTES`; warplens/host/launch.py): a launch
// may use up to 228 x 1024 bytes per CTA. The simulated device reports an opt-in maximum of
// 227 x 1024 (e2e/device_attributes), so the three sizes straddle the two numbers. If only
// 227 KiB launches on hardware, MAX_SMEM_BYTES is too large by 1 KiB.
//
// A refused request is an error RETURNED by a call (the context stays usable); its name goes
// on an info: line. A launch that is accepted and then fails ends the program (KERNEL
// FAILED), and an error that says the kernels could not be loaded is LAUNCH FAILED, as in
// every probe.
//
//   smem_limit
#include "../common.h"

extern "C" __global__ void smem_touch(uint32_t bytes, uint32_t* out) {
    extern __shared__ uint8_t dynamic[];
    if (threadIdx.x == 0) {
        dynamic[0] = 0x5A;
        dynamic[bytes - 1] = 0xA5;
        out[0] = dynamic[0];
        out[1] = dynamic[bytes - 1];
    }
}

int main() {
    cudaDeviceProp prop;
    HOST_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("info: sharedMemPerBlockOptin %zu, sharedMemPerMultiprocessor %zu\n", prop.sharedMemPerBlockOptin,
           prop.sharedMemPerMultiprocessor);
    uint32_t* d_out = dev_alloc<uint32_t>(2);
    for (uint32_t bytes : {227u * 1024u, 227u * 1024u + 16u, 228u * 1024u}) {
        char label[96];
        snprintf(label, sizeof label, "%u bytes (227 KiB + %u)", bytes, bytes - 227u * 1024u);
        cudaError_t opt_in = cudaFuncSetAttribute(smem_touch, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
        (void)cudaGetLastError();
        if (opt_in != cudaSuccess) {
            if (probe_is_load_error(opt_in)) probe_report_failure(label, opt_in, "at cudaFuncSetAttribute");
            printf("info: %s: cudaFuncSetAttribute returned %s\n", label, cudaGetErrorName(opt_in));
        }
        dev_fill(d_out, 2, 0);
        smem_touch<<<1, 32, bytes>>>(bytes, d_out);
        cudaError_t at_launch = cudaGetLastError();
        if (at_launch != cudaSuccess) {
            if (probe_is_load_error(at_launch)) probe_report_failure(label, at_launch, "at launch");
            printf("%s: opt-in %s; launch refused\n", label, opt_in == cudaSuccess ? "accepted" : "refused");
            printf("info: %s: the launch returned %s\n", label, cudaGetErrorName(at_launch));
            continue;
        }
        probe_sync(label);
        uint32_t out[2];
        dev_read(out, d_out, 2);
        printf("%s: opt-in %s; launched; first byte %02x, last byte %02x\n", label,
               opt_in == cudaSuccess ? "accepted" : "refused", out[0], out[1]);
        fflush(stdout);
    }
    return 0;
}
