// How many CTAs a cluster may have, with little and with the most shared memory per CTA.
//
// A trivial kernel (each CTA writes its last dynamic shared byte, reads it back and records
// 1) is launched as ONE cluster of 8, 16 and 32 CTAs along x, first with 1024 bytes of
// dynamic shared memory per CTA and then with the device's opt-in maximum
// (cudaDeviceProp::sharedMemPerBlockOptin; printed on an info: line). For every launch the
// program prints whether the launch call accepted it and how many CTAs ran.
//
// Claim tested (docs/m5.md; warplens/host/launch.py): a cluster holds up to 16 CTAs; a
// larger shape is refused at the launch. Only "16 CTAs at 1 KB" is something a kernel in
// this repository relies on (CUTLASS tutorial 3); the rest is recorded.
//
// A refused launch is an error returned by cudaLaunchKernelEx, which leaves the context
// usable; its name goes on an info: line. A launch that is accepted and then fails ends the
// program (KERNEL FAILED), as in every probe. An error that says the program's kernels
// could not be loaded is never counted as "refused" (LAUNCH FAILED).
//
//   max_size
#include "../common.h"

extern "C" __global__ void cluster_touch(uint32_t bytes, uint32_t* ran) {
    extern __shared__ uint8_t dynamic[];
    if (threadIdx.x == 0) {
        dynamic[bytes - 1] = (uint8_t)(0x40 + blockIdx.x);
        ran[blockIdx.x] = dynamic[bytes - 1] == (uint8_t)(0x40 + blockIdx.x) ? 1 : 2;
    }
}

int main() {
    cudaDeviceProp prop;
    HOST_CHECK(cudaGetDeviceProperties(&prop, 0));
    uint32_t most = (uint32_t)prop.sharedMemPerBlockOptin;
    printf("info: sharedMemPerBlockOptin %u bytes\n", most);
    HOST_CHECK(cudaFuncSetAttribute(cluster_touch, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most));
    uint32_t* d_ran = dev_alloc<uint32_t>(32);
    uint32_t ran[32];
    for (int pass = 0; pass < 2; ++pass)
        for (unsigned size : {8u, 16u, 32u}) {
            uint32_t bytes = pass == 0 ? 1024u : most;
            dev_fill(d_ran, 32, 0);
            cudaLaunchConfig_t config;
            memset(&config, 0, sizeof config);
            config.gridDim = dim3(size, 1, 1);
            config.blockDim = dim3(32, 1, 1);
            config.dynamicSmemBytes = bytes;
            cudaLaunchAttribute attribute;
            attribute.id = cudaLaunchAttributeClusterDimension;
            attribute.val.clusterDim.x = size;
            attribute.val.clusterDim.y = 1;
            attribute.val.clusterDim.z = 1;
            config.attrs = &attribute;
            config.numAttrs = 1;
            cudaError_t at_launch = cudaLaunchKernelEx(&config, cluster_touch, bytes, d_ran);
            (void)cudaGetLastError();  // the same error; cleared so that it is not reported twice
            char label[128];
            snprintf(label, sizeof label, "cluster of %2u CTAs, %s of dynamic shared memory per CTA", size,
                     pass == 0 ? "1024 bytes" : "the opt-in maximum");
            if (at_launch != cudaSuccess) {
                if (probe_is_load_error(at_launch)) probe_report_failure(label, at_launch, "at launch");
                printf("%s: refused at the launch\n", label);
                printf("info: cluster of %u CTAs, %u bytes: %s\n", size, bytes, cudaGetErrorName(at_launch));
                continue;
            }
            probe_sync(label);
            dev_read(ran, d_ran, 32);
            unsigned ok = 0;
            for (unsigned i = 0; i < size; ++i) ok += ran[i] == 1;
            printf("%s: launched; %u of %u CTAs ran and read back their last shared byte\n", label, ok, size);
            fflush(stdout);
        }
    return 0;
}
