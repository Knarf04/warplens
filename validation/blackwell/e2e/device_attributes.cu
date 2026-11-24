// What the device says about itself. Host only; no kernel is launched.
//
// The harness runs this first. It answers "is this really a B200" (compute capability
// 10.0, sm_100a; a consumer Blackwell card reports 12.0 and lacks parts of this suite) and
// records the limits the simulated device claims.
//
// Claim tested (warplens/shim/cudart_shim.c `warplens_device_attribute`): the simulated B200
// reports compute capability 10.0, 148 SMs, warp size 32, 1024 threads per block, block
// dimensions up to 1024 x 1024 x 64, 48 KiB of shared memory per block by default and
// 227 KiB on request, 228 KiB per SM, and that it supports cluster launch
// (cudaDevAttrClusterLaunch = 1, which CUTLASS-style hosts query before a cluster launch).
// The attribute is printed from both cudaDeviceGetAttribute and cudaDeviceProp, so a GPU
// that answers otherwise shows as a difference. The name and the memory size go on "info:"
// lines.
//
//   device_attributes
#include "../common.h"

int main() {
    int count = 0, device = 0;
    HOST_CHECK(cudaGetDeviceCount(&count));
    HOST_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p;
    HOST_CHECK(cudaGetDeviceProperties(&p, device));
    size_t free_bytes = 0, total_bytes = 0;
    HOST_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("info: %d device(s); using device %d: %s\n", count, device, p.name);
    printf("info: memory %zu bytes total, %zu free\n", total_bytes, free_bytes);
    printf("compute capability %d.%d\n", p.major, p.minor);
    printf("multiprocessors %d\n", p.multiProcessorCount);
    printf("warp size %d\n", p.warpSize);
    printf("max threads per block %d, block dimensions %d x %d x %d\n", p.maxThreadsPerBlock, p.maxThreadsDim[0],
           p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("max grid %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf("shared memory per block %zu, with opt-in %zu, per multiprocessor %zu\n", p.sharedMemPerBlock,
           p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor);
    printf("registers per block %d, per multiprocessor %d\n", p.regsPerBlock, p.regsPerMultiprocessor);
    printf("max threads per multiprocessor %d, max blocks per multiprocessor %d\n", p.maxThreadsPerMultiProcessor,
           p.maxBlocksPerMultiProcessor);
    int cluster_launch = -1;
    HOST_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    printf("cluster launch %d (attribute), %d (properties)\n", cluster_launch, p.clusterLaunch);
    return 0;
}
