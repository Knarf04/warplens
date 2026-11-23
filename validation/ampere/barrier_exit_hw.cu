// Host program: does __syncthreads() complete when some threads have already
// returned? Prints one line per case, or hangs (run it under `timeout`).
//   case 0: whole warps return before the barrier (threads >= 64 of 128)
//   case 1: half of every warp returns before the barrier (odd lanes)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)

extern "C" __global__ void barrier_exit(int* out, int which) {
    __shared__ int box[128];
    int t = threadIdx.x;
    box[t] = t;
    if (which == 0 ? t >= 64 : (t & 1)) return;
    __syncthreads();
    out[t] = box[t ^ 2] + 1000;
}

int main(int argc, char** argv) {
    int which = argc > 1 ? atoi(argv[1]) : 0;
    int *d_out, h_out[128];
    CHECK(cudaMalloc(&d_out, sizeof(h_out)));
    CHECK(cudaMemset(d_out, 0, sizeof(h_out)));
    barrier_exit<<<1, 128>>>(d_out, which);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    int wrong = 0;
    for (int t = 0; t < 128; ++t) {
        bool exited = which == 0 ? t >= 64 : (t & 1);
        wrong += h_out[t] != (exited ? 0 : (t ^ 2) + 1000);
    }
    printf("case %d: completed, %s\n", which, wrong ? "WRONG output" : "output ok");
    return 0;
}
