// Probe for the shared-memory wavefront offset (validation matrix TIM-17).
//
// ncu's l1tex__data_pipe_lsu_wavefronts_mem_shared was 4160 above the model for
// the transpose kernels at grid 64, block 32x32. In that launch the number of
// CTAs, warps, shared loads, shared stores, global loads, global stores and
// barriers are all tied together, so the offset cannot be attributed. This
// program launches one kernel many times and changes one of them at a time.
//
//   wavefront_probe [configs.csv]
//
// Every launch is one row in ncu's output, in the order launched. The same
// order is written to configs.csv (default wavefront_probe_configs.csv) so
// that wavefront_fit.py can join the two.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)

#define SLOTS 1024  // one word per thread of the largest block

// All accesses are volatile so that the compiler keeps each one. Within a warp
// lane i touches word base+i: 32 different banks, never a conflict.
extern "C" __global__ void wf_probe(const float* in, float* out,
                                    int n_st, int n_ld, int n_gld, int n_gst, int n_bar) {
    __shared__ float buf[SLOTS];
    int t = threadIdx.x + threadIdx.y * blockDim.x;
    volatile float* s = buf;
    volatile const float* gi = in + blockIdx.x * SLOTS;
    volatile float* go = out + blockIdx.x * SLOTS;
    float acc = (float)t;
    for (int i = 0; i < n_st; ++i) s[t] = acc + (float)i;
    for (int i = 0; i < n_bar; ++i) __syncthreads();
    for (int i = 0; i < n_ld; ++i) acc += s[t];
    for (int i = 0; i < n_gld; ++i) acc += gi[t];
    for (int i = 0; i < n_gst; ++i) go[t] = acc;
}

// The same without any shared memory in the kernel at all.
extern "C" __global__ void wf_noshared(const float* in, float* out,
                                       int n_st, int n_ld, int n_gld, int n_gst, int n_bar) {
    int t = threadIdx.x + threadIdx.y * blockDim.x;
    volatile const float* gi = in + blockIdx.x * SLOTS;
    volatile float* go = out + blockIdx.x * SLOTS;
    float acc = (float)t;
    for (int i = 0; i < n_bar; ++i) __syncthreads();
    for (int i = 0; i < n_gld; ++i) acc += gi[t];
    for (int i = 0; i < n_gst; ++i) go[t] = acc;
}

struct Config { int shared, grid, bx, by, n_st, n_ld, n_gld, n_gst, n_bar; };

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "wavefront_probe_configs.csv";
    std::vector<Config> cfg;
    // The transpose launch: expect the known offset of 4160.
    const Config base = {1, 64, 32, 32, 1, 1, 1, 1, 1};
    cfg.push_back(base);
    cfg.push_back(base);  // again, to see whether the counter repeats exactly
    // Grid.
    for (int g : {1, 2, 3, 32, 100, 128, 500}) { Config c = base; c.grid = g; cfg.push_back(c); }
    // Block shape, whole warps: 1, 2, 4, 8, 16 warps per CTA.
    for (int by : {1, 2, 4, 8, 16}) { Config c = base; c.by = by; cfg.push_back(c); }
    for (int bx : {64, 128, 1024}) { Config c = base; c.bx = bx; c.by = 1; cfg.push_back(c); }
    // Partial warps.
    for (int bx : {1, 16, 31, 33, 48}) { Config c = base; c.bx = bx; c.by = 1; cfg.push_back(c); }
    // Shared stores and loads per thread.
    for (int n : {0, 2, 4}) { Config c = base; c.n_st = n; cfg.push_back(c); }
    for (int n : {0, 2, 4}) { Config c = base; c.n_ld = n; cfg.push_back(c); }
    { Config c = base; c.n_st = 0; c.n_ld = 0; cfg.push_back(c); }
    { Config c = base; c.n_st = 3; c.n_ld = 5; cfg.push_back(c); }
    // Global loads and stores per thread.
    for (int n : {0, 2, 4}) { Config c = base; c.n_gld = n; cfg.push_back(c); }
    for (int n : {0, 2, 4}) { Config c = base; c.n_gst = n; cfg.push_back(c); }
    { Config c = base; c.n_gld = 0; c.n_gst = 0; cfg.push_back(c); }
    // Barriers.
    for (int n : {0, 2, 4}) { Config c = base; c.n_bar = n; cfg.push_back(c); }
    // Nothing but shared accesses; nothing at all.
    { Config c = base; c.n_gld = 0; c.n_gst = 0; c.n_bar = 0; cfg.push_back(c); }
    { Config c = base; c.n_st = 0; c.n_ld = 0; c.n_gld = 0; c.n_gst = 0; c.n_bar = 0; cfg.push_back(c); }
    // No shared memory in the kernel.
    { Config c = base; c.shared = 0; c.n_st = 0; c.n_ld = 0; cfg.push_back(c); }
    { Config c = base; c.shared = 0; c.n_st = 0; c.n_ld = 0; c.n_bar = 0; cfg.push_back(c); }
    { Config c = base; c.shared = 0; c.n_st = 0; c.n_ld = 0; c.n_gld = 0; c.n_gst = 0; c.n_bar = 0; cfg.push_back(c); }

    // Barriers at other block shapes and grids. With one barrier per thread, or
    // with 32 warps per CTA, "one per CTA per barrier" and "1/32 per warp per
    // barrier" give the same number; these launches tell them apart.
    for (int by : {1, 4, 5})
        for (int n : {0, 2, 3}) { Config c = base; c.by = by; c.n_bar = n; cfg.push_back(c); }
    for (int n : {0, 3}) { Config c = base; c.bx = 48; c.by = 1; c.n_bar = n; cfg.push_back(c); }
    for (int n : {0, 3}) { Config c = base; c.grid = 7; c.by = 2; c.n_bar = n; cfg.push_back(c); }
    { Config c = base; c.shared = 0; c.by = 4; c.n_st = 0; c.n_ld = 0; c.n_bar = 3; cfg.push_back(c); }

    int max_grid = 0;
    for (const Config& c : cfg) max_grid = c.grid > max_grid ? c.grid : max_grid;
    size_t bytes = (size_t)max_grid * SLOTS * sizeof(float);
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemset(d_in, 0, bytes));
    CHECK(cudaMemset(d_out, 0, bytes));

    FILE* f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    fprintf(f, "launch,kernel,grid,bx,by,n_st,n_ld,n_gld,n_gst,n_bar\n");
    for (size_t i = 0; i < cfg.size(); ++i) {
        const Config& c = cfg[i];
        if (c.shared)
            wf_probe<<<c.grid, dim3(c.bx, c.by)>>>(d_in, d_out, c.n_st, c.n_ld, c.n_gld, c.n_gst, c.n_bar);
        else
            wf_noshared<<<c.grid, dim3(c.bx, c.by)>>>(d_in, d_out, c.n_st, c.n_ld, c.n_gld, c.n_gst, c.n_bar);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        fprintf(f, "%zu,%s,%d,%d,%d,%d,%d,%d,%d,%d\n", i, c.shared ? "wf_probe" : "wf_noshared",
                c.grid, c.bx, c.by, c.n_st, c.n_ld, c.n_gld, c.n_gst, c.n_bar);
    }
    fclose(f);
    printf("%zu launches, configurations in %s\n", cfg.size(), path);
    return 0;
}
