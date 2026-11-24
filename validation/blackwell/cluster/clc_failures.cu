// Cluster launch control: how many requests of a real launch succeed (matrix CL-15).
//
// The kernel is examples/09_clc/clc.cu, included unchanged: clusters of two CTAs, one tile
// per CTA; a cluster scales its tiles, then asks to cancel a cluster that has not started
// (multicast to both CTAs) and scales that cluster's tiles too, until a request fails.
// worker[tile] is the blockIdx.x of the CTA that computed the tile, so a CTA started exactly
// when worker[its own tile] names itself.
//
// The launch is repeated with 12 tiles (the example's size) and with larger grids, every CTA
// asking for 100 KB of shared memory it never touches so that an SM holds two CTAs and the
// larger grids are below, near and far above what a B200 (148 SMs) starts at once.
//
// Claim: none about how many requests succeed; the simulator's two extremes are "every
// request while clusters are left" (what `warplens run` does, and the committed prediction:
// 2 CTAs start) and "none" (clc_limit=0 in the Python API: every CTA starts).
//
// Compared lines: per grid size, in how many launches the output was right and every tile
// was computed by a CTA of its own parity that started. Those must be all of them.
// "info:" lines: the smallest and largest number of CTAs that started over the launches of
// a size, and for the first launch of each size the tiles of the first 12 workers.
// Whether a request failed while clusters were still pending cannot be read from worker[]
// alone; clc_order records that.
//
//   clc_failures [repetitions]         default: 30 launches of 12 tiles, 5 of each larger grid
#include "clc_common.h"

namespace example {  // its helpers have the names of common.h's
#include "../../../examples/09_clc/clc.cu"
}

int main(int argc, char** argv) {
    int small_reps = argc > 1 ? atoi(argv[1]) : 30;
    struct Size { uint32_t tiles; int reps; };
    const Size sizes[] = {{12, small_reps}, {128, 5}, {512, 5}, {2048, 5}};
    const float factor = 3.0f;
    HOST_CHECK(cudaFuncSetAttribute(example::scale_tiles, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    CLC_OCCUPANCY_BYTES));
    for (const Size& size : sizes) {
        uint32_t n = size.tiles * TILE;
        float* in = (float*)malloc(n * sizeof(float));
        float* out = (float*)malloc(n * sizeof(float));
        uint32_t* worker = (uint32_t*)malloc(size.tiles * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; ++i) in[i] = (float)(i % 4093) * 0.25f;
        float* d_in = dev_copy(in, n);
        float* d_out = dev_alloc<float>(n);
        uint32_t* d_worker = dev_alloc<uint32_t>(size.tiles);
        int correct = 0, consistent = 0;
        uint32_t fewest = size.tiles, most = 0, first[12], first_started = 0;
        for (int rep = 0; rep < size.reps; ++rep) {
            dev_fill(d_out, n, 0);
            dev_fill(d_worker, size.tiles, 0xff);
            cudaLaunchConfig_t config;
            memset(&config, 0, sizeof config);
            config.gridDim = dim3(size.tiles, 1, 1);
            config.blockDim = dim3(TILE, 1, 1);
            config.dynamicSmemBytes = CLC_OCCUPANCY_BYTES;
            cudaLaunchAttribute attribute;
            attribute.id = cudaLaunchAttributeClusterDimension;
            attribute.val.clusterDim.x = 2;
            attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
            config.attrs = &attribute;
            config.numAttrs = 1;
            fflush(stdout);
            cudaError_t error = cudaLaunchKernelEx(&config, example::scale_tiles, (const float*)d_in, d_out, d_worker,
                                                   factor);
            (void)error;  // probe_sync() reports it
            probe_sync("scale_tiles");
            dev_read(out, d_out, n);
            dev_read(worker, d_worker, size.tiles);
            bool right = true, sane = true;
            uint32_t started = 0;
            for (uint32_t i = 0; i < n; ++i) right = right && out[i] == factor * in[i];
            for (uint32_t t = 0; t < size.tiles; ++t) {
                uint32_t w = worker[t];
                started += w == t;
                // The worker is a CTA of the launch, of the tile's rank, that computed its own tile.
                sane = sane && w < size.tiles && (w & 1) == (t & 1) && worker[w] == w;
            }
            correct += right;
            consistent += sane;
            fewest = started < fewest ? started : fewest;
            most = started > most ? started : most;
            if (rep == 0) {
                memcpy(first, worker, sizeof first);
                first_started = started;
            }
        }
        printf("%u tiles in clusters of two CTAs, %d launch(es):\n", size.tiles, size.reps);
        printf("  output correct in %d of %d\n", correct, size.reps);
        printf("  every tile computed once, by a CTA that started, in %d of %d\n", consistent, size.reps);
        printf("info: CTAs that started: between %u and %u of %u (first launch: %u)\n", fewest, most, size.tiles,
               first_started);
        printf("info: first launch, tile -> CTA that computed it, tiles 0-11:");
        for (int t = 0; t < 12; ++t) printf(" %u", first[t]);
        printf("\n");
        free(in); free(out); free(worker);
        HOST_CHECK(cudaFree(d_in));
        HOST_CHECK(cudaFree(d_out));
        HOST_CHECK(cudaFree(d_worker));
    }
    return 0;
}
