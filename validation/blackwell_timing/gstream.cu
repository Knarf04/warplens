// gstream: device memory bandwidth with ordinary loads and stores (STREAM Triad).
//
// Calibrates: `hbm_bandwidth_tb_s`. The config's current value is the STREAM Triad figure
// of arXiv:2512.02189 Table XIII, so this repeats that measurement:
//   a[i] = b[i] + s * c[i]  over n doubles, 24 bytes of traffic per element
//   hbm_bandwidth_tb_s = 24 / (ns per element) / 1000
// where "ns per element" is the slope of launch time against n (two sizes), which removes
// launch overhead. `reps` in the output is n; `norm=24` tells fit.py the bytes per rep.
//
// The simulator runs this (tiny n) but its time is not a prediction: the model simulates
// each CTA alone on an SM and charges ordinary global accesses a fixed latency, with no
// bandwidth limit. fit.py compares the measured bandwidth with the config value instead.
//
// Method: one pass per launch over three arrays far larger than L2, each thread one
// element, grid = n / 256. Sizes: the largest power of two n with 24 n <= 60% of free
// memory, at most 2^28 (2 GiB per array), and half of it. Valid on Ampere.
#include "common.h"

__global__ void triad_kernel(uint64_t* out, double* a, const double* b, const double* c, double s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = b[i] + s * c[i];
    if (i == 0) { out[OUT_CLK] = 0; out[OUT_CHECK] = (uint64_t)a[0]; }
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "gstream");
    if (!selected(o, "gstream/triad")) return 0;
    size_t n = o.sim ? 4096 : (size_t)1 << 28;
    while (!o.sim && 24 * n > o.free_mem * 6 / 10) n >>= 1;
    double *a, *b, *c;
    CHECK(cudaMalloc(&a, n * 8));
    CHECK(cudaMalloc(&b, n * 8));
    CHECK(cudaMalloc(&c, n * 8));
    CHECK(cudaMemset(a, 0, n * 8));
    CHECK(cudaMemset(b, 0, n * 8));
    CHECK(cudaMemset(c, 0, n * 8));
    // `measure` scales repetition counts; here they are array sizes, so launch directly.
    std::vector<long> sizes = {(long)(n / 2), (long)n};
    for (int t = o.sim ? 0 : -1; t < o.trials; ++t)
        for (long m : sizes) {
            timed_launch(o, "gstream/triad", "norm=24", m, t, [&] {
                triad_kernel<<<(unsigned)(m / 256), 256>>>(o.d_out, a, b, c, 3.0);
            });
            if (t < 0) break;  // one warm-up launch
        }
    return 0;
}
