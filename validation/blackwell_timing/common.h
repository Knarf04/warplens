// Host-side harness shared by every timing microbenchmark.
//
// Output contract (fit.py and predict.py depend on it):
//   * one `DEVICE key=value ...` line per process;
//   * with --sim, exactly one `BENCH key=value ...` line per kernel launch, in launch order.
//     predict.py pairs these lines with the simulator's per-launch records by position.
//   * `id=` names a configuration and is identical on hardware and in the simulator.
//     Sizes that differ between the two (repetitions, footprints) are other keys.
//
// A configuration is launched with several repetition counts. Everything fit.py reports
// is the SLOPE of time against repetitions, so launch overhead, kernel prologue and
// the first cold iteration cancel.
//
// Two builds of the same source:
//   BENCH_CLOCK=1  (hardware)  the kernel reads %clock64 before and after its loop
//   BENCH_CLOCK=0  (simulator) it does not; warplens does not implement %clock64
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#ifndef BENCH_CLOCK
#define BENCH_CLOCK 1
#endif
#if BENCH_CLOCK
#define BENCH_NOW() ((uint64_t)clock64())
#else
#define BENCH_NOW() ((uint64_t)0)
#endif

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("BENCH_ERROR where=\"%s\" error=%s\n", #call, cudaGetErrorName(e_)); \
    fflush(stdout); \
    fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(e_)); exit(1); } } while (0)

// What every kernel reports through its `out` buffer.
enum { OUT_CLK = 0, OUT_CHECK = 1, OUT_WORDS = 4 };

struct Opts {
    bool sim = false;         // --sim: tiny repetition counts, one trial, no warm-up launch
    int trials = 5;           // --trials N
    double scale = 1.0;       // --scale X: multiply hardware repetition counts
    const char* only = nullptr;  // --only S: run only ids containing S
    int sm_count = 0;
    size_t free_mem = 0;
    uint64_t* d_out = nullptr;
};

// Hardware only: keep every SM busy for a moment before measuring, so that a GPU that
// lowered its clocks while idle (seen on the 3080 Ti laptop GPU: the first launch after
// start ran at 1.45 ticks/ns of %clock64, later ones at 1.88) is back at speed.
// Not launched with --sim, and prints no BENCH line.
__global__ void bench_spin_kernel(int reps, uint64_t* sink) {
    uint32_t x = threadIdx.x + 1;
    for (int i = 0; i < reps; ++i) x = (x ^ (x >> 13)) * 1664525u + 1013904223u;
    if (x == 0x12345u && reps < 0) sink[3] = x;
}

inline Opts bench_init(int argc, char** argv, const char* bench) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--sim")) o.sim = true;
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) o.trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--scale") && i + 1 < argc) o.scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) o.only = argv[++i];
        else { fprintf(stderr, "usage: %s [--sim] [--trials N] [--scale X] [--only SUBSTRING]\n", argv[0]); exit(2); }
    }
    if (o.sim) o.trials = 1;
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    size_t total = 0;
    CHECK(cudaMemGetInfo(&o.free_mem, &total));
    o.sm_count = p.multiProcessorCount;
    std::string name = p.name;
    for (char& c : name) if (c == ' ') c = '_';
    printf("DEVICE bench=%s name=%s cc=%d.%d sm_count=%d total_mem_bytes=%zu free_mem_bytes=%zu "
           "bench_clock=%d sim=%d trials=%d scale=%g synthetic=0\n",
           bench, name.c_str(), p.major, p.minor, o.sm_count, total, o.free_mem, BENCH_CLOCK,
           (int)o.sim, o.trials, o.scale);
    CHECK(cudaMalloc(&o.d_out, OUT_WORDS * sizeof(uint64_t)));
    CHECK(cudaMemset(o.d_out, 0, OUT_WORDS * sizeof(uint64_t)));
    if (!o.sim) {
        bench_spin_kernel<<<o.sm_count, 128>>>(40000000, o.d_out);
        CHECK(cudaDeviceSynchronize());
    }
    return o;
}

inline bool selected(const Opts& o, const std::string& id) {
    return !o.only || id.find(o.only) != std::string::npos;
}

inline long scaled(const Opts& o, long reps) {
    long r = (long)(reps * o.scale);
    return r < 1 ? 1 : r;
}

// One timed launch; prints its BENCH line. `launch()` must launch exactly one kernel.
template <class Launch>
void timed_launch(const Opts& o, const std::string& id, const std::string& extra, long reps,
                  int trial, Launch launch) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaDeviceSynchronize());
    auto w0 = std::chrono::steady_clock::now();
    CHECK(cudaEventRecord(start));
    launch();
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaDeviceSynchronize());
    auto w1 = std::chrono::steady_clock::now();
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    uint64_t h[OUT_WORDS];
    CHECK(cudaMemcpy(h, o.d_out, sizeof(h), cudaMemcpyDeviceToHost));
    printf("BENCH id=%s %s%sreps=%ld trial=%d event_ns=%.0f wall_ns=%lld clk=%llu check=%llu\n",
           id.c_str(), extra.c_str(), extra.empty() ? "" : " ", reps, trial, (double)ms * 1e6,
           (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count(),
           (unsigned long long)h[OUT_CLK], (unsigned long long)h[OUT_CHECK]);
    fflush(stdout);
}

// A configuration: a warm-up launch (hardware only, trial=-1), then `trials` rounds of one
// launch per repetition count. `launch(reps)` launches the kernel once.
template <class Launch>
void measure(const Opts& o, const std::string& id, const std::string& extra,
             std::vector<long> reps_hw, std::vector<long> reps_sim, Launch launch) {
    if (!selected(o, id)) return;
    std::vector<long> reps = reps_sim;
    if (!o.sim) {
        reps.clear();
        for (long r : reps_hw) reps.push_back(scaled(o, r));
        timed_launch(o, id, extra, reps[0], -1, [&] { launch(reps[0]); });
    }
    for (int t = 0; t < o.trials; ++t)
        for (long r : reps)
            timed_launch(o, id, extra, r, t, [&] { launch(r); });
}
