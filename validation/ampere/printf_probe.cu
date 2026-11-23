// printf from device code with mixed argument types, plus a by-value struct
// parameter and a per-thread local array. Run on real hardware (printf_hw.cu)
// to record the expected output, and in warplens to compare.
struct Config {
    int offset;
    float scale;
    const int* table;
    short tag;
};

extern "C" __global__ void printf_probe(Config cfg, int* out) {
    int t = threadIdx.x;
    int scratch[4];
    for (int i = 0; i < 4; ++i) scratch[i] = cfg.table[(t + i) & 3] * (i + 1);
    int sum = 0;
    for (int i = 0; i < 4; ++i) sum += scratch[(i + t) & 3];
    out[t] = sum + cfg.offset;
    if (t < 3) {
        printf("thread %d of %u: sum=%d scale=%f half=%5.2f tag=%hd ptrdiff=%ld hex=%#x %s %c%%\n",
               t, blockDim.x, sum, cfg.scale, cfg.scale * 0.5f, cfg.tag,
               (long)(t - 7) * 1000000000L, 255 + t, "ok", 'A' + t);
    }
}
