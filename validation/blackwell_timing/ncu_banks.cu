// ncu_banks: the shared-memory transpose of examples/01_smem_transpose in three block
// shapes, to be profiled under Nsight Compute (validation matrix TIM-16 and TIM-17).
//
// Calibrates: nothing in the timing config. It is input for the bank model's counters:
// on the RTX 3080 Ti, ncu's total wavefront count was 4160 above the model's in every
// layout of the 32 x 32 block, grid 64 (validation/README.md). 4160 = 64 CTAs x 65, so the
// surplus may be per CTA, per warp (64 x 32 warps = 2048, no), or per instruction. Running
// the same transpose with 32 x 8 and 16 x 16 threads per CTA (8 warps instead of 32, and a
// different number of rows per warp) separates those: compare ncu's
// l1tex__data_pipe_lsu_wavefronts_mem_shared.sum with the simulator's count per kernel
// (ncu_banks_compare.py).
//
// Kernels: banks_<layout>_<TX>x<TY>, layout linear, padded or swizzled, block (TX, TY).
// Thread t = y * TX + x writes its input to tile[y][x] and reads the tile "transposed":
// element (t % TY, t / TY), so that a warp reads down a column. For 32 x 32 that is exactly
// the three kernels of examples/01_smem_transpose; the simulator's counters for them are
// identical (ncu_banks_compare.py --self-check).
//   linear    rows are TX words apart      32 x 32: a 32-way conflict on the read
//   padded    rows are TX + 1 words apart  32 x 32: no conflict
//   swizzled  the column is XORed with the row   32 x 32: no conflict
// In the other two shapes a warp reads an 8 x 4 (32 x 8) or 16 x 2 (16 x 16) patch of the
// tile, so the layouts conflict differently; what matters here is only that the degrees
// differ between kernels. The model's counts, grid 64 (ld = st executed; conflicts;
// wavefronts), from ncu_banks_compare.py:
//   32 x 32   2048 each   linear 63488 / 67584   padded 0 / 4096      swizzled 0 / 4096
//   32 x 8     512 each   linear  3584 /  4608   padded 1536 / 2560   swizzled 1536 / 2560
//   16 x 16    512 each   linear  3584 /  4608   padded 1024 / 2048   swizzled 0 / 1024
//
// One launch per kernel and trial, grid 64, one BENCH line each (id
// ncu_banks/<layout>/block=<TX>x<TY>, reps = the grid size). The output is checked on the
// host; a wrong result is a BENCH_ERROR line and exit status 1. Under ncu run it with
// `--trials 1`: then every kernel is launched once (plus bench_spin_kernel, the harness's
// warm-up, which has no shared memory).
//
// Nothing here needs more than Ampere: build.sh ampere builds it for the local GPU.
#include "common.h"

#define CTAS 64

#define BANKS_BODY(TX, TY, STRIDE, WRITE_COLUMN, READ_COLUMN)                 \
    int x = threadIdx.x, y = threadIdx.y, t = y * TX + x;                    \
    const float* src = in + blockIdx.x * (TX * TY);                          \
    float* dst = out + blockIdx.x * (TX * TY);                               \
    tile[y * (STRIDE) + (WRITE_COLUMN)] = src[t];                            \
    __syncthreads();                                                         \
    int r = t % TY, c = t / TY;                                              \
    dst[t] = tile[r * (STRIDE) + (READ_COLUMN)];

#define DEFINE_BANKS(TX, TY)                                                                   \
    extern "C" __global__ void banks_linear_##TX##x##TY(const float* in, float* out) {         \
        __shared__ float tile[TY * TX];                                                        \
        BANKS_BODY(TX, TY, TX, x, c)                                                           \
    }                                                                                          \
    extern "C" __global__ void banks_padded_##TX##x##TY(const float* in, float* out) {         \
        __shared__ float tile[TY * (TX + 1)];                                                  \
        BANKS_BODY(TX, TY, TX + 1, x, c)                                                       \
    }                                                                                          \
    extern "C" __global__ void banks_swizzled_##TX##x##TY(const float* in, float* out) {       \
        __shared__ float tile[TY * TX];                                                        \
        BANKS_BODY(TX, TY, TX, x ^ y, c ^ r)                                                   \
    }
DEFINE_BANKS(32, 32)
DEFINE_BANKS(32, 8)
DEFINE_BANKS(16, 16)

typedef void (*Kernel)(const float*, float*);

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "ncu_banks");
    struct Shape { int tx, ty; Kernel linear, padded, swizzled; } shapes[] = {
        {32, 32, banks_linear_32x32, banks_padded_32x32, banks_swizzled_32x32},
        {32, 8, banks_linear_32x8, banks_padded_32x8, banks_swizzled_32x8},
        {16, 16, banks_linear_16x16, banks_padded_16x16, banks_swizzled_16x16},
    };
    const char* layouts[] = {"linear", "padded", "swizzled"};
    int bad = 0;
    for (auto& s : shapes) {
        size_t per_cta = (size_t)s.tx * s.ty, n = CTAS * per_cta;
        std::vector<float> in(n), out(n);
        for (size_t i = 0; i < n; ++i) in[i] = (float)i;
        float *d_in, *d_out;
        CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CHECK(cudaMalloc(&d_out, n * sizeof(float)));
        CHECK(cudaMemcpy(d_in, in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        Kernel kernels[] = {s.linear, s.padded, s.swizzled};
        for (int k = 0; k < 3; ++k) {
            std::string block = std::to_string(s.tx) + "x" + std::to_string(s.ty);
            std::string id = std::string("ncu_banks/") + layouts[k] + "/block=" + block;
            if (!selected(o, id)) continue;
            for (int trial = 0; trial < o.trials; ++trial) {
                CHECK(cudaMemset(d_out, 0, n * sizeof(float)));
                timed_launch(o, id, "nctas=" + std::to_string(CTAS), CTAS, trial,
                             [&] { kernels[k]<<<CTAS, dim3(s.tx, s.ty)>>>(d_in, d_out); });
                CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
                size_t wrong = 0;
                for (size_t i = 0; i < n; ++i) {
                    size_t cta = i / per_cta, t = i % per_cta;
                    wrong += out[i] != in[cta * per_cta + (t % s.ty) * s.tx + t / s.ty];
                }
                if (wrong) {
                    printf("BENCH_ERROR where=\"%s\" error=wrong_result elements=%zu\n", id.c_str(), wrong);
                    fflush(stdout);
                    ++bad;
                }
            }
        }
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
    }
    return bad ? 1 : 0;
}
