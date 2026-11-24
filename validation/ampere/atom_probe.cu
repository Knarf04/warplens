// Atomic updates (`atom`) whose results do not depend on the order in which threads run, so
// that a GPU gives the same numbers every time: sums, minima and maxima, bit masks, counters
// that every thread bumps, increments built from compare-and-swap, and exchanges checked by
// what they leave behind plus everything they returned. Every thread of the launch updates
// the same global words, every thread of a CTA the same shared words, so lanes of one warp
// meet on one address, as do warps and CTAs. Run on real hardware to record golden results,
// and in warplens to compare.
//
// Words, in global memory (g32, g64, gf, gd) and again in each CTA's shared memory. The
// first half of each array is updated; the second half holds the values the first half
// started with, which the kernel copies into shared memory.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long s64;

enum { N32 = 64, N64 = 24, NF = 4 };  // words that are updated, per array

__device__ __forceinline__ u32 mix(u32 i) {
    u32 h = (i + 1u) * 2654435761u;
    return h ^ (h >> 15);
}

// Add `step` with compare-and-swap. It starts from a guess instead of loading the word: a
// plain load next to other threads' atomic updates is a data race (harmless here on a GPU,
// but warplens reports it for shared memory).
#define CAS_ADD(CAS, T, p, step)                         \
    do {                                                 \
        T seen_ = 0, assumed_;                           \
        do {                                             \
            assumed_ = seen_;                            \
            seen_ = CAS(p, assumed_, assumed_ + (step)); \
        } while (seen_ != assumed_);                     \
    } while (0)

// The same updates for global and for shared memory; inlined, so the compiler knows which.
__device__ __forceinline__ u32 words32(u32* w, u32 i, u32 h) {
    u32 ticket = atomicAdd(w + 0, 1u);  // a counter: every thread gets a different old value
    atomicAdd(w + 1, i + 1);
    atomicSub(w + 2, i + 1);
    atomicAdd((int*)w + 3, (i & 1) ? -(int)i : (int)(3 * i));
    atomicMin(w + 4, h);
    atomicMax(w + 5, h);
    atomicMin((int*)w + 6, (int)h);
    atomicMax((int*)w + 7, (int)h);
    atomicAnd(w + 8, ~(1u << ((mix(i >> 5) >> 13) & 31)));  // one bit per warp
    atomicOr(w + 9, 1u << ((mix(i >> 5) >> 8) & 31));
    atomicXor(w + 10, h);
    atomicInc(w + 11, 99u);  // counts 0..99 and wraps
    atomicDec(w + 12, 99u);
    CAS_ADD(atomicCAS, u32, w + 13, 3u);
    u32 before = atomicExch(w + 14, i + 1);  // what is left plus what was returned is fixed
    atomicAdd(w + 15, before);
    if (i % 3 == 0) atomicAdd(w + 16, h);  // only some lanes
    atomicAdd(w + 17 + (h & 3), 1u);       // a histogram: some lanes share a word, some do not
    atomicSub((int*)w + 21, (int)(h & 0xffff));
    atomicAdd(w + 32 + (i & 31), h);  // a different word for each lane of a warp
    return ticket;
}

__device__ __forceinline__ void words64(u64* w, u32 i, u64 h) {
    atomicAdd(w + 0, h);
    atomicMin(w + 1, h);
    atomicMax(w + 2, h);
    atomicMin((s64*)w + 3, (s64)h);
    atomicMax((s64*)w + 4, (s64)h);
    atomicAnd(w + 5, ~(1ull << ((mix(i >> 5) >> 13) & 63)));
    atomicOr(w + 6, 1ull << ((mix(i >> 5) >> 8) & 63));
    atomicXor(w + 7, h);
    CAS_ADD(atomicCAS, u64, w + 8, 0x100000001ull);
    u64 before = atomicExch(w + 9, h);
    atomicAdd(w + 10, before);
    atomicAnd((s64*)w + 11, (s64)~(2ull << ((mix(i >> 5) >> 11) & 31)));
    atomicOr((s64*)w + 12, (s64)(4ull << ((mix(i >> 5) >> 5) & 31)));
    atomicXor((s64*)w + 13, (s64)(h >> 1));
}

// Sums of multiples of 1/8 that stay far below 2^24: exact in any order.
__device__ __forceinline__ void floats(float* f, double* d, u32 i) {
    atomicAdd(f + 0, 0.25f * (float)((i & 7) + 1));
    float before = atomicExch(f + 1, (float)(i + 1));
    atomicAdd(f + 2, before);
    atomicAdd(d + 0, 0.125 * (double)((i & 15) + 1));
}

// g32: 2 * N32 words, then two words for each CTA. g64: 2 * N64. gf, gd: 2 * NF.
// ticket: two words per thread. s32_out, s64_out, sf_out, sd_out: each CTA's shared words.
extern "C" __global__ void atom_probe(u32* g32, u64* g64, float* gf, double* gd, u32* ticket,
                                      u32* s32_out, u64* s64_out, float* sf_out, double* sd_out,
                                      int sel) {
    __shared__ u32 s32[N32];
    __shared__ u64 s64[N64];
    __shared__ float sf[NF];
    __shared__ double sd[NF];
    u32 t = threadIdx.x, i = blockIdx.x * blockDim.x + t, h = mix(i);
    u64 h64 = ((u64)h << 32) | mix(i + 77777u);
    if (t < N32) s32[t] = g32[N32 + t];
    if (t < N64) s64[t] = g64[N64 + t];
    if (t < NF) { sf[t] = gf[NF + t]; sd[t] = gd[NF + t]; }
    __syncthreads();

    ticket[2 * i] = words32(g32, i, h);
    ticket[2 * i + 1] = words32(s32, i, h);
    words64(g64, i, h64);
    words64(s64, i, h64);
    floats(gf, gd, i);
    floats(sf, sd, i);

    // The scoped variants. _block only where no other CTA touches the word.
    atomicAdd_system(g32 + 22, h);
    atomicMin_system(g32 + 23, h);
    CAS_ADD(atomicCAS_system, u32, g32 + 24, 7u);
    atomicOr_system(g64 + 14, 1ull << (h & 63));
    atomicAdd_block(g32 + 2 * N32 + 2 * blockIdx.x, h);
    atomicXor_block(g32 + 2 * N32 + 2 * blockIdx.x + 1, h);
    atomicAdd_block(s32 + 22, h);
    atomicMax_block(s32 + 23, h);
    CAS_ADD(atomicCAS_block, u32, s32 + 24, 7u);
    atomicAdd_system(s32 + 25, i + 1);
    u64 before = atomicExch_block(s64 + 14, h64);
    atomicAdd_block(s64 + 15, before);
    atomicAdd_block(sf + 3, 0.5f);
    atomicAdd_system(sd + 3, 0.5);

    // Through pointers the compiler cannot place: in one warp, some lanes update shared
    // memory and the others global memory with the same instruction.
    u32* either = ((i ^ (u32)sel) & 1) ? s32 + 26 : g32 + 26;
    atomicAdd(either, h);
    atomicMax(either + 1, h);
    CAS_ADD(atomicCAS, u32, either + 2, 5u);
    u64* either64 = ((i ^ (u32)sel) & 2) ? s64 + 16 : g64 + 16;
    atomicAdd(either64, h64);
    atomicMin(either64 + 1, h64);

    __syncthreads();
    if (t < N32) s32_out[blockIdx.x * N32 + t] = s32[t];
    if (t < N64) s64_out[blockIdx.x * N64 + t] = s64[t];
    if (t < NF) { sf_out[blockIdx.x * NF + t] = sf[t]; sd_out[blockIdx.x * NF + t] = sd[t]; }
}
