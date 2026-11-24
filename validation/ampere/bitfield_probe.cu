// bfi.b32, bfi.b64 and shf.{l,r}.{wrap,clamp}.b32 on operands the compiler cannot fold.
// nvcc 12.4 emits bfi.b64 for `(u64)hi << 32 | lo` and shf.l.wrap.b32 for a 32-bit rotation
// (atom_probe.cu compiled by it has both); nvcc 13.5 for sm_100a emits neither.
// Run on real hardware to record golden results, and in warplens to compare.
typedef unsigned int u32;
typedef unsigned long long u64;

// out[7 * i ..]: bfi.b32, bfi.b64 (low, high), shf.l.wrap, shf.r.wrap, shf.l.clamp, shf.r.clamp.
// The 64-bit operands of bfi.b64 are built from a[i] and b[i].
extern "C" __global__ void bitfield_probe(const u32* a, const u32* b, const u32* pos, const u32* len,
                                          u32* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 x = a[i], y = b[i], p = pos[i], l = len[i], r32, s[4];
    u64 x64 = ((u64)x << 32) | (x * 2654435761u), y64 = ((u64)y << 32) | (y * 40503u + 1), r64;
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(r32) : "r"(x), "r"(y), "r"(p), "r"(l));
    asm volatile("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(r64) : "l"(x64), "l"(y64), "r"(p), "r"(l));
    asm volatile("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(s[0]) : "r"(x), "r"(y), "r"(p));
    asm volatile("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(s[1]) : "r"(x), "r"(y), "r"(p));
    asm volatile("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(s[2]) : "r"(x), "r"(y), "r"(p));
    asm volatile("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(s[3]) : "r"(x), "r"(y), "r"(p));
    out[7 * i] = r32;
    out[7 * i + 1] = (u32)r64;
    out[7 * i + 2] = (u32)(r64 >> 32);
    for (int k = 0; k < 4; ++k) out[7 * i + 3 + k] = s[k];
}
