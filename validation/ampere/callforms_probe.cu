// Call forms beyond plain calls with scalar arguments. Thread i of 256 computes twelve values
// through: structs passed and returned by value (one small enough for registers, one with mixed
// fields, one large enough that the callee takes its address), function pointers (from a local
// table, from a global table, passed on to another function, and with a second signature), and
// recursion (two recursive calls in one function, mutual recursion, a local array per call with a
// pointer into the caller's, and a struct argument). Run on real hardware to record golden
// results, and in warplens to compare.
struct Pair { float x, y; };
struct Mixed { int a[6]; double d; short s; char c[3]; };
struct Table { int v[64]; };

__device__ __noinline__ Pair swap_add(Pair p) { Pair r; r.x = p.y; r.y = p.x + 1.0f; return r; }

__device__ __noinline__ Mixed make_mixed(int i) {
    Mixed m;
    for (int k = 0; k < 6; ++k) m.a[k] = i * k - k;
    m.d = i * 0.5; m.s = (short)(-3 * i); m.c[0] = (char)i; m.c[1] = (char)(i >> 1); m.c[2] = 7;
    return m;
}

__device__ __noinline__ int sum_mixed(Mixed m, int k) {
    int s = 0;
    for (int j = 0; j < 6; ++j) s += m.a[j] * (j + k);
    return s + (int)m.d + m.s + m.c[0] + m.c[1] + m.c[2];
}

__device__ __noinline__ Table make_table(int i) { Table t; for (int k = 0; k < 64; ++k) t.v[k] = (i * 31) ^ (k * k); return t; }
__device__ __noinline__ int pick(Table t, int k) { return t.v[k & 63] - t.v[(k * 7 + 3) & 63]; }

__device__ __noinline__ int inc(int x) { return x + 1; }
__device__ __noinline__ int twice(int x) { return x * 2; }
__device__ __noinline__ int negate(int x) { return -x; }
__device__ __noinline__ float plus(float x, float y) { return x + y; }
__device__ __noinline__ float minus(float x, float y) { return x - y; }
typedef int (*unary)(int);
__device__ const unary GLOBAL_TABLE[3] = {negate, inc, twice};
__device__ __noinline__ int apply(unary f, int x) { return f(x) + f(x + 1); }

__device__ __noinline__ int fib(int n) { if (n < 2) return n; return fib(n - 1) + fib(n - 2) * 3; }
__device__ __noinline__ int is_odd(int n);
__device__ __noinline__ int is_even(int n) { if (n == 0) return 1; return is_odd(n - 1); }
__device__ __noinline__ int is_odd(int n) { if (n == 0) return 0; return is_even(n - 1); }

__device__ __noinline__ int walk(const int* above, int n) {  // a frame per call, read by the next
    int mine[4];
    for (int k = 0; k < 4; ++k) mine[k] = n * 10 + k + above[k];
    if (n == 0) return mine[3] + above[1];
    int below = walk(mine, n - 1);
    return below + mine[n & 3];
}

__device__ __noinline__ int fold(Pair p, int n) {  // a struct argument in a recursive call
    if (n == 0) return (int)(p.x * 3.0f + p.y);
    Pair q; q.x = p.y + (float)n; q.y = p.x * 0.5f;
    int below = fold(q, n - 1);
    return below ^ (int)(p.x + (float)(below & 15));
}

extern "C" __global__ void callforms_probe(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int* o = out + 12 * i;
    Pair p; p.x = (float)i; p.y = 2.0f * i;
    Pair q = swap_add(p);
    o[0] = (int)(q.x * 4.0f + q.y);
    o[1] = sum_mixed(make_mixed(i), i & 7);
    o[2] = pick(make_table(i), i);
    unary local_table[3] = {inc, twice, negate};
    o[3] = local_table[i % 3](i);
    o[4] = GLOBAL_TABLE[(i / 3) % 3](i);
    o[5] = apply((i & 4) ? twice : negate, i);
    float (*g)(float, float) = (i & 1) ? plus : minus;
    o[6] = (int)g((float)i, 2.5f);
    o[7] = fib(i % 11);
    o[8] = is_even(i % 9);
    int seed[4] = {i, 1, 2, 3};
    o[9] = walk(seed, i % 6);
    o[10] = fold(p, i % 5);
    o[11] = -1;
    if (i % 3 == 1) o[11] = fib(i % 7) + local_table[i & 1](i);  // calls only some lanes make
}
