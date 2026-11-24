// Calls to device functions that the compiler may not inline. Thread i of 256 computes a
// value through: a function with early returns on several paths, a function that calls
// another, a function with a local array, a function with five arguments of mixed types, and
// a void function that writes through a pointer. Run on real hardware to record golden
// results, and in warplens to compare.
__device__ __noinline__ int steps(int n) {  // Collatz steps, with returns inside the loop
    int count = 0;
    while (true) {
        if (n <= 1) return count;
        if (count >= 40) return -1;
        n = (n & 1) ? 3 * n + 1 : n / 2;
        ++count;
    }
}

__device__ __noinline__ float halve(float x) { return x * 0.5f; }

__device__ __noinline__ float quarter(float x) {  // a call inside a call, twice
    return halve(halve(x));
}

__device__ __noinline__ int digits(unsigned v) {  // a frame of its own
    int digit[10];
    int n = 0;
    do { digit[n++] = v % 10; v /= 10; } while (v);
    int sum = 0;
    for (int k = 0; k < n; ++k) sum += digit[k] * (k + 1);
    return sum;
}

__device__ __noinline__ long long mixed(int a, long long b, float c, double d, unsigned char e) {
    return a + b * 3 + (long long)(c * 2.0f) + (long long)d - e;
}

__device__ __noinline__ void store(int* where, int value) {
    if (value & 1) *where = value;
    else *where = -value;
}

extern "C" __global__ void call_probe(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int a = steps(i);
    int b = (int)(quarter((float)i) * 8.0f);
    int c = digits((unsigned)i * 2654435761u);
    int d = (int)mixed(i, (long long)i << 33, i * 0.25f, i * 1.5, (unsigned char)i);
    if (i % 3 == 0) a += steps(i + 7);  // a call that only some lanes make
    store(out + 5 * i + 4, i);
    out[5 * i] = a; out[5 * i + 1] = b; out[5 * i + 2] = c; out[5 * i + 3] = d;
}
