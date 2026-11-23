// Kernels whose lanes take different paths, to exercise SIMT divergence.

// Data-dependent loop: each lane runs a different number of iterations.
// steps[i] = number of Collatz steps for x[i] to reach 1.
extern "C" __global__ void collatz(const unsigned* x, unsigned* steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned v = x[i], count = 0;
    while (v > 1) {
        v = (v & 1) ? 3 * v + 1 : v / 2;
        ++count;
    }
    steps[i] = count;
}

// If/else with side effects on both sides, nested inside a loop with an early exit.
// For each i: walk a row of `data`; negative entries go to `neg` (count), positive
// entries are summed into `sum`; a zero entry ends the row early.
extern "C" __global__ void classify(const int* data, int cols, int* sum, int* neg, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    sum[r] = 0;
    neg[r] = 0;
    for (int c = 0; c < cols; ++c) {
        int v = data[r * cols + c];
        if (v == 0) break;
        if (v < 0) {
            neg[r] += 1;
        } else {
            sum[r] += v;
        }
    }
}
