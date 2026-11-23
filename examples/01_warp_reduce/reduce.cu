// Sum each block's 256 inputs: a warp-level reduction with shuffles, then a
// block-level reduction through dynamically sized shared memory.

extern "C" __global__ void block_sum(const int* in, int* out) {
    extern __shared__ int partial[];  // one slot per warp; sized at launch
    int tid = threadIdx.x;
    int v = in[blockIdx.x * blockDim.x + tid];

    // Warp reduction: after 5 butterfly steps lane 0 holds the warp's sum.
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);

    if (tid % 32 == 0) partial[tid / 32] = v;
    __syncthreads();

    // Warp 0 sums the per-warp partials.
    if (tid < 32) {
        int nwarps = blockDim.x / 32;
        int s = tid < nwarps ? partial[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffff, s, offset);
        if (tid == 0) out[blockIdx.x] = s;
    }
}
