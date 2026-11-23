// One warp; lane i holds 100 + i and shuffles with the given mode, operand and width.
// Run on real hardware to record golden results, and in warplens to compare.
extern "C" __global__ void shfl_probe(int* out, int mode, int b, int width) {
    int lane = threadIdx.x;
    int v = 100 + lane;
    int r;
    if (mode == 0)      r = __shfl_sync(0xffffffff, v, b, width);
    else if (mode == 1) r = __shfl_up_sync(0xffffffff, v, b, width);
    else if (mode == 2) r = __shfl_down_sync(0xffffffff, v, b, width);
    else                r = __shfl_xor_sync(0xffffffff, v, b, width);
    out[lane] = r;
}
