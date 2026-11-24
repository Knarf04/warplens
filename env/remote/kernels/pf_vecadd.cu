// Preflight: the smallest kernel. c[i] = a[i] + b[i].
// Compiled twice by preflight.sh: to machine code (-cubin) and to PTX (-ptx).
extern "C" __global__ void pf_vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
