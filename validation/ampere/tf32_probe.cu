// out[i] = cvt.rna.tf32.f32(in[i]): round a float32 to TF32 precision (10 fraction
// bits), nearest with ties away from zero. Available from Ampere on.
extern "C" __global__ void tf32_probe(const unsigned* in, unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = in[i], y;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "r"(x));
    out[i] = y;
}
