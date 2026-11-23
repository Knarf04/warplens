// ldmatrix and bfe on real hardware.
//
// ldmatrix: shared memory holds u16 values equal to their own index. Lane l passes the
// address of row (l * 5 + 3) % 32, a permutation, so the result shows which lane's
// address supplies which row and which elements each lane receives.
// out[variant][lane][4]: variant 0 = x1, 1 = x2, 2 = x4, 3 = x1.trans, 4 = x2.trans, 5 = x4.trans
extern "C" __global__ void ldmatrix_probe(unsigned* out) {
    __shared__ unsigned short rows[32][8];
    int lane = threadIdx.x;
    for (int i = lane; i < 256; i += 32) rows[i / 8][i % 8] = (unsigned short)i;
    __syncthreads();
    unsigned addr = (unsigned)__cvta_generic_to_shared(&rows[(lane * 5 + 3) % 32][0]);
    unsigned r[6][4] = {};
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(r[0][0]) : "r"(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(r[1][0]), "=r"(r[1][1]) : "r"(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[2][0]), "=r"(r[2][1]), "=r"(r[2][2]), "=r"(r[2][3]) : "r"(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(r[3][0]) : "r"(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(r[4][0]), "=r"(r[4][1]) : "r"(addr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[5][0]), "=r"(r[5][1]), "=r"(r[5][2]), "=r"(r[5][3]) : "r"(addr));
    for (int v = 0; v < 6; ++v)
        for (int k = 0; k < 4; ++k) out[(v * 32 + lane) * 4 + k] = r[v][k];
}

// out[i] = bfe.u32(a[i], pos[i], len[i]); out[n + i] = bfe.s32(...)
extern "C" __global__ void bfe_probe(const unsigned* a, const unsigned* pos, const unsigned* len,
                                     unsigned* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned u, s;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(u) : "r"(a[i]), "r"(pos[i]), "r"(len[i]));
    asm volatile("bfe.s32 %0, %1, %2, %3;" : "=r"(s) : "r"(a[i]), "r"(pos[i]), "r"(len[i]));
    out[i] = u;
    out[n + i] = s;
}
