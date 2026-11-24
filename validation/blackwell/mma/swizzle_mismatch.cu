// An MMA whose descriptor names another swizzle than the TMA load that filled the buffer.
//
// The readout trick: B is the 16 x 16 identity (B[n] = e_n, placed honestly with ordinary
// stores), so D[m][n] = A[m][n] as the tensor core reads it: the result names the operand
// element that was fetched for (m, n). A is NOT placed by the host. It is a tensor of 128
// rows x 64 half floats (128 bytes per row) loaded by one TMA load with CU_TENSOR_MAP_
// SWIZZLE_128B into a 1024-aligned shared buffer, which is how a K-major operand with a
// 128B layout is filled in real kernels. Two tensors are used, one holding its row number
// in every element and one holding the element's position in the row (0-63), so two runs
// give the (row, position) of the tensor element behind every D[m][n]. All values are small
// integers, exact in half precision.
//
// The MMA (M = 128, N = 16, kind::f16, K = 16) is then issued with a descriptor of
//   layout 128B (leading offset 16, stride 1024)   honest: D[m][n] comes from (m, n)
//   layout 64B  (leading offset 16, stride 512)    the mismatch
//
// Claim tested (docs/m3.md; warplens/units/tcgen05.py `_operand_bytes`, finding
// `mma-swizzle-mismatch`): the tensor core reads by ITS descriptor. For the 64B descriptor
// the simulator's answer is what the 64B K-major formula of mma/operand_layouts.cu selects
// from the 128B-swizzled bytes. This can only agree with hardware if tma/swizzle_aligned
// and mma/operand_layouts do.
//
//   swizzle_mismatch
#include "../tcgen05_common.h"

#define N 16
#define A_BYTES 16384

extern "C" __global__ void mma_mismatch(const __grid_constant__ CUtensorMap map, const uint32_t* image,
                                        uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t* out) {
    __shared__ alignas(1024) uint32_t operands[OPERAND_BYTES / 4];
    __shared__ uint64_t loaded, done;
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;
    // B by ordinary stores; A is left to the TMA load.
    for (uint32_t i = B_OFFSET / 4 + tid; i < OPERAND_BYTES / 4; i += blockDim.x) operands[i] = image[i];
    fence_proxy_async();
    if (tid == 0) {
        mbar_init(&loaded, 1);
        mbar_init(&done, 1);
        mbar_init_fence();
    }
    __syncthreads();  // see mma_dump in ../tcgen05_common.h
    if (warp == 0) tmem_alloc(&tmem_base, 32);
    __syncthreads();
    uint32_t d = tmem_base, own = d + ((warp * 32) << 16);
    if (tid == 0) {
        mbar_arrive_expect_tx(&loaded, A_BYTES);
        tma_load_2d(operands, &map, &loaded, 0, 0);
        mbar_wait(&loaded, 0);
        uint64_t where = (uint64_t)((smem_addr(operands) >> 4) & 0x3fff);
        mma_f16(d, desc_a + where, desc_b + where, idesc, 0);
        mma_commit(&done);
    }
    mbar_wait(&done, 0);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (uint32_t c = 0; c < N; ++c) {
        uint32_t v = tmem_ld_32x32b(own + c);
        tmem_wait_ld();
        out[tid * N + c] = v;
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(d, 32);
}

static uint8_t image[OPERAND_BYTES];
static uint32_t rows_read[128 * N], positions_read[128 * N];

int main() {
    static uint16_t by_row[128 * 64], by_position[128 * 64], b[N * 16];
    for (int r = 0; r < 128; ++r)
        for (int j = 0; j < 64; ++j) {
            by_row[r * 64 + j] = half_bits((float)r);
            by_position[r * 64 + j] = half_bits((float)j);
        }
    for (int n = 0; n < N; ++n) b[n * 16 + n] = half_bits(1.0f);
    uint64_t desc_b = place_k_major(image, B_OFFSET, b, N, 2, LAYOUTS[3]);
    uint32_t* d_image = dev_copy((const uint32_t*)image, OPERAND_BYTES / 4);
    uint16_t* d_tensor[2] = {dev_copy(by_row, 128 * 64), dev_copy(by_position, 128 * 64)};
    uint32_t* d_out = dev_alloc<uint32_t>(128 * N);
    const uint64_t dims[2] = {64, 128};
    const uint32_t box[2] = {64, 128};
    struct { const char* name; uint64_t desc; } descriptors[2] = {
        {"128B (as loaded)", mma_smem_desc(0, 16, 1024, 2)}, {"64B (mismatch)", mma_smem_desc(0, 16, 512, 4)}};
    for (auto& descriptor : descriptors) {
        for (int which = 0; which < 2; ++which) {
            CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, 2, d_tensor[which], dims, box, nullptr,
                                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            dev_fill(d_out, 128 * N, 0xff);
            mma_mismatch<<<1, 128>>>(map, d_image, descriptor.desc, desc_b, mma_idesc(128, N, 0, 0, 0), d_out);
            probe_sync("mma_mismatch");
            dev_read(which ? positions_read : rows_read, d_out, 128 * N);
        }
        int honest = 0;
        for (int m = 0; m < 128; ++m)
            for (int n = 0; n < N; ++n)
                honest += rows_read[m * N + n] == float_bits((float)m)
                          && positions_read[m * N + n] == float_bits((float)n);
        printf("A loaded with a 128B swizzle, descriptor layout %s, descriptor %016llx\n", descriptor.name,
               (unsigned long long)descriptor.desc);
        printf("  elements D[m][n] that came from tensor element (row m, position n): %d of %d\n", honest, 128 * N);
        printf("  tensor element (row.position) behind D[m][0..15], for m = 0-17 and 126-127:\n");
        for (int m = 0; m < 128; ++m) {
            if (m >= 18 && m < 126) continue;
            printf("  m %3d:", m);
            for (int n = 0; n < N; ++n)
                printf(" %3d.%-2d", (int)bits_float(rows_read[m * N + n]), (int)bits_float(positions_read[m * N + n]));
            printf("\n");
        }
    }
    return 0;
}
