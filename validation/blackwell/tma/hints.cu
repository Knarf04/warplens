// Cache hints and prefetches around a TMA load: they must not change what is loaded.
//
// The rank-2 case of tma/ranks.cu (uint32 tensor 8 x 6 holding 0x10000000 + linear index,
// box 4 x 2 at coordinate (4, 1)), loaded by one thread in five ways:
//   plain            cp.async.bulk.tensor.2d ... [buffer], [map, {4, 1}], [bar]
//   cache hint       the same with .L2::cache_hint and a policy operand, for the three policies
//                    CUTLASS uses (evict_normal, evict_first, evict_last, as 64-bit constants)
//   prefetch map     after prefetch.tensormap [map]
//   prefetch tensor  after cp.async.bulk.prefetch.tensor.2d.L2.global [map, {4, 1}]
//   prefetch bytes   after cp.async.bulk.prefetch.L2.global [tensor], 192
// For each, the 8 loaded words and whether the phase completed with expect_tx 32.
//
// Claim tested (docs/m2.md; warplens/units/tma.py `_cp`, `_prefetch`): none of these has a
// functional effect. Every variant must print the words of `plain`, which are the rank-2
// words of tma/ranks.
//
//   hints
#include "../common.h"

enum { PLAIN, HINT_NORMAL, HINT_FIRST, HINT_LAST, PREFETCH_MAP, PREFETCH_TENSOR, PREFETCH_BYTES, VARIANTS };
static const char* names[VARIANTS] = {"plain", "cache hint evict_normal", "cache hint evict_first",
                                      "cache hint evict_last", "after prefetch.tensormap",
                                      "after cp.async.bulk.prefetch.tensor", "after cp.async.bulk.prefetch"};

extern "C" __global__ void tma_hints(const __grid_constant__ CUtensorMap map, const uint32_t* tensor, int variant,
                                     uint32_t* out) {
    __shared__ alignas(128) uint32_t buffer[32];
    __shared__ uint64_t bar;
    for (int i = 0; i < 32; ++i) buffer[i] = 0xEEEEEEEEu;
    fence_proxy_async();
    mbar_init(&bar, 1);
    mbar_init_fence();
    if (variant == PREFETCH_MAP) asm volatile("prefetch.tensormap [%0];" :: "l"(&map) : "memory");
    if (variant == PREFETCH_TENSOR)
        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];" :: "l"(&map), "r"(4), "r"(1)
                     : "memory");
    if (variant == PREFETCH_BYTES)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(tensor), "r"(192) : "memory");
    mbar_arrive_expect_tx(&bar, 32);
    if (variant >= HINT_NORMAL && variant <= HINT_LAST) {
        // The policies of cute/arch/copy_sm90_tma.hpp (CacheHintSm90), which createpolicy
        // would produce.
        uint64_t policy = variant == HINT_NORMAL ? 0x1000000000000000ull
                          : variant == HINT_FIRST ? 0x12F0000000000000ull : 0x14F0000000000000ull;
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                     " [%0], [%1, {%3, %4}], [%2], %5;"
                     :: "r"(smem_addr(buffer)), "l"(&map), "r"(smem_addr(&bar)), "r"(4), "r"(1), "l"(policy)
                     : "memory");
    } else {
        tma_load_2d(buffer, &map, &bar, 4, 1);
    }
    out[0] = mbar_poll(&bar, 0);
    for (int i = 0; i < 9; ++i) out[1 + i] = smem_peek(&buffer[i]);
}

int main() {
    static uint32_t tensor[8 * 6];
    for (uint32_t i = 0; i < 8 * 6; ++i) tensor[i] = 0x10000000u + i;
    uint32_t* d_tensor = dev_copy(tensor, 8 * 6);
    const uint64_t dims[2] = {8, 6};
    const uint32_t box[2] = {4, 2};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 2, d_tensor, dims, box, nullptr,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    uint32_t* d_out = dev_alloc<uint32_t>(10);
    uint32_t plain[10];
    for (int variant = 0; variant < VARIANTS; ++variant) {
        uint32_t out[10];
        dev_fill(d_out, 10, 0);
        tma_hints<<<1, 1>>>(map, d_tensor, variant, d_out);
        probe_sync(names[variant]);
        dev_read(out, d_out, 10);
        if (variant == PLAIN) memcpy(plain, out, sizeof out);
        printf("%-36s phase completed %u; words", names[variant], out[0]);
        for (int i = 0; i < 8; ++i) printf(" %08x", out[1 + i]);
        printf("; word after the box %08x; %s\n", out[9],
               memcmp(plain, out, sizeof out) ? "DIFFERS from plain" : "same as plain");
    }
    return 0;
}
