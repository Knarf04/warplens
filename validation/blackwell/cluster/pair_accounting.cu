// Which mbarrier a TMA load reports to when it is issued with .cta_group::2.
//
// A cluster of four CTAs: two pairs, (0, 1) and (2, 3). Rank 0 issues one 256-byte load.
// Each scenario arms the mbarriers the simulator predicts will receive completions, with
// the byte count it predicts; the CTAs then poll (bounded) and print whether their phase
// completed and whether their buffer received the box. If hardware sends the completions
// elsewhere, or in other amounts, an armed mbarrier does not complete and it shows.
// An mbarrier that is not armed has had no arrival, so its phase cannot complete whatever
// bytes it is sent; those lines only show where the data went.
//
// Claims tested (docs/m5.md, "TMA multicast"; warplens/units/tma.py `_load_targets`):
//   A. .cta_group::1 with multicast: every receiving CTA's own mbarrier gets the box size.
//   B. .cta_group::2 with multicast, mbarrier address in the issuer (even rank): the
//      completions of both members of a pair go to the pair's EVEN member, which therefore
//      receives twice the box size ("leader" accounting, as CUTLASS's 2-SM kernels expect).
//   C. the same with the mbarrier address mapped to rank 1: they go to each pair's ODD member.
//   D. .cta_group::2 without multicast, destination in the issuer, mbarrier in its peer:
//      the peer's mbarrier gets the box size.
//
//   pair_accounting
#include "../common.h"

#define BYTES 256

// scenario: 0 = A, 1 = B, 2 = C, 3 = D. expect[r]: bytes CTA r arms its mbarrier with (0: not armed).
extern "C" __global__ void pair_accounting(const __grid_constant__ CUtensorMap map, uint32_t scenario,
                                           const uint32_t* expect, uint32_t* out) {
    __shared__ alignas(128) uint32_t buffer[BYTES / 4];
    __shared__ uint64_t bar;
    uint32_t rank = cluster_ctarank();
    bool lead = threadIdx.x == 0;
    uint32_t mine = expect[rank];
    if (lead) {
        for (int i = 0; i < BYTES / 4; ++i) buffer[i] = 0xEEEEEEEEu;
        mbar_init(&bar, 1);
        mbar_init_fence();
        if (mine) mbar_arrive_expect_tx(&bar, mine);
    }
    cluster_sync();
    if (lead && rank == 0) {
        uint16_t all = 0xF;
        uint32_t dst = smem_addr(buffer), own_bar = smem_addr(&bar), peer_bar = mapa(&bar, 1);
        const void* tm = &map;
        switch (scenario) {
            case 0:
                asm volatile("cp.async.bulk.tensor.1d.cta_group::1.shared::cluster.global"
                             ".mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3}], [%2], %4;"
                             :: "r"(dst), "l"(tm), "r"(own_bar), "r"(0), "h"(all) : "memory");
                break;
            case 1:
                asm volatile("cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global"
                             ".mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3}], [%2], %4;"
                             :: "r"(dst), "l"(tm), "r"(own_bar), "r"(0), "h"(all) : "memory");
                break;
            case 2:
                asm volatile("cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global"
                             ".mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3}], [%2], %4;"
                             :: "r"(dst), "l"(tm), "r"(peer_bar), "r"(0), "h"(all) : "memory");
                break;
            case 3:
                asm volatile("cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global"
                             ".mbarrier::complete_tx::bytes [%0], [%1, {%3}], [%2];"
                             :: "r"(dst), "l"(tm), "r"(peer_bar), "r"(0) : "memory");
                break;
        }
    }
    uint32_t done = 0;
    if (lead) {
        if (mine) {
            done = mbar_poll(&bar, 0);
        } else {
            // Not armed: wait (bounded) for the data instead, so that "untouched" below
            // means it never came and not that it was still on its way.
            for (int i = 0; i < POLL_BUDGET && smem_peek(&buffer[BYTES / 4 - 1]) == 0xEEEEEEEEu; ++i) {}
            done = mbar_test(&bar, 0);
        }
    }
    cluster_sync();
    if (lead) {
        out[rank * 3] = done;
        out[rank * 3 + 1] = buffer[0];
        out[rank * 3 + 2] = buffer[BYTES / 4 - 1];
    }
}

static void run(uint32_t scenario, const char* title, const uint32_t* expect, const CUtensorMap& map) {
    uint32_t out[12];
    uint32_t* d_out = dev_alloc<uint32_t>(12);
    uint32_t* d_expect = dev_copy(expect, 4);
    dev_fill(d_out, 12, 0);
    printf("%s\n", title);
    fflush(stdout);
    launch_cluster(pair_accounting, 4, 4, 32, map, scenario, (const uint32_t*)d_expect, d_out);
    probe_sync("pair_accounting");
    dev_read(out, d_out, 12);
    for (int r = 0; r < 4; ++r) {
        bool data = out[r * 3 + 1] == 0x10000000u && out[r * 3 + 2] == 0x1000003fu;
        bool untouched = out[r * 3 + 1] == 0xEEEEEEEEu && out[r * 3 + 2] == 0xEEEEEEEEu;
        if (expect[r]) printf("  CTA rank %d: armed with expect_tx %3u, phase completed %u;", r, expect[r], out[r * 3]);
        else printf("  CTA rank %d: not armed;                                ", r);
        printf(" buffer %s\n", data ? "holds the box" : untouched ? "untouched" : "holds something else");
    }
}

int main() {
    uint32_t tensor[BYTES / 4];
    for (uint32_t i = 0; i < BYTES / 4; ++i) tensor[i] = 0x10000000u + i;
    uint32_t* d_tensor = dev_copy(tensor, BYTES / 4);
    const uint64_t dims[1] = {BYTES / 4};
    const uint32_t box[1] = {BYTES / 4};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 1, d_tensor, dims, box, nullptr,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const uint32_t a[4] = {256, 256, 256, 256}, b[4] = {512, 0, 512, 0}, c[4] = {0, 512, 0, 512}, d[4] = {0, 256, 0, 0};
    run(0, "A: cta_group::1, multicast to ranks 0-3, mbarrier address in rank 0", a, map);
    run(1, "B: cta_group::2, multicast to ranks 0-3, mbarrier address in rank 0 (even)", b, map);
    run(2, "C: cta_group::2, multicast to ranks 0-3, mbarrier address mapped to rank 1 (odd)", c, map);
    run(3, "D: cta_group::2, no multicast, destination in rank 0, mbarrier address mapped to rank 1", d, map);
    return 0;
}
