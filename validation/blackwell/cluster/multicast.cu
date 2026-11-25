// TMA multicast: which CTAs of a cluster receive a load issued with a CTA mask.
//
// A cluster of four CTAs. Every CTA fills a 256-byte buffer with a sentinel and arms its own
// mbarrier with expect_tx 256. One CTA (the issuer) then issues a single 1-D tensor load of
// 256 bytes with .multicast::cluster and a mask. Every CTA polls its own mbarrier (bounded)
// and reports whether its phase completed and what its buffer holds.
//
// Claims tested (docs/history/m5.md, "TMA multicast"):
//   * the box is written at the same shared offset in every CTA whose rank bit is set in the
//     mask, and in no other, whether or not the issuer is in the mask;
//   * each receiving CTA's mbarrier at the same offset gets complete_tx of the box size.
// Case `outside` (own process): a mask naming rank 4 in a cluster of four. The simulator
// refuses it; what hardware does is not known.
//
//   multicast [outside]
#include "../common.h"

#define BYTES 256

extern "C" __global__ void multicast(const __grid_constant__ CUtensorMap map, uint32_t issuer, uint32_t mask,
                                     uint32_t* out) {
    __shared__ alignas(128) uint32_t buffer[BYTES / 4];
    __shared__ uint64_t bar;
    uint32_t rank = cluster_ctarank();
    bool lead = threadIdx.x == 0;
    if (lead) {
        for (int i = 0; i < BYTES / 4; ++i) buffer[i] = 0xEEEEEEEEu;
        mbar_init(&bar, 1);
        mbar_init_fence();
        mbar_arrive_expect_tx(&bar, BYTES);
    }
    cluster_sync();  // every CTA's mbarrier is armed before the load is issued
    if (lead && rank == issuer) {
        uint16_t m = (uint16_t)mask;
        asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     ".multicast::cluster [%0], [%1, {%3}], [%2], %4;"
                     :: "r"(smem_addr(buffer)), "l"(&map), "r"(smem_addr(&bar)), "r"(0), "h"(m) : "memory");
    }
    uint32_t done = 0;
    if (lead) done = mbar_poll(&bar, 0);
    cluster_sync();  // every CTA has finished polling: any load has long landed
    if (lead) {
        out[rank * 3] = done;
        out[rank * 3 + 1] = buffer[0];
        out[rank * 3 + 2] = buffer[BYTES / 4 - 1];
    }
}

static void run(uint32_t issuer, uint32_t mask, const CUtensorMap& map, uint32_t* d_out) {
    uint32_t out[12];
    dev_fill(d_out, 12, 0);
    printf("issuer rank %u, mask 0x%x (ranks", issuer, mask);
    for (int r = 0; r < 16; ++r)
        if (mask >> r & 1) printf(" %d", r);
    printf("):\n");
    fflush(stdout);
    launch_cluster(multicast, 4, 4, 32, map, issuer, mask, d_out);
    probe_sync("multicast");
    dev_read(out, d_out, 12);
    for (int r = 0; r < 4; ++r) {
        bool data = out[r * 3 + 1] == 0x10000000u && out[r * 3 + 2] == 0x1000003fu;
        bool untouched = out[r * 3 + 1] == 0xEEEEEEEEu && out[r * 3 + 2] == 0xEEEEEEEEu;
        printf("  CTA rank %d: phase completed %u; buffer %s (first word %08x, last word %08x)\n", r, out[r * 3],
               data ? "holds the box" : untouched ? "untouched" : "holds something else", out[r * 3 + 1],
               out[r * 3 + 2]);
    }
}

int main(int argc, char** argv) {
    uint32_t tensor[BYTES / 4];
    for (uint32_t i = 0; i < BYTES / 4; ++i) tensor[i] = 0x10000000u + i;
    uint32_t* d_tensor = dev_copy(tensor, BYTES / 4);
    const uint64_t dims[1] = {BYTES / 4};
    const uint32_t box[1] = {BYTES / 4};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, 1, d_tensor, dims, box, nullptr,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    uint32_t* d_out = dev_alloc<uint32_t>(12);
    if (argc > 1 && !strcmp(argv[1], "outside")) {
        run(0, 0x11, map, d_out);
        return 0;
    }
    run(0, 0xF, map, d_out);  // everyone
    run(0, 0x5, map, d_out);  // the issuer and one other
    run(0, 0xA, map, d_out);  // not the issuer
    run(3, 0x3, map, d_out);  // issued from the last rank, to the first two
    return 0;
}
