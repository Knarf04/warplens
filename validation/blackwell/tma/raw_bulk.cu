// cp.async.bulk without a tensor map: a byte range, global to shared and shared to global.
//
// One thread. The source is `bytes` bytes of global memory holding 0x30000000 + i in word i.
//   1. fill a shared buffer with the sentinel 0xEEEEEEEE
//   2. arrive on a fresh mbarrier with expect_tx = bytes and issue
//      cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [buffer], [src], bytes, [bar]
//   3. poll the mbarrier (bounded) and record whether phase 0 completed, how many of the
//      buffer's words equal the source, and the word just past the range
//   4. issue cp.async.bulk.global.shared::cta.bulk_group [dst], [buffer], bytes; commit_group;
//      wait_group 0
// The host compares the second global buffer with the source and checks the guard words
// after it.
//
// Claims tested (docs/history/m2.md; warplens/units/tma.py `_cp`):
//   * the copy is verbatim in both directions;
//   * the load reports exactly `bytes` to the mbarrier (the phase completes with that expect_tx);
//   * nothing past the range is written, in shared or in global memory.
// Sizes 16, 64 and 4096 bytes. Addresses and sizes are multiples of 16; what an unaligned
// one does is not asked here.
//
//   raw_bulk
#include "../common.h"

#define MAX_WORDS 1024
#define GUARD 16

extern "C" __global__ void raw_bulk(const uint32_t* src, uint32_t* dst, uint32_t bytes, uint32_t* out) {
    __shared__ alignas(128) uint32_t buffer[MAX_WORDS + GUARD];
    __shared__ uint64_t bar;
    for (int i = 0; i < MAX_WORDS + GUARD; ++i) buffer[i] = 0xEEEEEEEEu;
    fence_proxy_async();
    mbar_init(&bar, 1);
    mbar_init_fence();
    mbar_arrive_expect_tx(&bar, bytes);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(smem_addr(buffer)), "l"(src), "r"(bytes), "r"(smem_addr(&bar)) : "memory");
    out[0] = mbar_poll(&bar, 0);
    uint32_t same = 0;
    for (uint32_t i = 0; i < bytes / 4; ++i) same += smem_peek(&buffer[i]) == src[i];
    out[1] = same;
    out[2] = smem_peek(&buffer[bytes / 4]);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(dst), "r"(smem_addr(buffer)), "r"(bytes) : "memory");
    tma_store_commit_and_wait();
}

int main() {
    static uint32_t source[MAX_WORDS], back[MAX_WORDS + GUARD];
    for (uint32_t i = 0; i < MAX_WORDS; ++i) source[i] = 0x30000000u + i;
    uint32_t* d_src = dev_copy(source, MAX_WORDS);
    uint32_t* d_dst = dev_alloc<uint32_t>(MAX_WORDS + GUARD);
    uint32_t* d_out = dev_alloc<uint32_t>(3);
    for (uint32_t bytes : {16u, 64u, 4096u}) {
        uint32_t out[3];
        for (uint32_t& v : back) v = 0x11111111u;
        HOST_CHECK(cudaMemcpy(d_dst, back, sizeof back, cudaMemcpyHostToDevice));
        dev_fill(d_out, 3, 0);
        raw_bulk<<<1, 1>>>(d_src, d_dst, bytes, d_out);
        probe_sync("raw_bulk");
        dev_read(out, d_out, 3);
        dev_read(back, d_dst, MAX_WORDS + GUARD);
        uint32_t words = bytes / 4, same = 0, guard_changed = 0;
        for (uint32_t i = 0; i < words; ++i) same += back[i] == source[i];
        for (uint32_t i = words; i < MAX_WORDS + GUARD; ++i) guard_changed += back[i] != 0x11111111u;
        printf("%4u bytes: load with expect_tx %u: phase completed %u; shared words equal to the source %u of %u; "
               "word after the range %08x\n", bytes, bytes, out[0], out[1], words, out[2]);
        printf("            store back: global words equal to the source %u of %u (first %08x, last %08x); "
               "words changed after the range %u\n", same, words, back[0], back[words - 1], guard_changed);
    }
    return 0;
}
