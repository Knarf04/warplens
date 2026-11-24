// Shared by the cluster-launch-control probes (cluster/clc_*.cu): the PTX wrappers, two
// kernels and the host code that launches them.
//
// The forms are those of docs/clc.md (PTX ISA 9.5, sections 9.7.15.18 and 9.7.15.19). A
// response is 16 opaque bytes of shared memory; only query_cancel may interpret them, so
// raw response bytes are printed on "info:" lines only.
//
// What is deterministic. In the simulator one cluster runs at a time and `warplens run`
// grants every request, so the first cluster takes over the whole grid: that extreme is
// what the committed predictions show. On a GPU, how many clusters start before a request
// is served depends on the machine. The probes therefore print
//   * on compared lines: what must hold for ANY mix of started and canceled clusters (the
//     set equalities and counts of violations the validation matrix names), and whether a
//     phase completed;
//   * on "info:" lines: how many clusters started, in which order coordinates were handed
//     out, and the like.
#pragma once
#include "../common.h"

// Polls before a CLC request counts as "did not complete". Larger than POLL_BUDGET: how
// long a GPU takes to answer a request is not known (matrix row TIM-20).
#define CLC_POLL_BUDGET 100000
#define CLC_SENTINEL 0xEEEEEEEEu
// Dynamic shared memory the grid launches ask for and never touch. It limits an SM to two
// CTAs, so that a grid of a few thousand CTAs is several times what a B200 (148 SMs)
// starts at once and some requests can succeed.
#define CLC_OCCUPANCY_BYTES (100 * 1024)

__device__ __forceinline__ void clc_try_cancel(const void* response, uint64_t* bar) {
    asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
                 :: "r"(smem_addr(response)), "r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ void clc_try_cancel_multicast(const void* response, uint64_t* bar) {
    asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"
                 ".multicast::cluster::all.b128 [%0], [%1];"
                 :: "r"(smem_addr(response)), "r"(smem_addr(bar)) : "memory");
}
// Decode a response the way kernels do: the coordinate is read only if the request succeeded.
__device__ __forceinline__ uint32_t clc_query(const void* response, uint32_t& x, uint32_t& y, uint32_t& z) {
    uint32_t canceled, cx = 0, cy = 0, cz = 0;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 ".reg .b128 r;\n"
                 "ld.shared.b128 r, [%4];\n"
                 "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, r;\n"
                 "}" : "=r"(canceled), "+r"(cx), "+r"(cy), "+r"(cz) : "r"(smem_addr(response)) : "memory");
    x = cx; y = cy; z = cz;
    return canceled;
}
// Every query form, unconditionally: v[0] is_canceled, v[1..4] the .v4 elements, v[5..7]
// the ::x, ::y and ::z forms. On a failed response the coordinates are undefined in PTX.
__device__ __forceinline__ void clc_query_all(const void* response, uint32_t* v) {
    uint32_t c, a0, a1, a2, a3, x, y, z;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 ".reg .b128 r;\n"
                 "ld.shared.b128 r, [%8];\n"
                 "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, %4}, r;\n"
                 "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %5, r;\n"
                 "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %6, r;\n"
                 "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %7, r;\n"
                 "}" : "=r"(c), "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3), "=r"(x), "=r"(y), "=r"(z)
                 : "r"(smem_addr(response)) : "memory");
    v[0] = c; v[1] = a0; v[2] = a1; v[3] = a2; v[4] = a3; v[5] = x; v[6] = y; v[7] = z;
}

// -------------------------------------------------------------------------------------
// Kernel 1: one request in one cluster, and what became of it.
// -------------------------------------------------------------------------------------
// Thread 0 of every CTA puts a sentinel in its response slot and arms its own mbarrier
// (one arrival, expect_tx `expect` bytes). After a cluster barrier, thread 0 of rank 0
// issues ONE try_cancel, with or without .multicast::cluster::all. Thread 0 of every CTA
// then polls its own mbarrier a bounded number of times, and records per rank, in 24 words:
//   [0]      1 if the phase completed within the budget
//   [1..4]   the four words of its response slot
//   [5..12]  only if the phase completed (else 0xFFFFFFFF): is_canceled, and with `detail`
//            the rest of clc_query_all (get_first_ctaid of a response that may have failed)
// With `repeats` > 0, rank 0 then issues that many further requests, each announced with 16
// bytes and polled for; bit i of [13] says request i completed, bit i of [14] that it
// reported a canceled cluster, and [15..17] hold the last coordinate a repeat returned.
#define CLC_ONE_WORDS 24
extern "C" __global__ void clc_one_request(uint32_t expect, uint32_t multicast, uint32_t repeats, uint32_t detail,
                                           uint32_t* out) {
    __shared__ alignas(16) uint32_t response[4];
    __shared__ uint64_t bar;
    uint32_t rank = cluster_ctarank();
    bool lead = threadIdx.x == 0;
    if (lead) {
        for (int i = 0; i < 4; ++i) response[i] = CLC_SENTINEL;
        mbar_init(&bar, 1);
        mbar_init_fence();
        fence_proxy_async();
        mbar_arrive_expect_tx(&bar, expect);
    }
    cluster_sync();  // every CTA's slot and mbarrier are ready before the request
    if (lead && rank == 0) {
        if (multicast) clc_try_cancel_multicast(response, &bar);
        else clc_try_cancel(response, &bar);
    }
    uint32_t done = 0;
    if (lead) done = mbar_poll(&bar, 0, CLC_POLL_BUDGET);
    cluster_sync();  // every CTA has finished polling
    if (lead) {
        uint32_t* mine = out + rank * CLC_ONE_WORDS;
        uint32_t v[8];
        for (int i = 0; i < 8; ++i) v[i] = 0xFFFFFFFFu;
        if (done && detail) {
            clc_query_all(response, v);
        } else if (done) {
            uint32_t x, y, z;
            v[0] = clc_query(response, x, y, z);
        }
        mine[0] = done;
        for (int i = 0; i < 4; ++i) mine[1 + i] = smem_peek(response + i);
        for (int i = 0; i < 8; ++i) mine[5 + i] = v[i];
        uint32_t completed = 0, canceled = 0, x = 0, y = 0, z = 0;
        if (rank == 0 && done) {
            for (uint32_t i = 0; i < repeats; ++i) {
                fence_proxy_async();
                mbar_arrive_expect_tx(&bar, 16);
                clc_try_cancel(response, &bar);
                if (!mbar_poll(&bar, (i + 1) & 1, CLC_POLL_BUDGET)) break;
                completed |= 1u << i;
                if (clc_query(response, x, y, z)) canceled |= 1u << i;
            }
        }
        mine[13] = completed;
        mine[14] = canceled;
        mine[15] = x; mine[16] = y; mine[17] = z;
    }
}

// Launch clc_one_request on one cluster of `ctas` CTAs and read the records back.
static inline void clc_run_one(uint32_t ctas, uint32_t expect, uint32_t multicast, uint32_t repeats, uint32_t detail,
                               uint32_t* records, const char* what) {
    uint32_t* d_out = dev_alloc<uint32_t>(ctas * CLC_ONE_WORDS);
    dev_fill(d_out, ctas * CLC_ONE_WORDS, 0);
    fflush(stdout);
    launch_cluster(clc_one_request, ctas, ctas, 32u, expect, multicast, repeats, detail, d_out);
    probe_sync(what);
    dev_read(records, d_out, ctas * CLC_ONE_WORDS);
    HOST_CHECK(cudaFree(d_out));
}

// -------------------------------------------------------------------------------------
// Kernel 2: a grid of persistent workers.
// -------------------------------------------------------------------------------------
// One thread per CTA. Every CTA marks itself started. The first CTA of each cluster then
// works as a CLC kernel does: request a cancellation, wait, and on success take over the
// coordinate returned and request again; stop at the first failure.
// With `again` != 0 a worker issues ONE more request after the failure it observed (PTX
// leaves that undefined); if that succeeds it goes on working.
//
// Nothing here is atomic (the simulator has no `atom`), so every word has one writer unless
// noted, and the host derives the counts:
//   started[c]      1 if CTA c ran                                  (written by c)
//   taken[c]        how many clusters the worker c took over         (written by c)
//   returned_by[c]  1 + the worker that was handed coordinate c. Two workers handed the
//                   same coordinate would both write here; the host sees that as
//                   sum(taken) exceeding the number of coordinates handed out
//   seq[c]          c was the seq[c]-th coordinate its worker took over
//   saw_failure[c]  for a worker: the shared flag `any_failed` as c read it when it started.
//                   A worker sets the flag when its request fails, so a 1 here means a
//                   cluster started after some request had already failed
//   repeats[c]      with `again`: 1 if c issued the extra request, 3 if that succeeded
//   stray[c]        1 if c was handed a coordinate outside the grid
struct ClcGridBuffers {
    uint32_t *started, *taken, *returned_by, *seq, *saw_failure, *repeats, *stray, *any_failed;
};
extern "C" __global__ void clc_worker(uint32_t nx, uint32_t ny, uint32_t total, uint32_t again, ClcGridBuffers b) {
    __shared__ alignas(16) uint32_t response[4];
    __shared__ uint64_t bar;
    uint32_t me = blockIdx.x + nx * (blockIdx.y + ny * blockIdx.z);
    b.started[me] = 1;
    if (cluster_ctarank() != 0) return;
    b.saw_failure[me] = *b.any_failed;
    mbar_init(&bar, 1);
    mbar_init_fence();
    uint32_t taken = 0, phase = 0, failures = 0;
    for (;;) {
        mbar_arrive_expect_tx(&bar, 16);
        clc_try_cancel(response, &bar);
        mbar_wait(&bar, phase);
        phase ^= 1;
        uint32_t x, y, z;
        uint32_t canceled = clc_query(response, x, y, z);
        fence_proxy_async();
        if (!canceled) {
            *b.any_failed = 1;
            if (failures++ || !again) break;
            b.repeats[me] = 1;
            continue;
        }
        if (failures) b.repeats[me] = 3;
        uint32_t coord = x + nx * (y + ny * z);
        if (x >= nx || y >= ny || coord >= total) {
            b.stray[me] = 1;
            break;
        }
        b.returned_by[coord] = me + 1;
        b.seq[coord] = ++taken;
        b.taken[me] = taken;
    }
}

struct ClcGrid {
    uint32_t nx, ny, nz, cx, cy, cz;  // grid and cluster shape; cx = 0: launched with no cluster shape
    uint32_t total = 0;
    uint32_t *started = nullptr, *taken = nullptr, *returned_by = nullptr, *seq = nullptr, *saw_failure = nullptr,
             *repeats = nullptr, *stray = nullptr;
    // What the checks below count.
    uint32_t origins = 0, once = 0, surplus = 0, both = 0, not_origin = 0, orphans = 0, strays = 0;
    uint32_t clusters_started = 0, granted = 0, late_starts = 0, repeats_issued = 0, repeats_granted = 0;

    uint32_t origin_of(uint32_t c) const {
        uint32_t x = c % nx, y = c / nx % ny, z = c / nx / ny;
        uint32_t sx = cx ? cx : 1, sy = cx ? cy : 1, sz = cx ? cz : 1;
        return (x - x % sx) + nx * ((y - y % sy) + ny * (z - z % sz));
    }

    void run(uint32_t again, const char* what) {
        total = nx * ny * nz;
        ClcGridBuffers d;
        uint32_t** host[] = {&started, &taken, &returned_by, &seq, &saw_failure, &repeats, &stray};
        uint32_t** dev[] = {&d.started, &d.taken, &d.returned_by, &d.seq, &d.saw_failure, &d.repeats, &d.stray};
        for (int i = 0; i < 7; ++i) {
            *dev[i] = dev_alloc<uint32_t>(total);
            dev_fill(*dev[i], total, 0);
            *host[i] = (uint32_t*)malloc(total * sizeof(uint32_t));
        }
        d.any_failed = dev_alloc<uint32_t>(1);
        dev_fill(d.any_failed, 1, 0);
        HOST_CHECK(cudaFuncSetAttribute(clc_worker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        CLC_OCCUPANCY_BYTES));
        cudaLaunchConfig_t config;
        memset(&config, 0, sizeof config);
        config.gridDim = dim3(nx, ny, nz);
        config.blockDim = dim3(1, 1, 1);
        config.dynamicSmemBytes = CLC_OCCUPANCY_BYTES;
        cudaLaunchAttribute attribute;
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = cx;
        attribute.val.clusterDim.y = cy;
        attribute.val.clusterDim.z = cz;
        config.attrs = &attribute;
        config.numAttrs = cx ? 1 : 0;
        fflush(stdout);
        cudaError_t error = cudaLaunchKernelEx(&config, clc_worker, nx, ny, total, again, d);
        (void)error;  // probe_sync() reports it
        probe_sync(what);
        for (int i = 0; i < 7; ++i) {
            dev_read(*host[i], *dev[i], total);
            HOST_CHECK(cudaFree(*dev[i]));
        }
        HOST_CHECK(cudaFree(d.any_failed));
        uint32_t handed_out = 0;
        for (uint32_t c = 0; c < total; ++c) {
            bool returned = returned_by[c] != 0;
            if (origin_of(c) == c) {
                ++origins;
                clusters_started += started[c];
                once += started[c] + returned == 1;
                both += started[c] && returned;
                late_starts += started[c] && saw_failure[c];
                repeats_issued += repeats[c] & 1;
                repeats_granted += repeats[c] >> 1;
            } else {
                not_origin += returned;
            }
            handed_out += returned;
            granted += taken[c];
            strays += stray[c];
            orphans += started[c] != started[origin_of(c)];
        }
        surplus = granted - handed_out;
    }

    // The lines that must be the same for every mix of started and canceled clusters.
    void print_invariants() const {
        printf("  cluster origins: %u\n", origins);
        printf("  origins that either started or were handed to a running cluster, not both and not neither: %u\n",
               once);
        printf("  successful requests beyond one per coordinate handed out: %u\n", surplus);
        printf("  coordinates returned although that cluster started: %u\n", both);
        printf("  coordinates returned that are not the first CTA of a cluster: %u\n", not_origin + strays);
        printf("  CTAs whose start disagrees with their cluster's first CTA: %u\n", orphans);
    }
    void print_info() const {
        printf("info: clusters started %u of %u, requests granted %u\n", clusters_started, origins, granted);
        printf("info: clusters that started after some request had already failed: %u\n", late_starts);
    }
    void release() {
        uint32_t* all[] = {started, taken, returned_by, seq, saw_failure, repeats, stray};
        for (uint32_t* p : all) free(p);
    }
};
