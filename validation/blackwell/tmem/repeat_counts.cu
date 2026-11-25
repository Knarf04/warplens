// Repeat counts of tcgen05.ld: every (shape, count) pair the simulator accepts.
//
// The claim (docs/history/m3.md; warplens/units/tcgen05.py `transfer_cells`): the repeat count is a
// power of two that gives at most 128 registers per thread, i.e.
//   32x32b  x1 .. x128     (count registers)
//   16x64b  x1 .. x128     (count registers)
//   16x128b x1 .. x64      (2 x count registers)
//   16x256b x1 .. x32      (4 x count registers)
// Whether an instruction EXISTS is the assembler's decision, so the accepted set is tested
// where it is decided: this file contains every pair above, and build.sh assembles it for
// sm_100a (bin/tmem/repeat_counts.o). If ptxas rejected one of them, the probe would not
// build. That the pairs just beyond the limits (x256, x128, x64) and counts that are not a
// power of two are REJECTED is recorded by tmem/repeat_counts_ptxas.py, which needs only
// the toolkit; its output for toolkit 13.5 is tmem/repeat_counts_ptxas.txt.
//
// Running it adds what TM-05 (tmem/transfer_shapes) shows for small counts: 256 columns are
// written through the reference addressing (32x32b.x1 at each warp's own lanes) with
// values naming (lane, column); each pair then loads at lane offset 0 and the host checks
// every register of every thread against
//   32x32b:   (t, q)
//   16x64b:   (8 (t & 1) + (t >> 2), ((t >> 1) & 1) + 2 q)
//   16x128b:  ((t >> 2) + 8 (q & 1), (t & 3) + 4 (q >> 1))
//   16x256b:  ((t >> 2) + 8 ((q >> 1) & 1), 2 (t & 3) + (q & 1) + 8 (q >> 2))
// for thread t of the warp and register q.
//
//   repeat_counts
#include "../common.h"

#define COLS 256
#define MAX_REGS 128

// Register lists r0.., as binary-numbered names: N4(r) is "r00, r01, r10, r11".
#define N1(p) #p
#define N2(p) N1(p##0) ", " N1(p##1)
#define N4(p) N2(p##0) ", " N2(p##1)
#define N8(p) N4(p##0) ", " N4(p##1)
#define N16(p) N8(p##0) ", " N8(p##1)
#define N32(p) N16(p##0) ", " N16(p##1)
#define N64(p) N32(p##0) ", " N32(p##1)
#define N128(p) N64(p##0) ", " N64(p##1)
// The same registers, each stored to [ptr] in order.
#define E1(p) "st.global.u32 [ptr], " #p ";\nadd.u64 ptr, ptr, 4;\n"
#define E2(p) E1(p##0) E1(p##1)
#define E4(p) E2(p##0) E2(p##1)
#define E8(p) E4(p##0) E4(p##1)
#define E16(p) E8(p##0) E8(p##1)
#define E32(p) E16(p##0) E16(p##1)
#define E64(p) E32(p##0) E32(p##1)
#define E128(p) E64(p##0) E64(p##1)

#define LD(code, shape, NAMES, STORES)                                                  \
    case code:                                                                          \
        asm volatile("{\n.reg .b32 " NAMES(r) ";\n.reg .u64 ptr;\nmov.u64 ptr, %1;\n"   \
                     "tcgen05.ld.sync.aligned." shape ".b32 {" NAMES(r) "}, [%0];\n"    \
                     "tcgen05.wait::ld.sync.aligned;\n" STORES(r) "}"                   \
                     :: "r"(at), "l"(dst) : "memory");                                  \
        break;

// code = shape * 8 + log2(count). Writes out[thread][register], MAX_REGS per thread.
extern "C" __global__ void repeat_counts(uint32_t code, uint32_t* out) {
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;
    if (warp == 0) tmem_alloc(&tmem_base, COLS);
    __syncthreads();
    uint32_t base = tmem_base;
    uint32_t at = base + ((warp * 32) << 16);
    for (uint32_t c = 0; c < COLS; ++c) tmem_st_32x32b(at + c, 0xC0000000u | (tid << 8) | c);
    tmem_wait_st();
    uint32_t* dst = out + tid * MAX_REGS;
    switch (code) {
        LD(0, "32x32b.x1", N1, E1) LD(1, "32x32b.x2", N2, E2) LD(2, "32x32b.x4", N4, E4) LD(3, "32x32b.x8", N8, E8)
        LD(4, "32x32b.x16", N16, E16) LD(5, "32x32b.x32", N32, E32) LD(6, "32x32b.x64", N64, E64)
        LD(7, "32x32b.x128", N128, E128)
        LD(8, "16x64b.x1", N1, E1) LD(9, "16x64b.x2", N2, E2) LD(10, "16x64b.x4", N4, E4) LD(11, "16x64b.x8", N8, E8)
        LD(12, "16x64b.x16", N16, E16) LD(13, "16x64b.x32", N32, E32) LD(14, "16x64b.x64", N64, E64)
        LD(15, "16x64b.x128", N128, E128)
        LD(16, "16x128b.x1", N2, E2) LD(17, "16x128b.x2", N4, E4) LD(18, "16x128b.x4", N8, E8)
        LD(19, "16x128b.x8", N16, E16) LD(20, "16x128b.x16", N32, E32) LD(21, "16x128b.x32", N64, E64)
        LD(22, "16x128b.x64", N128, E128)
        LD(24, "16x256b.x1", N4, E4) LD(25, "16x256b.x2", N8, E8) LD(26, "16x256b.x4", N16, E16)
        LD(27, "16x256b.x8", N32, E32) LD(28, "16x256b.x16", N64, E64) LD(29, "16x256b.x32", N128, E128)
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(base, COLS);
}

static const char* SHAPES[4] = {"32x32b", "16x64b", "16x128b", "16x256b"};
static const int REGS_PER_COUNT[4] = {1, 1, 2, 4};

int main() {
    static uint32_t out[128 * MAX_REGS];
    uint32_t* d_out = dev_alloc<uint32_t>(128 * MAX_REGS);
    for (int shape = 0; shape < 4; ++shape)
        for (int log = 0; (REGS_PER_COUNT[shape] << log) <= MAX_REGS; ++log) {
            int registers = REGS_PER_COUNT[shape] << log;
            char name[64];
            snprintf(name, sizeof name, "tcgen05.ld %s.x%d", SHAPES[shape], 1 << log);
            dev_fill(d_out, 128 * MAX_REGS, 0);
            repeat_counts<<<1, 128>>>((uint32_t)(shape * 8 + log), d_out);
            probe_sync(name);
            dev_read(out, d_out, 128 * MAX_REGS);
            int right = 0;
            for (int w = 0; w < 4; ++w)
                for (int t = 0; t < 32; ++t)
                    for (int q = 0; q < registers; ++q) {
                        int lane, column;
                        switch (shape) {
                            case 0: lane = t; column = q; break;
                            case 1: lane = 8 * (t & 1) + (t >> 2); column = ((t >> 1) & 1) + 2 * q; break;
                            case 2: lane = (t >> 2) + 8 * (q & 1); column = (t & 3) + 4 * (q >> 1); break;
                            default:
                                lane = (t >> 2) + 8 * ((q >> 1) & 1);
                                column = 2 * (t & 3) + (q & 1) + 8 * (q >> 2);
                                break;
                        }
                        uint32_t expect = 0xC0000000u | ((uint32_t)(32 * w + lane) << 8) | (uint32_t)column;
                        right += out[(w * 32 + t) * MAX_REGS + q] == expect;
                    }
            printf("%-24s %3d registers per thread; registers holding the cell the mapping names: %5d of %5d\n",
                   name, registers, right, 128 * registers);
        }
    return 0;
}
