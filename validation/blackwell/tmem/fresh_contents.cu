// What tensor memory holds before anything was written to it.
//
// 128 threads (4 warps), 32 columns. Three launches of one kernel in one process:
//   1. read only:   allocate, load every cell (32x32b at each warp's own lanes), deallocate
//   2. write, read: allocate, store 0xA5000000 | lane << 8 | column to every cell, load,
//                   deallocate
//   3. read only:   as 1., after launch 2 left its values behind
// For each launch: how many of the 4096 cells read as zero, and how many read as the
// pattern of launch 2.
//
// Claim tested (docs/m3.md; warplens/units/tcgen05.py, finding `tmem-uninitialized-read`):
// the simulator gives zeros for never-written tensor memory, in launch 1 and in launch 3
// (a launch starts with fresh tensor memory), and reports the read. Hardware promises
// nothing, and a B200 showed as much on 2026-11-25 (validation/blackwell/measured/b200/tmem/
// fresh_contents.txt): zeros in launch 1, and in launch 3 the values launch 2 had stored,
// each warp reading what another warp's lanes had written (no cell held its own pattern, and
// none was zero). Tensor memory survives a launch and is not cleared. So only launch 2 is
// compared with a prediction; the two read-only launches go on info: lines, with the count
// of cells that hold any value of launch 2's pattern (0xA5 in the top byte). The first four
// words of each warp go on info: lines too.
//
//   fresh_contents
#include "../common.h"

#define COLS 32

extern "C" __global__ void fresh_contents(uint32_t write, uint32_t* out) {
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32;
    if (warp == 0) tmem_alloc(&tmem_base, COLS);
    __syncthreads();
    uint32_t base = tmem_base;
    uint32_t own = base + ((warp * 32) << 16);
    if (write) {
        for (uint32_t c = 0; c < COLS; ++c) tmem_st_32x32b(own + c, 0xA5000000u | (tid << 8) | c);
        tmem_wait_st();
    }
    for (uint32_t c = 0; c < COLS; ++c) {
        uint32_t v = tmem_ld_32x32b(own + c);
        tmem_wait_ld();
        out[tid * COLS + c] = v;
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(base, COLS);
}

int main() {
    static uint32_t out[128 * COLS];
    uint32_t* d_out = dev_alloc<uint32_t>(128 * COLS);
    static const char* names[3] = {"launch 1, read only", "launch 2, write then read", "launch 3, read only"};
    for (int launch = 0; launch < 3; ++launch) {
        dev_fill(d_out, 128 * COLS, 0xff);
        fresh_contents<<<1, 128>>>(launch == 1, d_out);
        probe_sync(names[launch]);
        dev_read(out, d_out, 128 * COLS);
        int zero = 0, pattern = 0, residue = 0;
        for (uint32_t t = 0; t < 128; ++t)
            for (uint32_t c = 0; c < COLS; ++c) {
                uint32_t v = out[t * COLS + c];
                zero += v == 0;
                pattern += v == (0xA5000000u | (t << 8) | c);
                residue += (v >> 24) == 0xA5u;
            }
        // What a read-only launch finds is whatever was left there: not compared.
        printf("%s%-26s cells reading zero: %4d of 4096; cells holding launch 2's pattern: %4d\n",
               launch == 1 ? "" : "info: ", names[launch], zero, pattern);
        if (launch != 1)
            printf("info: %s: cells holding a value some lane stored in launch 2: %4d\n", names[launch], residue);
        for (int w = 0; w < 4; ++w)
            printf("info: %s, warp %d, lane 0, columns 0-3: %08x %08x %08x %08x\n", names[launch], w,
                   out[w * 32 * COLS], out[w * 32 * COLS + 1], out[w * 32 * COLS + 2], out[w * 32 * COLS + 3]);
    }
    return 0;
}
