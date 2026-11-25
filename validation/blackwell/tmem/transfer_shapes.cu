// Which tensor-memory cell each register of each thread maps to, for the four shapes of
// tcgen05.ld and tcgen05.st.
//
// Reference addressing: tcgen05.st/ld.32x32b.x1 at lane field 32 * warp, one column at a
// time, where the thread on lane l of warp w reaches TMEM lane 32 w + l. That is what the
// CUTLASS tutorials use on hardware.
//   load test:  every cell is first written through the reference with a value naming its
//               (lane, column); the shape under test then loads, and each register's value
//               says which cell it came from.
//   store test: every cell is first written with a sentinel; the shape under test stores
//               values naming (thread, register); the reference then reads everything back
//               and the host finds where each value went.
// Both are printed in the same form, so the two mappings can be compared with each other.
//
// Claims tested (docs/history/m3.md; warplens/units/tcgen05.py `transfer_cells`, derived from
// cute/atom/copy_traits_sm100.hpp). Thread t, register q -> (lane offset, column offset):
//   32x32b:   (t, q)
//   16x64b:   (8 (t & 1) + (t >> 2), ((t >> 1) & 1) + 2 q)
//   16x128b:  ((t >> 2) + 8 (q & 1), (t & 3) + 4 (q >> 1))
//   16x256b:  ((t >> 2) + 8 ((q >> 1) & 1), 2 (t & 3) + (q & 1) + 8 (q >> 2))
// The 16-lane shapes are run at lane offsets 0 and 16 of the warp's window.
//
//   transfer_shapes
#include "../common.h"

#define COLS 32
#define LOAD_TEST 0
#define STORE_TEST 1

struct Shape { const char* name; int registers; };
static const Shape SHAPES[] = {{"32x32b.x1", 1},  {"32x32b.x4", 4},  {"16x64b.x1", 1},  {"16x64b.x2", 2},
                               {"16x128b.x1", 2}, {"16x128b.x2", 4}, {"16x256b.x1", 4}, {"16x256b.x2", 8}};

#define LD1(shape, r, a) asm volatile("tcgen05.ld.sync.aligned." shape ".b32 {%0}, [%1];" : "=r"(r[0]) : "r"(a))
#define LD2(shape, r, a) \
    asm volatile("tcgen05.ld.sync.aligned." shape ".b32 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a))
#define LD4(shape, r, a)                                                    \
    asm volatile("tcgen05.ld.sync.aligned." shape ".b32 {%0, %1, %2, %3}, [%4];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a))
#define LD8(shape, r, a)                                                                        \
    asm volatile("tcgen05.ld.sync.aligned." shape ".b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" \
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) \
                 : "r"(a))
#define ST1(shape, r, a) \
    asm volatile("tcgen05.st.sync.aligned." shape ".b32 [%0], {%1};" :: "r"(a), "r"(r[0]) : "memory")
#define ST2(shape, r, a) \
    asm volatile("tcgen05.st.sync.aligned." shape ".b32 [%0], {%1, %2};" :: "r"(a), "r"(r[0]), "r"(r[1]) : "memory")
#define ST4(shape, r, a)                                                    \
    asm volatile("tcgen05.st.sync.aligned." shape ".b32 [%0], {%1, %2, %3, %4};" \
                 :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory")
#define ST8(shape, r, a)                                                                        \
    asm volatile("tcgen05.st.sync.aligned." shape ".b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" \
                 :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) \
                 : "memory")

// 128 threads. `row0` is the lane offset inside each warp's window for the shape under test.
// LOAD_TEST writes out[thread][register] (8 per thread); STORE_TEST writes out[thread][column].
extern "C" __global__ void transfer_shapes(uint32_t test, uint32_t shape, uint32_t row0, uint32_t* out) {
    __shared__ uint32_t tmem_base;
    uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    if (warp == 0) tmem_alloc(&tmem_base, COLS);
    __syncthreads();
    uint32_t base = tmem_base;
    uint32_t own = base + ((warp * 32) << 16);
    uint32_t at = own + (row0 << 16);

    for (uint32_t c = 0; c < COLS; ++c)
        tmem_st_32x32b(own + c, test == LOAD_TEST ? (0xC0000000u | (tid << 8) | c) : 0xEEEEEEEEu);
    tmem_wait_st();

    uint32_t r[8];
    for (uint32_t q = 0; q < 8; ++q) r[q] = test == LOAD_TEST ? 0u : (0xD0000000u | (lane << 8) | q);
    if (test == LOAD_TEST) {
        switch (shape) {
            case 0: LD1("32x32b.x1", r, at); break;
            case 1: LD4("32x32b.x4", r, at); break;
            case 2: LD1("16x64b.x1", r, at); break;
            case 3: LD2("16x64b.x2", r, at); break;
            case 4: LD2("16x128b.x1", r, at); break;
            case 5: LD4("16x128b.x2", r, at); break;
            case 6: LD4("16x256b.x1", r, at); break;
            case 7: LD8("16x256b.x2", r, at); break;
        }
        tmem_wait_ld();
        for (uint32_t q = 0; q < 8; ++q) out[tid * 8 + q] = r[q];
    } else {
        switch (shape) {
            case 0: ST1("32x32b.x1", r, at); break;
            case 1: ST4("32x32b.x4", r, at); break;
            case 2: ST1("16x64b.x1", r, at); break;
            case 3: ST2("16x64b.x2", r, at); break;
            case 4: ST2("16x128b.x1", r, at); break;
            case 5: ST4("16x128b.x2", r, at); break;
            case 6: ST4("16x256b.x1", r, at); break;
            case 7: ST8("16x256b.x2", r, at); break;
        }
        tmem_wait_st();
        for (uint32_t c = 0; c < COLS; ++c) {
            uint32_t v = tmem_ld_32x32b(own + c);
            tmem_wait_ld();
            out[tid * COLS + c] = v;
        }
    }
    __syncthreads();
    if (warp == 0) tmem_dealloc(base, COLS);
}

// cell[w][t][q] = (lane relative to the warp's window, column), or (-1, -1).
struct Cell { int lane, column; };

static void report(const Shape& shape, const char* test, uint32_t row0, Cell cells[4][32][8], int extra) {
    printf("%s %s at lane offset %u: thread -> (lane, column) of each register\n", test, shape.name, row0);
    for (int t = 0; t < 32; ++t) {
        printf("  thread %2d:", t);
        for (int q = 0; q < shape.registers; ++q) {
            if (cells[0][t][q].lane < 0) printf(" (  ?,  ?)");
            else printf(" (%3d,%3d)", cells[0][t][q].lane, cells[0][t][q].column);
        }
        printf("\n");
    }
    bool same = true;
    for (int w = 1; w < 4; ++w)
        for (int t = 0; t < 32; ++t)
            for (int q = 0; q < shape.registers; ++q)
                same = same && cells[w][t][q].lane == cells[0][t][q].lane
                       && cells[w][t][q].column == cells[0][t][q].column;
    printf("  warps 1-3 show the same pattern in their own windows: %s", same ? "yes" : "NO");
    if (extra >= 0) printf("; cells changed that no (thread, register) value explains: %d", extra);
    printf("\n");
}

static void run(int shape_index, uint32_t row0, uint32_t* d_out) {
    const Shape& shape = SHAPES[shape_index];
    static uint32_t out[128 * COLS];
    static Cell cells[4][32][8];

    dev_fill(d_out, 128 * COLS, 0);
    transfer_shapes<<<1, 128>>>(LOAD_TEST, (uint32_t)shape_index, row0, d_out);
    probe_sync("transfer_shapes, load test");
    dev_read(out, d_out, 128 * 8);
    for (int w = 0; w < 4; ++w)
        for (int t = 0; t < 32; ++t)
            for (int q = 0; q < 8; ++q) {
                uint32_t v = out[(w * 32 + t) * 8 + q];
                bool valid = (v >> 28) == 0xC && ((v >> 8) & 0xfffff) < 128 && (v & 0xff) < COLS;
                cells[w][t][q].lane = valid ? (int)((v >> 8) & 0xfffff) - 32 * w : -1;
                cells[w][t][q].column = valid ? (int)(v & 0xff) : -1;
            }
    report(shape, "ld", row0, cells, -1);

    dev_fill(d_out, 128 * COLS, 0);
    transfer_shapes<<<1, 128>>>(STORE_TEST, (uint32_t)shape_index, row0, d_out);
    probe_sync("transfer_shapes, store test");
    dev_read(out, d_out, 128 * COLS);
    int extra = 0;
    for (int w = 0; w < 4; ++w) {
        for (int t = 0; t < 32; ++t)
            for (int q = 0; q < 8; ++q) cells[w][t][q].lane = cells[w][t][q].column = -1;
        for (int lane = 0; lane < 32; ++lane)
            for (int c = 0; c < COLS; ++c) {
                uint32_t v = out[(w * 32 + lane) * COLS + c];
                if (v == 0xEEEEEEEEu) continue;
                uint32_t t = (v >> 8) & 0xfffff, q = v & 0xff;
                if ((v >> 28) == 0xD && t < 32 && q < (uint32_t)shape.registers && cells[w][t][q].lane < 0) {
                    cells[w][t][q].lane = lane;
                    cells[w][t][q].column = c;
                } else {
                    ++extra;
                }
            }
    }
    report(shape, "st", row0, cells, extra);
}

int main() {
    uint32_t* d_out = dev_alloc<uint32_t>(128 * COLS);
    for (int s = 0; s < 8; ++s) {
        run(s, 0, d_out);
        if (s >= 2) run(s, 16, d_out);
    }
    return 0;
}
