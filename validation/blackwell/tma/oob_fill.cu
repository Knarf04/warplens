// TMA loads whose box is partly or wholly outside the tensor.
//
// The tensor is 6 rows x 8 columns. Element (row r, column c) holds the number 100 r + c + 1
// in the tensor's type, never zero. A box of 4 rows x 8 columns is loaded at coordinates that
// leave some of it outside, with out-of-bounds fill ZERO and NAN, and printed as raw bits.
//
// Claims tested (docs/history/m2.md, "Assumptions"):
//   * elements outside the tensor are filled, with zero or with a NaN, and elements inside
//     are loaded; coordinates may be negative;
//   * the NaN is 0x7ff7 in every 16 bits of the element (0x7ff7, 0x7ff77ff7,
//     0x7ff77ff77ff77ff7): measured on a B200, 2026-11-25. It is not documented;
//   * complete_tx counts the WHOLE box, filled bytes included: every load here expects the
//     full box and reports whether the mbarrier phase completed.
// Case `inbounds`: the same loads with expect_tx equal to the bytes actually inside the
// tensor. On a B200 these phases complete, and that alone does not say the fill is not
// counted: a copy reports its bytes in pieces, so a count smaller than the copy reaches zero
// part of the way through (mbarrier/expect_tx, "too small", shows the same with no fill at
// all). The buffer is therefore copied only after a settle (tma_dump.h): read at once, one
// filled row still held the sentinel.
//
//   oob_fill [inbounds]
#include "tma_dump.h"

#define ROWS 6
#define COLS 8
#define BOX_ROWS 4

struct Type {
    const char* name;
    CUtensorMapDataType type;
    int bytes;
    bool is_float;
};
static const Type TYPES[] = {{"float32", CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, true},
                             {"float16", CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, true},
                             {"float64", CU_TENSOR_MAP_DATA_TYPE_FLOAT64, 8, true},
                             {"uint32", CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, false}};

// IEEE half for a small positive integer.
static uint16_t half_of(int v) {
    int e = 0;
    while ((v >> (e + 1)) != 0) ++e;
    return (uint16_t)(((e + 15) << 10) | (((v << (10 - e)) & 0x3ff)));
}

static void fill_tensor(const Type& t, uint8_t* raw) {
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            int v = 100 * r + c + 1;
            uint8_t* p = raw + (r * COLS + c) * t.bytes;
            if (t.bytes == 2) { uint16_t h = half_of(v); memcpy(p, &h, 2); }
            else if (t.bytes == 8) { double d = v; memcpy(p, &d, 8); }
            else if (t.is_float) { float f = (float)v; memcpy(p, &f, 4); }
            else { uint32_t u = (uint32_t)v; memcpy(p, &u, 4); }
        }
}

static void print_element(const uint8_t* p, int bytes) {
    for (int i = bytes - 1; i >= 0; --i) printf("%02x", p[i]);
}

static void run(const Type& t, bool nan_fill, int c0, int c1, bool inbounds_only, void* d_tensor) {
    const uint64_t dims[2] = {COLS, ROWS};
    const uint32_t box[2] = {COLS, BOX_ROWS};
    CUtensorMap map = tensor_map(t.type, t.bytes, 2, d_tensor, dims, box, nullptr, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 nan_fill ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
                                          : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    int inside = 0;
    for (int r = c1; r < c1 + BOX_ROWS; ++r)
        for (int c = c0; c < c0 + COLS; ++c) inside += r >= 0 && r < ROWS && c >= 0 && c < COLS;
    uint32_t box_bytes = COLS * BOX_ROWS * t.bytes;
    uint32_t expect = inbounds_only ? (uint32_t)(inside * t.bytes) : box_bytes;
    const int coords[2] = {c0, c1};
    Dump dump = load_and_dump(map, 2, coords, 0, expect, "tma_dump", inbounds_only);
    int first, last;
    dump.extent(&first, &last);
    printf("%s, fill %s, box at column %d, row %d: %d of %d elements inside\n", t.name, nan_fill ? "NaN" : "zero",
           c0, c1, inside, COLS * BOX_ROWS);
    printf("  expect_tx %u of %u box bytes: phase completed %u; buffer bytes changed in [%d, %d)\n", expect,
           box_bytes, dump.completed, first, last);
    for (int r = 0; r < BOX_ROWS; ++r) {
        printf("  box row %d (tensor row %2d):", r, c1 + r);
        for (int c = 0; c < COLS; ++c) {
            printf(" ");
            print_element(dump.bytes() + (r * COLS + c) * t.bytes, t.bytes);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    bool inbounds = argc > 1 && !strcmp(argv[1], "inbounds");
    static uint8_t raw[ROWS * COLS * 8];
    // Column coordinates are multiples of 16 bytes for every type (8 float16 = 16 bytes).
    static const int coords[][2] = {{0, 0},   {8, 0},  {-8, 0},   {0, 4},   {0, -2},
                                    {0, 6},   {0, -4}, {0, 1000}, {0, -1000}, {-8, -2}};
    for (const Type& t : TYPES) {
        fill_tensor(t, raw);
        uint8_t* d_tensor = dev_copy(raw, sizeof raw);
        for (int fill = 0; fill < (t.is_float ? 2 : 1); ++fill) {
            if (inbounds) {
                if (t.bytes != 4) continue;
                // Two rows inside. (A box wholly outside would expect zero bytes, and the
                // phase would complete at the arrival whatever the load reported.)
                run(t, fill, 0, 4, true, d_tensor);
            } else {
                for (const auto& c : coords) run(t, fill, c[0], c[1], false, d_tensor);
            }
        }
    }
    if (!inbounds) {
        // Half-width boxes of float32, so that a column offset of 4 elements (16 bytes) puts
        // the box across the right and left edges.
        const Type& t = TYPES[0];
        fill_tensor(t, raw);
        uint8_t* d_tensor = dev_copy(raw, sizeof raw);
        printf("float32 with column offsets of 4 elements:\n");
        static const int more[][2] = {{4, 0}, {-4, 0}, {4, 4}, {-4, -2}};
        for (int fill = 0; fill < 2; ++fill)
            for (const auto& c : more) run(t, fill, c[0], c[1], false, d_tensor);
    }
    return 0;
}
