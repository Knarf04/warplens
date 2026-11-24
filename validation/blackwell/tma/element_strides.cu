// Element strides of a tiled tensor map.
//
// The tensor is uint32, 16 columns x 16 rows (x 4 planes for the rank-3 cases). Element
// (plane p, row r, column c) holds 0x10000000 | p << 16 | r << 8 | c. One box is loaded at
// the origin and every word the load changed is printed with the element it names.
//
// Claims tested (warplens/host/tensor_map.py, `TensorMapInfo.counts`):
//   * with element stride s in dimension d > 0, the box size counts TRAVERSED elements, and
//     ceil(box / s) of them are transferred: indices 0, s, 2s, ...;
//   * the transferred elements are packed densely in shared memory;
//   * complete_tx reports the bytes transferred, not box-size bytes. Each load expects
//     ceil-model bytes and reports whether the phase completed, and how many buffer bytes
//     changed (which is the answer even if the phase did not complete).
// Case `inner` (own process; the driver may refuse to encode it): an element stride other
// than 1 in dimension 0. The simulator ignores it, as cuTensorMapEncodeTiled documents for
// non-interleaved tensors.
//
//   element_strides [inner]
#include "tma_dump.h"

static void run(int rank, const uint32_t* box, const uint32_t* strides, uint32_t* d_tensor) {
    const uint64_t dims[3] = {16, 16, 4};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, rank, d_tensor, dims, box, strides,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    uint32_t expect = 4 * box[0];  // dimension 0 is assumed to move every element
    long long b[3], s[3];
    for (int i = 0; i < rank; ++i) {
        if (i) expect *= (box[i] + strides[i] - 1) / strides[i];
        b[i] = box[i];
        s[i] = strides[i];
    }
    const int coords[3] = {0, 0, 0};
    Dump dump = load_and_dump(map, rank, coords, 0, expect, "tma_dump");
    int first, last;
    dump.extent(&first, &last);
    printf("rank %d: ", rank);
    print_tuple("box", b, rank);
    print_tuple(", element strides", s, rank);
    printf(" (dimension 0 first)\n");
    printf("  expect_tx %u bytes: phase completed %u; buffer bytes changed in [%d, %d)\n", expect, dump.completed,
           first, last);
    int words = last < 0 ? 0 : (last + 3) / 4;
    for (int i = 0; i < words; ++i) {
        uint32_t v = dump.words[i];
        if (i % 4 == 0) printf("  words %3d..:", i);
        if (v == SENTINEL) printf("  (unchanged)");
        else if ((v >> 24) == 0x10) printf("  (p%u r%2u c%2u)", (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff);
        else printf("  %08x    ", v);
        if (i % 4 == 3 || i == words - 1) printf("\n");
    }
}

int main(int argc, char** argv) {
    static uint32_t tensor[4 * 16 * 16];
    for (uint32_t p = 0; p < 4; ++p)
        for (uint32_t r = 0; r < 16; ++r)
            for (uint32_t c = 0; c < 16; ++c) tensor[(p * 16 + r) * 16 + c] = 0x10000000u | p << 16 | r << 8 | c;
    uint32_t* d_tensor = dev_copy(tensor, 4 * 16 * 16);
    struct Case { int rank; uint32_t box[3], strides[3]; };
    if (argc > 1 && !strcmp(argv[1], "inner")) {
        static const Case cases[] = {{2, {8, 2}, {2, 1}}, {2, {8, 4}, {2, 2}}, {2, {8, 2}, {4, 1}}};
        for (const Case& c : cases) run(c.rank, c.box, c.strides, d_tensor);
        return 0;
    }
    static const Case cases[] = {
        {2, {4, 4}, {1, 1}},        // control
        {2, {4, 4}, {1, 2}},        // rows 0, 2
        {2, {4, 4}, {1, 3}},        // rows 0, 3: ceil(4 / 3) = 2
        {2, {4, 5}, {1, 2}},        // rows 0, 2, 4: ceil(5 / 2) = 3
        {2, {4, 8}, {1, 8}},        // row 0 only
        {2, {4, 1}, {1, 2}},        // box smaller than the stride
        {3, {4, 4, 4}, {1, 2, 1}},  // rows 0, 2 of planes 0..3
        {3, {4, 4, 4}, {1, 2, 3}},  // rows 0, 2 of planes 0, 3
        {3, {4, 3, 3}, {1, 2, 2}},  // rows 0, 2 of planes 0, 2
    };
    for (const Case& c : cases) run(c.rank, c.box, c.strides, d_tensor);
    return 0;
}
