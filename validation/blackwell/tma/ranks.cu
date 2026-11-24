// TMA tensor loads of rank 1 to 5.
//
// The tensor holds uint32 elements whose value is 0x10000000 + the element's linear index,
// so every loaded word names the element it came from. One box is loaded at a non-zero
// coordinate in every dimension and the shared buffer is printed in order.
//
// Claims tested (docs/m2.md; warplens/host/tensor_map.py `element_addresses`):
//   * coordinates and box sizes are in elements, dimension 0 innermost;
//   * the box lands in shared memory densely, dimension 0 fastest;
//   * complete_tx reports box-size bytes (the mbarrier phase completes with that expect_tx);
//   * nothing outside the box's bytes is written.
// Case `unaligned` (own process): an innermost coordinate that is not a multiple of 16
// bytes. The simulator refuses it; whether hardware accepts it is not known.
//
//   ranks [unaligned]
#include "tma_dump.h"

static const uint64_t DIMS[5] = {8, 6, 5, 4, 3};
static const uint32_t BOX[5] = {4, 2, 2, 2, 2};

static void run(int rank, const int* coords, uint32_t* d_tensor) {
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, rank, d_tensor, DIMS, BOX, nullptr,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    uint32_t box_bytes = 4;
    long long dims[5], box[5], at[5];
    for (int i = 0; i < rank; ++i) {
        box_bytes *= BOX[i];
        dims[i] = (long long)DIMS[i];
        box[i] = BOX[i];
        at[i] = coords[i];
    }
    Dump dump = load_and_dump(map, rank, coords, 0, box_bytes, "tma_dump");
    printf("rank %d: ", rank);
    print_tuple("dims", dims, rank);
    print_tuple(", box", box, rank);
    print_tuple(", coordinates", at, rank);
    printf(" (dimension 0 first)\n");
    int first, last;
    dump.extent(&first, &last);
    printf("  expect_tx %u bytes: phase completed %u; bytes written %d, in buffer range [%d, %d)\n", box_bytes,
           dump.completed, dump.written(), first, last);
    for (uint32_t i = 0; i < box_bytes / 4; ++i) {
        uint32_t v = dump.words[i];
        printf("  word %2u = %08x", i, v);
        uint32_t index = v - 0x10000000u;
        if (v != SENTINEL && index < 8 * 6 * 5 * 4 * 3) {
            printf("  element (");
            for (int d = 0; d < rank; ++d) {
                printf("%s%u", d ? ", " : "", (unsigned)(index % DIMS[d]));
                index /= (uint32_t)DIMS[d];
            }
            printf(")");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    static uint32_t tensor[8 * 6 * 5 * 4 * 3];
    for (uint32_t i = 0; i < 8 * 6 * 5 * 4 * 3; ++i) tensor[i] = 0x10000000u + i;
    uint32_t* d_tensor = dev_copy(tensor, 8 * 6 * 5 * 4 * 3);
    if (argc > 1 && !strcmp(argv[1], "unaligned")) {
        const int coords[5] = {1, 1, 0, 0, 0};  // innermost coordinate 1 = byte offset 4
        printf("innermost coordinate 1 (4 bytes, not a multiple of 16):\n");
        run(2, coords, d_tensor);
        return 0;
    }
    const int coords[5] = {4, 1, 2, 1, 1};
    for (int rank = 1; rank <= 5; ++rank) run(rank, coords, d_tensor);
    return 0;
}
