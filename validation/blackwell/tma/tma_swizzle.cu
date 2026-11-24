// Where a swizzled TMA load puts each 16-byte chunk, with aligned and misaligned buffers.
//
// The tensor is 16 rows of uint16; a row is as wide as the swizzle span (32, 64 or 128
// bytes; 128 for no swizzle). Chunk c of the box (16 bytes, in row-major order) holds the
// values 0x4000 + 8c .. 8c + 7, so every chunk found in shared memory names its source.
// The whole box is loaded to the address buffer + 1024 + offset, where the buffer is
// aligned to 1024 bytes, and the buffer is printed as a map of 128-byte rows.
//
// Claim tested (docs/m2.md, "Assumptions"): the swizzle is a function of the shared-memory
// ADDRESS (CuTe's Swizzle<B,4,3>, B = 1, 2, 3 for 32B, 64B, 128B: the 16-byte chunk index
// within a 128-byte row is XORed with the low B bits of the row index), not of the offset
// into the destination buffer. The two agree when the destination is aligned to the swizzle
// period (256, 512, 1024 bytes) and differ at the misaligned offsets used here. If hardware
// swizzled by offset, every offset would print the same pattern as offset 0, shifted.
//
// The probe also assumes that the shared address of an `alignas(1024)` buffer is a multiple
// of 1024, which it prints.
//
//   tma_swizzle                  offsets 0, 128, 256, 512 for every mode
//   tma_swizzle sixteen          offset 16: the destination is not aligned to a 128-byte row
#include "tma_dump.h"

static const int SPAN[4] = {128, 32, 64, 128};  // bytes per tensor row, by swizzle mode

static void run(int mode, int offset, uint16_t* d_tensor) {
    const uint64_t dims[2] = {(uint64_t)SPAN[mode] / 2, 16};
    const uint32_t box[2] = {(uint32_t)SPAN[mode] / 2, 16};
    CUtensorMap map = tensor_map(CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, 2, d_tensor, dims, box, nullptr,
                                 (CUtensorMapSwizzle)mode, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    const int coords[2] = {0, 0};
    uint32_t bytes = (uint32_t)SPAN[mode] * 16;
    Dump dump = load_and_dump(map, 2, coords, 1024 + offset, bytes, "tma_dump");
    printf("swizzle %s, box 16 rows x %d bytes = %u bytes (%u chunks), destination = buffer + 1024 + %d\n",
           swizzle_name(mode), SPAN[mode], bytes, bytes / 16, offset);
    printf("  buffer address %% 1024 = %u; phase completed %u\n", dump.shared_address % 1024, dump.completed);
    printf("info: shared address of the buffer 0x%x\n", dump.shared_address);
    printf("  buffer row (128 bytes each): source chunk found in each of its eight 16-byte chunks\n");
    const uint16_t* v = (const uint16_t*)dump.words;
    int strange = 0;
    for (int row = 0; row < DUMP_BYTES / 128; ++row) {
        char text[8][8];
        bool any = false;
        for (int chunk = 0; chunk < 8; ++chunk) {
            const uint16_t* p = v + (row * 128 + chunk * 16) / 2;
            bool untouched = true, clean = p[0] >= 0x4000 && (p[0] - 0x4000) % 8 == 0;
            for (int i = 0; i < 8; ++i) {
                untouched = untouched && p[i] == 0xEEEE;
                clean = clean && p[i] == p[0] + i;
            }
            if (untouched) snprintf(text[chunk], 8, "  .");
            else if (clean) snprintf(text[chunk], 8, "%3d", (p[0] - 0x4000) / 8);
            else { snprintf(text[chunk], 8, " ??"); ++strange; }
            any = any || !untouched;
        }
        if (any) {
            printf("  row %2d (bytes %4d..):", row, row * 128);
            for (int chunk = 0; chunk < 8; ++chunk) printf(" %s", text[chunk]);
            printf("\n");
        }
    }
    if (strange) {
        printf("  %d chunk(s) hold something other than one whole source chunk; raw buffer follows\n", strange);
        for (int i = 0; i < DUMP_BYTES / 16; ++i) {
            bool untouched = true;
            for (int j = 0; j < 8; ++j) untouched = untouched && v[i * 8 + j] == 0xEEEE;
            if (untouched) continue;
            printf("  bytes %4d:", i * 16);
            for (int j = 0; j < 8; ++j) printf(" %04x", v[i * 8 + j]);
            printf("\n");
        }
    }
}

int main(int argc, char** argv) {
    static uint16_t tensor[1024];
    for (int i = 0; i < 1024; ++i) tensor[i] = (uint16_t)(0x4000 + i);
    uint16_t* d_tensor = dev_copy(tensor, 1024);
    if (argc > 1 && !strcmp(argv[1], "sixteen")) {
        for (int mode = 0; mode < 4; ++mode) run(mode, 16, d_tensor);
        return 0;
    }
    const int offsets[] = {0, 128, 256, 512};
    for (int mode = 0; mode < 4; ++mode)
        for (int offset : offsets) run(mode, offset, d_tensor);
    return 0;
}
