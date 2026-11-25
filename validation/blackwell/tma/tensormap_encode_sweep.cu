// Sweep of the driver's cuTensorMapEncodeTiled: every parameter set with the 128 bytes the
// driver wrote, or the error it returned.
//
// Host only; no kernel is launched. This probe has NO predicted output: the simulator has
// its own tensor-map encoding. Its output is the ground truth for writing a decoder of real
// tensor maps (docs/history/m6.md, "Not working: kernels that use TMA": the CuTe DSL writes the
// driver's layout itself). The driver encodes only on a GPU that has TMA, so this cannot be
// produced on the development machine.
//
// Each line is
//   type rank address dims strides box elementStrides interleave swizzle l2 fill -> status bytes
// with dimension 0 (innermost) first, `strides` the rank - 1 byte strides passed to the
// driver, and `bytes` the tensor map as 16 little-endian 64-bit words, lowest address first.
// A parameter is varied alone from a baseline first (so its field can be located), then in
// combinations, then one bit at a time.
//
//   tensormap_encode_sweep
#include "../common.h"

static probe_encode_fn encode;
static uint64_t BASE;  // a real device allocation
static int lines = 0, failures = 0;

struct Params {
    int type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    int rank = 2;
    uint64_t address = 0;
    uint64_t dims[5] = {64, 32, 4, 4, 4};
    uint64_t strides[5] = {0, 0, 0, 0, 0};  // 0: derived (dense, rounded up to 16 bytes)
    uint32_t box[5] = {16, 8, 2, 2, 2};
    uint32_t estrides[5] = {1, 1, 1, 1, 1};
    int interleave = 0, swizzle = 0, l2 = 0, fill = 0;
};

static int element_bytes(int type) {
    switch (type) {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8: return 1;
        case CU_TENSOR_MAP_DATA_TYPE_UINT16: case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 2;
        case CU_TENSOR_MAP_DATA_TYPE_UINT64: case CU_TENSOR_MAP_DATA_TYPE_INT64:
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: return 8;
        default: return 4;
    }
}

static void emit(const char* group, Params p) {
    uint64_t stride = element_bytes(p.type);
    cuuint64_t dims[5], strides[5];
    cuuint32_t box[5], estrides[5];
    for (int i = 0; i < 5; ++i) {
        dims[i] = p.dims[i];
        box[i] = p.box[i];
        estrides[i] = p.estrides[i];
        stride *= p.dims[i];
        stride = (stride + 15) / 16 * 16;
        if (p.strides[i]) stride = p.strides[i];
        strides[i] = stride;
    }
    CUtensorMap map;
    memset(&map, 0xA5, sizeof map);  // bytes the driver leaves alone stay visible
    CUresult status = encode(&map, (CUtensorMapDataType)p.type, (cuuint32_t)p.rank, (void*)p.address, dims,
                             strides, box, estrides, (CUtensorMapInterleave)p.interleave,
                             (CUtensorMapSwizzle)p.swizzle, (CUtensorMapL2promotion)p.l2,
                             (CUtensorMapFloatOOBfill)p.fill);
    int shown = p.rank >= 1 && p.rank <= 5 ? p.rank : 5;
    printf("%s type=%d rank=%d address=0x%llx base+0x%llx dims=", group, p.type, p.rank,
           (unsigned long long)p.address, (unsigned long long)(p.address - BASE));
    for (int i = 0; i < shown; ++i) printf("%s%llu", i ? "," : "", (unsigned long long)dims[i]);
    printf(" strides=");
    for (int i = 0; i < shown - 1; ++i) printf("%s%llu", i ? "," : "", (unsigned long long)strides[i]);
    printf(" box=");
    for (int i = 0; i < shown; ++i) printf("%s%u", i ? "," : "", box[i]);
    printf(" estrides=");
    for (int i = 0; i < shown; ++i) printf("%s%u", i ? "," : "", estrides[i]);
    printf(" interleave=%d swizzle=%d l2=%d fill=%d -> %d", p.interleave, p.swizzle, p.l2, p.fill, (int)status);
    if (status == CUDA_SUCCESS) {
        const uint64_t* words = (const uint64_t*)&map;
        for (int i = 0; i < 16; ++i) printf(" %016llx", (unsigned long long)words[i]);
    } else {
        ++failures;
    }
    printf("\n");
    ++lines;
}

int main() {
    encode = probe_encoder();
    BASE = (uint64_t)dev_alloc<uint8_t>(1 << 20);
    Params base;
    base.address = BASE;
    printf("# base allocation at 0x%llx; CU_TENSOR_MAP_NUM_QWORDS = %d\n", (unsigned long long)BASE,
           (int)CU_TENSOR_MAP_NUM_QWORDS);

    emit("baseline", base);

    // A. data types, including values past the end of the enumeration.
    for (int type = 0; type <= 18; ++type)
        for (int swizzle : {0, 3}) {
            Params p = base;
            p.type = type;
            p.swizzle = swizzle;
            emit("type", p);
        }

    // B. ranks, including invalid ones.
    for (int rank = 0; rank <= 6; ++rank) {
        Params p = base;
        p.rank = rank;
        emit("rank", p);
    }

    // C. one dimension at a time (rank 3, uint8 so that any size is a whole number of bytes).
    static const uint64_t sizes[] = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 255, 256, 257, 4095, 4096,
                                     65535, 65536, 65537, 1ull << 20, 1ull << 24, (1ull << 31) - 1, 1ull << 31,
                                     (1ull << 32) - 1, 1ull << 32, (1ull << 32) + 1};
    for (int d = 0; d < 3; ++d)
        for (uint64_t size : sizes) {
            Params p = base;
            p.type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
            p.rank = 3;
            p.dims[0] = 64; p.dims[1] = 4; p.dims[2] = 4;
            p.box[0] = 16; p.box[1] = 1; p.box[2] = 1;
            p.dims[d] = size;
            emit("dim", p);
        }

    // D. byte strides, one at a time (rank 3 float32, dims 64 x 32 x 4: dense strides 256, 8192).
    static const uint64_t stride0[] = {16, 128, 240, 256, 257, 264, 272, 512, 4096, 1ull << 20, 1ull << 32,
                                       1ull << 36, (1ull << 40) - 16, 1ull << 40};
    for (uint64_t s : stride0) {
        Params p = base;
        p.rank = 3;
        p.strides[0] = s;
        p.strides[1] = s * 32 < 8192 ? 8192 : s * 32;
        emit("stride0", p);
    }
    static const uint64_t stride1[] = {256, 4096, 8176, 8192, 8208, 16384, 1ull << 20, 1ull << 32, 1ull << 36,
                                       (1ull << 40) - 16, 1ull << 40};
    for (uint64_t s : stride1) {
        Params p = base;
        p.rank = 3;
        p.strides[1] = s;
        emit("stride1", p);
    }

    // E. box sizes, one dimension at a time (uint8: the inner box must be a multiple of 16 bytes).
    static const uint32_t inner[] = {1, 8, 15, 16, 17, 32, 48, 64, 96, 128, 240, 256, 257, 512};
    static const uint32_t outer[] = {0, 1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 128, 255, 256, 257};
    for (uint32_t b : inner) {
        Params p = base;
        p.type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        p.rank = 3;
        p.dims[0] = 1024; p.dims[1] = 512; p.dims[2] = 512;
        p.box[0] = b;
        emit("box0", p);
    }
    for (int d = 1; d < 3; ++d)
        for (uint32_t b : outer) {
            Params p = base;
            p.type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
            p.rank = 3;
            p.dims[0] = 1024; p.dims[1] = 512; p.dims[2] = 512;
            p.box[d] = b;
            emit(d == 1 ? "box1" : "box2", p);
        }

    // F. element strides, one dimension at a time.
    for (int d = 0; d < 3; ++d)
        for (uint32_t s = 0; s <= 9; ++s) {
            Params p = base;
            p.rank = 3;
            p.estrides[d] = s;
            emit("estride", p);
        }

    // G. interleave with every type size, with and without swizzle.
    for (int interleave = 0; interleave <= 3; ++interleave)
        for (int type : {CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_UINT16, CU_TENSOR_MAP_DATA_TYPE_UINT32,
                         CU_TENSOR_MAP_DATA_TYPE_FLOAT32, CU_TENSOR_MAP_DATA_TYPE_UINT64})
            for (int swizzle : {0, 1, 3})
                for (uint32_t estride0 : {1u, 2u, 4u}) {
                    Params p = base;
                    p.type = type;
                    p.rank = 3;
                    p.interleave = interleave;
                    p.swizzle = swizzle;
                    p.estrides[0] = estride0;
                    p.box[0] = 32 / element_bytes(type);
                    emit("interleave", p);
                }

    // H. swizzle modes (past the documented ones too) against the inner box size in bytes.
    for (int swizzle = 0; swizzle <= 8; ++swizzle)
        for (uint32_t bytes : {16u, 32u, 64u, 128u, 256u})
            for (int type : {CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                             CU_TENSOR_MAP_DATA_TYPE_FLOAT32}) {
                Params p = base;
                p.type = type;
                p.swizzle = swizzle;
                p.dims[0] = 1024;
                p.box[0] = bytes / element_bytes(type);
                emit("swizzle", p);
            }

    // I. L2 promotion.
    for (int l2 = 0; l2 <= 4; ++l2)
        for (int swizzle : {0, 3})
            for (int type : {CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_FLOAT32}) {
                Params p = base;
                p.type = type;
                p.l2 = l2;
                p.swizzle = swizzle;
                p.dims[0] = 1024;
                p.box[0] = 64 / element_bytes(type);
                emit("l2", p);
            }

    // J. out-of-bounds fill for every type.
    for (int fill = 0; fill <= 2; ++fill)
        for (int type = 0; type <= 15; ++type) {
            Params p = base;
            p.type = type;
            p.fill = fill;
            emit("fill", p);
        }

    // K. addresses: offsets into the allocation, misaligned ones, and values that are not
    // allocations at all (the encoder may or may not look).
    static const uint64_t offsets[] = {0, 1, 4, 8, 16, 32, 48, 64, 128, 256, 4096, 0x10000, 0xffff0};
    for (uint64_t offset : offsets) {
        Params p = base;
        p.address = BASE + offset;
        emit("address", p);
    }
    static const uint64_t synthetic[] = {0, 0x10, 0x1000, 0x7f0123456780ull, 0x0000fffffffffff0ull,
                                         0x00fffffffffffff0ull, 0xfffffffffffffff0ull};
    for (uint64_t address : synthetic) {
        Params p = base;
        p.address = address;
        emit("address", p);
    }

    // L. combinations: type x rank x swizzle x fill, with distinct sizes in every dimension.
    for (int type : {CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_UINT16, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                     CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_DATA_TYPE_UINT32, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT64})
        for (int rank = 1; rank <= 5; ++rank)
            for (int swizzle = 0; swizzle <= 3; ++swizzle)
                for (int fill = 0; fill <= 1; ++fill) {
                    Params p = base;
                    p.type = type;
                    p.rank = rank;
                    p.swizzle = swizzle;
                    p.fill = fill;
                    p.address = BASE + 0x100 * rank;
                    const uint64_t dims[5] = {96, 40, 24, 12, 6};
                    const uint32_t box[5] = {(uint32_t)(32 / element_bytes(type)), 5, 3, 2, 1};
                    const uint32_t estrides[5] = {1, 1, 2, 1, 3};
                    for (int i = 0; i < 5; ++i) {
                        p.dims[i] = dims[i];
                        p.box[i] = box[i];
                        p.estrides[i] = estrides[i];
                    }
                    emit("combo", p);
                }
    // The shapes CUTLASS's Blackwell tutorials use: float16, rank 2, 128B swizzle, 64-wide rows.
    for (uint64_t rows : {128ull, 256ull, 512ull, 1024ull})
        for (uint32_t box_rows : {32u, 64u, 128u}) {
            Params p = base;
            p.type = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
            p.swizzle = 3;
            p.dims[0] = 128; p.dims[1] = rows;
            p.box[0] = 64; p.box[1] = box_rows;
            emit("gemm", p);
        }

    // M. one bit at a time, rank 5, uint8, so that every field can be located and sized.
    Params walk = base;
    walk.type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    walk.rank = 5;
    for (int i = 0; i < 5; ++i) { walk.dims[i] = 16; walk.box[i] = 1; }
    walk.box[0] = 16;
    walk.address = 0;
    emit("walk-baseline", walk);
    for (int bit = 4; bit < 64; ++bit) {
        Params p = walk;
        p.address = 1ull << bit;
        emit("walk-address", p);
    }
    for (int d = 0; d < 5; ++d)
        for (int bit = 0; bit <= 32; ++bit) {
            Params p = walk;
            p.dims[d] = 1ull << bit;
            // Keep the strides fixed and large, so that only the dimension changes.
            for (int i = 0; i < 4; ++i) p.strides[i] = 1ull << (36 + i);
            emit("walk-dim", p);
        }
    for (int d = 0; d < 4; ++d)
        for (int bit = 4; bit <= 40; ++bit) {
            Params p = walk;
            for (int i = 0; i < 4; ++i) p.strides[i] = 1ull << 20;
            p.strides[d] = 1ull << bit;
            emit("walk-stride", p);
        }
    for (int d = 0; d < 5; ++d)
        for (int bit = 0; bit <= 8; ++bit) {
            Params p = walk;
            for (int i = 0; i < 5; ++i) p.dims[i] = 4096;
            p.box[d] = 1u << bit;
            emit("walk-box", p);
        }
    for (int d = 0; d < 5; ++d)
        for (uint32_t s = 1; s <= 8; ++s) {
            Params p = walk;
            p.estrides[d] = s;
            emit("walk-estride", p);
        }

    printf("# %d parameter sets, %d refused by the encoder\n", lines, failures);
    return 0;
}
