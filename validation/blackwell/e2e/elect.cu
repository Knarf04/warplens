// elect.sync: which lane is elected.
//
// One warp. A subset of the lanes executes `elect.sync d|p, membermask` and every lane that
// did records the lane id it got in d and its predicate.
//
//   elect            the well-formed cases: the member mask is exactly the set of lanes
//                    that execute the instruction
//                      mask 0xFFFFFFFF, all 32 lanes
//                      mask 0xFFFF0000, lanes 16-31
//                      mask 0x80000001, lanes 0 and 31
//                      mask 0xAAAAAAAA, the odd lanes
//   elect all        masks 0xFFFF0000 and 0x80000001 executed by ALL 32 lanes, so lanes
//                    outside the mask execute it too
//   elect partial    mask 0xFFFFFFFF executed only by the odd lanes, so the mask names
//                    lanes that do not execute it
//
// Claim tested (docs/history/m1.md; warplens/units/alu.py `_elect`): the lowest lane that is in the
// mask and executes the instruction is elected; every executing lane gets its id, and the
// predicate is true only in that lane. Expected leaders: 0, 16, 0, 1; `all`: 16, 0;
// `partial`: 1. The PTX manual defines only the well-formed cases, so `all` and `partial`
// run in their own processes and hardware may do anything there. CUTLASS needs only that
// exactly one lane is elected.
#include "../common.h"

// who: bit l set when lane l executes the instruction.
extern "C" __global__ void elect(uint32_t mask, uint32_t who, uint32_t* out) {
    uint32_t lane = threadIdx.x, leader = 0xEEu, chosen = 0xEEu;
    if (who >> lane & 1) {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "elect.sync %0|p, %2;\n"
                     "selp.u32 %1, 1, 0, p;\n"
                     "}" : "=r"(leader), "=r"(chosen) : "r"(mask));
    }
    out[lane * 2] = leader;
    out[lane * 2 + 1] = chosen;
}

static void run(uint32_t mask, uint32_t who, uint32_t* d_out) {
    uint32_t out[64];
    dev_fill(d_out, 64, 0);
    char label[96];
    snprintf(label, sizeof label, "member mask 0x%08x, executed by the lanes of 0x%08x", mask, who);
    elect<<<1, 32>>>(mask, who, d_out);
    probe_sync(label);
    dev_read(out, d_out, 64);
    printf("%s\n", label);
    printf("  lane:      ");
    for (int l = 0; l < 32; ++l) printf(" %2d", l);
    printf("\n  leader:    ");
    for (int l = 0; l < 32; ++l) {
        if (who >> l & 1) printf(" %2u", out[l * 2]);
        else printf("  .");
    }
    printf("\n  predicate: ");
    int elected = 0;
    for (int l = 0; l < 32; ++l) {
        if (who >> l & 1) {
            printf(" %2u", out[l * 2 + 1]);
            elected += out[l * 2 + 1] == 1;
        } else {
            printf("  .");
        }
    }
    printf("\n  lanes with a true predicate: %d\n", elected);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    uint32_t* d_out = dev_alloc<uint32_t>(64);
    if (!strcmp(which, "all")) {
        run(0xFFFF0000u, 0xFFFFFFFFu, d_out);
        run(0x80000001u, 0xFFFFFFFFu, d_out);
    } else if (!strcmp(which, "partial")) {
        run(0xFFFFFFFFu, 0xAAAAAAAAu, d_out);
    } else {
        run(0xFFFFFFFFu, 0xFFFFFFFFu, d_out);
        run(0xFFFF0000u, 0xFFFF0000u, d_out);
        run(0x80000001u, 0x80000001u, d_out);
        run(0xAAAAAAAAu, 0xAAAAAAAAu, d_out);
    }
    return 0;
}
