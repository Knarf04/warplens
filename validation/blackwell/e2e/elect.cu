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
//   elect other-branch   mask 0xFFFFFFFF executed by lanes 16-31 while lanes 0-15 take a
//                        branch of their own, set a flag there and execute no elect.sync;
//                        each electing lane then reads the flag of the lane 16 below it.
//                        A flag that reads 1 says the instruction waited for the lanes its
//                        mask names until they had finished their branch, as vote.sync,
//                        shfl.sync and bar.warp.sync do on an RTX 3080 Ti
//                        (validation/README.md); 0 says it went on without them
//                        Then the same with the halves exchanged: which branch a GPU runs
//                        first is not a rule, and a flag tells only where the electing
//                        half would have run first
//   elect other-callers  the same two with a mask that names only the callers: nothing to
//                        wait for. This is the control that shows which half runs first
//   elect other-sites    lanes 0-15 and lanes 16-31 at two call sites, both with the mask
//                        0xFFFFFFFF. May never return (each group waits for the other), or
//                        may elect one lane for both if the compiler gives the two call
//                        sites one machine instruction
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

// One half of the warp is the other group. sites == 0: its lanes set flag[lane] and execute
// no elect.sync; sites == 1: they set it and execute one of their own with `mask`. The lanes
// of the electing half execute elect.sync with `mask` and then read the flag of their
// partner, lane ^ 16. `low` chooses the half that elects: which of two branches a GPU runs
// first is its own business, and a flag says something only if the electing half would
// otherwise have run first (the control with a mask of the callers alone shows which).
extern "C" __global__ void elect_other(uint32_t mask, int sites, int low, int* flag, uint32_t* out) {
    uint32_t lane = threadIdx.x, leader = 0xEEu, chosen = 0xEEu, seen = 0xEEu;
    const bool electing = low ? lane < 16 : lane >= 16;
    if (!electing) {
        flag[lane] = 1;
        if (sites)
            asm volatile("{\n.reg .pred p;\nelect.sync %0|p, %2;\nselp.u32 %1, 1, 0, p;\n}"
                         : "=r"(leader), "=r"(chosen) : "r"(mask));
    } else {
        asm volatile("{\n.reg .pred p;\nelect.sync %0|p, %2;\nselp.u32 %1, 1, 0, p;\n}"
                     : "=r"(leader), "=r"(chosen) : "r"(mask));
        seen = (uint32_t)flag[lane ^ 16];
    }
    out[lane * 3] = leader;
    out[lane * 3 + 1] = chosen;
    out[lane * 3 + 2] = seen;
}

static void run_other(const char* label, uint32_t mask, int sites, int low) {
    uint32_t out[96];
    int flag[32] = {};
    uint32_t* d_out = dev_alloc<uint32_t>(96);
    int* d_flag = dev_copy(flag, 32);
    dev_fill(d_out, 96, 0);
    printf("%s\n", label);
    fflush(stdout);
    elect_other<<<1, 32>>>(mask, sites, low, d_flag, d_out);
    probe_sync(label);
    dev_read(out, d_out, 96);
    auto electing = [&](int l) { return low ? l < 16 : l >= 16; };
    printf("  lane:      ");
    for (int l = 0; l < 32; ++l) printf(" %2d", l);
    printf("\n  leader:    ");
    for (int l = 0; l < 32; ++l) {
        if (electing(l) || sites) printf(" %2u", out[l * 3]);
        else printf("  .");
    }
    printf("\n  predicate: ");
    for (int l = 0; l < 32; ++l) {
        if (electing(l) || sites) printf(" %2u", out[l * 3 + 1]);
        else printf("  .");
    }
    printf("\n  flag seen: ");
    for (int l = 0; l < 32; ++l) {
        if (electing(l)) printf(" %2u", out[l * 3 + 2]);
        else printf("  .");
    }
    printf("\n");
    fflush(stdout);
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
    } else if (!strcmp(which, "other-branch")) {
        run_other("member mask 0xffffffff, executed by lanes 16-31; lanes 0-15 set a flag on a branch of their own",
                  0xFFFFFFFFu, 0, 0);
        run_other("member mask 0xffffffff, executed by lanes 0-15; lanes 16-31 set a flag on a branch of their own",
                  0xFFFFFFFFu, 0, 1);
    } else if (!strcmp(which, "other-callers")) {
        run_other("member mask 0xffff0000, executed by lanes 16-31; lanes 0-15 set a flag on a branch of their own",
                  0xFFFF0000u, 0, 0);
        run_other("member mask 0x0000ffff, executed by lanes 0-15; lanes 16-31 set a flag on a branch of their own",
                  0x0000FFFFu, 0, 1);
    } else if (!strcmp(which, "other-sites")) {
        run_other("member mask 0xffffffff at two call sites: lanes 0-15 at one, lanes 16-31 at the other",
                  0xFFFFFFFFu, 1, 0);
    } else {
        run(0xFFFFFFFFu, 0xFFFFFFFFu, d_out);
        run(0xFFFF0000u, 0xFFFF0000u, d_out);
        run(0x80000001u, 0x80000001u, d_out);
        run(0xAAAAAAAAu, 0xAAAAAAAAu, d_out);
    }
    return 0;
}
