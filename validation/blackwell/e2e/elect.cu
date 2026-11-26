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
//   elect other-branch   mask 0xFFFFFFFF executed by one half of the warp while the other
//                        half is on a branch of its own, where each lane sets a flag and
//                        executes no elect.sync. Each electing lane reads its partner's
//                        flag before and after the instruction: "after" 1 with "before" 0
//                        says the instruction waited for the lanes its mask names until
//                        they had finished their branch, as bar.warp.sync does; 0 and 0
//                        says it went on without them. Run for both halves
//   elect other-callers  the same with a mask that names only the callers: nothing to wait
//                        for. The control
//   elect other-sites    both halves at call sites of their own, each with the mask
//                        0xFFFFFFFF. May never return (each half waits for the other), or
//                        may elect one lane for both if the compiler gives the two call
//                        sites one machine instruction
//
// Measured on a B200 (2026-11-26): "before" and "after" are 0 in other-branch as in the
// control. elect.sync does not wait for lanes of its mask that are elsewhere.
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

// One half of the warp elects and the other half is on another branch, where its lanes set
// flag[lane] (and, with sites == 1, execute an elect.sync of their own with `mask`). The
// electing lanes read the flag of their partner, lane ^ 16, before and after the
// instruction. Which of two branches a GPU runs first is not a rule: this kernel is written
// so that a B200 runs the electing half first ("before" reads 0), and only then does "after"
// say whether the instruction waited for the other half (1) or went on without it (0).
extern "C" __global__ void elect_other(uint32_t mask, int sites, int low, int* flag, uint32_t* out) {
    uint32_t lane = threadIdx.x, leader = 0xEEu, chosen = 0xEEu, before = 0xEEu, after = 0xEEu;
    const bool electing = low ? lane < 16 : lane >= 16;
    if (electing) {
        before = (uint32_t)flag[lane ^ 16];
        asm volatile("{\n.reg .pred p;\nelect.sync %0|p, %2;\nselp.u32 %1, 1, 0, p;\n}"
                     : "=r"(leader), "=r"(chosen) : "r"(mask));
        after = (uint32_t)flag[lane ^ 16];
    } else {
        flag[lane] = 1;
        if (sites)
            asm volatile("{\n.reg .pred p;\nelect.sync %0|p, %2;\nselp.u32 %1, 1, 0, p;\n}"
                         : "=r"(leader), "=r"(chosen) : "r"(mask));
    }
    out[lane * 4] = leader;
    out[lane * 4 + 1] = chosen;
    out[lane * 4 + 2] = before;
    out[lane * 4 + 3] = after;
}

static void run_other(const char* label, uint32_t mask, int sites, int low) {
    uint32_t out[128];
    int flag[32] = {};
    uint32_t* d_out = dev_alloc<uint32_t>(128);
    int* d_flag = dev_copy(flag, 32);
    dev_fill(d_out, 128, 0);
    printf("%s\n", label);
    fflush(stdout);
    elect_other<<<1, 32>>>(mask, sites, low, d_flag, d_out);
    probe_sync(label);
    dev_read(out, d_out, 128);
    auto electing = [&](int l) { return low ? l < 16 : l >= 16; };
    const char* names[] = {"leader:     ", "predicate:  ", "flag before:", "flag after: "};
    printf("  lane:        ");
    for (int l = 0; l < 32; ++l) printf(" %2d", l);
    for (int field = 0; field < 4; ++field) {
        printf("\n  %s ", names[field]);
        for (int l = 0; l < 32; ++l) {
            if (electing(l) || (sites && field < 2)) printf(" %2u", out[l * 4 + field]);
            else printf("  .");
        }
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
