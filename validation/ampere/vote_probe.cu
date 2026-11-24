// Up to two warps vote. Thread t has a predicate, a member mask and a role:
//
//   role 0  votes with the instruction `mode` selects
//   role 1  returns before any vote
//   role 2  takes a branch without a vote while the others vote
//   role 3  takes a branch with a ballot of its own while the others vote
//
// Afterwards every thread that has not returned takes a ballot with the mask 0xffffffff.
// out[3 t] is the vote, out[3 t + 1] the ballot of role 3, out[3 t + 2] the last ballot; a
// word the thread does not write keeps what the host put there.
//
//   mode 0 to 3  __all_sync, __any_sync, __uni_sync, __ballot_sync
//                (vote.sync.{all,any,uni}.pred, vote.sync.ballot.b32)
//   mode 4       vote.sync.ballot.b32 with a negated source (`!p`)
//   mode 5 to 9  with -DVOTE_LEGACY: the same five without .sync and without a mask
//                (vote.all.pred, ...). ptxas refuses these for sm_70 and newer ("not
//                supported on .target sm_70 and higher from PTX ISA version 6.4"), so they are
//                built for sm_60 only and reach an Ampere GPU through the driver's compiler.
//
// Run on real hardware to record golden results, and in warplens to compare.
#define VOTE_ASM(text) \
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.s32 p, %1, 0;\n\t" text "\n\t}" : "=r"(r) : "r"(p))
#define VOTE_PRED(text) VOTE_ASM(text " p, p;\n\tselp.u32 %0, 1, 0, p;")

extern "C" __global__ void vote_probe(unsigned* out, const int* pred, const unsigned* mask,
                                      const int* role, int mode) {
    int t = threadIdx.x;
    int p = pred[t];
    unsigned m = mask[t];
    int what = role[t];
    if (what == 1) return;
    unsigned r = 0;
    if (what == 2) {
        out[3 * t + 1] = 0x22220000u + t;
    } else if (what == 3) {
        out[3 * t + 1] = __ballot_sync(m, p);
    } else {
        if (mode == 0)      r = __all_sync(m, p);
        else if (mode == 1) r = __any_sync(m, p);
        else if (mode == 2) r = __uni_sync(m, p);
        else if (mode == 3) r = __ballot_sync(m, p);
        else if (mode == 4)
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.s32 p, %1, 0;\n\tvote.sync.ballot.b32 %0, !p, %2;\n\t}"
                         : "=r"(r) : "r"(p), "r"(m));
#ifdef VOTE_LEGACY
        else if (mode == 5) VOTE_PRED("vote.all.pred");
        else if (mode == 6) VOTE_PRED("vote.any.pred");
        else if (mode == 7) VOTE_PRED("vote.uni.pred");
        else if (mode == 8) VOTE_ASM("vote.ballot.b32 %0, p;");
        else if (mode == 9) VOTE_ASM("vote.ballot.b32 %0, !p;");
#endif
        out[3 * t] = r;
    }
    out[3 * t + 2] = __ballot_sync(0xffffffffu, p);
}
