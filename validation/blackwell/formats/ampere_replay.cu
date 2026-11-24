// The numeric programs of validation/ampere, rebuilt for this GPU and rerun.
//
// Three behaviours the simulator takes from an RTX 3080 Ti where PTX gives no rule, so a
// B200 may do something else: which bits a NaN result has, what an integer division by zero
// returns, and what min/max and float-to-integer conversion do with NaNs. The other probes
// of this directory print such results as the word "nan" or on "info:" lines, which the
// comparison skips. This one prints the bits.
//
// It runs the SAME host programs and kernels that recorded the Ampere golden files (they are
// included, not copied), and prints the lines of the golden file, in its order. The
// committed prediction (what this program prints under `warplens run`) is therefore the
// golden file itself, byte for byte (for `nan`, without the 44 lines of `nan_cvt_ftz`);
// tests/test_ampere_replay.py checks that. A line that differs on the B200 is a difference
// between Blackwell and Ampere, and says which instruction and operands.
//
//   ampere_replay nan         validation/ampere/nan_golden.cu: NaN results of add, sub, mul,
//                             div, fma, sqrt, neg, abs, mov and cvt between f16, f32 and f64,
//                             "instruction a b c result" (matrix row NUM-13): the 14226 of
//                             the file's 14270 lines that are not the next case
//   ampere_replay nan_cvt_ftz the other 44: cvt.ftz.f64.f32 and cvt.rn.ftz.f32.f64. The
//                             simulator does not implement cvt with .ftz and refuses a
//                             kernel that has one, so these are in a kernel and a run of
//                             their own, which has no prediction: its output on the B200 is
//                             to be read against the same lines of nan_golden.txt
//   ampere_replay intdiv      intdiv_golden.cu: div and rem on s16 to u64, with every
//                             dividend over zero and the minimum over -1, 5808 lines
//                             "instruction a b result" (NUM-12)
//   ampere_replay minmax_cvt  minmax_cvt_golden.cu: min and max over zeros, infinities and
//                             NaNs; cvt from f32 and f64 to the four integer types in the
//                             four rounding modes, 15844 lines "instruction a b result"
//                             (NUM-11)
//   ampere_replay bitfield    bitfield_golden.cu: bfi.b32, bfi.b64 and the four shf forms,
//                             2925 lines "a b pos len" and the seven results (NUM-12)
//
// The whole of each file is printed, not a sample: a difference in NaN selection shows in
// particular operand positions, the files are what tests/test_hardware_golden.py holds the
// simulator to line by line, and each case is one launch of well under a second.
//
// Nothing here is newer than Ampere: the Ampere build (build.sh --ampere) runs on the
// development machine's GPU and must reproduce the golden files.
//
// Failures are reported as in the other probes (HOST ERROR, KERNEL FAILED, LAUNCH FAILED),
// through the two macros the Ampere programs use for their checks.
#include "../common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(call) HOST_CHECK(call)
#define CHECK_LAUNCH() probe_sync("ampere_replay")

#define NAN_NO_MAIN
#include "../../ampere/nan_golden.cu"
#define main replay_intdiv
#include "../../ampere/intdiv_golden.cu"
#undef main
#define main replay_minmax_cvt
#include "../../ampere/minmax_cvt_golden.cu"
#undef main
#define main replay_bitfield
#include "../../ampere/bitfield_golden.cu"
#undef main

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    if (!strcmp(which, "nan")) return nan_run(NAN_WITHOUT_CVT_FTZ);
    if (!strcmp(which, "nan_cvt_ftz")) return nan_run(NAN_ONLY_CVT_FTZ);
    if (!strcmp(which, "intdiv")) return replay_intdiv();
    if (!strcmp(which, "minmax_cvt")) return replay_minmax_cvt();
    if (!strcmp(which, "bitfield")) return replay_bitfield();
    printf("HOST ERROR: usage: ampere_replay nan|nan_cvt_ftz|intdiv|minmax_cvt|bitfield\n");
    return 2;
}
