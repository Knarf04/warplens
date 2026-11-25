// .ftz arithmetic, replayed from validation/ampere.
//
// validation/ampere/ftz_golden.cu (included, not copied) runs add, sub, mul, div, fma and
// sqrt with .rn.ftz on f32 operands around the subnormal range and prints
// "op a b c result" in hex. Its output on an RTX 3080 Ti is validation/ampere/ftz_golden.txt,
// which the simulator reproduces except for the payload of NaN results (the GPU gave
// 7fffffff, the simulator gives ffc00000; tests/validation/golden_replay/test_hardware_golden.py compares NaNs as
// NaNs). So this probe prints a NaN result as the word "nan" and its bits on an "info:"
// line: the committed prediction is the golden file with its 642 NaN results spelled "nan".
//
// Claim tested (docs/numerics.md; matrix row NUM-09): .ftz flushes subnormal inputs, and
// results that are tiny after rounding, the same way on Blackwell as on Ampere. A
// difference would mean tininess detection changed between architectures.
//
//   ftz
#include "../common.h"

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call) HOST_CHECK(call)
#define CHECK_LAUNCH() probe_sync("ftz_probe")

#define FTZ_NO_MAIN
#include "../../ampere/ftz_golden.cu"

int main() { return ftz_run(true); }
