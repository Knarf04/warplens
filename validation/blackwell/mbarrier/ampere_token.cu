// What the token of mbarrier.arrive encodes: validation/ampere/mbarrier_token_golden.cu,
// rebuilt for this GPU and rerun.
//
// The kernel (validation/ampere/mbarrier_token_probe.cu; included, not copied) makes 8
// phases of 1, 2, 3 and 5 arrivals on one mbarrier and, after every arrival, asks
// mbarrier.test_wait about the token of every arrival so far, about the made-up tokens
// 1 << b, and mbarrier.test_wait.parity about 0 to 3. One line per arrival:
//
//   count step token-bit-63 parity-answers own-token-answers made-up-token-answers
//
// On an RTX 3080 Ti the answer depends on bit 63 of the token alone, the parity of the
// phase the arrival was made in (validation/ampere/mbarrier_token_golden.txt), and the
// simulator does the same. So the committed prediction is that file, byte for byte
// (tests/validation/golden_replay/test_ampere_replay.py).
//
// What this run can and cannot say (matrix row MB-03). The answers for the token of the
// current and of the preceding phase, and all the parity answers, are defined by PTX; a
// difference there is a simulator error. The answers for tokens two or more phases old and
// for made-up tokens are invalid PTX use: the simulator reports `mbarrier-invalid-token`
// for them (predicted/mbarrier/ampere_token.sim.txt) and returns what Ampere returned. A
// difference in only those columns means Blackwell encodes or tests the token differently
// from Ampere. No valid program can observe that, but it is the measurement that
// mbarrier/arrive_token.stale can only put on `info:` lines.
//
// The kernel uses only the sm_80 spellings (mbarrier.init/arrive/test_wait with `.shared`,
// no `::cta`, no expect_tx), so the Ampere build (build.sh --ampere) runs it on the
// development machine's GPU, and the sm_100a build shows that ptxas still takes them.
//
//   ampere_token
#include "../common.h"

#define CHECK(call) HOST_CHECK(call)
#define CHECK_LAUNCH() probe_sync("mbarrier_token_probe")

#include "../../ampere/mbarrier_token_golden.cu"
