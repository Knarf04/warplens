// The programs of validation/ampere, rebuilt for this GPU and rerun.
//
// Everything the simulator shares with Ampere was checked bit for bit on an RTX 3080 Ti:
// each program of validation/ampere printed a golden file, and the simulator reproduces it
// (tests/test_hardware_golden.py). This probe runs the SAME host programs and kernels (they
// are included, not copied) on the B200, so that an instruction the simulator took as
// settled is seen to behave the same on Blackwell. The expected output of every case is its
// Ampere golden file, validation/ampere/<case>_golden.txt, and the committed prediction
// (what this program prints under `warplens run`) is identical to it; for `printf` the
// lines are the same and their order differs (see below).
//
//   ampere_replay shfl          shfl.sync up/down/bfly/idx over widths and distances
//   ampere_replay printf        device printf with a by-value struct and a local array
//   ampere_replay ldmatrix      ldmatrix in six variants, and bfe
//   ampere_replay prmt          prmt.b32
//   ampere_replay cpasync       cp.async with a source size
//   ampere_replay call          calls to device functions (matrix row RT-02)
//   ampere_replay barrier_exit  __syncthreads() with exited threads, cases 0 and 1
//
// `printf` prints from the device. The device's lines and the host's "out:" line reach
// standard output through different buffers, so their relative order is not defined;
// compare that case as a set of lines.
//
// Failures are reported as in the other probes (HOST ERROR, KERNEL FAILED, LAUNCH FAILED),
// through the two macros the Ampere programs use for their checks.
#include "../common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) HOST_CHECK(call)
#define CHECK_LAUNCH() probe_sync("ampere_replay")

#define main replay_shfl
#include "../../ampere/shfl_golden.cu"
#undef main
#define main replay_printf
#include "../../ampere/printf_hw.cu"
#undef main
#define main replay_ldmatrix
#include "../../ampere/ldmatrix_golden.cu"
#undef main
#define main replay_prmt
#include "../../ampere/prmt_golden.cu"
#undef main
#define main replay_cpasync
#include "../../ampere/cpasync_golden.cu"
#undef main
#define main replay_call
#include "../../ampere/call_golden.cu"
#undef main
#define main replay_barrier_exit
#include "../../ampere/barrier_exit_hw.cu"
#undef main

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "";
    if (!strcmp(which, "shfl")) return replay_shfl();
    if (!strcmp(which, "printf")) return replay_printf();
    if (!strcmp(which, "ldmatrix")) return replay_ldmatrix();
    if (!strcmp(which, "prmt")) return replay_prmt();
    if (!strcmp(which, "cpasync")) return replay_cpasync();
    if (!strcmp(which, "call")) return replay_call();
    if (!strcmp(which, "barrier_exit")) {
        char zero[] = "0", one[] = "1";
        char* first[] = {argv[0], zero};
        char* second[] = {argv[0], one};
        int status = replay_barrier_exit(2, first);
        fflush(stdout);
        return status ? status : replay_barrier_exit(2, second);
    }
    printf("HOST ERROR: usage: ampere_replay shfl|printf|ldmatrix|prmt|cpasync|call|barrier_exit\n");
    return 2;
}
