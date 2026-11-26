// What the CUDA runtime returns for requests it refuses before any kernel runs, on this GPU.
//
// The program is validation/ampere/launch_errors_hw.cu: it asks the device for its
// limits (shared memory per block, with and without opt-in) and tries one step inside and one
// step outside each, so it measures a B200's limits as it measured an RTX 3080 Ti's
// (validation/ampere/launch_errors_3080ti.txt). Under `warplens run` the shim answers with
// the simulated device's limits; the prediction is that output.
//
// Claim tested (docs/validation-matrix.md, RT-04 and RT-12): the code each refusal carries,
// and where it shows up afterwards (the call, cudaPeekAtLastError, cudaDeviceSynchronize,
// cudaGetLastError twice).
//
//   launch_errors
#include "../common.h"  // for its line-buffered standard output: the runner reads a pipe
#define LAUNCH_ERRORS_DEVICE_LABEL "the device"  // its name is the one thing a prediction cannot have
#include "../../ampere/launch_errors_hw.cu"
