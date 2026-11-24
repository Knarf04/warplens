// Error checks shared by the host programs here. A failed CUDA call, a launch the driver
// refused, or a kernel that faulted ends the program with a message on stderr and status 1,
// so that an output file is never silently the product of a kernel that did not run.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// A program that includes these sources may define both macros first (validation/blackwell
// does, to report failures the way its other probes do).
#ifndef CHECK
#define CHECK(call)                                                                              \
    do {                                                                                         \
        cudaError_t check_error_ = (call);                                                       \
        if (check_error_ != cudaSuccess) {                                                       \
            fflush(stdout);                                                                      \
            fprintf(stderr, "%s:%d: %s failed: %s (%s)\n", __FILE__, __LINE__, #call,            \
                    cudaGetErrorName(check_error_), cudaGetErrorString(check_error_));           \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)
#endif

// Directly after a launch: the launch's own error (a module the driver would not load, a
// bad configuration), then the error the kernel ended with, then anything left behind.
#ifndef CHECK_LAUNCH
#define CHECK_LAUNCH()                     \
    do {                                   \
        CHECK(cudaGetLastError());         \
        CHECK(cudaDeviceSynchronize());    \
        CHECK(cudaGetLastError());         \
    } while (0)
#endif
