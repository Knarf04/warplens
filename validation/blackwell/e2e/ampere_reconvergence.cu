// Where diverged lanes rejoin: the kernels of validation/ampere/reconvergence_golden.py,
// rerun on this GPU.
//
// Every kernel is `if (tid == 0) shared = 7;` followed by a join that all or some lanes
// reach: `[bar.sync 0;] out[tid] = shared;`. One warp runs it. A lane that passes the join
// reads 7 only if lane 0 got there with it and not after it. The twenty kernels differ in
// how the other lanes get to the join (layouts A to D of reconvergence_golden.py), in
// whether the join has a barrier, in whether all lanes or lanes 0 to 15 reach it, and in
// what follows it. On an RTX 3080 Ti every lane that passes the join reads 7 in all of them
// (validation/ampere/reconvergence_golden.txt), and the simulator does the same; the
// committed prediction is that file, byte for byte (tests/validation/golden_replay/test_ampere_replay.py).
//
// The ten kernels with `bar.sync` test something CUDA defines (matrix row RT-14). The ten
// without (`_nobar`) have no synchronization between lane 0's store and the other lanes'
// load: where the lanes of a warp rejoin is the hardware's business, the simulator copies
// what Ampere did, and a different line on the B200 says Blackwell schedules those lanes
// differently, not that the simulator broke a rule.
//
// The kernels are PTX text, because the layouts are the point and a compiler picks its own.
// They are the very modules the Ampere recording used (`.target sm_86`): the header
// validation/ampere/reconvergence_kernels.h is written by reconvergence_golden.py --header
// and a test keeps it so. This program loads each one with cuModuleLoadData, so the driver
// of the machine compiles it. The driver functions are found through the runtime, as
// common.h finds cuTensorMapEncodeTiled, so the program does not link libcuda, and they use
// the context the runtime made.
//
// One line per kernel: its name, then out[0..63] in decimal (99 where nothing was stored),
// as in the golden file. Each kernel is launched LAUNCHES times (100, as for the recording) and
// every launch must give the same 64 words; if one does not, a line "UNSTABLE <name> launch
// <k>: ..." follows the kernel's line and so differs from the prediction.
//
// Failures: a module the driver will not load is LAUNCH FAILED (exit 4), a launch the driver
// refuses or a kernel that faults is KERNEL FAILED (exit 3), as in the other probes.
//
//   ampere_reconvergence
#include "../common.h"

#include "../../ampere/reconvergence_kernels.h"

#define LAUNCHES 100

typedef CUresult (*module_load_fn)(CUmodule*, const void*);
typedef CUresult (*module_get_function_fn)(CUfunction*, CUmodule, const char*);
typedef CUresult (*module_unload_fn)(CUmodule);
typedef CUresult (*launch_kernel_fn)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, CUstream, void**, void**);

static void* driver_function(const char* name) {
    void* found = nullptr;
    cudaDriverEntryPointQueryResult status;
#ifdef PROBE_AMPERE  // the toolkit of the Ampere dry run predates the versioned query
    HOST_CHECK(cudaGetDriverEntryPoint(name, &found, cudaEnableDefault, &status));
#else
    HOST_CHECK(cudaGetDriverEntryPointByVersion(name, &found, 12000, cudaEnableDefault, &status));
#endif
    if (!found || status != cudaDriverEntryPointSuccess) {
        printf("HOST ERROR: the driver has no %s\n", name);
        fflush(stdout);
        exit(2);
    }
    return found;
}

int main() {
    enum { WORDS = 64 };
    uint32_t fill[WORDS], first[WORDS], again[WORDS];
    for (int i = 0; i < WORDS; ++i) fill[i] = 99;
    uint32_t* out = dev_alloc<uint32_t>(WORDS);  // also makes the runtime's context current
    module_load_fn load = (module_load_fn)driver_function("cuModuleLoadData");
    module_get_function_fn get = (module_get_function_fn)driver_function("cuModuleGetFunction");
    module_unload_fn unload = (module_unload_fn)driver_function("cuModuleUnload");
    launch_kernel_fn launch = (launch_kernel_fn)driver_function("cuLaunchKernel");

    for (const ReconvergenceKernel& kernel : RECONVERGENCE_KERNELS) {
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        CUresult error = load(&module, kernel.ptx);
        if (error == CUDA_SUCCESS) error = get(&function, module, "k");
        if (error != CUDA_SUCCESS) {
            printf("LAUNCH FAILED: %s\n", kernel.name);
            printf("info: the driver did not load the PTX module of this kernel (CUresult %d)\n", (int)error);
            fflush(stdout);
            fprintf(stderr, "LAUNCH FAILED: %s: cuModuleLoadData or cuModuleGetFunction returned %d\n", kernel.name,
                    (int)error);
            return 4;
        }
        for (int k = 0; k < LAUNCHES; ++k) {
            HOST_CHECK(cudaMemcpy(out, fill, sizeof fill, cudaMemcpyHostToDevice));
            void* parameters[] = {&out};
            error = launch(function, 1, 1, 1, 32, 1, 1, 0, nullptr, parameters, nullptr);
            if (error != CUDA_SUCCESS) {
                printf("KERNEL FAILED: %s\n", kernel.name);
                printf("info: cuLaunchKernel returned CUresult %d\n", (int)error);
                fflush(stdout);
                fprintf(stderr, "KERNEL FAILED: %s: cuLaunchKernel returned %d\n", kernel.name, (int)error);
                return 3;
            }
            probe_sync(kernel.name);
            dev_read(k ? again : first, out, WORDS);
            if (k == 0) {
                printf("%s", kernel.name);
                for (int i = 0; i < WORDS; ++i) printf(" %u", first[i]);
                printf("\n");
            } else if (memcmp(first, again, sizeof first)) {
                printf("UNSTABLE %s launch %d:", kernel.name, k);
                for (int i = 0; i < WORDS; ++i) printf(" %u", again[i]);
                printf("\n");
            }
        }
        if (unload(module) != CUDA_SUCCESS) {
            printf("HOST ERROR: cuModuleUnload failed for %s\n", kernel.name);
            return 2;
        }
    }
    return 0;
}
