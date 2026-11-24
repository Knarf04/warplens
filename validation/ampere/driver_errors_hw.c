/* Host program: the driver API's codes for the refusals of launch_errors_hw.cu. Recorded
 * in driver_errors_3080ti.txt (RTX 3080 Ti, driver 580.173.02):
 *   bin/driver_errors_hw > driver_errors_3080ti.txt */
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>

static const char *ptx =
    ".version 7.0\n.target sm_80\n.address_size 64\n"
    ".visible .entry k(.param .u64 out)\n{\n.reg .b64 %p;\nld.param.u64 %p, [out];\nret;\n}\n";

static void say(const char *what, CUresult r) {
    const char *name = 0;
    cuGetErrorName(r, &name);
    printf("%-64s %d %s\n", what, (int)r, name ? name : "?");
}

int main(void) {
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction f; CUdeviceptr out;
    cuInit(0); cuDeviceGet(&dev, 0); cuCtxCreate(&ctx, 0, dev);
    say("cuModuleLoadData", cuModuleLoadData(&mod, ptx));
    cuModuleGetFunction(&f, mod, "k");
    cuMemAlloc(&out, 64);
    int optin = 0, plain = 0;
    cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    cuDeviceGetAttribute(&plain, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev);
    printf("plain %d optin %d\n", plain, optin);
    void *args[] = {&out};
    char what[128];
    say("cuLaunchKernel, no opt-in, default maximum + 16", cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, plain + 16, 0, args, 0));
    int values[] = {plain, optin, optin + 1, optin + 1024, -1, 0};
    for (int i = 0; i < 6; ++i) {
        snprintf(what, sizeof what, "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, %d)", values[i]);
        say(what, cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, values[i]));
    }
    say("cuFuncSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT, 101)", cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 101));
    say("cuFuncSetAttribute(MAX_THREADS_PER_BLOCK, 5) (read-only)", cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 5));
    say("cuFuncSetAttribute(opt-in maximum)", cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin));
    say("cuLaunchKernel, opted in, opt-in maximum", cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, optin, 0, args, 0));
    say("cuLaunchKernel, opted in, opt-in maximum + 16", cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, optin + 16, 0, args, 0));
    say("cuLaunchKernel, block of 1025", cuLaunchKernel(f, 1, 1, 1, 1025, 1, 1, 0, 0, args, 0));
    say("cuLaunchKernel, block (0, 1, 1)", cuLaunchKernel(f, 1, 1, 1, 0, 1, 1, 0, 0, args, 0));
    say("cuLaunchKernel, grid (0, 1, 1)", cuLaunchKernel(f, 0, 1, 1, 32, 1, 1, 0, 0, args, 0));
    say("cuCtxSynchronize", cuCtxSynchronize());
    for (int k = 1; k <= 2; ++k) {
        CUlaunchConfig config = {0};
        CUlaunchAttribute attribute[1];
        config.gridDimX = 4; config.gridDimY = 1; config.gridDimZ = 1;
        config.blockDimX = 32; config.blockDimY = 1; config.blockDimZ = 1;
        attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attribute[0].value.clusterDim.x = k; attribute[0].value.clusterDim.y = 1; attribute[0].value.clusterDim.z = 1;
        config.attrs = attribute; config.numAttrs = 1;
        snprintf(what, sizeof what, "cuLaunchKernelEx, grid 4, cluster (%d, 1, 1)", k);
        say(what, cuLaunchKernelEx(&config, f, args, 0));
    }
    say("cuCtxSynchronize", cuCtxSynchronize());
    say("cuLaunchKernel, ordinary", cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
    say("cuCtxSynchronize", cuCtxSynchronize());
    return 0;
}
