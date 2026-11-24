/* Host program: the driver API's codes for the refusals of launch_errors_hw.cu. Recorded
 * in driver_errors_3080ti.txt (RTX 3080 Ti, driver 580.173.02):
 *   bin/driver_errors_hw > driver_errors_3080ti.txt */
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>

static const char *ptx =
    ".version 7.0\n.target sm_80\n.address_size 64\n"
    ".visible .entry k(.param .u64 out)\n{\n.reg .b64 %p;\nld.param.u64 %p, [out];\nret;\n}\n"
    /* For the opt-in rule: a kernel nothing else touches, and one with 40960 bytes of static
     * shared memory (under the default of 49152). */
    ".visible .entry k2(.param .u64 out)\n{\n.reg .b64 %p;\nld.param.u64 %p, [out];\nret;\n}\n"
    ".shared .align 1 .b8 fixed40[40960];\n"
    ".visible .entry s40(.param .u64 out)\n{\n.reg .b64 %p;\n.reg .b32 %r;\n"
    "ld.param.u64 %p, [out];\nmov.u32 %r, %tid.x;\nst.shared.u8 [fixed40+40959], %r;\nret;\n}\n";
/* Static shared memory of the default, and of one byte more. The driver compiles the second
 * for no target but the architecture-specific ones (sm_90a, sm_100a, ...), so it is a module
 * of its own: see the "Launch errors" section of ../README.md. */
static const char *ptx_static =
    ".version 7.0\n.target sm_80\n.address_size 64\n.shared .align 1 .b8 fixed[%d];\n"
    ".visible .entry s(.param .u64 out)\n{\n.reg .b64 %%p;\n.reg .b32 %%r;\n"
    "ld.param.u64 %%p, [out];\nmov.u32 %%r, %%tid.x;\nst.shared.u8 [fixed+%d], %%r;\nret;\n}\n";

static void say(const char *what, CUresult r) {
    const char *name = 0;
    cuGetErrorName(r, &name);
    printf("%-64s %d %s\n", what, (int)r, name ? name : "?");
}

static void attributes(const char *what, CUfunction f) {
    int fixed = -1, dynamic = -1;
    cuFuncGetAttribute(&fixed, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
    cuFuncGetAttribute(&dynamic, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f);
    printf("%-64s SHARED_SIZE_BYTES %d, MAX_DYNAMIC_SHARED_SIZE_BYTES %d\n", what, fixed, dynamic);
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

    /* The opt-in rule, as launch_errors_hw.cu asks it of the runtime. */
    CUfunction k2, s40;
    cuModuleGetFunction(&k2, mod, "k2"); cuModuleGetFunction(&s40, mod, "s40");
    attributes("attributes, kernel opted in to the maximum above", f);
    attributes("attributes, fresh kernel", k2);
    say("cuLaunchKernel, fresh kernel, default maximum", cuLaunchKernel(k2, 1, 1, 1, 32, 1, 1, plain, 0, args, 0));
    say("cuLaunchKernel, fresh kernel, default maximum + 1", cuLaunchKernel(k2, 1, 1, 1, 32, 1, 1, plain + 1, 0, args, 0));
    say("cuFuncSetAttribute(fresh, 60000)", cuFuncSetAttribute(k2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 60000));
    attributes("attributes, opted in to 60000", k2);
    say("cuLaunchKernel, opted in to 60000, 60000", cuLaunchKernel(k2, 1, 1, 1, 32, 1, 1, 60000, 0, args, 0));
    say("cuLaunchKernel, opted in to 60000, 60001", cuLaunchKernel(k2, 1, 1, 1, 32, 1, 1, 60001, 0, args, 0));
    say("cuFuncSetAttribute(the same, 1024)", cuFuncSetAttribute(k2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1024));
    say("cuLaunchKernel, set to 1024, 1024", cuLaunchKernel(k2, 1, 1, 1, 32, 1, 1, 1024, 0, args, 0));
    say("cuLaunchKernel, set to 1024, default maximum", cuLaunchKernel(k2, 1, 1, 1, 32, 1, 1, plain, 0, args, 0));
    attributes("attributes, 40960 static", s40);
    say("cuLaunchKernel, 40960 static + 8192, no opt-in", cuLaunchKernel(s40, 1, 1, 1, 32, 1, 1, plain - 40960, 0, args, 0));
    say("cuLaunchKernel, 40960 static + 8208, no opt-in", cuLaunchKernel(s40, 1, 1, 1, 32, 1, 1, plain - 40960 + 16, 0, args, 0));
    say("cuFuncSetAttribute(40960 static, 8208)", cuFuncSetAttribute(s40, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 8208));
    say("cuLaunchKernel, 40960 static + 8208, set to 8208", cuLaunchKernel(s40, 1, 1, 1, 32, 1, 1, 8208, 0, args, 0));
    say("cuLaunchKernel, 40960 static + 8224, set to 8208", cuLaunchKernel(s40, 1, 1, 1, 32, 1, 1, 8224, 0, args, 0));
    say("cuFuncSetAttribute(40960 static, 60416 (opt-in maximum - static))", cuFuncSetAttribute(s40, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin - 40960));
    say("cuFuncSetAttribute(40960 static, 60417)", cuFuncSetAttribute(s40, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin - 40960 + 1));
    attributes("attributes, 40960 static, after the refused 60417", s40);
    for (int extra = 0; extra <= 1; ++extra) {
        static char text[1024];
        CUmodule big = 0; CUfunction s = 0;
        snprintf(text, sizeof text, ptx_static, plain + extra, plain + extra - 1);
        snprintf(what, sizeof what, "cuModuleLoadData, kernel with %d static bytes", plain + extra);
        say(what, cuModuleLoadData(&big, text));
        if (!big || cuModuleGetFunction(&s, big, "s") != CUDA_SUCCESS) continue;
        attributes("attributes of it", s);
        say("cuLaunchKernel, 49152 static + 0, no opt-in", cuLaunchKernel(s, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
        say("cuLaunchKernel, 49152 static + 16, no opt-in", cuLaunchKernel(s, 1, 1, 1, 32, 1, 1, 16, 0, args, 0));
    }
    say("cuCtxSynchronize", cuCtxSynchronize());
    say("cuLaunchKernel, ordinary", cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, 0, args, 0));
    say("cuCtxSynchronize", cuCtxSynchronize());
    return 0;
}
