// [reg+offset] with a 32-bit address register that has wrapped below zero.
//
// For a reversed index into a shared array, `fixed[N - 1 - tid]`, nvcc 13.5 (sm_100a and
// sm_86 alike) computes `fixed - 4 * tid` in a 32-bit register and puts the constant in the
// instruction:
//     sub.s32        %r6, %r4, %r3;          // %r4 = fixed, %r3 = 4 * tid
//     ld.shared.b32  %r7, [%r6+252];
// The array is at shared address 0, so the register is below zero (0xfffffffc, ...) for
// every lane but lane 0, and the access is right only if the sum wraps at 32 bits. nvcc 12.4,
// which builds the program that runs on the RTX 3080 Ti, does not write that: it computes
// `fixed + 4 * (63 - tid)` and uses `[%r9]` without an offset.
//
// So the two `cxx` kernels are that C++ (the wrapping form under 13.5, an ordinary one under
// 12.4), and the four `ptx` kernels spell the wrapping form out in inline PTX, which every
// toolkit passes through as written and the GPU therefore executes whatever compiled it.
// Each kernel is run by one CTA of 64 threads and has its own array.
//
// Run on real hardware to record golden results, and in warplens to compare.
typedef unsigned int u32;

#define ADDRWRAP_N 64

// The inline PTX starts from the array's address in shared space, in a 32-bit register as the
// compiler has it (`mov.b32 %r4, fixed`), less 4 * tid. `fixed` reaches the block as a
// generic address in a 64-bit register, because the array's name is the compiler's to choose.
#define ADDRWRAP_REGISTER(array, bytes)                    \
    "{ .reg .b64 %s; .reg .b32 %a;\n\t"                    \
    "cvta.to.shared.u64 %s, " array ";\n\t"                \
    "cvt.u32.u64 %a, %s;\n\t"                              \
    "sub.s32 %a, %a, " bytes ";\n\t"

// out[tid] = fixed[63 - tid], as the compiler writes it.
extern "C" __global__ void addrwrap_cxx_load(int* out) {
    __shared__ int fixed[ADDRWRAP_N];
    int tid = threadIdx.x;
    fixed[tid] = 1000 + tid;
    __syncthreads();
    out[tid] = fixed[ADDRWRAP_N - 1 - tid];
}

// fixed[63 - tid] = ..., as the compiler writes it (13.5: `st.shared.b32 [%r5+252], %r2`).
extern "C" __global__ void addrwrap_cxx_store(int* out) {
    __shared__ int fixed[ADDRWRAP_N];
    int tid = threadIdx.x;
    fixed[ADDRWRAP_N - 1 - tid] = 2000 + tid;
    __syncthreads();
    out[tid] = fixed[tid];
}

// The load of addrwrap_cxx_load as nvcc 13.5 compiles it.
extern "C" __global__ void addrwrap_ptx_load(int* out) {
    __shared__ int fixed[ADDRWRAP_N];
    u32 tid = threadIdx.x;
    fixed[tid] = 3000 + tid;
    __syncthreads();
    int value;
    asm volatile(ADDRWRAP_REGISTER("%1", "%2") "ld.shared.b32 %0, [%a+252]; }"
                 : "=r"(value) : "l"(fixed), "r"(4 * tid) : "memory");
    out[tid] = value;
}

// The store of addrwrap_cxx_store as nvcc 13.5 compiles it.
extern "C" __global__ void addrwrap_ptx_store(int* out) {
    __shared__ int fixed[ADDRWRAP_N];
    u32 tid = threadIdx.x;
    fixed[tid] = -1;
    __syncthreads();
    asm volatile(ADDRWRAP_REGISTER("%0", "%1") "st.shared.b32 [%a+252], %2; }"
                 :: "l"(fixed), "r"(4 * tid), "r"(4000 + tid) : "memory");
    __syncthreads();
    out[tid] = fixed[tid];
}

// The register a further 2^31 - 256 below, and the largest offset that is a multiple of 4:
// the register has wrapped for every lane, lane 0 included, wherever the array is.
extern "C" __global__ void addrwrap_ptx_far(int* out) {
    __shared__ int fixed[ADDRWRAP_N];
    u32 tid = threadIdx.x;
    fixed[tid] = 5000 + tid;
    __syncthreads();
    int value;
    asm volatile(ADDRWRAP_REGISTER("%1", "%2") "sub.s32 %a, %a, 2147483392;\n\t"
                 "ld.shared.b32 %0, [%a+2147483644]; }"
                 : "=r"(value) : "l"(fixed), "r"(4 * tid) : "memory");
    out[tid] = value;
}

// For comparison, a negative offset and a register that has not wrapped: fixed + 256 - 4 * tid
// and -4. The sum is the same in 32 and in 64 bits.
extern "C" __global__ void addrwrap_ptx_back(int* out) {
    __shared__ int fixed[ADDRWRAP_N];
    u32 tid = threadIdx.x;
    fixed[tid] = 6000 + tid;
    __syncthreads();
    int value;
    asm volatile(ADDRWRAP_REGISTER("%1", "%2") "add.s32 %a, %a, 256;\n\t"
                 "ld.shared.b32 %0, [%a+-4]; }"
                 : "=r"(value) : "l"(fixed), "r"(4 * tid) : "memory");
    out[tid] = value;
}
