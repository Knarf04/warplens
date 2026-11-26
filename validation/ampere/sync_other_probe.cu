// shfl.sync and bar.warp.sync with a member mask that names lanes which are elsewhere.
//
// One warp. Thread t holds the value 100 + t and has a role:
//
//   role 0  executes the instruction under test with the mask it was given
//   role 1  returns before it
//   role 2  takes another branch: sets flag[t] = 1 and its value to 1000 + t, with no
//           instruction that has a mask
//   role 3  takes another branch and executes the same kind of instruction there, at a
//           call site of its own, with the mask it was given
//
// mode 0  __shfl_xor_sync(mask, value, 16)  (shfl.sync.bfly.b32): the value of lane t ^ 16
// mode 1  __syncwarp(mask)                  (bar.warp.sync), then read flag[t ^ 16]
//
// out[2 t] is what a role 0 thread got (the shuffled value; or the flag it read), out[2 t + 1]
// what a role 3 thread got. A word the thread does not write keeps what the host put there
// (eeeeeeee). What is asked: does the launch return, and if it does, did the instruction wait
// for the lanes on the other branch (it then sees 1000 + lane, or flag 1) or go on without
// them (100 + lane, or flag 0)?
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

extern "C" __global__ void sync_other(unsigned* out, const int* role, const unsigned* mask, int* flag, int mode) {
    int t = threadIdx.x;
    unsigned m = mask[t];
    int what = role[t];
    int value = 100 + t;
    if (what == 1) return;
    if (what == 2) {
        flag[t] = 1;
        value = 1000 + t;
    } else if (what == 3) {
        flag[t] = 1;
        value = 1000 + t;
        if (mode == 0) out[2 * t + 1] = __shfl_xor_sync(m, value, 16);
        else { __syncwarp(m); out[2 * t + 1] = flag[t ^ 16]; }
    } else {
        if (mode == 0) out[2 * t] = __shfl_xor_sync(m, value, 16);
        else { __syncwarp(m); out[2 * t] = flag[t ^ 16]; }
    }
}

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(e_)); exit(3); } } while (0)

struct Case { const char* name; int mode; int role[32]; unsigned mask[32]; };

static std::vector<Case> cases() {
    std::vector<Case> all;
    for (int mode : {0, 1}) {
        auto add = [&](const char* name, int other, int from, int to, unsigned mask_of_role0, unsigned mask_of_other) {
            Case c{name, mode, {}, {}};
            for (int t = 0; t < 32; ++t) {
                c.role[t] = t >= from && t < to ? other : 0;
                c.mask[t] = c.role[t] ? mask_of_other : mask_of_role0;
            }
            all.push_back(c);
        };
        add("every lane, full mask (the ordinary case)", 0, 0, 0, 0xffffffffu, 0);
        add("lanes 0-15 returned, full mask", 1, 0, 16, 0xffffffffu, 0);
        add("lanes 0-15 on a branch without the instruction, full mask", 2, 0, 16, 0xffffffffu, 0);
        add("lane 0 on a branch without the instruction, full mask", 2, 0, 1, 0xffffffffu, 0);
        add("lanes 0-15 at a call site of their own, both with the full mask", 3, 0, 16, 0xffffffffu, 0xffffffffu);
        add("lanes 0-15 at a call site of their own, each group with its own lanes as the mask", 3, 0, 16, 0xffff0000u, 0x0000ffffu);
        add("lanes 0-15 on a branch without the instruction, the mask names only the callers", 2, 0, 16, 0xffff0000u, 0);
        add("every lane, the mask names only lanes 0-15", 0, 0, 0, 0x0000ffffu, 0);
    }
    return all;
}

int main(int argc, char** argv) {
    std::vector<Case> all = cases();
    if (argc < 2) { printf("%zu\n", all.size()); return 0; }
    const Case& c = all[atoi(argv[1])];
    printf("mode %d (%s): %s:", c.mode, c.mode ? "bar.warp.sync" : "shfl.sync", c.name);
    fflush(stdout);
    unsigned out[64]; int flag[32] = {};
    for (unsigned& word : out) word = 0xeeeeeeeeu;
    unsigned *d_out, *d_mask; int *d_role, *d_flag;
    CHECK(cudaMalloc(&d_out, sizeof out)); CHECK(cudaMalloc(&d_mask, sizeof c.mask));
    CHECK(cudaMalloc(&d_role, sizeof c.role)); CHECK(cudaMalloc(&d_flag, sizeof flag));
    CHECK(cudaMemcpy(d_out, out, sizeof out, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, c.mask, sizeof c.mask, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_role, c.role, sizeof c.role, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_flag, flag, sizeof flag, cudaMemcpyHostToDevice));
    sync_other<<<1, 32>>>(d_out, d_role, d_mask, d_flag, c.mode);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost));
    printf(" role 0 got");
    for (int t = 0; t < 32; ++t) if (c.role[t] == 0) printf(" %d:%x", t, out[2 * t]);
    printf("; role 3 got");
    for (int t = 0; t < 32; ++t) if (c.role[t] == 3) printf(" %d:%x", t, out[2 * t + 1]);
    printf("\n");
    return 0;
}
