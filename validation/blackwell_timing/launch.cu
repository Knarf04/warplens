// launch: the cost of launching a kernel that does nothing.
//
// Calibrates: no model parameter. The timing model starts every kernel at t = 0 and has
// no launch cost. This is the baseline that every other benchmark removes by taking the
// slope of time against repetitions; it also tells how long a launch must run before
// event timing is trustworthy (launch time and its spread).
// Also prints the DEVICE line, which carries `sm_count` and the memory size
// (`hbm_capacity_gb`) as the driver reports them.
//
// Method: one empty kernel per launch; cudaEvent time and host wall time (launch to
// cudaDeviceSynchronize) for each of many launches. fit.py reports min and median.
// Valid on Ampere.
#include "common.h"

__global__ void empty_kernel(uint64_t* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[OUT_CLK] = 0; out[OUT_CHECK] = 1; }
}

int main(int argc, char** argv) {
    Opts o = bench_init(argc, argv, "launch");
    int launches = o.sim ? 2 : o.trials * 40;
    struct { const char* name; int ctas; } grids[] = {{"1", 1}, {"allsm", o.sm_count}};
    for (auto& g : grids) {
        std::string id = std::string("launch/empty/ctas=") + g.name;
        if (!selected(o, id)) continue;
        std::string extra = "nctas=" + std::to_string(g.ctas);
        for (int t = o.sim ? 0 : -1; t < launches; ++t)
            timed_launch(o, id, extra, 0, t, [&] { empty_kernel<<<g.ctas, 32>>>(o.d_out); });
    }
    return 0;
}
