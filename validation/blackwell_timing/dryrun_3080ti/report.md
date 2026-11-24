# Timing calibration report

**WARNING: Measured on NVIDIA_GeForce_RTX_3080_Ti_Laptop_GPU (compute capability 8.6), NOT a Blackwell data-center GPU. These are not B200 parameters.**

Device: NVIDIA_GeForce_RTX_3080_Ti_Laptop_GPU, compute capability 8.6, 58 SMs, 16758472704 bytes.

## (a) Measured

ns per repetition is the slope of launch time against repetitions. `min` uses the fastest launch at each repetition count; `median` is the median of the per-trial slopes; spread is (max - min) / median of the per-trial slopes.

| configuration | ns/rep min | ns/rep median | spread % | trials | clock64 ticks/rep |
|---|---|---|---|---|---|
| alu_chain/warps=1 | 7.635 | 7.635 | 0.888 | 5 | 14.37 |
| alu_chain/warps=2 | 7.635 | 7.634 | 1.7 | 5 | 14.38 |
| alu_chain/warps=4 | 7.634 | 7.634 | 1.74 | 5 | 14.37 |
| alu_chain/warps=5 | 7.701 | 7.701 | 1.72 | 5 | 14.5 |
| alu_chain/warps=8 | 7.702 | 7.702 | 1.72 | 5 | 14.5 |
| alu_chain/warps=16 | 9.428 | 9.513 | 3.11 | 5 | 17.75 |
| alu_chain/warps=32 | 18.44 | 18.48 | 1.98 | 5 | 34.71 |
| cp_async/warm | 133.4 | 133.4 | 0.132 | 5 | 254.2 |
| cp_async/cold | 298 | 302 | 5.89 | 5 | 567.8 |
| global_load/load_chain/footprint=l1 | 44.44 | 44.52 | 1.48 | 5 | 84.65 |
| global_load/store/footprint=l1 | 32.39 | 32.79 | 3.22 | 5 | 61.71 |
| global_load/load_chain/footprint=l2 | 163.1 | 163.2 | 4.2 | 5 | 310.8 |
| global_load/store/footprint=l2 | 33.68 | 33.68 | 2.4 | 5 | 64.17 |
| global_load/load_chain/footprint=dram | 364.4 | 364.4 | 3.2 | 5 | 694.1 |
| global_load/store/footprint=dram | 33.69 | 33.69 | 0.0608 | 5 | 64.2 |
| gstream/triad | 0.002077 | 0.002092 | 2.71 | 5 | 0 |
| launch/empty/ctas=1 (whole launch) | 3872 | 4736 | | 200 | |
| launch/empty/ctas=allsm (whole launch) | 3904 | 4736 | | 200 | |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.73 | 14.73 | 2.72 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.78 | 15.78 | 1.69 | 5 | 30.06 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.88 | 17.88 | 1.47 | 5 | 34.06 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 22.08 | 21.95 | 1.8 | 5 | 42.06 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 30.48 | 30.48 | 0.864 | 5 | 58.06 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 47.28 | 47.54 | 1.1 | 5 | 90.06 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.73 | 14.73 | 0.916 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 67.33 | 67.45 | 1.09 | 5 | 128.2 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.822 | 4.822 | 0.0216 | 5 | 9.188 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.856 | 4.856 | 2.71 | 5 | 9.25 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.102 | 6.102 | 0.116 | 5 | 11.62 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.24 | 10.24 | 0.0422 | 5 | 19.5 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.63 | 18.63 | 2.32 | 5 | 35.5 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 35.44 | 35.69 | 0.734 | 5 | 67.5 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.46 | 2.46 | 0.0397 | 5 | 4.688 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.461 | 2.461 | 0.311 | 5 | 4.688 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.199 | 4.196 | 0.143 | 5 | 8 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.399 | 8.399 | 0.0617 | 5 | 16 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.8 | 16.8 | 2.35 | 5 | 32 |
| smem_wavefronts/store/conflict=32/warps=1 | 33.6 | 33.6 | 0.782 | 5 | 64 |

## (b) Derived model parameters

| parameter | derived | config now | derived / config | how |
|---|---|---|---|---|
| `sm_count` | 58 SMs | 148 | 0.3919 | cudaGetDeviceProperties().multiProcessorCount.  |
| `hbm_capacity_gb` | 15.6 GB | 192 | 0.08125 | cudaMemGetInfo() total bytes / 2^30. The driver reports usable memory, which is below the nominal size. |
| `clock_ghz` | 0.49117 GHz | 1.5 | 0.3274 | clock_ghz(config) 1.5 * predicted 2.5 / measured 7.635 ns per repetition of alu_chain/warps=1 (3.75 PTX instructions per repetition). Effective issue clock for one warp of register-only instructions, not the core clock. |
| `smem_wavefront_clk` | 0.5157 clk | 1 | 0.5157 | slope 1.05 ns per extra wavefront (load_chain, conflict 1..32) * clock_ghz. Clocks converted with the fitted clock_ghz = 0.4912. |
| `global_access_latency_clk` | 73.883 clk | 30 | 2.463 | config 30 + measured 163.1 ns * clock_ghz - predicted 24.17 ns * 1.5 (global_load/load_chain/footprint=l2, one load per repetition). Clocks converted with the fitted clock_ghz = 0.4912. |
| `hbm_bandwidth_tb_s` | 0.48153 TB/s | 4.141 | 0.1163 | STREAM Triad: 1 / (0.002077 ns per byte) / 1000 (gstream/triad, 24 bytes per element). Same definition as the config's published figure. |

Not derived (benchmark missing from the results, or unusable): `l2_bandwidth_per_sm_gb_s`, `tma_latency_ns`, `tma_bytes_per_clk`, `tmem_access_clk`, `tensor_core_tflops_f16`, `tensor_core_tflops_tf32`, `mma_latency_clk`

Other measured quantities:

| quantity | value | unit | note |
|---|---|---|---|
| %clock64 ticks per ns | 1.8828 | ticks/ns | alu_chain/warps=1: clock64 slope / event-time slope. 0 means the build did not read the clock |
| shared load, no conflict | 14.73 | ns | smem_wavefronts/load_chain fit at conflict=1 |

## (c) Measured against the simulator's current prediction

Prediction: `warplens run --timing` on the same program (predict.py), ns per repetition.

| configuration | measured | predicted | measured / predicted |
|---|---|---|---|
| alu_chain/warps=1 | 7.635 | 2.5 | 3.054 |
| alu_chain/warps=2 | 7.635 | 2.5 | 3.054 |
| alu_chain/warps=4 | 7.634 | 2.5 | 3.054 |
| alu_chain/warps=5 | 7.701 | 5 | 1.54 |
| alu_chain/warps=8 | 7.702 | 5 | 1.54 |
| alu_chain/warps=16 | 9.428 | 10 | 0.9428 |
| alu_chain/warps=32 | 18.44 | 20 | 0.9218 |
| cp_async/warm | 133.4 | 122.1 | 1.092 |
| cp_async/cold | 298 | 122.1 | 2.44 |
| global_load/load_chain/footprint=l1 | 44.44 | 24.17 | 1.839 |
| global_load/store/footprint=l1 | 32.39 | 3.667 | 8.835 |
| global_load/load_chain/footprint=l2 | 163.1 | 24.17 | 6.751 |
| global_load/store/footprint=l2 | 33.68 | 3.667 | 9.186 |
| global_load/load_chain/footprint=dram | 364.4 | 24.17 | 15.08 |
| global_load/store/footprint=dram | 33.69 | 3.667 | 9.188 |
| gstream/triad | 0.002077 | 0 | n/a: the model predicts no cost |
| launch/empty/ctas=1 (launch) | 3872 | 6.667 | 580.8 |
| launch/empty/ctas=allsm (launch) | 3904 | 6.667 | 585.6 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.73 | 3.333 | 4.419 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.78 | 4 | 3.945 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.88 | 5.333 | 3.353 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 22.08 | 8 | 2.76 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 30.48 | 13.33 | 2.286 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 47.28 | 24 | 1.97 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.73 | 3.333 | 4.419 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 67.33 | 85.33 | 0.789 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.822 | 6 | 0.8037 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.856 | 6.667 | 0.7284 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.102 | 8 | 0.7627 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.24 | 10.67 | 0.9596 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.63 | 16 | 1.165 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 35.44 | 26.67 | 1.329 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.46 | 2.5 | 0.984 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.461 | 3.167 | 0.7771 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.199 | 4.5 | 0.9332 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.399 | 7.167 | 1.172 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.8 | 12.5 | 1.344 |
| smem_wavefronts/store/conflict=32/warps=1 | 33.6 | 23.17 | 1.45 |

85 predicted configuration(s) have no measurement, e.g. `mbar/arrive_wait/threads=1`, `mbar/arrive_wait/threads=32`, `mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1`, `mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1`, `mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1`
