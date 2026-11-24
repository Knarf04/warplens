# Predicted by the simulator

`warplens run --timing --model B200 bin/sim/<bench> --sim`, config `b200.yaml` (uncalibrated unless stated), 148 SMs assumed to share HBM.

| configuration | ns per repetition | PTX warp-instructions per repetition | limited by |
|---|---|---|---|
| alu_chain/warps=1 | 2.5 | 3.75 | latency |
| alu_chain/warps=2 | 2.5 | 7.5 | latency |
| alu_chain/warps=4 | 2.5 | 15 | issue_slots |
| alu_chain/warps=5 | 5 | 18.75 | latency |
| alu_chain/warps=8 | 5 | 30 | issue_slots |
| alu_chain/warps=16 | 10 | 60 | issue_slots |
| alu_chain/warps=32 | 20 | 120 | issue_slots |
| clc_latency/fail/plain/ctas=1 | 11 | 16.5 | latency |
| clc_latency/fail/plain/ctas=2 | 12.3333 | 25.75 | latency |
| clc_latency/fail/multicast/ctas=2 | 13 | 34 | latency |
| clc_latency/succeed/plain/ctas=1 | 22 | 30 | latency |
| clc_latency/succeed/multicast/ctas=2 | 24 | 62 | latency |
| cluster_costs/load/multicast/ctas=1 | 193.667 | 20.5 | latency |
| cluster_costs/load/multicast/ctas=2 | 193.667 | 36 | latency |
| cluster_costs/load/multicast/ctas=4 | 193.667 | 67 | latency |
| cluster_costs/load/separate/ctas=1 | 196.333 | 20.5 | latency |
| cluster_costs/load/separate/ctas=2 | 196.333 | 41 | latency |
| cluster_costs/load/separate/ctas=4 | 196.333 | 82 | latency |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 176.385 | 18 | latency |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 659.605 | 33 | latency |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 95.9589 | 36 | Tensor_Core |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 337.902 | 66 | Tensor_Core |
| cp_async/warm | 122.132 | 8.75 | latency |
| cp_async/cold | 122.132 | 8.75 | latency |
| global_load/load_chain/footprint=l1 | 24.1667 | 6.25 | latency |
| global_load/store/footprint=l1 | 3.66667 | 5.5 | latency |
| global_load/load_chain/footprint=l2 | 24.1667 | 6.25 | latency |
| global_load/store/footprint=l2 | 3.66667 | 5.5 | latency |
| global_load/load_chain/footprint=dram | 24.1667 | 6.25 | latency |
| global_load/store/footprint=dram | 3.66667 | 5.5 | latency |
| gstream/triad | 0 | 0.0325521 | latency |
| launch/empty/ctas=1 (whole launch) | 6.66667 | | latency |
| launch/empty/ctas=allsm (whole launch) | 6.66667 | | latency |
| mbar/arrive_wait/threads=1 | 3 | 4.5 | latency |
| mbar/arrive_wait/threads=32 | 3 | 4.5 | latency |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 21.8933 | 20 | latency |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 26.9199 | 26 | latency |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 37.6397 | 35 | latency |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 57.7461 | 51 | latency |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 36.9731 | 20 | latency |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 57.0795 | 26 | latency |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 97.9589 | 35 | latency |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 178.385 | 51 | Tensor_Core |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 57.0795 | 20 | latency |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 97.2923 | 26 | latency |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 178.385 | 35 | Tensor_Core |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 339.236 | 51 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 26.9199 | 20 | latency |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 36.9731 | 26 | latency |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 57.7461 | 35 | latency |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 97.9589 | 51 | latency |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 57.0795 | 20 | latency |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 97.2923 | 26 | latency |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 178.385 | 35 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 339.236 | 51 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 97.2923 | 20 | latency |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 177.718 | 26 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 339.236 | 35 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 660.938 | 51 | Tensor_Core |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 21.8948 | 20 | latency |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 26.923 | 26 | latency |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 37.646 | 35 | latency |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 57.7586 | 51 | latency |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 36.9793 | 20 | latency |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 57.092 | 26 | latency |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 97.984 | 35 | latency |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 178.435 | 51 | Tensor_Core |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 57.092 | 20 | latency |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 97.3173 | 26 | latency |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 178.435 | 35 | Tensor_Core |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 339.336 | 51 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 26.923 | 20 | latency |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 36.9793 | 26 | latency |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 57.7586 | 35 | latency |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 97.984 | 51 | latency |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 57.092 | 20 | latency |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 97.3173 | 26 | latency |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 178.435 | 35 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 339.336 | 51 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 97.3173 | 20 | latency |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 177.768 | 26 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 339.336 | 35 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 661.138 | 51 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 97.2923 | 2960 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 177.718 | 3848 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 339.236 | 5180 | Tensor_Core |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 660.938 | 7548 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 97.3173 | 2960 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 177.768 | 3848 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 339.336 | 5180 | Tensor_Core |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 661.138 | 7548 | Tensor_Core |
| ncu_banks/linear/block=32x32 (whole launch) | 851.333 | | latency |
| ncu_banks/padded/block=32x32 (whole launch) | 190 | | issue_slots |
| ncu_banks/swizzled/block=32x32 (whole launch) | 211.333 | | issue_slots |
| ncu_banks/linear/block=32x8 (whole launch) | 104 | | latency |
| ncu_banks/padded/block=32x8 (whole launch) | 82.6667 | | latency |
| ncu_banks/swizzled/block=32x8 (whole launch) | 88 | | latency |
| ncu_banks/linear/block=16x16 (whole launch) | 104 | | latency |
| ncu_banks/padded/block=16x16 (whole launch) | 77.3333 | | latency |
| ncu_banks/swizzled/block=16x16 (whole launch) | 72 | | latency |
| pipelined_gemm/stages=1/ctas=8 | 1081.14 | 696 | latency |
| pipelined_gemm/stages=2/ctas=8 | 878.35 | 696 | HBM |
| pipelined_gemm/stages=4/ctas=8 | 878.35 | 696 | HBM |
| pipelined_gemm/stages=1/ctas=allsm | 1081.14 | 12876 | latency |
| pipelined_gemm/stages=2/ctas=allsm | 878.35 | 12876 | HBM |
| pipelined_gemm/stages=4/ctas=allsm | 878.35 | 12876 | HBM |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 3.33333 | 4 | latency |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 4 | 4 | latency |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 5.33333 | 4 | latency |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 8 | 4 | latency |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 13.3333 | 4 | latency |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 24 | 4 | latency |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 3.33333 | 16 | latency |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 85.3333 | 16 | SMEM |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 6 | 8 | latency |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 6.66667 | 8 | latency |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 8 | 8 | latency |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.6667 | 8 | latency |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 16 | 8 | latency |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 26.6667 | 8 | latency |
| smem_wavefronts/store/conflict=1/warps=1 | 2.5 | 2.75 | latency |
| smem_wavefronts/store/conflict=2/warps=1 | 3.16667 | 2.75 | latency |
| smem_wavefronts/store/conflict=4/warps=1 | 4.5 | 2.75 | latency |
| smem_wavefronts/store/conflict=8/warps=1 | 7.16667 | 2.75 | latency |
| smem_wavefronts/store/conflict=16/warps=1 | 12.5 | 2.75 | latency |
| smem_wavefronts/store/conflict=32/warps=1 | 23.1667 | 2.75 | latency |
| tma/load/warm/bytes=64/ctas=1 | 104.833 | 9 | latency |
| tma/load/warm/bytes=256/ctas=1 | 105.833 | 9 | latency |
| tma/load/warm/bytes=1024/ctas=1 | 109.833 | 9 | latency |
| tma/load/warm/bytes=4096/ctas=1 | 125.833 | 9 | latency |
| tma/load/warm/bytes=16384/ctas=1 | 189.833 | 9 | latency |
| tma/load/warm/bytes=32768/ctas=1 | 275.167 | 9 | latency |
| tma/store/bytes=64/ctas=1 | 103.333 | 7.5 | latency |
| tma/store/bytes=256/ctas=1 | 104.333 | 7.5 | latency |
| tma/store/bytes=1024/ctas=1 | 108.333 | 7.5 | latency |
| tma/store/bytes=4096/ctas=1 | 124.333 | 7.5 | latency |
| tma/store/bytes=16384/ctas=1 | 188.333 | 7.5 | latency |
| tma/store/bytes=32768/ctas=1 | 273.667 | 7.5 | latency |
| tma/load/cold/bytes=64/ctas=1 | 105.81 | 9 | latency |
| tma/load/cold/bytes=256/ctas=1 | 113.649 | 9 | latency |
| tma/load/cold/bytes=1024/ctas=1 | 141.098 | 9 | latency |
| tma/load/cold/bytes=4096/ctas=1 | 250.892 | 9 | latency |
| tma/load/cold/bytes=16384/ctas=1 | 690.067 | 9 | HBM |
| tma/load/cold/bytes=32768/ctas=1 | 1275.63 | 9 | HBM |
| tma/load/cold/bytes=16384/ctas=8 | 689.9 | 64 | latency |
| tma/load/cold/bytes=16384/ctas=37 | 689.9 | 296 | latency |
| tma/load/cold/bytes=16384/ctas=allsm | 689.9 | 1184 | latency |
| tmem_access/ld/x1 | 5.33333 | 4 | latency |
| tmem_access/ld/x8 | 10 | 11 | latency |
| tmem_access/st/x1 | 4.66667 | 3 | latency |
| tmem_access/st/x8 | 4.66667 | 3 | latency |
| tmem_access/ld/x1/nowait | 4.5 | 2.75 | latency |
| tmem_access/st/x1/nowait | 3.83333 | 1.75 | latency |

Not simulated:
- `mma_rate/kind=i8`: warplens implements tcgen05.mma for kind::f16 and kind::tf32 only, and the timing model has no rate for other kinds; the i8 variant is hardware-only
- `%clock64`: warplens does not implement the %clock64 register; the simulator build (BENCH_CLOCK=0) omits the two clock reads around each loop, so clk is 0 in every predicted line
