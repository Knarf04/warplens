# Timing calibration report

Device: NVIDIA_B200, compute capability 10.0, 148 SMs, 191490555904 bytes.

## (a) Measured

ns per repetition is the slope of launch time against repetitions. `min` uses the fastest launch at each repetition count; `median` is the median of the per-trial slopes; spread is (max - min) / median of the per-trial slopes.

| configuration | ns/rep min | ns/rep median | spread % | trials | clock64 ticks/rep |
|---|---|---|---|---|---|
| alu_chain/warps=1 | 7.315 | 7.315 | 0.0139 | 5 | 14.38 |
| alu_chain/warps=2 | 7.316 | 7.316 | 0.0113 | 5 | 14.38 |
| alu_chain/warps=4 | 7.315 | 7.316 | 0.00513 | 5 | 14.38 |
| alu_chain/warps=5 | 7.379 | 7.379 | 0.00586 | 5 | 14.5 |
| alu_chain/warps=8 | 7.379 | 7.379 | 0.00803 | 5 | 14.5 |
| alu_chain/warps=16 | 8.925 | 8.925 | 0.0114 | 5 | 17.53 |
| alu_chain/warps=32 | 17.03 | 17.05 | 0.375 | 5 | 33.5 |
| clc_latency/fail/plain/ctas=1 | 260.8 | 260.8 | 0.0565 | 5 | 512.2 |
| clc_latency/fail/plain/ctas=2 | 463.2 | 463.3 | 0.0946 | 5 | 910.2 |
| clc_latency/fail/multicast/ctas=2 | 451 | 451 | 0.243 | 5 | 886.3 |
| clc_latency/succeed/plain/ctas=1 | 1.43 | 1.488 | 48.2 | 5 | 2.836 |
| clc_latency/succeed/multicast/ctas=2 | 4.294 | 4.294 | 6.26 | 5 | 8.509 |
| cluster_costs/load/multicast/ctas=1 | 849.3 | 849.3 | 0.0347 | 5 | 1669 |
| cluster_costs/load/multicast/ctas=2 | 849.4 | 849.5 | 0.0481 | 5 | 1669 |
| cluster_costs/load/multicast/ctas=4 | 841.1 | 841.1 | 0.0295 | 5 | 1653 |
| cluster_costs/load/separate/ctas=1 | 886.5 | 886.5 | 0.0242 | 5 | 1742 |
| cluster_costs/load/separate/ctas=2 | 890.7 | 890.7 | 0.018 | 5 | 1750 |
| cluster_costs/load/separate/ctas=4 | 886.3 | 886.6 | 0.0785 | 5 | 1742 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 295.9 | 296 | 0.0177 | 5 | 546 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 478.1 | 478.1 | 0.00871 | 5 | 882 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 272.1 | 272.1 | 0.0306 | 5 | 502 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 454.2 | 454.2 | 0.0121 | 5 | 838 |
| cp_async/warm | 147.4 | 147.4 | 0.159 | 5 | 289.7 |
| cp_async/cold | 325.6 | 154.3 | 181 | 5 | 639.9 |
| global_load/load_chain/footprint=l1 | 43.99 | 43.99 | 0.00737 | 5 | 86.45 |
| global_load/store/footprint=l1 | 23.06 | 23.06 | 0.00347 | 5 | 45.31 |
| global_load/load_chain/footprint=l2 | 189 | 189 | 0.00807 | 5 | 371.4 |
| global_load/store/footprint=l2 | 20.82 | 20.82 | 0.0659 | 5 | 40.91 |
| global_load/load_chain/footprint=dram | 691.5 | 691.5 | 1.01 | 5 | 1359 |
| global_load/store/footprint=dram | 21.03 | 21.02 | 0.242 | 5 | 41.34 |
| gstream/triad | 0.0001624 | 0.0001623 | 0.41 | 5 | 0 |
| launch/empty/ctas=1 (whole launch) | 3744 | 4720 | | 200 | |
| launch/empty/ctas=allsm (whole launch) | 3744 | 4640 | | 200 | |
| mbar/arrive_wait/threads=1 | 32.7 | 32.7 | 0.0554 | 5 | 64.25 |
| mbar/arrive_wait/threads=32 | 32.7 | 32.7 | 0.00783 | 5 | 64.25 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 209.2 | 209.2 | 0.0138 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 261.8 | 261.8 | 0.0435 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 310 | 310 | 0.0186 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 426.6 | 426.6 | 0.017 | 5 | 787 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 209.2 | 209.2 | 0.0193 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 261.8 | 261.8 | 0.0264 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 310 | 310 | 0.0155 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 513.3 | 513.3 | 0.0186 | 5 | 947 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 295.9 | 295.9 | 0.0143 | 5 | 546 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 348.5 | 348.5 | 0.0195 | 5 | 643 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 478 | 478 | 0.0182 | 5 | 882 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 757.2 | 757.2 | 0.00556 | 5 | 1397 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 209.2 | 209.2 | 0.0229 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 261.8 | 261.8 | 0.0171 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 310 | 310 | 0.00556 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 426.6 | 426.6 | 0.0148 | 5 | 787 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 209.2 | 209.2 | 0.00824 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 261.8 | 261.8 | 0.0203 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 310 | 310 | 0.00867 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 513.3 | 513.3 | 0.0274 | 5 | 947 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 295.9 | 295.9 | 0.0184 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 348.5 | 348.5 | 0.0119 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 478.1 | 478.1 | 0.0071 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 773.4 | 773.5 | 0.00538 | 5 | 1427 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 211.9 | 211.9 | 0.0414 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 264.5 | 264.5 | 0.0208 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 312.8 | 312.7 | 0.0119 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 429.3 | 429.3 | 0.00805 | 5 | 792 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 211.9 | 211.9 | 0.032 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 264.5 | 264.5 | 0.0152 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 312.7 | 312.8 | 0.0201 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 516 | 516 | 0.00967 | 5 | 952 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 298.7 | 298.6 | 0.0167 | 5 | 551 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 351.2 | 351.2 | 0.00564 | 5 | 648 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 480.8 | 480.8 | 0.0157 | 5 | 887 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 759.9 | 759.9 | 0.0114 | 5 | 1402 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 211.9 | 211.9 | 0.0212 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 264.5 | 264.5 | 0.00751 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 312.8 | 312.8 | 0.0143 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 429.3 | 429.3 | 0.00776 | 5 | 792 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 211.9 | 211.9 | 0.0112 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 264.5 | 264.5 | 0.0218 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 312.7 | 312.7 | 0.0385 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 516 | 516 | 0.0164 | 5 | 952 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 298.7 | 298.6 | 0.0124 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 351.2 | 351.2 | 0.0184 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 480.8 | 480.7 | 0.0333 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 776.2 | 776.2 | 0.00767 | 5 | 1432 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 297.5 | 297.5 | 0.134 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.2 | 351.2 | 0.255 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 481.1 | 481.1 | 0.104 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 777.3 | 777.2 | 0.0225 | 5 | 1427 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 300.7 | 300.6 | 0.1 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 353.2 | 353.5 | 0.382 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 483.5 | 483.5 | 0.0818 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 779.9 | 780 | 0.0486 | 5 | 1432 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 288.9 | 288.9 | 0.0221 | 5 | 546 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 340.2 | 340.2 | 0.00734 | 5 | 643 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 468.8 | 468.7 | 0.0356 | 5 | 886 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 755.1 | 755.1 | 0.00738 | 5 | 1427 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 290.1 | 290.1 | 0.0249 | 5 | 546 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 340.3 | 342.5 | 0.692 | 5 | 643 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 471.9 | 471.8 | 0.0696 | 5 | 886 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 759.9 | 759.9 | 0.0246 | 5 | 1427 |
| ncu_banks/linear/block=32x32 (whole launch) | 6880 | 7808 | | 5 | |
| ncu_banks/padded/block=32x32 (whole launch) | 6272 | 6816 | | 5 | |
| ncu_banks/swizzled/block=32x32 (whole launch) | 5536 | 6624 | | 5 | |
| ncu_banks/linear/block=32x8 (whole launch) | 5024 | 6944 | | 5 | |
| ncu_banks/padded/block=32x8 (whole launch) | 5056 | 5376 | | 5 | |
| ncu_banks/swizzled/block=32x8 (whole launch) | 5536 | 6464 | | 5 | |
| ncu_banks/linear/block=16x16 (whole launch) | 4992 | 5888 | | 5 | |
| ncu_banks/padded/block=16x16 (whole launch) | 4864 | 6656 | | 5 | |
| ncu_banks/swizzled/block=16x16 (whole launch) | 5504 | 6880 | | 5 | |
| pipelined_gemm/stages=1/ctas=8 | 1033 | 892.3 | 16.4 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=8 | 539.3 | 456.1 | 21.4 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=8 | 291.2 | 290.4 | 0.603 | 5 | 0 |
| pipelined_gemm/stages=1/ctas=allsm | 1110 | 1109 | 0.789 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=allsm | 618.4 | 620.5 | 2 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=allsm | 393.2 | 397.1 | 1.97 | 5 | 0 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.28 | 14.28 | 0.0113 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.3 | 15.3 | 0.00794 | 5 | 30.06 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.33 | 17.33 | 0.0331 | 5 | 34.06 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.41 | 21.41 | 0.0104 | 5 | 42.06 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.55 | 29.55 | 0.00932 | 5 | 58.06 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.83 | 45.83 | 0.00275 | 5 | 90.06 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.28 | 14.28 | 0.00783 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.14 | 65.14 | 0.0017 | 5 | 128 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.771 | 4.77 | 0.0332 | 5 | 9.375 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.834 | 4.834 | 0.0172 | 5 | 9.5 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.138 | 6.139 | 0.0135 | 5 | 12.06 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.21 | 10.21 | 0.00672 | 5 | 20.06 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.35 | 18.35 | 0.0214 | 5 | 36.06 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.64 | 34.64 | 0.00576 | 5 | 68.06 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.545 | 2.545 | 0.0578 | 5 | 5 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.545 | 2.545 | 0.0396 | 5 | 5 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.072 | 4.072 | 0.0318 | 5 | 8 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.142 | 8.142 | 0.00865 | 5 | 16 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.29 | 16.29 | 0.00746 | 5 | 32 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.57 | 32.57 | 0.003 | 5 | 64 |
| tma/load/warm/bytes=64/ctas=1 | 153.4 | 153.4 | 0.138 | 5 | 301.4 |
| tma/load/warm/bytes=256/ctas=1 | 152.2 | 152.6 | 0.383 | 5 | 299.1 |
| tma/load/warm/bytes=1024/ctas=1 | 162 | 162 | 0.161 | 5 | 318.1 |
| tma/load/warm/bytes=4096/ctas=1 | 198.8 | 198.9 | 0.395 | 5 | 390.5 |
| tma/load/warm/bytes=16384/ctas=1 | 234.9 | 235 | 0.342 | 5 | 461.5 |
| tma/load/warm/bytes=32768/ctas=1 | 302.6 | 302.6 | 0.337 | 5 | 594.5 |
| tma/store/bytes=64/ctas=1 | 23.58 | 23.62 | 1.23 | 5 | 46.25 |
| tma/store/bytes=256/ctas=1 | 26.53 | 26.52 | 0.718 | 5 | 52.25 |
| tma/store/bytes=1024/ctas=1 | 38.82 | 38.79 | 0.293 | 5 | 76.25 |
| tma/store/bytes=4096/ctas=1 | 87.6 | 87.61 | 0.11 | 5 | 172.2 |
| tma/store/bytes=16384/ctas=1 | 284.1 | 284.1 | 0.0191 | 5 | 558.2 |
| tma/store/bytes=32768/ctas=1 | 544.7 | 544.7 | 0.0164 | 5 | 1070 |
| tma/load/cold/bytes=64/ctas=1 | 301.7 | 300.7 | 1.44 | 5 | 592.3 |
| tma/load/cold/bytes=256/ctas=1 | 436.2 | 435.9 | 0.692 | 5 | 856.3 |
| tma/load/cold/bytes=1024/ctas=1 | 461.2 | 462.7 | 1.41 | 5 | 906.5 |
| tma/load/cold/bytes=4096/ctas=1 | 503.2 | 521.4 | 4.61 | 5 | 988.7 |
| tma/load/cold/bytes=16384/ctas=1 | 657.9 | 659.1 | 1.35 | 5 | 1292 |
| tma/load/cold/bytes=32768/ctas=1 | 716.4 | 718.2 | 0.709 | 5 | 1407 |
| tma/load/cold/bytes=16384/ctas=8 | 663.1 | 661.6 | 0.876 | 5 | 1286 |
| tma/load/cold/bytes=16384/ctas=37 | 659.5 | 659.8 | 0.71 | 5 | 1279 |
| tma/load/cold/bytes=16384/ctas=allsm | 701.7 | 702.8 | 0.858 | 5 | 1358 |
| tmem_access/ld/x1 | 14.79 | 14.79 | 0.0199 | 5 | 29.06 |
| tmem_access/ld/x8 | 25.7 | 25.7 | 0.00598 | 5 | 50.5 |
| tmem_access/st/x1 | 6.14 | 6.139 | 0.0334 | 5 | 12.06 |
| tmem_access/st/x8 | 8.079 | 8.079 | 0.038 | 5 | 15.88 |
| tmem_access/ld/x1/nowait | 15.04 | 15.04 | 0.0166 | 5 | 29.56 |
| tmem_access/st/x1/nowait | 0.9217 | 0.9213 | 0.139 | 5 | 1.812 |

## (b) Derived model parameters

| parameter | derived | config now | derived / config | how |
|---|---|---|---|---|
| `sm_count` | 148 SMs | 148 | 1 | cudaGetDeviceProperties().multiProcessorCount.  |
| `hbm_capacity_gb` | 178.3 GB | 192 | 0.9286 | cudaMemGetInfo() total bytes / 2^30. The driver reports usable memory, which is below the nominal size. |
| `clock_ghz` | 0.51262 GHz | 1.5 | 0.3417 | clock_ghz(config) 1.5 * predicted 2.5 / measured 7.315 ns per repetition of alu_chain/warps=1 (3.75 PTX instructions per repetition). Effective issue clock for one warp of register-only instructions, not the core clock. |
| `smem_wavefront_clk` | 0.52177 clk | 1 | 0.5218 | slope 1.018 ns per extra wavefront (load_chain, conflict 1..32) * clock_ghz. Clocks converted with the fitted clock_ghz = 0.5126. |
| `global_access_latency_clk` | 90.629 clk | 30 | 3.021 | config 30 + measured 189 ns * clock_ghz - predicted 24.17 ns * 1.5 (global_load/load_chain/footprint=l2, one load per repetition). Clocks converted with the fitted clock_ghz = 0.5126. |
| `tmem_access_clk` | 3.5817 clk | 4 | 0.8954 | config 4 + measured 14.79 ns * clock_ghz - predicted 5.333 ns * 1.5 (tmem_access/ld/x1, one tcgen05.ld per repetition). Clocks converted with the fitted clock_ghz = 0.5126. |
| `clc_latency_ns` | 236.37 ns | 100 | 2.364 | measured 260.8 ns per request - 24.38 ns of instruction time the simulator counts for the same loop (clc_latency/fail/plain/ctas=1, requests that fail in a launch of one cluster). One latency for every form of the request, as the model has. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5126. NOT ONE NUMBER on this device: clc_latency/fail/plain/ctas=2 implies 435 ns, clc_latency/fail/multicast/ctas=2 implies 422.7 ns, clc_latency/succeed/plain/ctas=1 implies 303.3 ns, clc_latency/succeed/multicast/ctas=2 implies 493 ns; the model would need a latency per form. |
| `tma_bytes_per_clk` | 438.76 bytes/clk | 128 | 3.428 | 224.9 bytes/ns (1 / slope of round trip against bytes, load/warm) / clock_ghz. The faster of warm loads and stores is taken as the engine rate: the model charges max(engine, memory) per copy, so each slope is the slower of the two. Clocks converted with the fitted clock_ghz = 0.5126. |
| `l2_bandwidth_per_sm_gb_s` | 224.92 GB/s | 400 | 0.5623 | 1 / slope of round trip against bytes, warm loads, in bytes/ns. LOWER BOUND: warm loads run at the engine rate, so L2 is at least this fast; the model reproduces the measurement with any value from here up. |
| `tma_latency_ns` | 147.02 ns | 100 | 1.47 | intercept 160.2 ns of warm-load round trip against bytes - 13.17 ns of instruction time the simulator counts for the same loop. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5126. |
| `hbm_bandwidth_tb_s` | 6.1575 TB/s | 4.141 | 1.487 | STREAM Triad: 1 / (0.0001624 ns per byte) / 1000 (gstream/triad, 24 bytes per element). Same definition as the config's published figure. Cold TMA loads on every SM gave 8.57 TB/s. |
| `tensor_core_tflops_f16` | 2240.9 TFLOPS | 1929.6 | 1.161 | 2 * 128 * 256 * 16 FLOP / 69.25 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |
| `mma_latency_clk` | 96.478 clk | 11.3 | 8.538 | config 11.3 + measured batch intercept 215.3 ns * clock_ghz - predicted 16.81 ns * 1.5 (f16 m128 n256). Includes whatever tcgen05.commit and the mbarrier wake-up cost on hardware beyond their instructions. Clocks converted with the fitted clock_ghz = 0.5126. |
| `tensor_core_tflops_tf32` | 1120.9 TFLOPS | 964.5 | 1.162 | 2 * 128 * 256 * 8 FLOP / 69.23 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |

Other measured quantities:

| quantity | value | unit | note |
|---|---|---|---|
| %clock64 ticks per ns | 1.965 | ticks/ns | alu_chain/warps=1: clock64 slope / event-time slope. 0 means the build did not read the clock |
| shared load, no conflict | 14.281 | ns | smem_wavefronts/load_chain fit at conflict=1 |
| CLC request, fail/plain/ctas=1 | 260.76 | ns | slope of launch time against requests; minus 24.38 ns of instructions the simulator counts for the same loop = 236.4 ns of latency |
| CLC request, fail/plain/ctas=2 | 463.24 | ns | slope of launch time against requests; minus 28.29 ns of instructions the simulator counts for the same loop = 435 ns of latency |
| CLC request, fail/multicast/ctas=2 | 451.02 | ns | slope of launch time against requests; minus 28.29 ns of instructions the simulator counts for the same loop = 422.7 ns of latency |
| CLC request, succeed/plain/ctas=1 | 361.82 | ns | 711 ticks per request of the first cluster (clk / (check + 1)) / 1.965 ticks per ns; minus 58.52 ns of instructions the simulator counts for the same loop = 303.3 ns of latency |
| CLC request, succeed/multicast/ctas=2 | 555.45 | ns | 1091 ticks per request of the first cluster (clk / (check + 1)) / 1.965 ticks per ns; minus 62.42 ns of instructions the simulator counts for the same loop = 493 ns of latency |
| TMA load/warm: ns per byte | 0.004446 | ns/byte | = 224.9 GB/s for one SM |
| TMA load/warm: intercept | 160.18 | ns | round trip of a zero-byte copy, extrapolated |
| TMA store: ns per byte | 0.015941 | ns/byte | = 62.73 GB/s for one SM |
| TMA store: intercept | 22.508 | ns | round trip of a zero-byte copy, extrapolated |
| TMA load/cold: ns per byte | 0.010336 | ns/byte | = 96.75 GB/s for one SM |
| TMA load/cold: intercept | 418.73 | ns | round trip of a zero-byte copy, extrapolated |
| cold TMA fetch rate, ctas=8 | 536.34 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 418.7 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=37 | 2517.3 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 418.7 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=allsm | 8569.5 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 418.7 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| MMA f16 m128 n256, ctas=allsm | 69.254 | ns/MMA | = 2240.9 TFLOPS for 148 SMs; batch intercept 217.1 ns |
| MMA f16 m128 n256, ctas=1 | 68.971 | ns/MMA | = 2250.1 TFLOPS for 148 SMs; batch intercept 215.3 ns |
| MMA tf32 m128 n256, ctas=allsm | 69.226 | ns/MMA | = 1120.9 TFLOPS for 148 SMs; batch intercept 219.7 ns |
| MMA tf32 m128 n256, ctas=1 | 68.97 | ns/MMA | = 1125 TFLOPS for 148 SMs; batch intercept 218.1 ns |
| MMA i8 m128 n256, ctas=allsm | 67.97 | ns/MMA | = 4566.4 TFLOPS for 148 SMs; batch intercept 210.7 ns |
| MMA i8 m128 n256, ctas=1 | 67.351 | ns/MMA | = 4608.4 TFLOPS for 148 SMs; batch intercept 210.7 ns |
| mbarrier arrive + wait, one thread | 32.7 | ns | mbar/arrive_wait/threads=1; predicted 3 ns |
| mbarrier arrive + wait, whole warp | 32.698 | ns | mbar/arrive_wait/threads=32; predicted 3 ns |
| tcgen05.wait::ld | -0.25382 | ns | tmem_access/ld/x1 minus the same loop without the wait; the model charges one clock |
| tcgen05.wait::st | 5.2179 | ns | tmem_access/st/x1 minus the same loop without the wait; the model charges one clock |

## (c) Measured against the simulator's current prediction

Prediction: `warplens run --timing` on the same program (predict.py), ns per repetition. "SMs assumed" is how many SMs the prediction took the launch to keep busy (they share HBM in the model); it comes from the launch's shape, and nothing in a run measures it directly. What a run can contradict is listed under "Problems in the results".

| configuration | measured | predicted | measured / predicted | SMs assumed |
|---|---|---|---|---|
| alu_chain/warps=1 | 7.315 | 2.5 | 2.926 | 1 |
| alu_chain/warps=2 | 7.316 | 2.5 | 2.926 | 1 |
| alu_chain/warps=4 | 7.315 | 2.5 | 2.926 | 1 |
| alu_chain/warps=5 | 7.379 | 5 | 1.476 | 1 |
| alu_chain/warps=8 | 7.379 | 5 | 1.476 | 1 |
| alu_chain/warps=16 | 8.925 | 10 | 0.8925 | 1 |
| alu_chain/warps=32 | 17.03 | 20 | 0.8516 | 1 |
| clc_latency/fail/plain/ctas=1 | 260.8 | 108.3 | 2.407 | 1 |
| clc_latency/fail/plain/ctas=2 | 463.2 | 109.7 | 4.224 | 2 |
| clc_latency/fail/multicast/ctas=2 | 451 | 109.7 | 4.113 | 2 |
| clc_latency/succeed/plain/ctas=1 | 1.43 | 120 | 0.01192 | 148 |
| clc_latency/succeed/multicast/ctas=2 | 4.294 | 121.3 | 0.03539 | 148 |
| cluster_costs/load/multicast/ctas=1 | 849.3 | 193.7 | 4.385 | 1 |
| cluster_costs/load/multicast/ctas=2 | 849.4 | 193.7 | 4.386 | 2 |
| cluster_costs/load/multicast/ctas=4 | 841.1 | 193.7 | 4.343 | 4 |
| cluster_costs/load/separate/ctas=1 | 886.5 | 196.3 | 4.515 | 1 |
| cluster_costs/load/separate/ctas=2 | 890.7 | 196.3 | 4.536 | 2 |
| cluster_costs/load/separate/ctas=4 | 886.3 | 196.3 | 4.514 | 4 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 295.9 | 176.4 | 1.678 | 2 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 478.1 | 659.6 | 0.7248 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 272.1 | 95.96 | 2.835 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 454.2 | 337.9 | 1.344 | 2 |
| cp_async/warm | 147.4 | 106.5 | 1.384 | 1 |
| cp_async/cold | 325.6 | 106.5 | 3.058 | 1 |
| global_load/load_chain/footprint=l1 | 43.99 | 24.17 | 1.82 | 1 |
| global_load/store/footprint=l1 | 23.06 | 3.667 | 6.289 | 1 |
| global_load/load_chain/footprint=l2 | 189 | 24.17 | 7.82 | 1 |
| global_load/store/footprint=l2 | 20.82 | 3.667 | 5.678 | 1 |
| global_load/load_chain/footprint=dram | 691.5 | 24.17 | 28.61 | 1 |
| global_load/store/footprint=dram | 21.03 | 3.667 | 5.735 | 1 |
| gstream/triad | 0.0001624 | 0 | n/a: the model predicts no cost | 148 |
| launch/empty/ctas=1 (launch) | 3744 | 6.667 | 561.6 | 1 |
| launch/empty/ctas=allsm (launch) | 3744 | 6.667 | 561.6 | 148 |
| mbar/arrive_wait/threads=1 | 32.7 | 3 | 10.9 | 1 |
| mbar/arrive_wait/threads=32 | 32.7 | 3 | 10.9 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 209.2 | 21.89 | 9.555 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 261.8 | 26.92 | 9.725 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 310 | 37.64 | 8.237 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 426.6 | 57.75 | 7.387 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 209.2 | 36.97 | 5.658 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 261.8 | 57.08 | 4.587 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 310 | 97.96 | 3.165 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 513.3 | 178.4 | 2.877 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 295.9 | 57.08 | 5.184 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 348.5 | 97.29 | 3.582 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 478 | 178.4 | 2.68 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 757.2 | 339.2 | 2.232 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 209.2 | 26.92 | 7.771 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 261.8 | 36.97 | 7.081 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 310 | 57.75 | 5.369 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 426.6 | 97.96 | 4.355 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 209.2 | 57.08 | 3.665 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 261.8 | 97.29 | 2.691 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 310 | 178.4 | 1.738 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 513.3 | 339.2 | 1.513 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 295.9 | 97.29 | 3.042 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 348.5 | 177.7 | 1.961 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 478.1 | 339.2 | 1.409 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 773.4 | 660.9 | 1.17 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 211.9 | 21.89 | 9.678 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 264.5 | 26.92 | 9.824 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 312.8 | 37.65 | 8.308 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 429.3 | 57.76 | 7.432 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 211.9 | 36.98 | 5.731 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 264.5 | 57.09 | 4.633 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 312.7 | 97.98 | 3.192 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 516 | 178.4 | 2.892 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 298.7 | 57.09 | 5.231 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 351.2 | 97.32 | 3.609 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 480.8 | 178.4 | 2.694 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 759.9 | 339.3 | 2.239 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 211.9 | 26.92 | 7.871 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 264.5 | 36.98 | 7.152 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 312.8 | 57.76 | 5.415 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 429.3 | 97.98 | 4.381 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 211.9 | 57.09 | 3.712 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 264.5 | 97.32 | 2.718 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 312.7 | 178.4 | 1.753 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 516 | 339.3 | 1.521 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 298.7 | 97.32 | 3.069 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 351.2 | 177.8 | 1.976 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 480.8 | 339.3 | 1.417 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 776.2 | 661.1 | 1.174 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 297.5 | 97.29 | 3.057 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.2 | 177.7 | 1.976 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 481.1 | 339.2 | 1.418 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 777.3 | 660.9 | 1.176 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 300.7 | 97.32 | 3.09 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 353.2 | 177.8 | 1.987 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 483.5 | 339.3 | 1.425 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 779.9 | 661.1 | 1.18 | 148 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 288.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 340.2 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 468.8 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 755.1 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 290.1 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 340.3 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 471.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 759.9 | n/a | no prediction |  |
| ncu_banks/linear/block=32x32 (launch) | 6880 | 851.3 | 8.081 | 64 |
| ncu_banks/padded/block=32x32 (launch) | 6272 | 190 | 33.01 | 64 |
| ncu_banks/swizzled/block=32x32 (launch) | 5536 | 211.3 | 26.2 | 64 |
| ncu_banks/linear/block=32x8 (launch) | 5024 | 104 | 48.31 | 64 |
| ncu_banks/padded/block=32x8 (launch) | 5056 | 82.67 | 61.16 | 64 |
| ncu_banks/swizzled/block=32x8 (launch) | 5536 | 88 | 62.91 | 64 |
| ncu_banks/linear/block=16x16 (launch) | 4992 | 104 | 48 | 64 |
| ncu_banks/padded/block=16x16 (launch) | 4864 | 77.33 | 62.9 | 64 |
| ncu_banks/swizzled/block=16x16 (launch) | 5504 | 72 | 76.44 | 64 |
| pipelined_gemm/stages=1/ctas=8 | 1033 | 330.8 | 3.123 | 8 |
| pipelined_gemm/stages=2/ctas=8 | 539.3 | 165.8 | 3.253 | 8 |
| pipelined_gemm/stages=4/ctas=8 | 291.2 | 128 | 2.275 | 8 |
| pipelined_gemm/stages=1/ctas=allsm | 1110 | 1081 | 1.027 | 148 |
| pipelined_gemm/stages=2/ctas=allsm | 618.4 | 878.4 | 0.7041 | 148 |
| pipelined_gemm/stages=4/ctas=allsm | 393.2 | 878.4 | 0.4477 | 148 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.28 | 3.333 | 4.284 | 1 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.3 | 4 | 3.825 | 1 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.33 | 5.333 | 3.25 | 1 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.41 | 8 | 2.676 | 1 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.55 | 13.33 | 2.216 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.83 | 24 | 1.91 | 1 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.28 | 3.333 | 4.284 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.14 | 85.33 | 0.7634 | 1 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.771 | 6 | 0.7951 | 1 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.834 | 6.667 | 0.7252 | 1 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.138 | 8 | 0.7673 | 1 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.21 | 10.67 | 0.9571 | 1 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.35 | 16 | 1.147 | 1 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.64 | 26.67 | 1.299 | 1 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.545 | 2.5 | 1.018 | 1 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.545 | 3.167 | 0.8036 | 1 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.072 | 4.5 | 0.9049 | 1 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.142 | 7.167 | 1.136 | 1 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.29 | 12.5 | 1.303 | 1 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.57 | 23.17 | 1.406 | 1 |
| tma/load/warm/bytes=64/ctas=1 | 153.4 | 104.8 | 1.463 | 1 |
| tma/load/warm/bytes=256/ctas=1 | 152.2 | 105.8 | 1.438 | 1 |
| tma/load/warm/bytes=1024/ctas=1 | 162 | 109.8 | 1.475 | 1 |
| tma/load/warm/bytes=4096/ctas=1 | 198.8 | 125.8 | 1.58 | 1 |
| tma/load/warm/bytes=16384/ctas=1 | 234.9 | 189.8 | 1.237 | 1 |
| tma/load/warm/bytes=32768/ctas=1 | 302.6 | 275.2 | 1.1 | 1 |
| tma/store/bytes=64/ctas=1 | 23.58 | 103.3 | 0.2282 | 1 |
| tma/store/bytes=256/ctas=1 | 26.53 | 104.3 | 0.2542 | 1 |
| tma/store/bytes=1024/ctas=1 | 38.82 | 108.3 | 0.3583 | 1 |
| tma/store/bytes=4096/ctas=1 | 87.6 | 124.3 | 0.7045 | 1 |
| tma/store/bytes=16384/ctas=1 | 284.1 | 188.3 | 1.508 | 1 |
| tma/store/bytes=32768/ctas=1 | 544.7 | 273.7 | 1.99 | 1 |
| tma/load/cold/bytes=64/ctas=1 | 301.7 | 104.8 | 2.878 | 1 |
| tma/load/cold/bytes=256/ctas=1 | 436.2 | 105.8 | 4.122 | 1 |
| tma/load/cold/bytes=1024/ctas=1 | 461.2 | 109.8 | 4.199 | 1 |
| tma/load/cold/bytes=4096/ctas=1 | 503.2 | 125.8 | 3.999 | 1 |
| tma/load/cold/bytes=16384/ctas=1 | 657.9 | 189.8 | 3.465 | 1 |
| tma/load/cold/bytes=32768/ctas=1 | 716.4 | 275.2 | 2.603 | 1 |
| tma/load/cold/bytes=16384/ctas=8 | 663.1 | 189.7 | 3.496 | 8 |
| tma/load/cold/bytes=16384/ctas=37 | 659.5 | 250.7 | 2.631 | 37 |
| tma/load/cold/bytes=16384/ctas=allsm | 701.7 | 689.9 | 1.017 | 148 |
| tmem_access/ld/x1 | 14.79 | 5.333 | 2.773 | 1 |
| tmem_access/ld/x8 | 25.7 | 10 | 2.57 | 1 |
| tmem_access/st/x1 | 6.14 | 4.667 | 1.316 | 1 |
| tmem_access/st/x8 | 8.079 | 4.667 | 1.731 | 1 |
| tmem_access/ld/x1/nowait | 15.04 | 4.5 | 3.343 | 1 |
| tmem_access/st/x1/nowait | 0.9217 | 3.833 | 0.2404 | 1 |

`clc_latency/succeed/*`: on a GPU many clusters start at once, so these two slopes are not round trips and their ratios mean nothing; the kernel's own count is under "Other measured quantities" (CLC request, succeed/...).
