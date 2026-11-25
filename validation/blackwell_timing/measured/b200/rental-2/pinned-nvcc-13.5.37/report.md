# Timing calibration report

Device: NVIDIA_B200, compute capability 10.0, 148 SMs, 191490555904 bytes.

## (a) Measured

ns per repetition is the slope of launch time against repetitions. `min` uses the fastest launch at each repetition count; `median` is the median of the per-trial slopes; spread is (max - min) / median of the per-trial slopes.

| configuration | ns/rep min | ns/rep median | spread % | trials | clock64 ticks/rep |
|---|---|---|---|---|---|
| alu_chain/warps=1 | 7.333 | 7.333 | 0.0579 | 5 | 14.38 |
| alu_chain/warps=2 | 7.329 | 7.331 | 0.116 | 5 | 14.38 |
| alu_chain/warps=4 | 7.332 | 7.331 | 0.0537 | 5 | 14.38 |
| alu_chain/warps=5 | 7.395 | 7.394 | 0.058 | 5 | 14.5 |
| alu_chain/warps=8 | 7.396 | 7.396 | 0.0639 | 5 | 14.5 |
| alu_chain/warps=16 | 8.945 | 8.946 | 0.033 | 5 | 17.53 |
| alu_chain/warps=32 | 17.09 | 17.09 | 0.0182 | 5 | 33.5 |
| clc_latency/fail/plain/ctas=1 | 261.3 | 261.1 | 0.181 | 5 | 512.3 |
| clc_latency/fail/plain/ctas=2 | 471.6 | 471.5 | 0.0556 | 5 | 924.3 |
| clc_latency/fail/multicast/ctas=2 | 474.4 | 474.3 | 0.0695 | 5 | 929.8 |
| clc_latency/succeed/plain/ctas=1 | 1.488 | 1.485 | 16.6 | 5 | 2.836 |
| clc_latency/succeed/multicast/ctas=2 | 4.31 | 4.25 | 8.21 | 5 | 8.555 |
| cluster_costs/load/multicast/ctas=1 | 863.9 | 863.9 | 0.0347 | 5 | 1694 |
| cluster_costs/load/multicast/ctas=2 | 864.1 | 864.1 | 0.0522 | 5 | 1694 |
| cluster_costs/load/multicast/ctas=4 | 837.7 | 837.7 | 0.0273 | 5 | 1642 |
| cluster_costs/load/separate/ctas=1 | 901.4 | 901.5 | 0.036 | 5 | 1767 |
| cluster_costs/load/separate/ctas=2 | 905.8 | 905.9 | 0.042 | 5 | 1775 |
| cluster_costs/load/separate/ctas=4 | 885.5 | 885.6 | 0.0333 | 5 | 1736 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 296.7 | 296.8 | 0.0427 | 5 | 546 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 479.4 | 479.4 | 0.0395 | 5 | 882 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 273.5 | 273.5 | 0.0826 | 5 | 502 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 456.7 | 456.7 | 0.031 | 5 | 838 |
| cp_async/warm | 150.9 | 151 | 0.0876 | 5 | 295.9 |
| cp_async/cold | 332.8 | 163.6 | 170 | 5 | 652 |
| global_load/load_chain/footprint=l1 | 44.1 | 44.1 | 0.0332 | 5 | 86.45 |
| global_load/store/footprint=l1 | 23.11 | 23.12 | 0.0796 | 5 | 45.31 |
| global_load/load_chain/footprint=l2 | 200 | 200 | 0.0322 | 5 | 392.2 |
| global_load/store/footprint=l2 | 20.88 | 20.87 | 0.118 | 5 | 40.91 |
| global_load/load_chain/footprint=dram | 697.7 | 697.6 | 1.01 | 5 | 1367 |
| global_load/store/footprint=dram | 21.05 | 21.07 | 0.216 | 5 | 41.28 |
| gstream/triad | 0.0001618 | 0.0001621 | 0.331 | 5 | 0 |
| launch/empty/ctas=1 (whole launch) | 3808 | 4672 | | 200 | |
| launch/empty/ctas=allsm (whole launch) | 3648 | 4704 | | 200 | |
| mbar/arrive_wait/threads=1 | 32.78 | 32.78 | 0.0432 | 5 | 64.25 |
| mbar/arrive_wait/threads=32 | 32.78 | 32.78 | 0.027 | 5 | 64.25 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 209.7 | 209.7 | 0.0732 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 262.4 | 262.5 | 0.0758 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 310.8 | 310.8 | 0.0521 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 427.6 | 427.5 | 0.0843 | 5 | 787 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 209.7 | 209.7 | 0.0348 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 262.3 | 262.3 | 0.0403 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 310.7 | 310.7 | 0.0214 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 514.3 | 514.5 | 0.0804 | 5 | 947 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 296.7 | 296.7 | 0.0684 | 5 | 546 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 349.5 | 349.4 | 0.0412 | 5 | 643 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 479.2 | 479.2 | 0.0414 | 5 | 882 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 759.2 | 759 | 0.0571 | 5 | 1397 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 209.7 | 209.7 | 0.0522 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 262.4 | 262.5 | 0.0429 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 310.8 | 310.8 | 0.0424 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 427.7 | 427.6 | 0.0395 | 5 | 787 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 209.7 | 209.7 | 0.0891 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 262.4 | 262.4 | 0.0493 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 310.8 | 310.8 | 0.0243 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 514.5 | 514.6 | 0.0532 | 5 | 947 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 296.7 | 296.7 | 0.0423 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 349.3 | 349.3 | 0.0396 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 479.1 | 479.2 | 0.0648 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 775.1 | 775.2 | 0.0647 | 5 | 1427 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 212.4 | 212.4 | 0.0642 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 265 | 265.1 | 0.0546 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 313.4 | 313.5 | 0.0619 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 430.2 | 430.2 | 0.0297 | 5 | 792 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 212.4 | 212.4 | 0.0443 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 265 | 265 | 0.0413 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 313.3 | 313.4 | 0.0829 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 517 | 517.1 | 0.0512 | 5 | 952 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 299.3 | 299.3 | 0.0291 | 5 | 551 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 352 | 352.1 | 0.0413 | 5 | 648 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 481.7 | 482 | 0.0955 | 5 | 887 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 761.8 | 761.8 | 0.0243 | 5 | 1402 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 212.4 | 212.5 | 0.0762 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 265.1 | 265.1 | 0.0961 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 313.4 | 313.4 | 0.0494 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 430.4 | 430.3 | 0.0265 | 5 | 792 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 212.4 | 212.4 | 0.0331 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 265.1 | 265.2 | 0.125 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 313.5 | 313.6 | 0.0339 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 517.3 | 517.3 | 0.0427 | 5 | 952 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 299.3 | 299.4 | 0.0366 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 352.1 | 352.1 | 0.0433 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 482 | 482 | 0.0477 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 778.2 | 778.2 | 0.0708 | 5 | 1432 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 298.7 | 298.8 | 0.0726 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.6 | 352 | 0.163 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 481.7 | 482 | 0.0984 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 778 | 778.1 | 0.109 | 5 | 1427 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 301.4 | 301.5 | 0.222 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 354.5 | 354.5 | 0.146 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 483.7 | 483.8 | 0.114 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 780.3 | 780.5 | 0.194 | 5 | 1432 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 289.7 | 289.7 | 0.0597 | 5 | 546 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 341.1 | 341.1 | 0.107 | 5 | 643 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 470 | 470 | 0.053 | 5 | 886 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 757 | 757 | 0.0195 | 5 | 1427 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 291.2 | 291.4 | 0.18 | 5 | 546 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 342.6 | 342.5 | 0.108 | 5 | 643 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 472.1 | 472.1 | 0.0226 | 5 | 886 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 759.4 | 759.4 | 0.0298 | 5 | 1427 |
| ncu_banks/linear/block=32x32 (whole launch) | 7104 | 8544 | | 5 | |
| ncu_banks/padded/block=32x32 (whole launch) | 4864 | 5472 | | 5 | |
| ncu_banks/swizzled/block=32x32 (whole launch) | 4960 | 5344 | | 5 | |
| ncu_banks/linear/block=32x8 (whole launch) | 5120 | 5376 | | 5 | |
| ncu_banks/padded/block=32x8 (whole launch) | 4608 | 5344 | | 5 | |
| ncu_banks/swizzled/block=32x8 (whole launch) | 6784 | 7616 | | 5 | |
| ncu_banks/linear/block=16x16 (whole launch) | 4704 | 6464 | | 5 | |
| ncu_banks/padded/block=16x16 (whole launch) | 5312 | 5440 | | 5 | |
| ncu_banks/swizzled/block=16x16 (whole launch) | 4864 | 6528 | | 5 | |
| pipelined_gemm/stages=1/ctas=8 | 1026 | 893.1 | 18.4 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=8 | 547.6 | 454.2 | 23.3 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=8 | 291.8 | 291.8 | 1.78 | 5 | 0 |
| pipelined_gemm/stages=1/ctas=allsm | 1103 | 1107 | 1.07 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=allsm | 615.4 | 616.3 | 1.06 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=allsm | 399.1 | 396.9 | 1.76 | 5 | 0 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.31 | 14.31 | 0.0805 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.34 | 15.34 | 0.0552 | 5 | 30.06 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.38 | 17.38 | 0.0543 | 5 | 34.06 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.44 | 21.45 | 0.103 | 5 | 42.06 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.62 | 29.62 | 0.0246 | 5 | 58.06 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.94 | 45.95 | 0.0655 | 5 | 90.06 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.32 | 14.32 | 0.0193 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.28 | 65.29 | 0.0362 | 5 | 128 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.782 | 4.782 | 0.0937 | 5 | 9.375 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.845 | 4.845 | 0.122 | 5 | 9.5 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.152 | 6.152 | 0.0411 | 5 | 12.06 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.23 | 10.23 | 0.0432 | 5 | 20.06 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.39 | 18.4 | 0.0506 | 5 | 36.06 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.71 | 34.72 | 0.0425 | 5 | 68.06 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.55 | 2.551 | 0.1 | 5 | 5 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.55 | 2.55 | 0.0652 | 5 | 5 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.08 | 4.081 | 0.0639 | 5 | 8 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.16 | 8.161 | 0.0274 | 5 | 16 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.32 | 16.32 | 0.054 | 5 | 32 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.64 | 32.65 | 0.0498 | 5 | 64 |
| tma/load/warm/bytes=64/ctas=1 | 156.4 | 156.5 | 0.0746 | 5 | 306.5 |
| tma/load/warm/bytes=256/ctas=1 | 155.3 | 155.6 | 0.686 | 5 | 304.3 |
| tma/load/warm/bytes=1024/ctas=1 | 165.7 | 165.4 | 0.434 | 5 | 325 |
| tma/load/warm/bytes=4096/ctas=1 | 210.8 | 210.4 | 0.304 | 5 | 413.3 |
| tma/load/warm/bytes=16384/ctas=1 | 244.6 | 244.6 | 0.116 | 5 | 479.2 |
| tma/load/warm/bytes=32768/ctas=1 | 311.1 | 311.4 | 0.966 | 5 | 609.9 |
| tma/store/bytes=64/ctas=1 | 23.65 | 23.64 | 0.21 | 5 | 46.25 |
| tma/store/bytes=256/ctas=1 | 26.63 | 26.66 | 0.69 | 5 | 52.25 |
| tma/store/bytes=1024/ctas=1 | 38.93 | 38.92 | 0.288 | 5 | 76.25 |
| tma/store/bytes=4096/ctas=1 | 87.82 | 87.81 | 0.248 | 5 | 172.2 |
| tma/store/bytes=16384/ctas=1 | 284.7 | 284.6 | 0.103 | 5 | 558.2 |
| tma/store/bytes=32768/ctas=1 | 545.8 | 545.9 | 0.0736 | 5 | 1070 |
| tma/load/cold/bytes=64/ctas=1 | 305.9 | 306.9 | 2.29 | 5 | 597.2 |
| tma/load/cold/bytes=256/ctas=1 | 444.7 | 445.6 | 1.41 | 5 | 873.6 |
| tma/load/cold/bytes=1024/ctas=1 | 470.7 | 471.5 | 0.57 | 5 | 923.8 |
| tma/load/cold/bytes=4096/ctas=1 | 517.7 | 526 | 3.65 | 5 | 1016 |
| tma/load/cold/bytes=16384/ctas=1 | 666.8 | 668.7 | 0.801 | 5 | 1307 |
| tma/load/cold/bytes=32768/ctas=1 | 722.7 | 722.7 | 0.534 | 5 | 1416 |
| tma/load/cold/bytes=16384/ctas=8 | 665.1 | 666.2 | 0.375 | 5 | 1303 |
| tma/load/cold/bytes=16384/ctas=37 | 660.4 | 662.9 | 0.702 | 5 | 1296 |
| tma/load/cold/bytes=16384/ctas=allsm | 698.8 | 699.8 | 0.607 | 5 | 1365 |
| tmem_access/ld/x1 | 14.82 | 14.82 | 0.0438 | 5 | 29.06 |
| tmem_access/ld/x8 | 25.76 | 25.76 | 0.0451 | 5 | 50.5 |
| tmem_access/st/x1 | 6.153 | 6.152 | 0.0765 | 5 | 12.06 |
| tmem_access/st/x8 | 8.096 | 8.098 | 0.0427 | 5 | 15.88 |
| tmem_access/ld/x1/nowait | 15.08 | 15.08 | 0.0685 | 5 | 29.56 |
| tmem_access/st/x1/nowait | 0.9238 | 0.9245 | 0.336 | 5 | 1.812 |

## (b) Derived model parameters

| parameter | derived | config now | derived / config | how |
|---|---|---|---|---|
| `sm_count` | 148 SMs | 148 | 1 | cudaGetDeviceProperties().multiProcessorCount.  |
| `hbm_capacity_gb` | 178.3 GB | 192 | 0.9286 | cudaMemGetInfo() total bytes / 2^30. The driver reports usable memory, which is below the nominal size. |
| `clock_ghz` | 0.51142 GHz | 1.5 | 0.3409 | clock_ghz(config) 1.5 * predicted 2.5 / measured 7.333 ns per repetition of alu_chain/warps=1 (3.75 PTX instructions per repetition). Effective issue clock for one warp of register-only instructions, not the core clock. |
| `smem_wavefront_clk` | 0.52186 clk | 1 | 0.5219 | slope 1.02 ns per extra wavefront (load_chain, conflict 1..32) * clock_ghz. Clocks converted with the fitted clock_ghz = 0.5114. |
| `global_access_latency_clk` | 96.052 clk | 30 | 3.202 | config 30 + measured 200 ns * clock_ghz - predicted 24.17 ns * 1.5 (global_load/load_chain/footprint=l2, one load per repetition). Clocks converted with the fitted clock_ghz = 0.5114. |
| `tmem_access_clk` | 3.5797 clk | 4 | 0.8949 | config 4 + measured 14.82 ns * clock_ghz - predicted 5.333 ns * 1.5 (tmem_access/ld/x1, one tcgen05.ld per repetition). Clocks converted with the fitted clock_ghz = 0.5114. |
| `clc_latency_ns` | 236.85 ns | 100 | 2.368 | measured 261.3 ns per request - 24.44 ns of instruction time the simulator counts for the same loop (clc_latency/fail/plain/ctas=1, requests that fail in a launch of one cluster). One latency for every form of the request, as the model has. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5114. NOT ONE NUMBER on this device: clc_latency/fail/plain/ctas=2 implies 443.3 ns, clc_latency/fail/multicast/ctas=2 implies 446.1 ns, clc_latency/succeed/plain/ctas=1 implies 304.2 ns, clc_latency/succeed/multicast/ctas=2 implies 504.9 ns; the model would need a latency per form. |
| `tma_bytes_per_clk` | 425.58 bytes/clk | 128 | 3.325 | 217.6 bytes/ns (1 / slope of round trip against bytes, load/warm) / clock_ghz. The faster of warm loads and stores is taken as the engine rate: the model charges max(engine, memory) per copy, so each slope is the slower of the two. Clocks converted with the fitted clock_ghz = 0.5114. |
| `l2_bandwidth_per_sm_gb_s` | 217.65 GB/s | 400 | 0.5441 | 1 / slope of round trip against bytes, warm loads, in bytes/ns. LOWER BOUND: warm loads run at the engine rate, so L2 is at least this fast; the model reproduces the measurement with any value from here up. |
| `tma_latency_ns` | 152.31 ns | 100 | 1.523 | intercept 165.5 ns of warm-load round trip against bytes - 13.2 ns of instruction time the simulator counts for the same loop. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5114. |
| `hbm_bandwidth_tb_s` | 6.1795 TB/s | 4.141 | 1.492 | STREAM Triad: 1 / (0.0001618 ns per byte) / 1000 (gstream/triad, 24 bytes per element). Same definition as the config's published figure. Cold TMA loads on every SM gave 8.951 TB/s. |
| `tensor_core_tflops_f16` | 2241.6 TFLOPS | 1929.6 | 1.162 | 2 * 128 * 256 * 16 FLOP / 69.23 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |
| `mma_latency_clk` | 96.51 clk | 11.3 | 8.541 | config 11.3 + measured batch intercept 215.9 ns * clock_ghz - predicted 16.81 ns * 1.5 (f16 m128 n256). Includes whatever tcgen05.commit and the mbarrier wake-up cost on hardware beyond their instructions. Clocks converted with the fitted clock_ghz = 0.5114. |
| `tensor_core_tflops_tf32` | 1122.1 TFLOPS | 964.5 | 1.163 | 2 * 128 * 256 * 8 FLOP / 69.15 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |

Other measured quantities:

| quantity | value | unit | note |
|---|---|---|---|
| %clock64 ticks per ns | 1.9604 | ticks/ns | alu_chain/warps=1: clock64 slope / event-time slope. 0 means the build did not read the clock |
| shared load, no conflict | 14.311 | ns | smem_wavefronts/load_chain fit at conflict=1 |
| CLC request, fail/plain/ctas=1 | 261.29 | ns | slope of launch time against requests; minus 24.44 ns of instructions the simulator counts for the same loop = 236.8 ns of latency |
| CLC request, fail/plain/ctas=2 | 471.61 | ns | slope of launch time against requests; minus 28.35 ns of instructions the simulator counts for the same loop = 443.3 ns of latency |
| CLC request, fail/multicast/ctas=2 | 474.41 | ns | slope of launch time against requests; minus 28.35 ns of instructions the simulator counts for the same loop = 446.1 ns of latency |
| CLC request, succeed/plain/ctas=1 | 362.88 | ns | 711.4 ticks per request of the first cluster (clk / (check + 1)) / 1.96 ticks per ns; minus 58.66 ns of instructions the simulator counts for the same loop = 304.2 ns of latency |
| CLC request, succeed/multicast/ctas=2 | 567.42 | ns | 1112 ticks per request of the first cluster (clk / (check + 1)) / 1.96 ticks per ns; minus 62.57 ns of instructions the simulator counts for the same loop = 504.9 ns of latency |
| TMA load/warm: ns per byte | 0.0045945 | ns/byte | = 217.6 GB/s for one SM |
| TMA load/warm: intercept | 165.51 | ns | round trip of a zero-byte copy, extrapolated |
| TMA store: ns per byte | 0.015972 | ns/byte | = 62.61 GB/s for one SM |
| TMA store: intercept | 22.593 | ns | round trip of a zero-byte copy, extrapolated |
| TMA load/cold: ns per byte | 0.010282 | ns/byte | = 97.25 GB/s for one SM |
| TMA load/cold: intercept | 427.85 | ns | round trip of a zero-byte copy, extrapolated |
| cold TMA fetch rate, ctas=8 | 552.44 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 427.9 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=37 | 2607.1 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 427.9 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=allsm | 8951.1 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 427.9 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| MMA f16 m128 n256, ctas=allsm | 69.232 | ns/MMA | = 2241.6 TFLOPS for 148 SMs; batch intercept 217.9 ns |
| MMA f16 m128 n256, ctas=1 | 69.103 | ns/MMA | = 2245.8 TFLOPS for 148 SMs; batch intercept 215.9 ns |
| MMA tf32 m128 n256, ctas=allsm | 69.151 | ns/MMA | = 1122.1 TFLOPS for 148 SMs; batch intercept 220.7 ns |
| MMA tf32 m128 n256, ctas=1 | 69.162 | ns/MMA | = 1121.9 TFLOPS for 148 SMs; batch intercept 218.5 ns |
| MMA i8 m128 n256, ctas=allsm | 67.668 | ns/MMA | = 4586.8 TFLOPS for 148 SMs; batch intercept 212.6 ns |
| MMA i8 m128 n256, ctas=1 | 67.517 | ns/MMA | = 4597 TFLOPS for 148 SMs; batch intercept 211.3 ns |
| mbarrier arrive + wait, one thread | 32.776 | ns | mbar/arrive_wait/threads=1; predicted 3 ns |
| mbarrier arrive + wait, whole warp | 32.779 | ns | mbar/arrive_wait/threads=32; predicted 3 ns |
| tcgen05.wait::ld | -0.2615 | ns | tmem_access/ld/x1 minus the same loop without the wait; the model charges one clock |
| tcgen05.wait::st | 5.2296 | ns | tmem_access/st/x1 minus the same loop without the wait; the model charges one clock |

## (c) Measured against the simulator's current prediction

Prediction: `warplens run --timing` on the same program (predict.py), ns per repetition. "SMs assumed" is how many SMs the prediction took the launch to keep busy (they share HBM in the model); it comes from the launch's shape, and nothing in a run measures it directly. What a run can contradict is listed under "Problems in the results".

| configuration | measured | predicted | measured / predicted | SMs assumed |
|---|---|---|---|---|
| alu_chain/warps=1 | 7.333 | 2.5 | 2.933 | 1 |
| alu_chain/warps=2 | 7.329 | 2.5 | 2.931 | 1 |
| alu_chain/warps=4 | 7.332 | 2.5 | 2.933 | 1 |
| alu_chain/warps=5 | 7.395 | 5 | 1.479 | 1 |
| alu_chain/warps=8 | 7.396 | 5 | 1.479 | 1 |
| alu_chain/warps=16 | 8.945 | 10 | 0.8945 | 1 |
| alu_chain/warps=32 | 17.09 | 20 | 0.8545 | 1 |
| clc_latency/fail/plain/ctas=1 | 261.3 | 108.3 | 2.412 | 1 |
| clc_latency/fail/plain/ctas=2 | 471.6 | 109.7 | 4.3 | 2 |
| clc_latency/fail/multicast/ctas=2 | 474.4 | 109.7 | 4.326 | 2 |
| clc_latency/succeed/plain/ctas=1 | 1.488 | 120 | 0.0124 | 148 |
| clc_latency/succeed/multicast/ctas=2 | 4.31 | 121.3 | 0.03553 | 148 |
| cluster_costs/load/multicast/ctas=1 | 863.9 | 193.7 | 4.461 | 1 |
| cluster_costs/load/multicast/ctas=2 | 864.1 | 193.7 | 4.462 | 2 |
| cluster_costs/load/multicast/ctas=4 | 837.7 | 193.7 | 4.326 | 4 |
| cluster_costs/load/separate/ctas=1 | 901.4 | 196.3 | 4.591 | 1 |
| cluster_costs/load/separate/ctas=2 | 905.8 | 196.3 | 4.614 | 2 |
| cluster_costs/load/separate/ctas=4 | 885.5 | 196.3 | 4.51 | 4 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 296.7 | 176.4 | 1.682 | 2 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 479.4 | 659.6 | 0.7268 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 273.5 | 95.96 | 2.85 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 456.7 | 337.9 | 1.352 | 2 |
| cp_async/warm | 150.9 | 106.5 | 1.417 | 1 |
| cp_async/cold | 332.8 | 106.5 | 3.124 | 1 |
| global_load/load_chain/footprint=l1 | 44.1 | 24.17 | 1.825 | 1 |
| global_load/store/footprint=l1 | 23.11 | 3.667 | 6.302 | 1 |
| global_load/load_chain/footprint=l2 | 200 | 24.17 | 8.277 | 1 |
| global_load/store/footprint=l2 | 20.88 | 3.667 | 5.694 | 1 |
| global_load/load_chain/footprint=dram | 697.7 | 24.17 | 28.87 | 1 |
| global_load/store/footprint=dram | 21.05 | 3.667 | 5.742 | 1 |
| gstream/triad | 0.0001618 | 0 | n/a: the model predicts no cost | 148 |
| launch/empty/ctas=1 (launch) | 3808 | 6.667 | 571.2 | 1 |
| launch/empty/ctas=allsm (launch) | 3648 | 6.667 | 547.2 | 148 |
| mbar/arrive_wait/threads=1 | 32.78 | 3 | 10.93 | 1 |
| mbar/arrive_wait/threads=32 | 32.78 | 3 | 10.93 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 209.7 | 21.89 | 9.578 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 262.4 | 26.92 | 9.746 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 310.8 | 37.64 | 8.257 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 427.6 | 57.75 | 7.404 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 209.7 | 36.97 | 5.671 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 262.3 | 57.08 | 4.596 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 310.7 | 97.96 | 3.172 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 514.3 | 178.4 | 2.883 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 296.7 | 57.08 | 5.198 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 349.5 | 97.29 | 3.592 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 479.2 | 178.4 | 2.686 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 759.2 | 339.2 | 2.238 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 209.7 | 26.92 | 7.79 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 262.4 | 36.97 | 7.098 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 310.8 | 57.75 | 5.382 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 427.7 | 97.96 | 4.366 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 209.7 | 57.08 | 3.673 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 262.4 | 97.29 | 2.697 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 310.8 | 178.4 | 1.742 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 514.5 | 339.2 | 1.517 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 296.7 | 97.29 | 3.049 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 349.3 | 177.7 | 1.966 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 479.1 | 339.2 | 1.412 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 775.1 | 660.9 | 1.173 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 212.4 | 21.89 | 9.7 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 265 | 26.92 | 9.843 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 313.4 | 37.65 | 8.326 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 430.2 | 57.76 | 7.448 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 212.4 | 36.98 | 5.743 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 265 | 57.09 | 4.642 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 313.3 | 97.98 | 3.197 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 517 | 178.4 | 2.898 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 299.3 | 57.09 | 5.242 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 352 | 97.32 | 3.617 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 481.7 | 178.4 | 2.7 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 761.8 | 339.3 | 2.245 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 212.4 | 26.92 | 7.89 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 265.1 | 36.98 | 7.168 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 313.4 | 57.76 | 5.425 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 430.4 | 97.98 | 4.392 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 212.4 | 57.09 | 3.721 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 265.1 | 97.32 | 2.724 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 313.5 | 178.4 | 1.757 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 517.3 | 339.3 | 1.524 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 299.3 | 97.32 | 3.076 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 352.1 | 177.8 | 1.981 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 482 | 339.3 | 1.42 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 778.2 | 661.1 | 1.177 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 298.7 | 97.29 | 3.07 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.6 | 177.7 | 1.978 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 481.7 | 339.2 | 1.42 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 778 | 660.9 | 1.177 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 301.4 | 97.32 | 3.097 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 354.5 | 177.8 | 1.994 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 483.7 | 339.3 | 1.426 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 780.3 | 661.1 | 1.18 | 148 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 289.7 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 341.1 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 470 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 757 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 291.2 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 342.6 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 472.1 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 759.4 | n/a | no prediction |  |
| ncu_banks/linear/block=32x32 (launch) | 7104 | 851.3 | 8.345 | 64 |
| ncu_banks/padded/block=32x32 (launch) | 4864 | 190 | 25.6 | 64 |
| ncu_banks/swizzled/block=32x32 (launch) | 4960 | 211.3 | 23.47 | 64 |
| ncu_banks/linear/block=32x8 (launch) | 5120 | 104 | 49.23 | 64 |
| ncu_banks/padded/block=32x8 (launch) | 4608 | 82.67 | 55.74 | 64 |
| ncu_banks/swizzled/block=32x8 (launch) | 6784 | 88 | 77.09 | 64 |
| ncu_banks/linear/block=16x16 (launch) | 4704 | 104 | 45.23 | 64 |
| ncu_banks/padded/block=16x16 (launch) | 5312 | 77.33 | 68.69 | 64 |
| ncu_banks/swizzled/block=16x16 (launch) | 4864 | 72 | 67.56 | 64 |
| pipelined_gemm/stages=1/ctas=8 | 1026 | 330.8 | 3.1 | 8 |
| pipelined_gemm/stages=2/ctas=8 | 547.6 | 165.8 | 3.303 | 8 |
| pipelined_gemm/stages=4/ctas=8 | 291.8 | 128 | 2.28 | 8 |
| pipelined_gemm/stages=1/ctas=allsm | 1103 | 1081 | 1.021 | 148 |
| pipelined_gemm/stages=2/ctas=allsm | 615.4 | 878.4 | 0.7007 | 148 |
| pipelined_gemm/stages=4/ctas=allsm | 399.1 | 878.4 | 0.4544 | 148 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.31 | 3.333 | 4.292 | 1 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.34 | 4 | 3.834 | 1 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.38 | 5.333 | 3.258 | 1 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.44 | 8 | 2.68 | 1 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.62 | 13.33 | 2.222 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.94 | 24 | 1.914 | 1 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.32 | 3.333 | 4.295 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.28 | 85.33 | 0.7649 | 1 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.782 | 6 | 0.797 | 1 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.845 | 6.667 | 0.7268 | 1 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.152 | 8 | 0.769 | 1 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.23 | 10.67 | 0.9592 | 1 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.39 | 16 | 1.15 | 1 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.71 | 26.67 | 1.302 | 1 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.55 | 2.5 | 1.02 | 1 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.55 | 3.167 | 0.8053 | 1 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.08 | 4.5 | 0.9066 | 1 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.16 | 7.167 | 1.139 | 1 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.32 | 12.5 | 1.306 | 1 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.64 | 23.17 | 1.409 | 1 |
| tma/load/warm/bytes=64/ctas=1 | 156.4 | 104.8 | 1.492 | 1 |
| tma/load/warm/bytes=256/ctas=1 | 155.3 | 105.8 | 1.467 | 1 |
| tma/load/warm/bytes=1024/ctas=1 | 165.7 | 109.8 | 1.508 | 1 |
| tma/load/warm/bytes=4096/ctas=1 | 210.8 | 125.8 | 1.675 | 1 |
| tma/load/warm/bytes=16384/ctas=1 | 244.6 | 189.8 | 1.289 | 1 |
| tma/load/warm/bytes=32768/ctas=1 | 311.1 | 275.2 | 1.131 | 1 |
| tma/store/bytes=64/ctas=1 | 23.65 | 103.3 | 0.2289 | 1 |
| tma/store/bytes=256/ctas=1 | 26.63 | 104.3 | 0.2552 | 1 |
| tma/store/bytes=1024/ctas=1 | 38.93 | 108.3 | 0.3594 | 1 |
| tma/store/bytes=4096/ctas=1 | 87.82 | 124.3 | 0.7064 | 1 |
| tma/store/bytes=16384/ctas=1 | 284.7 | 188.3 | 1.512 | 1 |
| tma/store/bytes=32768/ctas=1 | 545.8 | 273.7 | 1.994 | 1 |
| tma/load/cold/bytes=64/ctas=1 | 305.9 | 104.8 | 2.918 | 1 |
| tma/load/cold/bytes=256/ctas=1 | 444.7 | 105.8 | 4.202 | 1 |
| tma/load/cold/bytes=1024/ctas=1 | 470.7 | 109.8 | 4.285 | 1 |
| tma/load/cold/bytes=4096/ctas=1 | 517.7 | 125.8 | 4.114 | 1 |
| tma/load/cold/bytes=16384/ctas=1 | 666.8 | 189.8 | 3.513 | 1 |
| tma/load/cold/bytes=32768/ctas=1 | 722.7 | 275.2 | 2.626 | 1 |
| tma/load/cold/bytes=16384/ctas=8 | 665.1 | 189.7 | 3.507 | 8 |
| tma/load/cold/bytes=16384/ctas=37 | 660.4 | 250.7 | 2.634 | 37 |
| tma/load/cold/bytes=16384/ctas=allsm | 698.8 | 689.9 | 1.013 | 148 |
| tmem_access/ld/x1 | 14.82 | 5.333 | 2.779 | 1 |
| tmem_access/ld/x8 | 25.76 | 10 | 2.576 | 1 |
| tmem_access/st/x1 | 6.153 | 4.667 | 1.319 | 1 |
| tmem_access/st/x8 | 8.096 | 4.667 | 1.735 | 1 |
| tmem_access/ld/x1/nowait | 15.08 | 4.5 | 3.352 | 1 |
| tmem_access/st/x1/nowait | 0.9238 | 3.833 | 0.241 | 1 |

`clc_latency/succeed/*`: on a GPU many clusters start at once, so these two slopes are not round trips and their ratios mean nothing; the kernel's own count is under "Other measured quantities" (CLC request, succeed/...).
