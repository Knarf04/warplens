# Timing calibration report

Device: NVIDIA_B200, compute capability 10.0, 148 SMs, 191490555904 bytes.

## (a) Measured

ns per repetition is the slope of launch time against repetitions. `min` uses the fastest launch at each repetition count; `median` is the median of the per-trial slopes; spread is (max - min) / median of the per-trial slopes.

| configuration | ns/rep min | ns/rep median | spread % | trials | clock64 ticks/rep |
|---|---|---|---|---|---|
| alu_chain/warps=1 | 7.34 | 7.34 | 0.0202 | 5 | 14.37 |
| alu_chain/warps=2 | 7.341 | 7.341 | 0.0119 | 5 | 14.38 |
| alu_chain/warps=4 | 7.341 | 7.341 | 0.0155 | 5 | 14.37 |
| alu_chain/warps=5 | 7.404 | 7.405 | 0.0172 | 5 | 14.5 |
| alu_chain/warps=8 | 7.405 | 7.405 | 0.00897 | 5 | 14.5 |
| alu_chain/warps=16 | 8.955 | 8.955 | 0.0149 | 5 | 17.53 |
| alu_chain/warps=32 | 17.11 | 17.11 | 0.00898 | 5 | 33.5 |
| clc_latency/fail/plain/ctas=1 | 258.7 | 258.7 | 0.26 | 5 | 506.7 |
| clc_latency/fail/plain/ctas=2 | 469.1 | 469.1 | 0.108 | 5 | 918.6 |
| clc_latency/fail/multicast/ctas=2 | 471.5 | 471.5 | 0.0421 | 5 | 923.2 |
| clc_latency/succeed/plain/ctas=1 | 1.526 | 1.517 | 11 | 5 | 2.828 |
| clc_latency/succeed/multicast/ctas=2 | 4.234 | 4.304 | 5.2 | 5 | 8.54 |
| cluster_costs/load/multicast/ctas=1 | 779 | 779 | 0.0179 | 5 | 1525 |
| cluster_costs/load/multicast/ctas=2 | 790.2 | 793.2 | 2.36 | 5 | 1547 |
| cluster_costs/load/multicast/ctas=4 | 760.6 | 760.7 | 0.041 | 5 | 1489 |
| cluster_costs/load/separate/ctas=1 | 898.4 | 898.3 | 0.029 | 5 | 1759 |
| cluster_costs/load/separate/ctas=2 | 902.2 | 902.3 | 0.0238 | 5 | 1767 |
| cluster_costs/load/separate/ctas=4 | 885.4 | 885.4 | 0.0275 | 5 | 1734 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 297 | 297 | 0.0121 | 5 | 546 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 479.8 | 479.8 | 0.0139 | 5 | 882 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 273.1 | 273.1 | 0.0199 | 5 | 502 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 455.9 | 455.9 | 0.0159 | 5 | 838 |
| cp_async/warm | 183.2 | 183.1 | 0.0886 | 5 | 358.6 |
| cp_async/cold | 334.5 | 165.6 | 167 | 5 | 655 |
| global_load/load_chain/footprint=l1 | 44.14 | 44.14 | 0.0203 | 5 | 86.45 |
| global_load/store/footprint=l1 | 22.86 | 22.86 | 0.018 | 5 | 44.77 |
| global_load/load_chain/footprint=l2 | 201.4 | 201.4 | 0.00598 | 5 | 394.5 |
| global_load/store/footprint=l2 | 20.89 | 20.89 | 0.0619 | 5 | 40.91 |
| global_load/load_chain/footprint=dram | 700.5 | 699.8 | 0.995 | 5 | 1372 |
| global_load/store/footprint=dram | 21.11 | 21.11 | 0.224 | 5 | 41.33 |
| gstream/triad | 0.0001618 | 0.0001616 | 0.565 | 5 | 0 |
| launch/empty/ctas=1 (whole launch) | 3744 | 4608 | | 200 | |
| launch/empty/ctas=allsm (whole launch) | 3648 | 4608 | | 200 | |
| mbar/arrive_wait/threads=1 | 32.81 | 32.81 | 0.0126 | 5 | 64.25 |
| mbar/arrive_wait/threads=32 | 32.81 | 32.82 | 0.0185 | 5 | 64.25 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 210 | 209.9 | 0.057 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 262.8 | 262.8 | 0.057 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 311.1 | 311.2 | 0.0154 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 428.1 | 428.2 | 0.0283 | 5 | 787 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 210 | 210 | 0.0247 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 262.8 | 262.8 | 0.0107 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 311.2 | 311.2 | 0.0272 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 515.2 | 515.2 | 0.0294 | 5 | 947 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 297.1 | 297.1 | 0.0368 | 5 | 546 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 349.8 | 349.8 | 0.0154 | 5 | 643 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 479.8 | 479.9 | 0.0215 | 5 | 882 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 760 | 760 | 0.0342 | 5 | 1397 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 210 | 210 | 0.0381 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 262.8 | 262.8 | 0.0317 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 311.2 | 311.2 | 0.0405 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 428.2 | 428.2 | 0.0317 | 5 | 787 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 210 | 210 | 0.0341 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 262.8 | 262.8 | 0.00927 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 311.2 | 311.2 | 0.0175 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 515.2 | 515.2 | 0.0127 | 5 | 947 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 297.1 | 297.1 | 0.00582 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 349.8 | 349.8 | 0.0218 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 479.8 | 479.8 | 0.0227 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 776.3 | 776.4 | 0.0249 | 5 | 1427 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 212.7 | 212.7 | 0.0403 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 265.5 | 265.5 | 0.0338 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 313.9 | 313.9 | 0.0214 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 430.9 | 430.9 | 0.0141 | 5 | 792 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 212.7 | 212.7 | 0.0954 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 265.5 | 265.5 | 0.0345 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 313.9 | 313.9 | 0.0691 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 517.9 | 517.9 | 0.0213 | 5 | 952 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 299.8 | 299.8 | 0.0365 | 5 | 551 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 352.5 | 352.6 | 0.0181 | 5 | 648 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 482.5 | 482.6 | 0.0423 | 5 | 887 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 762.7 | 762.7 | 0.0125 | 5 | 1402 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 212.7 | 212.7 | 0.037 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 265.5 | 265.5 | 0.0284 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 313.9 | 313.9 | 0.0447 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 430.9 | 430.9 | 0.0261 | 5 | 792 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 212.7 | 212.8 | 0.025 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 265.5 | 265.5 | 0.0212 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 313.9 | 313.9 | 0.0237 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 517.9 | 517.9 | 0.0163 | 5 | 952 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 299.8 | 299.8 | 0.0269 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 352.5 | 352.5 | 0.0269 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 482.5 | 482.6 | 0.0166 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 779 | 779 | 0.0204 | 5 | 1432 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 298.5 | 298.5 | 0.0418 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.7 | 351.9 | 0.0564 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 482.4 | 482.7 | 0.127 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 778.4 | 778.4 | 0.0638 | 5 | 1427 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 301.1 | 301.1 | 0.0495 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 354.5 | 354.5 | 0.0289 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 485.5 | 485.5 | 0.0539 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 781 | 781.2 | 0.073 | 5 | 1432 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 293.2 | 293.2 | 0.0349 | 5 | 550 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 344.9 | 344.9 | 0.0317 | 5 | 647 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 473.4 | 473.4 | 0.022 | 5 | 888 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 761.9 | 761.9 | 0.0223 | 5 | 1429 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 294.1 | 294.2 | 0.082 | 5 | 550 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 344.9 | 344.9 | 0.396 | 5 | 647 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 473.8 | 473.8 | 0.0549 | 5 | 888 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 760.7 | 761 | 0.04 | 5 | 1429 |
| ncu_banks/linear/block=32x32 (whole launch) | 6848 | 9280 | | 5 | |
| ncu_banks/padded/block=32x32 (whole launch) | 5152 | 6560 | | 5 | |
| ncu_banks/swizzled/block=32x32 (whole launch) | 4928 | 6592 | | 5 | |
| ncu_banks/linear/block=32x8 (whole launch) | 5344 | 5792 | | 5 | |
| ncu_banks/padded/block=32x8 (whole launch) | 5088 | 6176 | | 5 | |
| ncu_banks/swizzled/block=32x8 (whole launch) | 6624 | 6944 | | 5 | |
| ncu_banks/linear/block=16x16 (whole launch) | 5280 | 5600 | | 5 | |
| ncu_banks/padded/block=16x16 (whole launch) | 4864 | 5440 | | 5 | |
| ncu_banks/swizzled/block=16x16 (whole launch) | 5376 | 6464 | | 5 | |
| pipelined_gemm/stages=1/ctas=8 | 1037 | 902.4 | 16.7 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=8 | 546.6 | 451.9 | 22.2 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=8 | 294.6 | 296.5 | 1.39 | 5 | 0 |
| pipelined_gemm/stages=1/ctas=allsm | 1111 | 1112 | 1.37 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=allsm | 613.7 | 610.7 | 1.54 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=allsm | 403.6 | 400.9 | 2.87 | 5 | 0 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.33 | 14.33 | 0.0227 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.35 | 15.35 | 0.0138 | 5 | 30.06 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.4 | 17.4 | 0.0194 | 5 | 34.06 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.48 | 21.48 | 0.0179 | 5 | 42.06 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.65 | 29.65 | 0.0219 | 5 | 58.06 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.99 | 45.99 | 0.00514 | 5 | 90.06 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.33 | 14.33 | 0.0149 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.36 | 65.37 | 0.00969 | 5 | 128 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.819 | 4.82 | 0.0322 | 5 | 9.438 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.883 | 4.883 | 0.055 | 5 | 9.562 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.191 | 6.192 | 0.0351 | 5 | 12.12 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.28 | 10.28 | 0.00964 | 5 | 20.12 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.45 | 18.45 | 0.0129 | 5 | 36.12 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.79 | 34.79 | 0.0109 | 5 | 68.12 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.553 | 2.553 | 0.0457 | 5 | 5 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.553 | 2.553 | 0.0476 | 5 | 5 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.085 | 4.085 | 0.0595 | 5 | 8 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.17 | 8.171 | 0.0239 | 5 | 16 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.34 | 16.34 | 0.0111 | 5 | 32 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.68 | 32.68 | 0.0117 | 5 | 64 |
| tma/load/warm/bytes=64/ctas=1 | 188.4 | 188.3 | 0.191 | 5 | 369.2 |
| tma/load/warm/bytes=256/ctas=1 | 188.6 | 188.8 | 0.328 | 5 | 369.1 |
| tma/load/warm/bytes=1024/ctas=1 | 197.3 | 198.2 | 0.469 | 5 | 386.5 |
| tma/load/warm/bytes=4096/ctas=1 | 201.7 | 201.7 | 0.335 | 5 | 395.1 |
| tma/load/warm/bytes=16384/ctas=1 | 250.4 | 250.6 | 0.336 | 5 | 490.3 |
| tma/load/warm/bytes=32768/ctas=1 | 320.3 | 319.9 | 0.569 | 5 | 627.3 |
| tma/store/bytes=64/ctas=1 | 20.16 | 20.19 | 0.444 | 5 | 39.5 |
| tma/store/bytes=256/ctas=1 | 20.98 | 20.93 | 0.803 | 5 | 41 |
| tma/store/bytes=1024/ctas=1 | 25.1 | 25.11 | 0.121 | 5 | 49.25 |
| tma/store/bytes=4096/ctas=1 | 72.68 | 72.64 | 0.363 | 5 | 142.2 |
| tma/store/bytes=16384/ctas=1 | 269 | 269 | 0.161 | 5 | 526.8 |
| tma/store/bytes=32768/ctas=1 | 530.4 | 530.5 | 0.0326 | 5 | 1039 |
| tma/load/cold/bytes=64/ctas=1 | 308.8 | 309.6 | 1.46 | 5 | 603.6 |
| tma/load/cold/bytes=256/ctas=1 | 447.9 | 449.1 | 1.06 | 5 | 876.6 |
| tma/load/cold/bytes=1024/ctas=1 | 470.8 | 472.5 | 1.11 | 5 | 922.5 |
| tma/load/cold/bytes=4096/ctas=1 | 521.4 | 528.6 | 2.45 | 5 | 1021 |
| tma/load/cold/bytes=16384/ctas=1 | 675.7 | 676.6 | 0.464 | 5 | 1324 |
| tma/load/cold/bytes=32768/ctas=1 | 724.6 | 725 | 0.739 | 5 | 1417 |
| tma/load/cold/bytes=16384/ctas=8 | 673.7 | 673.2 | 0.265 | 5 | 1319 |
| tma/load/cold/bytes=16384/ctas=37 | 667.3 | 668.1 | 0.676 | 5 | 1306 |
| tma/load/cold/bytes=16384/ctas=allsm | 698.3 | 701.2 | 1.07 | 5 | 1371 |
| tmem_access/ld/x1 | 14.84 | 14.84 | 0.0298 | 5 | 29.06 |
| tmem_access/ld/x8 | 25.79 | 25.79 | 0.018 | 5 | 50.5 |
| tmem_access/st/x1 | 6.159 | 6.16 | 0.0675 | 5 | 12.06 |
| tmem_access/st/x8 | 8.106 | 8.106 | 0.0359 | 5 | 15.88 |
| tmem_access/ld/x1/nowait | 15.1 | 15.1 | 0.0282 | 5 | 29.56 |
| tmem_access/st/x1/nowait | 0.926 | 0.9258 | 0.187 | 5 | 1.812 |

## (b) Derived model parameters

| parameter | derived | config now | derived / config | how |
|---|---|---|---|---|
| `sm_count` | 148 SMs | 148 | 1 | cudaGetDeviceProperties().multiProcessorCount.  |
| `hbm_capacity_gb` | 178.3 GB | 192 | 0.9286 | cudaMemGetInfo() total bytes / 2^30. The driver reports usable memory, which is below the nominal size. |
| `clock_ghz` | 0.51089 GHz | 1.5 | 0.3406 | clock_ghz(config) 1.5 * predicted 2.5 / measured 7.34 ns per repetition of alu_chain/warps=1 (3.75 PTX instructions per repetition). Effective issue clock for one warp of register-only instructions, not the core clock. |
| `smem_wavefront_clk` | 0.52179 clk | 1 | 0.5218 | slope 1.021 ns per extra wavefront (load_chain, conflict 1..32) * clock_ghz. Clocks converted with the fitted clock_ghz = 0.5109. |
| `global_access_latency_clk` | 96.663 clk | 30 | 3.222 | config 30 + measured 201.4 ns * clock_ghz - predicted 24.17 ns * 1.5 (global_load/load_chain/footprint=l2, one load per repetition). Clocks converted with the fitted clock_ghz = 0.5109. |
| `tmem_access_clk` | 3.5833 clk | 4 | 0.8958 | config 4 + measured 14.84 ns * clock_ghz - predicted 5.333 ns * 1.5 (tmem_access/ld/x1, one tcgen05.ld per repetition). Clocks converted with the fitted clock_ghz = 0.5109. |
| `clc_latency_ns` | 234.23 ns | 100 | 2.342 | measured 258.7 ns per request - 24.47 ns of instruction time the simulator counts for the same loop (clc_latency/fail/plain/ctas=1, requests that fail in a launch of one cluster). One latency for every form of the request, as the model has. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5109. NOT ONE NUMBER on this device: clc_latency/fail/plain/ctas=2 implies 440.8 ns, clc_latency/fail/multicast/ctas=2 implies 443.1 ns, clc_latency/succeed/plain/ctas=1 implies 303.5 ns, clc_latency/succeed/multicast/ctas=2 implies 504.3 ns; the model would need a latency per form. |
| `tma_bytes_per_clk` | 492.09 bytes/clk | 128 | 3.844 | 251.4 bytes/ns (1 / slope of round trip against bytes, load/warm) / clock_ghz. The faster of warm loads and stores is taken as the engine rate: the model charges max(engine, memory) per copy, so each slope is the slower of the two. Clocks converted with the fitted clock_ghz = 0.5109. |
| `l2_bandwidth_per_sm_gb_s` | 251.41 GB/s | 400 | 0.6285 | 1 / slope of round trip against bytes, warm loads, in bytes/ns. LOWER BOUND: warm loads run at the engine rate, so L2 is at least this fast; the model reproduces the measurement with any value from here up. |
| `tma_latency_ns` | 175.05 ns | 100 | 1.751 | intercept 188.3 ns of warm-load round trip against bytes - 13.21 ns of instruction time the simulator counts for the same loop. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5109. |
| `hbm_bandwidth_tb_s` | 6.1814 TB/s | 4.141 | 1.493 | STREAM Triad: 1 / (0.0001618 ns per byte) / 1000 (gstream/triad, 24 bytes per element). Same definition as the config's published figure. Cold TMA loads on every SM gave 9.069 TB/s. |
| `tensor_core_tflops_f16` | 2239.6 TFLOPS | 1929.6 | 1.161 | 2 * 128 * 256 * 16 FLOP / 69.29 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |
| `mma_latency_clk` | 96.517 clk | 11.3 | 8.541 | config 11.3 + measured batch intercept 216.1 ns * clock_ghz - predicted 16.81 ns * 1.5 (f16 m128 n256). Includes whatever tcgen05.commit and the mbarrier wake-up cost on hardware beyond their instructions. Clocks converted with the fitted clock_ghz = 0.5109. |
| `tensor_core_tflops_tf32` | 1119.7 TFLOPS | 964.5 | 1.161 | 2 * 128 * 256 * 8 FLOP / 69.3 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |

Other measured quantities:

| quantity | value | unit | note |
|---|---|---|---|
| %clock64 ticks per ns | 1.9584 | ticks/ns | alu_chain/warps=1: clock64 slope / event-time slope. 0 means the build did not read the clock |
| shared load, no conflict | 14.332 | ns | smem_wavefronts/load_chain fit at conflict=1 |
| CLC request, fail/plain/ctas=1 | 258.69 | ns | slope of launch time against requests; minus 24.47 ns of instructions the simulator counts for the same loop = 234.2 ns of latency |
| CLC request, fail/plain/ctas=2 | 469.14 | ns | slope of launch time against requests; minus 28.38 ns of instructions the simulator counts for the same loop = 440.8 ns of latency |
| CLC request, fail/multicast/ctas=2 | 471.51 | ns | slope of launch time against requests; minus 28.38 ns of instructions the simulator counts for the same loop = 443.1 ns of latency |
| CLC request, succeed/plain/ctas=1 | 362.24 | ns | 709.4 ticks per request of the first cluster (clk / (check + 1)) / 1.958 ticks per ns; minus 58.72 ns of instructions the simulator counts for the same loop = 303.5 ns of latency |
| CLC request, succeed/multicast/ctas=2 | 566.98 | ns | 1110 ticks per request of the first cluster (clk / (check + 1)) / 1.958 ticks per ns; minus 62.64 ns of instructions the simulator counts for the same loop = 504.3 ns of latency |
| TMA load/warm: ns per byte | 0.0039776 | ns/byte | = 251.4 GB/s for one SM |
| TMA load/warm: intercept | 188.26 | ns | round trip of a zero-byte copy, extrapolated |
| TMA store: ns per byte | 0.015729 | ns/byte | = 63.58 GB/s for one SM |
| TMA store: intercept | 13.265 | ns | round trip of a zero-byte copy, extrapolated |
| TMA load/cold: ns per byte | 0.010324 | ns/byte | = 96.86 GB/s for one SM |
| TMA load/cold: intercept | 430.94 | ns | round trip of a zero-byte copy, extrapolated |
| cold TMA fetch rate, ctas=8 | 539.91 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 430.9 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=37 | 2564.9 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 430.9 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=allsm | 9069 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 430.9 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| MMA f16 m128 n256, ctas=allsm | 69.294 | ns/MMA | = 2239.6 TFLOPS for 148 SMs; batch intercept 217.9 ns |
| MMA f16 m128 n256, ctas=1 | 69.223 | ns/MMA | = 2241.9 TFLOPS for 148 SMs; batch intercept 216.1 ns |
| MMA tf32 m128 n256, ctas=allsm | 69.298 | ns/MMA | = 1119.7 TFLOPS for 148 SMs; batch intercept 220.6 ns |
| MMA tf32 m128 n256, ctas=1 | 69.221 | ns/MMA | = 1121 TFLOPS for 148 SMs; batch intercept 218.9 ns |
| MMA i8 m128 n256, ctas=allsm | 67.445 | ns/MMA | = 4601.9 TFLOPS for 148 SMs; batch intercept 215.5 ns |
| MMA i8 m128 n256, ctas=1 | 67.699 | ns/MMA | = 4584.7 TFLOPS for 148 SMs; batch intercept 214.5 ns |
| mbarrier arrive + wait, one thread | 32.813 | ns | mbar/arrive_wait/threads=1; predicted 3 ns |
| mbarrier arrive + wait, whole warp | 32.812 | ns | mbar/arrive_wait/threads=32; predicted 3 ns |
| tcgen05.wait::ld | -0.2544 | ns | tmem_access/ld/x1 minus the same loop without the wait; the model charges one clock |
| tcgen05.wait::st | 5.2333 | ns | tmem_access/st/x1 minus the same loop without the wait; the model charges one clock |

## (c) Measured against the simulator's current prediction

Prediction: `warplens run --timing` on the same program (predict.py), ns per repetition. "SMs assumed" is how many SMs the prediction took the launch to keep busy (they share HBM in the model); it comes from the launch's shape, and nothing in a run measures it directly. What a run can contradict is listed under "Problems in the results".

| configuration | measured | predicted | measured / predicted | SMs assumed |
|---|---|---|---|---|
| alu_chain/warps=1 | 7.34 | 2.5 | 2.936 | 1 |
| alu_chain/warps=2 | 7.341 | 2.5 | 2.936 | 1 |
| alu_chain/warps=4 | 7.341 | 2.5 | 2.936 | 1 |
| alu_chain/warps=5 | 7.404 | 5 | 1.481 | 1 |
| alu_chain/warps=8 | 7.405 | 5 | 1.481 | 1 |
| alu_chain/warps=16 | 8.955 | 10 | 0.8955 | 1 |
| alu_chain/warps=32 | 17.11 | 20 | 0.8554 | 1 |
| clc_latency/fail/plain/ctas=1 | 258.7 | 108.3 | 2.388 | 1 |
| clc_latency/fail/plain/ctas=2 | 469.1 | 109.7 | 4.278 | 2 |
| clc_latency/fail/multicast/ctas=2 | 471.5 | 109.7 | 4.3 | 2 |
| clc_latency/succeed/plain/ctas=1 | 1.526 | 120 | 0.01272 | 148 |
| clc_latency/succeed/multicast/ctas=2 | 4.234 | 121.3 | 0.03489 | 148 |
| cluster_costs/load/multicast/ctas=1 | 779 | 193.7 | 4.022 | 1 |
| cluster_costs/load/multicast/ctas=2 | 790.2 | 193.7 | 4.08 | 2 |
| cluster_costs/load/multicast/ctas=4 | 760.6 | 193.7 | 3.927 | 4 |
| cluster_costs/load/separate/ctas=1 | 898.4 | 196.3 | 4.576 | 1 |
| cluster_costs/load/separate/ctas=2 | 902.2 | 196.3 | 4.595 | 2 |
| cluster_costs/load/separate/ctas=4 | 885.4 | 196.3 | 4.51 | 4 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 297 | 176.4 | 1.684 | 2 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 479.8 | 659.6 | 0.7274 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 273.1 | 95.96 | 2.846 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 455.9 | 337.9 | 1.349 | 2 |
| cp_async/warm | 183.2 | 106.5 | 1.72 | 1 |
| cp_async/cold | 334.5 | 106.5 | 3.141 | 1 |
| global_load/load_chain/footprint=l1 | 44.14 | 24.17 | 1.827 | 1 |
| global_load/store/footprint=l1 | 22.86 | 3.667 | 6.235 | 1 |
| global_load/load_chain/footprint=l2 | 201.4 | 24.17 | 8.335 | 1 |
| global_load/store/footprint=l2 | 20.89 | 3.667 | 5.697 | 1 |
| global_load/load_chain/footprint=dram | 700.5 | 24.17 | 28.99 | 1 |
| global_load/store/footprint=dram | 21.11 | 3.667 | 5.758 | 1 |
| gstream/triad | 0.0001618 | 0 | n/a: the model predicts no cost | 148 |
| launch/empty/ctas=1 (launch) | 3744 | 6.667 | 561.6 | 1 |
| launch/empty/ctas=allsm (launch) | 3648 | 6.667 | 547.2 | 148 |
| mbar/arrive_wait/threads=1 | 32.81 | 3 | 10.94 | 1 |
| mbar/arrive_wait/threads=32 | 32.81 | 3 | 10.94 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 210 | 21.89 | 9.591 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 262.8 | 26.92 | 9.761 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 311.1 | 37.64 | 8.266 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 428.1 | 57.75 | 7.414 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 210 | 36.97 | 5.679 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 262.8 | 57.08 | 4.604 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 311.2 | 97.96 | 3.177 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 515.2 | 178.4 | 2.888 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 297.1 | 57.08 | 5.205 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 349.8 | 97.29 | 3.596 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 479.8 | 178.4 | 2.69 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 760 | 339.2 | 2.24 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 210 | 26.92 | 7.802 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 262.8 | 36.97 | 7.108 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 311.2 | 57.75 | 5.389 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 428.2 | 97.96 | 4.371 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 210 | 57.08 | 3.679 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 262.8 | 97.29 | 2.701 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 311.2 | 178.4 | 1.744 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 515.2 | 339.2 | 1.519 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 297.1 | 97.29 | 3.053 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 349.8 | 177.7 | 1.968 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 479.8 | 339.2 | 1.414 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 776.3 | 660.9 | 1.175 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 212.7 | 21.89 | 9.715 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 265.5 | 26.92 | 9.862 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 313.9 | 37.65 | 8.338 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 430.9 | 57.76 | 7.46 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 212.7 | 36.98 | 5.753 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 265.5 | 57.09 | 4.651 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 313.9 | 97.98 | 3.204 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 517.9 | 178.4 | 2.903 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 299.8 | 57.09 | 5.251 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 352.5 | 97.32 | 3.623 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 482.5 | 178.4 | 2.704 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 762.7 | 339.3 | 2.248 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 212.7 | 26.92 | 7.901 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 265.5 | 36.98 | 7.179 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 313.9 | 57.76 | 5.435 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 430.9 | 97.98 | 4.398 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 212.7 | 57.09 | 3.726 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 265.5 | 97.32 | 2.728 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 313.9 | 178.4 | 1.759 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 517.9 | 339.3 | 1.526 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 299.8 | 97.32 | 3.08 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 352.5 | 177.8 | 1.983 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 482.5 | 339.3 | 1.422 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 779 | 661.1 | 1.178 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 298.5 | 97.29 | 3.068 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.7 | 177.7 | 1.979 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 482.4 | 339.2 | 1.422 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 778.4 | 660.9 | 1.178 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 301.1 | 97.32 | 3.094 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 354.5 | 177.8 | 1.994 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 485.5 | 339.3 | 1.431 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 781 | 661.1 | 1.181 | 148 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 293.2 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 344.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 473.4 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 761.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 294.1 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 344.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 473.8 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 760.7 | n/a | no prediction |  |
| ncu_banks/linear/block=32x32 (launch) | 6848 | 851.3 | 8.044 | 64 |
| ncu_banks/padded/block=32x32 (launch) | 5152 | 190 | 27.12 | 64 |
| ncu_banks/swizzled/block=32x32 (launch) | 4928 | 211.3 | 23.32 | 64 |
| ncu_banks/linear/block=32x8 (launch) | 5344 | 104 | 51.38 | 64 |
| ncu_banks/padded/block=32x8 (launch) | 5088 | 82.67 | 61.55 | 64 |
| ncu_banks/swizzled/block=32x8 (launch) | 6624 | 88 | 75.27 | 64 |
| ncu_banks/linear/block=16x16 (launch) | 5280 | 104 | 50.77 | 64 |
| ncu_banks/padded/block=16x16 (launch) | 4864 | 77.33 | 62.9 | 64 |
| ncu_banks/swizzled/block=16x16 (launch) | 5376 | 72 | 74.67 | 64 |
| pipelined_gemm/stages=1/ctas=8 | 1037 | 330.8 | 3.135 | 8 |
| pipelined_gemm/stages=2/ctas=8 | 546.6 | 165.8 | 3.297 | 8 |
| pipelined_gemm/stages=4/ctas=8 | 294.6 | 128 | 2.302 | 8 |
| pipelined_gemm/stages=1/ctas=allsm | 1111 | 1081 | 1.028 | 148 |
| pipelined_gemm/stages=2/ctas=allsm | 613.7 | 878.4 | 0.6987 | 148 |
| pipelined_gemm/stages=4/ctas=allsm | 403.6 | 878.4 | 0.4595 | 148 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.33 | 3.333 | 4.299 | 1 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.35 | 4 | 3.838 | 1 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.4 | 5.333 | 3.262 | 1 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.48 | 8 | 2.685 | 1 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.65 | 13.33 | 2.224 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.99 | 24 | 1.916 | 1 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.33 | 3.333 | 4.299 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.36 | 85.33 | 0.766 | 1 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.819 | 6 | 0.8032 | 1 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.883 | 6.667 | 0.7325 | 1 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.191 | 8 | 0.7738 | 1 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.28 | 10.67 | 0.9635 | 1 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.45 | 16 | 1.153 | 1 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.79 | 26.67 | 1.304 | 1 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.553 | 2.5 | 1.021 | 1 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.553 | 3.167 | 0.8061 | 1 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.085 | 4.5 | 0.9078 | 1 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.17 | 7.167 | 1.14 | 1 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.34 | 12.5 | 1.307 | 1 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.68 | 23.17 | 1.411 | 1 |
| tma/load/warm/bytes=64/ctas=1 | 188.4 | 104.8 | 1.797 | 1 |
| tma/load/warm/bytes=256/ctas=1 | 188.6 | 105.8 | 1.782 | 1 |
| tma/load/warm/bytes=1024/ctas=1 | 197.3 | 109.8 | 1.796 | 1 |
| tma/load/warm/bytes=4096/ctas=1 | 201.7 | 125.8 | 1.603 | 1 |
| tma/load/warm/bytes=16384/ctas=1 | 250.4 | 189.8 | 1.319 | 1 |
| tma/load/warm/bytes=32768/ctas=1 | 320.3 | 275.2 | 1.164 | 1 |
| tma/store/bytes=64/ctas=1 | 20.16 | 103.3 | 0.1951 | 1 |
| tma/store/bytes=256/ctas=1 | 20.98 | 104.3 | 0.2011 | 1 |
| tma/store/bytes=1024/ctas=1 | 25.1 | 108.3 | 0.2317 | 1 |
| tma/store/bytes=4096/ctas=1 | 72.68 | 124.3 | 0.5845 | 1 |
| tma/store/bytes=16384/ctas=1 | 269 | 188.3 | 1.428 | 1 |
| tma/store/bytes=32768/ctas=1 | 530.4 | 273.7 | 1.938 | 1 |
| tma/load/cold/bytes=64/ctas=1 | 308.8 | 104.8 | 2.946 | 1 |
| tma/load/cold/bytes=256/ctas=1 | 447.9 | 105.8 | 4.232 | 1 |
| tma/load/cold/bytes=1024/ctas=1 | 470.8 | 109.8 | 4.287 | 1 |
| tma/load/cold/bytes=4096/ctas=1 | 521.4 | 125.8 | 4.144 | 1 |
| tma/load/cold/bytes=16384/ctas=1 | 675.7 | 189.8 | 3.559 | 1 |
| tma/load/cold/bytes=32768/ctas=1 | 724.6 | 275.2 | 2.633 | 1 |
| tma/load/cold/bytes=16384/ctas=8 | 673.7 | 189.7 | 3.552 | 8 |
| tma/load/cold/bytes=16384/ctas=37 | 667.3 | 250.7 | 2.661 | 37 |
| tma/load/cold/bytes=16384/ctas=allsm | 698.3 | 689.9 | 1.012 | 148 |
| tmem_access/ld/x1 | 14.84 | 5.333 | 2.783 | 1 |
| tmem_access/ld/x8 | 25.79 | 10 | 2.579 | 1 |
| tmem_access/st/x1 | 6.159 | 4.667 | 1.32 | 1 |
| tmem_access/st/x8 | 8.106 | 4.667 | 1.737 | 1 |
| tmem_access/ld/x1/nowait | 15.1 | 4.5 | 3.355 | 1 |
| tmem_access/st/x1/nowait | 0.926 | 3.833 | 0.2416 | 1 |

`clc_latency/succeed/*`: on a GPU many clusters start at once, so these two slopes are not round trips and their ratios mean nothing; the kernel's own count is under "Other measured quantities" (CLC request, succeed/...).
