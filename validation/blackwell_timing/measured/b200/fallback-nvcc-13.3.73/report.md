# Timing calibration report

Device: NVIDIA_B200, compute capability 10.0, 148 SMs, 191490555904 bytes.

## (a) Measured

ns per repetition is the slope of launch time against repetitions. `min` uses the fastest launch at each repetition count; `median` is the median of the per-trial slopes; spread is (max - min) / median of the per-trial slopes.

| configuration | ns/rep min | ns/rep median | spread % | trials | clock64 ticks/rep |
|---|---|---|---|---|---|
| alu_chain/warps=1 | 7.329 | 7.329 | 0.0234 | 5 | 14.38 |
| alu_chain/warps=2 | 7.329 | 7.329 | 0.0327 | 5 | 14.37 |
| alu_chain/warps=4 | 7.329 | 7.329 | 0.021 | 5 | 14.38 |
| alu_chain/warps=5 | 7.392 | 7.392 | 0.039 | 5 | 14.5 |
| alu_chain/warps=8 | 7.391 | 7.392 | 0.0343 | 5 | 14.5 |
| alu_chain/warps=16 | 8.942 | 8.943 | 0.0302 | 5 | 17.54 |
| alu_chain/warps=32 | 17.07 | 17.08 | 0.272 | 5 | 33.49 |
| clc_latency/fail/plain/ctas=1 | 246.4 | 246.4 | 0.23 | 5 | 482.5 |
| clc_latency/fail/plain/ctas=2 | 441.5 | 441.3 | 0.095 | 5 | 865.6 |
| clc_latency/fail/multicast/ctas=2 | 411.9 | 411.8 | 0.127 | 5 | 807.8 |
| clc_latency/succeed/plain/ctas=1 | 1.414 | 1.414 | 25.8 | 5 | 2.828 |
| clc_latency/succeed/multicast/ctas=2 | 4.317 | 4.214 | 7.97 | 5 | 8.49 |
| cluster_costs/load/multicast/ctas=1 | 745.4 | 745.3 | 0.0494 | 5 | 1462 |
| cluster_costs/load/multicast/ctas=2 | 745.2 | 745.2 | 0.0215 | 5 | 1462 |
| cluster_costs/load/multicast/ctas=4 | 757.4 | 757.4 | 0.0395 | 5 | 1485 |
| cluster_costs/load/separate/ctas=1 | 864.3 | 864.5 | 0.0231 | 5 | 1695 |
| cluster_costs/load/separate/ctas=2 | 868.9 | 868.9 | 0.0202 | 5 | 1704 |
| cluster_costs/load/separate/ctas=4 | 880.1 | 880.3 | 0.0396 | 5 | 1726 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 296.5 | 296.5 | 0.0384 | 5 | 546 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 479 | 479 | 0.0112 | 5 | 882 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 272.6 | 272.6 | 0.0291 | 5 | 502 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 455.1 | 455.2 | 0.036 | 5 | 838 |
| cp_async/warm | 162.7 | 162.8 | 0.224 | 5 | 319.2 |
| cp_async/cold | 320.5 | 150.4 | 186 | 5 | 628.7 |
| global_load/load_chain/footprint=l1 | 44.07 | 44.07 | 0.0395 | 5 | 86.45 |
| global_load/store/footprint=l1 | 22.73 | 22.74 | 0.0411 | 5 | 44.59 |
| global_load/load_chain/footprint=l2 | 181.3 | 181.3 | 0.00554 | 5 | 355.5 |
| global_load/store/footprint=l2 | 20.85 | 20.85 | 0.0388 | 5 | 40.88 |
| global_load/load_chain/footprint=dram | 686.5 | 686.3 | 1.01 | 5 | 1346 |
| global_load/store/footprint=dram | 21.05 | 21.05 | 0.198 | 5 | 41.27 |
| gstream/triad | 0.0001623 | 0.000162 | 0.601 | 5 | 0 |
| launch/empty/ctas=1 (whole launch) | 3744 | 4640 | | 200 | |
| launch/empty/ctas=allsm (whole launch) | 3712 | 4624 | | 200 | |
| mbar/arrive_wait/threads=1 | 32.76 | 32.76 | 0.0114 | 5 | 64.25 |
| mbar/arrive_wait/threads=32 | 32.75 | 32.76 | 0.0283 | 5 | 64.25 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 209.7 | 209.7 | 0.0943 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 262.3 | 262.3 | 0.0388 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 310.7 | 310.7 | 0.0245 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 427.4 | 427.4 | 0.0311 | 5 | 787 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 209.6 | 209.7 | 0.0379 | 5 | 386 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 262.3 | 262.3 | 0.0217 | 5 | 483 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 310.6 | 310.7 | 0.0251 | 5 | 572 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 514.3 | 514.3 | 0.0266 | 5 | 947 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 296.5 | 296.5 | 0.0298 | 5 | 546 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 349.2 | 349.3 | 0.0299 | 5 | 643 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 479 | 479.1 | 0.0258 | 5 | 882 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 758.7 | 758.7 | 0.0363 | 5 | 1397 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 209.6 | 209.6 | 0.0382 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 262.3 | 262.3 | 0.0146 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 310.6 | 310.7 | 0.027 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 427.4 | 427.5 | 0.0368 | 5 | 787 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 209.6 | 209.6 | 0.0113 | 5 | 386 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 262.3 | 262.3 | 0.0222 | 5 | 483 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 310.6 | 310.6 | 0.0214 | 5 | 572 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 514.3 | 514.2 | 0.0345 | 5 | 947 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 296.5 | 296.5 | 0.0352 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 349.2 | 349.2 | 0.0169 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 479 | 479 | 0.0197 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 775 | 775.1 | 0.0235 | 5 | 1427 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 212.4 | 212.4 | 0.022 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 265 | 265 | 0.0519 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 313.4 | 313.4 | 0.0192 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 430.1 | 430.1 | 0.0161 | 5 | 792 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 212.3 | 212.4 | 0.0232 | 5 | 391 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 265 | 265 | 0.0408 | 5 | 488 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 313.4 | 313.4 | 0.0368 | 5 | 577 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 517 | 517 | 0.0365 | 5 | 952 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 299.2 | 299.3 | 0.0349 | 5 | 551 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 351.9 | 351.9 | 0.0196 | 5 | 648 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 481.7 | 481.8 | 0.0223 | 5 | 887 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 761.5 | 761.5 | 0.024 | 5 | 1402 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 212.3 | 212.3 | 0.0949 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 265 | 265 | 0.0326 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 313.3 | 313.4 | 0.0333 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 430.1 | 430.1 | 0.036 | 5 | 792 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 212.4 | 212.4 | 0.0268 | 5 | 391 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 265 | 265 | 0.0304 | 5 | 488 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 313.4 | 313.4 | 0.0192 | 5 | 577 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 517.1 | 517.1 | 0.022 | 5 | 952 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 299.2 | 299.3 | 0.0393 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 351.9 | 352 | 0.0242 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 481.8 | 481.8 | 0.0207 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 777.6 | 777.7 | 0.024 | 5 | 1432 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 298.1 | 298.1 | 0.0389 | 5 | 546 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.4 | 351.2 | 0.196 | 5 | 643 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 482.2 | 482.4 | 0.0584 | 5 | 882 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 778.3 | 778.4 | 0.0284 | 5 | 1427 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 300.4 | 300.6 | 0.0945 | 5 | 551 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 354 | 354 | 0.277 | 5 | 648 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 484.7 | 484.7 | 0.0598 | 5 | 887 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 780.8 | 780.7 | 0.0293 | 5 | 1432 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 291.9 | 291.8 | 0.0322 | 5 | 550 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 343.3 | 343.3 | 0.0591 | 5 | 647 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 471.2 | 471.2 | 0.0432 | 5 | 888 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 758.2 | 758.2 | 0.0257 | 5 | 1429 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 292.8 | 292.8 | 0.103 | 5 | 550 |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 344.9 | 344.7 | 0.0674 | 5 | 647 |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 473.5 | 473.5 | 0.047 | 5 | 888 |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 760.7 | 760.8 | 0.0267 | 5 | 1429 |
| ncu_banks/linear/block=32x32 (whole launch) | 7040 | 8320 | | 5 | |
| ncu_banks/padded/block=32x32 (whole launch) | 5792 | 6816 | | 5 | |
| ncu_banks/swizzled/block=32x32 (whole launch) | 5120 | 7072 | | 5 | |
| ncu_banks/linear/block=32x8 (whole launch) | 5056 | 5312 | | 5 | |
| ncu_banks/padded/block=32x8 (whole launch) | 4800 | 6432 | | 5 | |
| ncu_banks/swizzled/block=32x8 (whole launch) | 5376 | 6592 | | 5 | |
| ncu_banks/linear/block=16x16 (whole launch) | 4800 | 5472 | | 5 | |
| ncu_banks/padded/block=16x16 (whole launch) | 5344 | 6944 | | 5 | |
| ncu_banks/swizzled/block=16x16 (whole launch) | 4928 | 6464 | | 5 | |
| pipelined_gemm/stages=1/ctas=8 | 1035 | 893.6 | 16.6 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=8 | 543.2 | 455.8 | 20.4 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=8 | 295.2 | 292.5 | 1.75 | 5 | 0 |
| pipelined_gemm/stages=1/ctas=allsm | 1109 | 1116 | 1.18 | 5 | 0 |
| pipelined_gemm/stages=2/ctas=allsm | 615.7 | 613.8 | 1.03 | 5 | 0 |
| pipelined_gemm/stages=4/ctas=allsm | 404.6 | 403.3 | 1.21 | 5 | 0 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.31 | 14.31 | 0.0396 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.32 | 15.33 | 0.0356 | 5 | 30.06 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.36 | 17.37 | 0.0406 | 5 | 34.06 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.45 | 21.45 | 0.0317 | 5 | 42.06 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.6 | 29.6 | 0.0356 | 5 | 58.06 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.91 | 45.92 | 0.0324 | 5 | 90.06 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.31 | 14.31 | 0.0164 | 5 | 28.06 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.26 | 65.26 | 0.0258 | 5 | 128 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.812 | 4.812 | 0.0416 | 5 | 9.438 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.875 | 4.876 | 0.0331 | 5 | 9.562 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.182 | 6.183 | 0.0207 | 5 | 12.12 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.26 | 10.26 | 0.0299 | 5 | 20.12 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.42 | 18.42 | 0.0121 | 5 | 36.12 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.73 | 34.73 | 0.00917 | 5 | 68.12 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.549 | 2.549 | 0.0477 | 5 | 5 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.549 | 2.55 | 0.037 | 5 | 5 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.079 | 4.078 | 0.0349 | 5 | 8 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.158 | 8.157 | 0.0335 | 5 | 16 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.31 | 16.31 | 0.0168 | 5 | 32 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.63 | 32.63 | 0.0122 | 5 | 64 |
| tma/load/warm/bytes=64/ctas=1 | 168.2 | 168.2 | 0.112 | 5 | 329.8 |
| tma/load/warm/bytes=256/ctas=1 | 168.4 | 168.2 | 0.29 | 5 | 330.3 |
| tma/load/warm/bytes=1024/ctas=1 | 175.9 | 177.9 | 1.18 | 5 | 345.2 |
| tma/load/warm/bytes=4096/ctas=1 | 185.1 | 185.4 | 0.712 | 5 | 362.9 |
| tma/load/warm/bytes=16384/ctas=1 | 230 | 230.3 | 0.42 | 5 | 451.2 |
| tma/load/warm/bytes=32768/ctas=1 | 301.2 | 301.3 | 0.551 | 5 | 590.4 |
| tma/store/bytes=64/ctas=1 | 20.13 | 20.14 | 1.11 | 5 | 39.5 |
| tma/store/bytes=256/ctas=1 | 20.88 | 20.86 | 0.399 | 5 | 41 |
| tma/store/bytes=1024/ctas=1 | 25.12 | 25.15 | 0.63 | 5 | 49.25 |
| tma/store/bytes=4096/ctas=1 | 72.57 | 72.59 | 0.0926 | 5 | 142.2 |
| tma/store/bytes=16384/ctas=1 | 268.5 | 268.5 | 0.0584 | 5 | 526.8 |
| tma/store/bytes=32768/ctas=1 | 529.6 | 529.6 | 0.0489 | 5 | 1039 |
| tma/load/cold/bytes=64/ctas=1 | 298.3 | 299.4 | 1.57 | 5 | 585.9 |
| tma/load/cold/bytes=256/ctas=1 | 436.2 | 435 | 1.71 | 5 | 854.6 |
| tma/load/cold/bytes=1024/ctas=1 | 458.8 | 458.8 | 0.507 | 5 | 897.7 |
| tma/load/cold/bytes=4096/ctas=1 | 497.7 | 522.4 | 5.52 | 5 | 978.5 |
| tma/load/cold/bytes=16384/ctas=1 | 656.6 | 655.6 | 0.906 | 5 | 1286 |
| tma/load/cold/bytes=32768/ctas=1 | 713.4 | 714.3 | 0.768 | 5 | 1400 |
| tma/load/cold/bytes=16384/ctas=8 | 657.4 | 657.8 | 0.903 | 5 | 1275 |
| tma/load/cold/bytes=16384/ctas=37 | 660.9 | 660.6 | 0.423 | 5 | 1266 |
| tma/load/cold/bytes=16384/ctas=allsm | 697.1 | 698.7 | 1.2 | 5 | 1355 |
| tmem_access/ld/x1 | 14.82 | 14.82 | 0.0374 | 5 | 29.06 |
| tmem_access/ld/x8 | 25.75 | 25.75 | 0.0317 | 5 | 50.5 |
| tmem_access/st/x1 | 6.149 | 6.151 | 0.0604 | 5 | 12.06 |
| tmem_access/st/x8 | 8.095 | 8.095 | 0.0269 | 5 | 15.88 |
| tmem_access/ld/x1/nowait | 15.07 | 15.08 | 0.0399 | 5 | 29.56 |
| tmem_access/st/x1/nowait | 0.9234 | 0.9237 | 0.201 | 5 | 1.812 |

## (b) Derived model parameters

| parameter | derived | config now | derived / config | how |
|---|---|---|---|---|
| `sm_count` | 148 SMs | 148 | 1 | cudaGetDeviceProperties().multiProcessorCount.  |
| `hbm_capacity_gb` | 178.3 GB | 192 | 0.9286 | cudaMemGetInfo() total bytes / 2^30. The driver reports usable memory, which is below the nominal size. |
| `clock_ghz` | 0.51169 GHz | 1.5 | 0.3411 | clock_ghz(config) 1.5 * predicted 2.5 / measured 7.329 ns per repetition of alu_chain/warps=1 (3.75 PTX instructions per repetition). Effective issue clock for one warp of register-only instructions, not the core clock. |
| `smem_wavefront_clk` | 0.52174 clk | 1 | 0.5217 | slope 1.02 ns per extra wavefront (load_chain, conflict 1..32) * clock_ghz. Clocks converted with the fitted clock_ghz = 0.5117. |
| `global_access_latency_clk` | 86.502 clk | 30 | 2.883 | config 30 + measured 181.3 ns * clock_ghz - predicted 24.17 ns * 1.5 (global_load/load_chain/footprint=l2, one load per repetition). Clocks converted with the fitted clock_ghz = 0.5117. |
| `tmem_access_clk` | 3.5834 clk | 4 | 0.8959 | config 4 + measured 14.82 ns * clock_ghz - predicted 5.333 ns * 1.5 (tmem_access/ld/x1, one tcgen05.ld per repetition). Clocks converted with the fitted clock_ghz = 0.5117. |
| `clc_latency_ns` | 222.01 ns | 100 | 2.22 | measured 246.4 ns per request - 24.43 ns of instruction time the simulator counts for the same loop (clc_latency/fail/plain/ctas=1, requests that fail in a launch of one cluster). One latency for every form of the request, as the model has. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5117. NOT ONE NUMBER on this device: clc_latency/fail/plain/ctas=2 implies 413.1 ns, clc_latency/fail/multicast/ctas=2 implies 383.6 ns, clc_latency/succeed/multicast/ctas=2 implies 455.1 ns; the model would need a latency per form. |
| `tma_bytes_per_clk` | 489.2 bytes/clk | 128 | 3.822 | 250.3 bytes/ns (1 / slope of round trip against bytes, load/warm) / clock_ghz. The faster of warm loads and stores is taken as the engine rate: the model charges max(engine, memory) per copy, so each slope is the slower of the two. Clocks converted with the fitted clock_ghz = 0.5117. |
| `l2_bandwidth_per_sm_gb_s` | 250.32 GB/s | 400 | 0.6258 | 1 / slope of round trip against bytes, warm loads, in bytes/ns. LOWER BOUND: warm loads run at the engine rate, so L2 is at least this fast; the model reproduces the measurement with any value from here up. |
| `tma_latency_ns` | 155.24 ns | 100 | 1.552 | intercept 168.4 ns of warm-load round trip against bytes - 13.19 ns of instruction time the simulator counts for the same loop. Includes whatever the mbarrier wake-up costs on hardware beyond its instructions. Clocks converted with the fitted clock_ghz = 0.5117. |
| `hbm_bandwidth_tb_s` | 6.1602 TB/s | 4.141 | 1.488 | STREAM Triad: 1 / (0.0001623 ns per byte) / 1000 (gstream/triad, 24 bytes per element). Same definition as the config's published figure. Cold TMA loads on every SM gave 8.631 TB/s. |
| `tensor_core_tflops_f16` | 2238.1 TFLOPS | 1929.6 | 1.16 | 2 * 128 * 256 * 16 FLOP / 69.34 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |
| `mma_latency_clk` | 96.481 clk | 11.3 | 8.538 | config 11.3 + measured batch intercept 215.7 ns * clock_ghz - predicted 16.81 ns * 1.5 (f16 m128 n256). Includes whatever tcgen05.commit and the mbarrier wake-up cost on hardware beyond their instructions. Clocks converted with the fitted clock_ghz = 0.5117. |
| `tensor_core_tflops_tf32` | 1118.9 TFLOPS | 964.5 | 1.16 | 2 * 128 * 256 * 8 FLOP / 69.35 ns per MMA * 148 SMs / 1000 (slope over batch size, ctas=allsm).  |

Other measured quantities:

| quantity | value | unit | note |
|---|---|---|---|
| %clock64 ticks per ns | 1.9615 | ticks/ns | alu_chain/warps=1: clock64 slope / event-time slope. 0 means the build did not read the clock |
| shared load, no conflict | 14.306 | ns | smem_wavefronts/load_chain fit at conflict=1 |
| CLC request, fail/plain/ctas=1 | 246.44 | ns | slope of launch time against requests; minus 24.43 ns of instructions the simulator counts for the same loop = 222 ns of latency |
| CLC request, fail/plain/ctas=2 | 441.48 | ns | slope of launch time against requests; minus 28.34 ns of instructions the simulator counts for the same loop = 413.1 ns of latency |
| CLC request, fail/multicast/ctas=2 | 411.93 | ns | slope of launch time against requests; minus 28.34 ns of instructions the simulator counts for the same loop = 383.6 ns of latency |
| CLC request, succeed/plain/ctas=1 | 305.56 | ns | 599.4 ticks per request of the first cluster (clk / (check + 1)) / 1.961 ticks per ns; minus 60.58 ns of instructions the simulator counts for the same loop = 245 ns of latency |
| CLC request, succeed/multicast/ctas=2 | 519.59 | ns | 1019 ticks per request of the first cluster (clk / (check + 1)) / 1.961 ticks per ns; minus 64.49 ns of instructions the simulator counts for the same loop = 455.1 ns of latency |
| TMA load/warm: ns per byte | 0.003995 | ns/byte | = 250.3 GB/s for one SM |
| TMA load/warm: intercept | 168.44 | ns | round trip of a zero-byte copy, extrapolated |
| TMA store: ns per byte | 0.015706 | ns/byte | = 63.67 GB/s for one SM |
| TMA store: intercept | 13.235 | ns | round trip of a zero-byte copy, extrapolated |
| TMA load/cold: ns per byte | 0.010334 | ns/byte | = 96.77 GB/s for one SM |
| TMA load/cold: intercept | 416.15 | ns | round trip of a zero-byte copy, extrapolated |
| cold TMA fetch rate, ctas=8 | 543.23 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 416.2 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=37 | 2476.5 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 416.2 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| cold TMA fetch rate, ctas=allsm | 8631.2 | GB/s | total over the active SMs: CTAs * 16384 bytes / (round trip - 416.2 ns fixed cost per copy, from the one-CTA cold fit). The model gives each SM hbm_bandwidth / active_sms |
| MMA f16 m128 n256, ctas=allsm | 69.339 | ns/MMA | = 2238.1 TFLOPS for 148 SMs; batch intercept 217.5 ns |
| MMA f16 m128 n256, ctas=1 | 69.117 | ns/MMA | = 2245.3 TFLOPS for 148 SMs; batch intercept 215.7 ns |
| MMA tf32 m128 n256, ctas=allsm | 69.347 | ns/MMA | = 1118.9 TFLOPS for 148 SMs; batch intercept 219.9 ns |
| MMA tf32 m128 n256, ctas=1 | 69.104 | ns/MMA | = 1122.9 TFLOPS for 148 SMs; batch intercept 218.5 ns |
| MMA i8 m128 n256, ctas=allsm | 67.573 | ns/MMA | = 4593.3 TFLOPS for 148 SMs; batch intercept 214.6 ns |
| MMA i8 m128 n256, ctas=1 | 67.366 | ns/MMA | = 4607.3 TFLOPS for 148 SMs; batch intercept 213.5 ns |
| mbarrier arrive + wait, one thread | 32.76 | ns | mbar/arrive_wait/threads=1; predicted 3 ns |
| mbarrier arrive + wait, whole warp | 32.754 | ns | mbar/arrive_wait/threads=32; predicted 3 ns |
| tcgen05.wait::ld | -0.25398 | ns | tmem_access/ld/x1 minus the same loop without the wait; the model charges one clock |
| tcgen05.wait::st | 5.2258 | ns | tmem_access/st/x1 minus the same loop without the wait; the model charges one clock |

## (c) Measured against the simulator's current prediction

Prediction: `warplens run --timing` on the same program (predict.py), ns per repetition. "SMs assumed" is how many SMs the prediction took the launch to keep busy (they share HBM in the model); it comes from the launch's shape, and nothing in a run measures it directly. What a run can contradict is listed under "Problems in the results".

| configuration | measured | predicted | measured / predicted | SMs assumed |
|---|---|---|---|---|
| alu_chain/warps=1 | 7.329 | 2.5 | 2.931 | 1 |
| alu_chain/warps=2 | 7.329 | 2.5 | 2.932 | 1 |
| alu_chain/warps=4 | 7.329 | 2.5 | 2.931 | 1 |
| alu_chain/warps=5 | 7.392 | 5 | 1.478 | 1 |
| alu_chain/warps=8 | 7.391 | 5 | 1.478 | 1 |
| alu_chain/warps=16 | 8.942 | 10 | 0.8942 | 1 |
| alu_chain/warps=32 | 17.07 | 20 | 0.8533 | 1 |
| clc_latency/fail/plain/ctas=1 | 246.4 | 108.3 | 2.275 | 1 |
| clc_latency/fail/plain/ctas=2 | 441.5 | 109.7 | 4.026 | 2 |
| clc_latency/fail/multicast/ctas=2 | 411.9 | 109.7 | 3.756 | 2 |
| clc_latency/succeed/plain/ctas=1 | 1.414 | 120.7 | 0.01172 | 148 |
| clc_latency/succeed/multicast/ctas=2 | 4.317 | 122 | 0.03538 | 148 |
| cluster_costs/load/multicast/ctas=1 | 745.4 | 193.7 | 3.849 | 1 |
| cluster_costs/load/multicast/ctas=2 | 745.2 | 193.7 | 3.848 | 2 |
| cluster_costs/load/multicast/ctas=4 | 757.4 | 193.7 | 3.911 | 4 |
| cluster_costs/load/separate/ctas=1 | 864.3 | 195 | 4.433 | 1 |
| cluster_costs/load/separate/ctas=2 | 868.9 | 195 | 4.456 | 2 |
| cluster_costs/load/separate/ctas=4 | 880.1 | 195 | 4.514 | 4 |
| cluster_costs/mma/pair/m=256/n=256/batch=1 | 296.5 | 176.4 | 1.681 | 2 |
| cluster_costs/mma/pair/m=256/n=256/batch=4 | 479 | 659.6 | 0.7262 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=1/ctas=2 | 272.6 | 95.96 | 2.841 | 2 |
| cluster_costs/mma/single/m=128/n=256/batch=4/ctas=2 | 455.1 | 337.9 | 1.347 | 2 |
| cp_async/warm | 162.7 | 106.5 | 1.528 | 1 |
| cp_async/cold | 320.5 | 106.5 | 3.01 | 1 |
| global_load/load_chain/footprint=l1 | 44.07 | 24.17 | 1.824 | 1 |
| global_load/store/footprint=l1 | 22.73 | 3.667 | 6.2 | 1 |
| global_load/load_chain/footprint=l2 | 181.3 | 24.17 | 7.501 | 1 |
| global_load/store/footprint=l2 | 20.85 | 3.667 | 5.685 | 1 |
| global_load/load_chain/footprint=dram | 686.5 | 24.17 | 28.4 | 1 |
| global_load/store/footprint=dram | 21.05 | 3.667 | 5.74 | 1 |
| gstream/triad | 0.0001623 | 0 | n/a: the model predicts no cost | 148 |
| launch/empty/ctas=1 (launch) | 3744 | 6.667 | 561.6 | 1 |
| launch/empty/ctas=allsm (launch) | 3712 | 6.667 | 556.8 | 148 |
| mbar/arrive_wait/threads=1 | 32.76 | 3 | 10.92 | 1 |
| mbar/arrive_wait/threads=32 | 32.75 | 3 | 10.92 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=1/ctas=1 | 209.7 | 21.89 | 9.576 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=2/ctas=1 | 262.3 | 26.92 | 9.745 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=4/ctas=1 | 310.7 | 37.64 | 8.254 | 1 |
| mma_rate/kind=f16/m=64/n=32/batch=8/ctas=1 | 427.4 | 57.75 | 7.402 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=1/ctas=1 | 209.6 | 36.97 | 5.67 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=2/ctas=1 | 262.3 | 57.08 | 4.596 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=4/ctas=1 | 310.6 | 97.96 | 3.171 | 1 |
| mma_rate/kind=f16/m=64/n=128/batch=8/ctas=1 | 514.3 | 178.4 | 2.883 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=1/ctas=1 | 296.5 | 57.08 | 5.195 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=2/ctas=1 | 349.2 | 97.29 | 3.59 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=4/ctas=1 | 479 | 178.4 | 2.685 | 1 |
| mma_rate/kind=f16/m=64/n=256/batch=8/ctas=1 | 758.7 | 339.2 | 2.236 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=1/ctas=1 | 209.6 | 26.92 | 7.788 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=2/ctas=1 | 262.3 | 36.97 | 7.095 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=4/ctas=1 | 310.6 | 57.75 | 5.379 | 1 |
| mma_rate/kind=f16/m=128/n=32/batch=8/ctas=1 | 427.4 | 97.96 | 4.363 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=1/ctas=1 | 209.6 | 57.08 | 3.672 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=2/ctas=1 | 262.3 | 97.29 | 2.696 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=4/ctas=1 | 310.6 | 178.4 | 1.741 | 1 |
| mma_rate/kind=f16/m=128/n=128/batch=8/ctas=1 | 514.3 | 339.2 | 1.516 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=1 | 296.5 | 97.29 | 3.048 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=1 | 349.2 | 177.7 | 1.965 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=1 | 479 | 339.2 | 1.412 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=1 | 775 | 660.9 | 1.173 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=1/ctas=1 | 212.4 | 21.89 | 9.699 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=2/ctas=1 | 265 | 26.92 | 9.842 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=4/ctas=1 | 313.4 | 37.65 | 8.325 | 1 |
| mma_rate/kind=tf32/m=64/n=32/batch=8/ctas=1 | 430.1 | 57.76 | 7.447 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=1/ctas=1 | 212.3 | 36.98 | 5.742 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=2/ctas=1 | 265 | 57.09 | 4.642 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=4/ctas=1 | 313.4 | 97.98 | 3.198 | 1 |
| mma_rate/kind=tf32/m=64/n=128/batch=8/ctas=1 | 517 | 178.4 | 2.897 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=1/ctas=1 | 299.2 | 57.09 | 5.241 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=2/ctas=1 | 351.9 | 97.32 | 3.616 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=4/ctas=1 | 481.7 | 178.4 | 2.7 | 1 |
| mma_rate/kind=tf32/m=64/n=256/batch=8/ctas=1 | 761.5 | 339.3 | 2.244 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=1/ctas=1 | 212.3 | 26.92 | 7.884 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=2/ctas=1 | 265 | 36.98 | 7.166 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=4/ctas=1 | 313.3 | 57.76 | 5.425 | 1 |
| mma_rate/kind=tf32/m=128/n=32/batch=8/ctas=1 | 430.1 | 97.98 | 4.389 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=1/ctas=1 | 212.4 | 57.09 | 3.72 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=2/ctas=1 | 265 | 97.32 | 2.723 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=4/ctas=1 | 313.4 | 178.4 | 1.756 | 1 |
| mma_rate/kind=tf32/m=128/n=128/batch=8/ctas=1 | 517.1 | 339.3 | 1.524 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=1 | 299.2 | 97.32 | 3.074 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=1 | 351.9 | 177.8 | 1.98 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=1 | 481.8 | 339.3 | 1.42 | 1 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=1 | 777.6 | 661.1 | 1.176 | 1 |
| mma_rate/kind=f16/m=128/n=256/batch=1/ctas=allsm | 298.1 | 97.29 | 3.064 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=2/ctas=allsm | 351.4 | 177.7 | 1.977 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=4/ctas=allsm | 482.2 | 339.2 | 1.421 | 148 |
| mma_rate/kind=f16/m=128/n=256/batch=8/ctas=allsm | 778.3 | 660.9 | 1.177 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=1/ctas=allsm | 300.4 | 97.32 | 3.087 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=2/ctas=allsm | 354 | 177.8 | 1.991 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=4/ctas=allsm | 484.7 | 339.3 | 1.428 | 148 |
| mma_rate/kind=tf32/m=128/n=256/batch=8/ctas=allsm | 780.8 | 661.1 | 1.181 | 148 |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=1 | 291.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=1 | 343.3 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=1 | 471.2 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=1 | 758.2 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=1/ctas=allsm | 292.8 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=2/ctas=allsm | 344.9 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=4/ctas=allsm | 473.5 | n/a | no prediction |  |
| mma_rate/kind=i8/m=128/n=256/batch=8/ctas=allsm | 760.7 | n/a | no prediction |  |
| ncu_banks/linear/block=32x32 (launch) | 7040 | 851.3 | 8.269 | 64 |
| ncu_banks/padded/block=32x32 (launch) | 5792 | 190 | 30.48 | 64 |
| ncu_banks/swizzled/block=32x32 (launch) | 5120 | 211.3 | 24.23 | 64 |
| ncu_banks/linear/block=32x8 (launch) | 5056 | 104 | 48.62 | 64 |
| ncu_banks/padded/block=32x8 (launch) | 4800 | 82.67 | 58.06 | 64 |
| ncu_banks/swizzled/block=32x8 (launch) | 5376 | 88 | 61.09 | 64 |
| ncu_banks/linear/block=16x16 (launch) | 4800 | 104 | 46.15 | 64 |
| ncu_banks/padded/block=16x16 (launch) | 5344 | 77.33 | 69.1 | 64 |
| ncu_banks/swizzled/block=16x16 (launch) | 4928 | 72 | 68.44 | 64 |
| pipelined_gemm/stages=1/ctas=8 | 1035 | 330.8 | 3.129 | 8 |
| pipelined_gemm/stages=2/ctas=8 | 543.2 | 165.8 | 3.276 | 8 |
| pipelined_gemm/stages=4/ctas=8 | 295.2 | 128 | 2.306 | 8 |
| pipelined_gemm/stages=1/ctas=allsm | 1109 | 1081 | 1.026 | 148 |
| pipelined_gemm/stages=2/ctas=allsm | 615.7 | 878.4 | 0.701 | 148 |
| pipelined_gemm/stages=4/ctas=allsm | 404.6 | 878.4 | 0.4606 | 148 |
| smem_wavefronts/load_chain/conflict=1/warps=1 | 14.31 | 3.333 | 4.292 | 1 |
| smem_wavefronts/load_chain/conflict=2/warps=1 | 15.32 | 4 | 3.831 | 1 |
| smem_wavefronts/load_chain/conflict=4/warps=1 | 17.36 | 5.333 | 3.256 | 1 |
| smem_wavefronts/load_chain/conflict=8/warps=1 | 21.45 | 8 | 2.681 | 1 |
| smem_wavefronts/load_chain/conflict=16/warps=1 | 29.6 | 13.33 | 2.22 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=1 | 45.91 | 24 | 1.913 | 1 |
| smem_wavefronts/load_chain/conflict=1/warps=4 | 14.31 | 3.333 | 4.292 | 1 |
| smem_wavefronts/load_chain/conflict=32/warps=4 | 65.26 | 85.33 | 0.7648 | 1 |
| smem_wavefronts/load_indep/conflict=1/warps=1 | 4.812 | 6 | 0.8021 | 1 |
| smem_wavefronts/load_indep/conflict=2/warps=1 | 4.875 | 6.667 | 0.7313 | 1 |
| smem_wavefronts/load_indep/conflict=4/warps=1 | 6.182 | 8 | 0.7728 | 1 |
| smem_wavefronts/load_indep/conflict=8/warps=1 | 10.26 | 10.67 | 0.962 | 1 |
| smem_wavefronts/load_indep/conflict=16/warps=1 | 18.42 | 16 | 1.151 | 1 |
| smem_wavefronts/load_indep/conflict=32/warps=1 | 34.73 | 26.67 | 1.302 | 1 |
| smem_wavefronts/store/conflict=1/warps=1 | 2.549 | 2.5 | 1.02 | 1 |
| smem_wavefronts/store/conflict=2/warps=1 | 2.549 | 3.167 | 0.805 | 1 |
| smem_wavefronts/store/conflict=4/warps=1 | 4.079 | 4.5 | 0.9064 | 1 |
| smem_wavefronts/store/conflict=8/warps=1 | 8.158 | 7.167 | 1.138 | 1 |
| smem_wavefronts/store/conflict=16/warps=1 | 16.31 | 12.5 | 1.305 | 1 |
| smem_wavefronts/store/conflict=32/warps=1 | 32.63 | 23.17 | 1.408 | 1 |
| tma/load/warm/bytes=64/ctas=1 | 168.2 | 104.8 | 1.604 | 1 |
| tma/load/warm/bytes=256/ctas=1 | 168.4 | 105.8 | 1.591 | 1 |
| tma/load/warm/bytes=1024/ctas=1 | 175.9 | 109.8 | 1.602 | 1 |
| tma/load/warm/bytes=4096/ctas=1 | 185.1 | 125.8 | 1.471 | 1 |
| tma/load/warm/bytes=16384/ctas=1 | 230 | 189.8 | 1.211 | 1 |
| tma/load/warm/bytes=32768/ctas=1 | 301.2 | 275.2 | 1.094 | 1 |
| tma/store/bytes=64/ctas=1 | 20.13 | 103.3 | 0.1948 | 1 |
| tma/store/bytes=256/ctas=1 | 20.88 | 104.3 | 0.2001 | 1 |
| tma/store/bytes=1024/ctas=1 | 25.12 | 108.3 | 0.2319 | 1 |
| tma/store/bytes=4096/ctas=1 | 72.57 | 124.3 | 0.5837 | 1 |
| tma/store/bytes=16384/ctas=1 | 268.5 | 188.3 | 1.426 | 1 |
| tma/store/bytes=32768/ctas=1 | 529.6 | 273.7 | 1.935 | 1 |
| tma/load/cold/bytes=64/ctas=1 | 298.3 | 104.8 | 2.846 | 1 |
| tma/load/cold/bytes=256/ctas=1 | 436.2 | 105.8 | 4.121 | 1 |
| tma/load/cold/bytes=1024/ctas=1 | 458.8 | 109.8 | 4.177 | 1 |
| tma/load/cold/bytes=4096/ctas=1 | 497.7 | 125.8 | 3.955 | 1 |
| tma/load/cold/bytes=16384/ctas=1 | 656.6 | 189.8 | 3.459 | 1 |
| tma/load/cold/bytes=32768/ctas=1 | 713.4 | 275.2 | 2.593 | 1 |
| tma/load/cold/bytes=16384/ctas=8 | 657.4 | 189.7 | 3.466 | 8 |
| tma/load/cold/bytes=16384/ctas=37 | 660.9 | 250.7 | 2.636 | 37 |
| tma/load/cold/bytes=16384/ctas=allsm | 697.1 | 689.9 | 1.01 | 148 |
| tmem_access/ld/x1 | 14.82 | 5.333 | 2.779 | 1 |
| tmem_access/ld/x8 | 25.75 | 10 | 2.575 | 1 |
| tmem_access/st/x1 | 6.149 | 4.667 | 1.318 | 1 |
| tmem_access/st/x8 | 8.095 | 4.667 | 1.735 | 1 |
| tmem_access/ld/x1/nowait | 15.07 | 4.5 | 3.35 | 1 |
| tmem_access/st/x1/nowait | 0.9234 | 3.833 | 0.2409 | 1 |

`clc_latency/succeed/*`: on a GPU many clusters start at once, so these two slopes are not round trips and their ratios mean nothing; the kernel's own count is under "Other measured quantities" (CLC request, succeed/...).
