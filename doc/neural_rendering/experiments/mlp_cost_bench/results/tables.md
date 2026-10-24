### gcc_fast: `g++ -O3 -march=native -ffast-math`
Peak microbenchmark in this build: 87.3 GFLOP/s (AVX2 8-wide FMA, 12 chains), 11.2 GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, 89.6 GFLOP/s.

| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert | single eval, value only | 19.4 | 24.1 | 19.0 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | single eval, value + pdf (as Cycles returns) | 24.0 | 35.6 | 22.4 | 1.2x | 1.2x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, plain loops | 7.0 | 8.6 | 5.4 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, explicit float8 | 5.9 | 6.7 | 3.7 | 1.0x | 1.0x | - | - | - | - |
| 2x16 (IN=28) | single, row-major weights (as specified) | 281.4 | 288.3 | - | 14.5x | - | 1504 | 5.3 | 6.9 | 6% |
| 2x16 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 197.5 | 217.1 | 190.4 | 10.2x | 10.0x | 1504 | 7.6 | 9.8 | 8% |
| 2x16 (IN=28) | single, input-major, hand-written float8 + split accumulators | 94.1 | 98.5 | 99.6 | 4.9x | 5.2x | 1504 | 16.0 | 20.6 | 18% |
| 2x16 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 106.4 | 128.0 | 99.4 | 17.9x | 26.7x | 1504 | 14.1 | 18.2 | 16% |
| 2x16 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 249.2 | 259.1 | 229.7 | 42.0x | 61.6x | 1504 | 6.0 | 7.8 | 7% |
| 2x16 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 40.3 | 50.0 | 38.6 | 6.8x | 10.3x | 1504 | 37.3 | 48.1 | 42% |
| 2x32 (IN=28) | single, row-major weights (as specified) | 584.0 | 600.1 | - | 30.1x | - | 4032 | 6.9 | 8.3 | 8% |
| 2x32 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 167.1 | 174.6 | 160.3 | 8.6x | 8.4x | 4032 | 24.1 | 28.9 | 27% |
| 2x32 (IN=28) | single, input-major, hand-written float8 + split accumulators | 158.5 | 190.1 | 137.4 | 8.2x | 7.2x | 4032 | 25.4 | 30.4 | 28% |
| 2x32 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 226.8 | 240.3 | 205.5 | 38.2x | 55.1x | 4032 | 17.8 | 21.3 | 20% |
| 2x32 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 509.4 | 515.3 | - | 85.8x | - | 4032 | 7.9 | 9.5 | 9% |
| 2x32 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 91.5 | 95.1 | 86.3 | 15.4x | 23.2x | 4032 | 44.1 | 52.7 | 49% |
| 3x64 (IN=28) | single, row-major weights (as specified) | 1845.6 | 1863.1 | - | 95.2x | - | 20352 | 11.0 | 12.2 | 12% |
| 3x64 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 662.3 | 705.2 | - | 34.2x | - | 20352 | 30.7 | 34.0 | 34% |
| 3x64 (IN=28) | single, input-major, hand-written float8 + split accumulators | 712.5 | 740.4 | - | 36.8x | - | 20352 | 28.6 | 31.6 | 32% |
| 3x64 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 1752.6 | 1773.4 | - | 295.1x | - | 20352 | 11.6 | 12.9 | 13% |
| 3x64 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 5066.4 | 5170.2 | - | 853.1x | - | 20352 | 4.0 | 4.5 | 4% |
| 3x64 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 392.2 | 445.8 | - | 66.0x | - | 20352 | 51.9 | 57.5 | 58% |

### gcc_strict: `g++ -O3 -march=native`
Peak microbenchmark in this build: 86.5 GFLOP/s (AVX2 8-wide FMA, 12 chains), 11.1 GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, 89.6 GFLOP/s.

| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert | single eval, value only | 27.7 | 35.6 | 26.6 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | single eval, value + pdf (as Cycles returns) | 32.6 | 43.5 | 32.4 | 1.2x | 1.2x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, plain loops | 31.8 | 39.8 | 27.9 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, explicit float8 | 5.7 | 7.3 | 3.6 | 1.0x | 1.0x | - | - | - | - |
| 2x16 (IN=28) | single, row-major weights (as specified) | 526.3 | 533.1 | - | 19.0x | - | 1504 | 2.9 | 3.7 | 3% |
| 2x16 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 554.5 | 566.5 | - | 20.1x | - | 1504 | 2.7 | 3.5 | 3% |
| 2x16 (IN=28) | single, input-major, hand-written float8 + split accumulators | 106.5 | 108.2 | 102.2 | 3.9x | 3.8x | 1504 | 14.1 | 18.2 | 16% |
| 2x16 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 500.8 | 539.0 | - | 87.6x | - | 1504 | 3.0 | 3.9 | 3% |
| 2x16 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 294.0 | 320.1 | - | 51.4x | - | 1504 | 5.1 | 6.6 | 6% |
| 2x16 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 45.0 | 55.3 | 37.8 | 7.9x | 10.6x | 1504 | 33.4 | 43.1 | 37% |
| 2x32 (IN=28) | single, row-major weights (as specified) | 1682.4 | 1816.6 | - | 60.8x | - | 4032 | 2.4 | 2.9 | 3% |
| 2x32 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 207.6 | 215.0 | 199.8 | 7.5x | 7.5x | 4032 | 19.4 | 23.2 | 22% |
| 2x32 (IN=28) | single, input-major, hand-written float8 + split accumulators | 158.7 | 171.7 | 149.9 | 5.7x | 5.6x | 4032 | 25.4 | 30.4 | 28% |
| 2x32 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 1357.2 | 1402.4 | - | 237.3x | - | 4032 | 3.0 | 3.6 | 3% |
| 2x32 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 613.0 | 645.5 | - | 107.2x | - | 4032 | 6.6 | 7.9 | 7% |
| 2x32 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 85.5 | 96.0 | 81.0 | 15.0x | 22.7x | 4032 | 47.1 | 56.4 | 53% |
| 3x64 (IN=28) | single, row-major weights (as specified) | 8620.7 | 9033.9 | - | 311.8x | - | 20352 | 2.4 | 2.6 | 3% |
| 3x64 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 767.2 | 831.7 | - | 27.7x | - | 20352 | 26.5 | 29.4 | 30% |
| 3x64 (IN=28) | single, input-major, hand-written float8 + split accumulators | 741.1 | 756.3 | - | 26.8x | - | 20352 | 27.5 | 30.4 | 31% |
| 3x64 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 6902.1 | 7166.2 | - | 1206.7x | - | 20352 | 2.9 | 3.3 | 3% |
| 3x64 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 469.7 | 500.5 | - | 82.1x | - | 20352 | 43.3 | 48.0 | 48% |

### clang_fast: `clang++ -O3 -march=native -ffast-math`
Peak microbenchmark in this build: 88.9 GFLOP/s (AVX2 8-wide FMA, 12 chains), 12.9 GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, 89.6 GFLOP/s.

| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert | single eval, value only | 20.6 | 25.4 | 21.0 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | single eval, value + pdf (as Cycles returns) | 25.7 | 41.7 | 25.5 | 1.2x | 1.2x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, plain loops | 5.7 | 8.0 | 3.8 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, explicit float8 | 5.5 | 6.4 | 3.1 | 1.0x | 1.0x | - | - | - | - |
| 2x16 (IN=28) | single, row-major weights (as specified) | 1699.5 | 1767.1 | - | 82.3x | - | 1504 | 0.9 | 1.1 | 1% |
| 2x16 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 1670.3 | 1676.6 | - | 80.9x | - | 1504 | 0.9 | 1.2 | 1% |
| 2x16 (IN=28) | single, input-major, hand-written float8 + split accumulators | 113.3 | 121.9 | 106.6 | 5.5x | 5.1x | 1504 | 13.3 | 17.1 | 15% |
| 2x16 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 138.9 | 149.7 | 127.0 | 25.1x | 41.3x | 1504 | 10.8 | 14.0 | 12% |
| 2x16 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 604.1 | 612.1 | - | 109.0x | - | 1504 | 2.5 | 3.2 | 3% |
| 2x16 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 49.8 | 53.2 | 42.6 | 9.0x | 13.9x | 1504 | 30.2 | 39.0 | 34% |
| 2x32 (IN=28) | single, row-major weights (as specified) | 4349.3 | 4410.9 | - | 210.7x | - | 4032 | 0.9 | 1.1 | 1% |
| 2x32 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 4698.6 | 4711.7 | - | 227.7x | - | 4032 | 0.9 | 1.0 | 1% |
| 2x32 (IN=28) | single, input-major, hand-written float8 + split accumulators | 155.1 | 168.9 | 153.2 | 7.5x | 7.3x | 4032 | 26.0 | 31.1 | 29% |
| 2x32 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 255.5 | 281.3 | - | 46.1x | - | 4032 | 15.8 | 18.9 | 18% |
| 2x32 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 1538.2 | 1621.9 | - | 277.6x | - | 4032 | 2.6 | 3.1 | 3% |
| 2x32 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 92.6 | 109.7 | 84.9 | 16.7x | 27.6x | 4032 | 43.5 | 52.1 | 49% |
| 3x64 (IN=28) | single, row-major weights (as specified) | 5503.3 | 5727.9 | - | 266.6x | - | 20352 | 3.7 | 4.1 | 4% |
| 3x64 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 672.9 | 736.3 | - | 32.6x | - | 20352 | 30.2 | 33.5 | 34% |
| 3x64 (IN=28) | single, input-major, hand-written float8 + split accumulators | 706.8 | 746.7 | - | 34.2x | - | 20352 | 28.8 | 31.9 | 32% |
| 3x64 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 790.2 | 829.3 | - | 142.6x | - | 20352 | 25.8 | 28.5 | 29% |
| 3x64 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 395.7 | 400.2 | - | 71.4x | - | 20352 | 51.4 | 57.0 | 57% |

### gcc_fast_novec: `g++ -O3 -march=native -ffast-math -fno-tree-vectorize`
Peak microbenchmark in this build: 89.6 GFLOP/s (AVX2 8-wide FMA, 12 chains), 7.0 GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, 89.6 GFLOP/s.

| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert | single eval, value only | 18.8 | 21.3 | 18.5 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | single eval, value + pdf (as Cycles returns) | 22.7 | 24.4 | 22.1 | 1.2x | 1.2x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, plain loops | 19.8 | 21.9 | 18.9 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, explicit float8 | 5.9 | 8.1 | 3.7 | 1.0x | 1.0x | - | - | - | - |
| 2x32 (IN=28) | single, row-major weights (as specified) | 1946.8 | 1982.3 | - | 103.4x | - | 4032 | 2.1 | 2.5 | 2% |
| 2x32 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 997.2 | 1070.3 | - | 53.0x | - | 4032 | 4.0 | 4.8 | 5% |
| 2x32 (IN=28) | single, input-major, hand-written float8 + split accumulators | 145.6 | 152.3 | 139.1 | 7.7x | 7.5x | 4032 | 27.7 | 33.1 | 31% |
| 2x32 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 578.2 | 603.5 | - | 98.0x | - | 4032 | 7.0 | 8.3 | 8% |
| 2x32 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 790.0 | 852.4 | - | 133.9x | - | 4032 | 5.1 | 6.1 | 6% |
| 2x32 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 85.5 | 91.0 | 76.7 | 14.5x | 20.8x | 4032 | 47.1 | 56.4 | 53% |

### gcc_cycles: `g++ -O3 -mavx2 -mfma + Cycles math flags (no fast-math)`
Peak microbenchmark in this build: 40.6 GFLOP/s (AVX2 8-wide FMA, 12 chains), 5.0 GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, 89.6 GFLOP/s.

| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert | single eval, value only | 28.9 | 39.7 | 28.8 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | single eval, value + pdf (as Cycles returns) | 35.1 | 50.7 | 34.7 | 1.2x | 1.2x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, plain loops | 36.8 | 46.0 | 32.4 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, explicit float8 | 6.2 | 7.9 | 3.8 | 1.0x | 1.0x | - | - | - | - |
| 2x16 (IN=28) | single, row-major weights (as specified) | 660.6 | 696.9 | - | 22.8x | - | 1504 | 2.3 | 2.9 | 3% |
| 2x16 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 730.1 | 791.8 | - | 25.2x | - | 1504 | 2.1 | 2.7 | 2% |
| 2x16 (IN=28) | single, input-major, hand-written float8 + split accumulators | 110.0 | 121.4 | 107.1 | 3.8x | 3.7x | 1504 | 13.7 | 17.6 | 15% |
| 2x16 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 490.0 | 514.0 | - | 78.7x | - | 1504 | 3.1 | 4.0 | 3% |
| 2x16 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 261.8 | 280.2 | - | 42.0x | - | 1504 | 5.7 | 7.4 | 6% |
| 2x16 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 49.0 | 56.9 | 46.8 | 7.9x | 12.5x | 1504 | 30.7 | 39.6 | 34% |
| 2x32 (IN=28) | single, row-major weights (as specified) | 1707.6 | 1779.8 | - | 59.0x | - | 4032 | 2.4 | 2.8 | 3% |
| 2x32 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 232.6 | 246.2 | 220.0 | 8.0x | 7.6x | 4032 | 17.3 | 20.7 | 19% |
| 2x32 (IN=28) | single, input-major, hand-written float8 + split accumulators | 158.7 | 165.7 | 151.9 | 5.5x | 5.3x | 4032 | 25.4 | 30.4 | 28% |
| 2x32 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 1761.1 | 1810.6 | - | 282.7x | - | 4032 | 2.3 | 2.7 | 3% |
| 2x32 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 627.0 | 652.6 | - | 100.6x | - | 4032 | 6.4 | 7.7 | 7% |
| 2x32 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 105.9 | 115.4 | 93.7 | 17.0x | 25.0x | 4032 | 38.1 | 45.5 | 43% |
| 3x64 (IN=28) | single, row-major weights (as specified) | 8760.9 | 8866.2 | - | 302.7x | - | 20352 | 2.3 | 2.6 | 3% |
| 3x64 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 807.0 | 839.1 | - | 27.9x | - | 20352 | 25.2 | 27.9 | 28% |
| 3x64 (IN=28) | single, input-major, hand-written float8 + split accumulators | 737.5 | 751.2 | - | 25.5x | - | 20352 | 27.6 | 30.6 | 31% |
| 3x64 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 9327.9 | 9488.0 | - | 1497.3x | - | 20352 | 2.2 | 2.4 | 2% |
| 3x64 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 448.6 | 480.0 | - | 72.0x | - | 20352 | 45.4 | 50.3 | 51% |

### clang_cycles: `clang++ -O3 -mavx2 -mfma + Cycles math flags (no fast-math)`
Peak microbenchmark in this build: 88.0 GFLOP/s (AVX2 8-wide FMA, 12 chains), 13.1 GFLOP/s (scalar FMA). '% of AVX2 peak' uses the machine-wide best, 89.6 GFLOP/s.

| config | variant | ns/eval (best of 3) | worst | cache-resident ns | ratio vs analytic | ratio (cache-resident) | matmul FLOPs | GFLOP/s (matmul) | GFLOP/s (all est.) | % of AVX2 peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert | single eval, value only | 25.7 | 45.6 | 24.3 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | single eval, value + pdf (as Cycles returns) | 31.4 | 37.4 | 31.0 | 1.2x | 1.3x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, plain loops | 6.4 | 8.6 | 4.4 | 1.0x | 1.0x | - | - | - | - |
| analytic GGX+Lambert | 8-batched, explicit float8 | 6.3 | 6.8 | 4.0 | 1.0x | 1.0x | - | - | - | - |
| 2x16 (IN=28) | single, row-major weights (as specified) | 278.6 | 306.5 | - | 10.8x | - | 1504 | 5.4 | 7.0 | 6% |
| 2x16 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 152.1 | 158.0 | 137.3 | 5.9x | 5.7x | 1504 | 9.9 | 12.8 | 11% |
| 2x16 (IN=28) | single, input-major, hand-written float8 + split accumulators | 105.6 | 109.0 | 100.9 | 4.1x | 4.2x | 1504 | 14.2 | 18.4 | 16% |
| 2x16 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 97.9 | 116.3 | 90.3 | 15.5x | 22.5x | 1504 | 15.4 | 19.8 | 17% |
| 2x16 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 542.9 | 546.9 | - | 85.9x | - | 1504 | 2.8 | 3.6 | 3% |
| 2x16 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 52.9 | 54.3 | 48.1 | 8.4x | 12.0x | 1504 | 28.4 | 36.7 | 32% |
| 2x32 (IN=28) | single, row-major weights (as specified) | 695.2 | 727.7 | - | 27.0x | - | 4032 | 5.8 | 6.9 | 6% |
| 2x32 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 206.2 | 213.1 | 199.0 | 8.0x | 8.2x | 4032 | 19.6 | 23.4 | 22% |
| 2x32 (IN=28) | single, input-major, hand-written float8 + split accumulators | 146.1 | 167.0 | 146.8 | 5.7x | 6.0x | 4032 | 27.6 | 33.0 | 31% |
| 2x32 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 264.3 | 280.6 | - | 41.8x | - | 4032 | 15.3 | 18.2 | 17% |
| 2x32 (IN=28) | 8-batched SoA, plain loops, 8-output register tile | 1436.6 | 1465.4 | - | 227.4x | - | 4032 | 2.8 | 3.4 | 3% |
| 2x32 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 107.5 | 119.5 | 112.1 | 17.0x | 28.0x | 4032 | 37.5 | 44.8 | 42% |
| 3x64 (IN=28) | single, row-major weights (as specified) | 3335.6 | 3462.6 | - | 129.7x | - | 20352 | 6.1 | 6.8 | 7% |
| 3x64 (IN=28) | single, input-major weights (compiler SIMD over outputs) | 766.3 | 874.6 | - | 29.8x | - | 20352 | 26.6 | 29.4 | 30% |
| 3x64 (IN=28) | single, input-major, hand-written float8 + split accumulators | 744.9 | 777.0 | - | 29.0x | - | 20352 | 27.3 | 30.3 | 30% |
| 3x64 (IN=28) | 8-batched SoA, plain loops (compiler SIMD over evals) | 1371.5 | 1456.7 | - | 217.1x | - | 20352 | 14.8 | 16.4 | 17% |
| 3x64 (IN=28) | 8-batched SoA, explicit float8, 8-output register tile | 442.5 | 450.3 | - | 70.0x | - | 20352 | 46.0 | 51.0 | 51% |

### gcc_fast, no latent code (IN=12)

| config | variant | ns/eval | ratio vs analytic | matmul FLOPs | GFLOP/s |
|---|---|---:|---:|---:|---:|
| 2x16 (IN=12) | rm | 265.1 | 13.7x | 992 | 3.7 |
| 2x16 (IN=12) | im | 117.7 | 6.1x | 992 | 8.4 |
| 2x16 (IN=12) | imv | 84.4 | 4.4x | 992 | 11.7 |
| 2x16 (IN=12) | b8n | 104.4 | 17.6x | 992 | 9.5 |
| 2x16 (IN=12) | b8t | 203.0 | 34.2x | 992 | 4.9 |
| 2x16 (IN=12) | b8v | 27.1 | 4.6x | 992 | 36.6 |
| 2x32 (IN=12) | rm | 537.9 | 27.8x | 3008 | 5.6 |
| 2x32 (IN=12) | im | 137.8 | 7.1x | 3008 | 21.8 |
| 2x32 (IN=12) | imv | 111.1 | 5.7x | 3008 | 27.1 |
| 2x32 (IN=12) | b8n | 227.4 | 38.3x | 3008 | 13.2 |
| 2x32 (IN=12) | b8t | 503.7 | 84.8x | 3008 | 6.0 |
| 2x32 (IN=12) | b8v | 76.5 | 12.9x | 3008 | 39.3 |
| 3x64 (IN=12) | rm | 1953.1 | 100.8x | 18304 | 9.4 |
| 3x64 (IN=12) | im | 622.1 | 32.1x | 18304 | 29.4 |
| 3x64 (IN=12) | imv | 658.6 | 34.0x | 18304 | 27.8 |
| 3x64 (IN=12) | b8n | 1786.8 | 300.9x | 18304 | 10.2 |
| 3x64 (IN=12) | b8t | 5062.6 | 852.4x | 18304 | 3.6 |
| 3x64 (IN=12) | b8v | 367.0 | 61.8x | 18304 | 49.9 |

### ns/eval, best of 3, IN=28, across builds (ratio vs that build's single-eval analytic in parentheses)

| variant | gcc_fast | gcc_strict | clang_fast | gcc_fast_novec | gcc_cycles | clang_cycles |
|---|---:|---:|---:|---:|---:|---:|
| analytic GGX+Lambert, single | 19.4 | 27.7 | 20.6 | 18.8 | 28.9 | 25.7 |
| analytic GGX+Lambert+pdf, single | 24.0 | 32.6 | 25.7 | 22.7 | 35.1 | 31.4 |
| analytic, 8-batched plain loops | 7.0 | 31.8 | 5.7 | 19.8 | 36.8 | 6.4 |
| analytic, 8-batched float8 | 5.9 | 5.7 | 5.5 | 5.9 | 6.2 | 6.3 |
| 2x16 rm | 281 (15x) | 526 (19x) | 1700 (82x) | - | 661 (23x) | 279 (11x) |
| 2x16 im | 197 (10x) | 555 (20x) | 1670 (81x) | - | 730 (25x) | 152 (6x) |
| 2x16 imv | 94 (5x) | 107 (4x) | 113 (5x) | - | 110 (4x) | 106 (4x) |
| 2x16 b8n | 106 (18x) | 501 (88x) | 139 (25x) | - | 490 (79x) | 98 (16x) |
| 2x16 b8t | 249 (42x) | 294 (51x) | 604 (109x) | - | 262 (42x) | 543 (86x) |
| 2x16 b8v | 40 (7x) | 45 (8x) | 50 (9x) | - | 49 (8x) | 53 (8x) |
| 2x32 rm | 584 (30x) | 1682 (61x) | 4349 (211x) | 1947 (103x) | 1708 (59x) | 695 (27x) |
| 2x32 im | 167 (9x) | 208 (8x) | 4699 (228x) | 997 (53x) | 233 (8x) | 206 (8x) |
| 2x32 imv | 158 (8x) | 159 (6x) | 155 (8x) | 146 (8x) | 159 (5x) | 146 (6x) |
| 2x32 b8n | 227 (38x) | 1357 (237x) | 255 (46x) | 578 (98x) | 1761 (283x) | 264 (42x) |
| 2x32 b8t | 509 (86x) | 613 (107x) | 1538 (278x) | 790 (134x) | 627 (101x) | 1437 (227x) |
| 2x32 b8v | 91 (15x) | 86 (15x) | 93 (17x) | 86 (14x) | 106 (17x) | 108 (17x) |
| 3x64 rm | 1846 (95x) | 8621 (312x) | 5503 (267x) | - | 8761 (303x) | 3336 (130x) |
| 3x64 im | 662 (34x) | 767 (28x) | 673 (33x) | - | 807 (28x) | 766 (30x) |
| 3x64 imv | 712 (37x) | 741 (27x) | 707 (34x) | - | 738 (25x) | 745 (29x) |
| 3x64 b8n | 1753 (295x) | 6902 (1207x) | 790 (143x) | - | 9328 (1497x) | 1371 (217x) |
| 3x64 b8t | 5066 (853x) | - | - | - | - | - |
| 3x64 b8v | 392 (66x) | 470 (82x) | 396 (71x) | - | 449 (72x) | 442 (70x) |

### Cost breakdown pieces (gcc_fast)

| piece | ns/eval stream | ns/eval cache-resident |
|---|---:|---:|
| memory-only baseline (reads the bytes of one MLP eval) | 11.6 | 1.6 |
| encode12 (h, shortest-arc wd') | 8.7 | 7.8 |
| 3 x Cycles fast_expf | 13.8 | 9.7 |
| 3 x libm expf | 15.4 | 11.6 |
| analytic GGX only | 15.0 | 14.8 |
| analytic Lambert only (layered weight) | 7.8 | 5.6 |

