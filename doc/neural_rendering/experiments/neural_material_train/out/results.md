## Variance: final held-out power-loss (n=3, L1) over seeds

| method | seeds | mean | std | min | max | mean wall/run (s) |
|---|---|---|---|---|---|---|
| 2x16 baseline (1 inst, bs 8192) | 8 | 0.1176 | 0.0149 | 0.0949 | 0.1314 | 81 |
| 2x16 Bitterli (64>16>4>1) | 8 | 0.1126 | 0.0052 | 0.1045 | 0.1184 | 97 |
| 2x32 baseline (1 inst, bs 8192) | 8 | 0.0766 | 0.0074 | 0.0617 | 0.0871 | 90 |
| 2x32 Bitterli (64>16>4>1) | 8 | 0.0853 | 0.0033 | 0.0808 | 0.0909 | 110 |

std ratio (Bitterli/baseline), bootstrap 5/50/95 pct: 2x16: 0.17, 0.35, 0.82; 2x32: 0.20, 0.45, 1.08

## Accuracy (mean over the 8 seeds; best seed in brackets) -- held-out 65,536 samples

| model | method | pow-loss | rel L1 (all) | rel L1 (top 1% brightest) | rel L1 (grazing >75deg) | median per-sample rel err | #decoder params |
|---|---|---|---|---|---|---|---|
| 2x16 | baseline | 0.1176 (0.0949) | 0.2405 (0.2103) | 0.3486 (0.3066) | 0.2611 (0.2292) | 0.1457 (0.1131) | 659 |
| 2x16 | bitterli | 0.1126 (0.1045) | 0.2574 (0.2408) | 0.3753 (0.3533) | 0.2785 (0.2610) | 0.1314 (0.1208) | 659 |
| 2x32 | baseline | 0.0766 (0.0617) | 0.1843 (0.1659) | 0.2780 (0.2471) | 0.2013 (0.1801) | 0.0857 (0.0625) | 1827 |
| 2x32 | bitterli | 0.0853 (0.0808) | 0.2183 (0.2090) | 0.3257 (0.3101) | 0.2368 (0.2257) | 0.0922 (0.0829) | 1827 |

## Ablations (2x16, baseline schedule, seeds 0-3; metrics on the same held-out set)

| variant | seeds | pow-loss mean | std | rel L1 | rel L1 top1% |
|---|---|---|---|---|---|
| power n=3 + LeakySmeLU (default) | 4 | 0.1226 | 0.0066 | 0.2464 | 0.3547 |
| log(1+x) loss + LeakySmeLU | 4 | 0.1707 | 0.0157 | 0.2639 | 0.3980 |
| power n=3 + ReLU | 4 | 0.1238 | 0.0158 | 0.2267 | 0.3122 |

## Bitterli phase diagnostics (selection score = mean fresh-batch train loss over last quarter of phase)

| model | phase (K@batch) | score min | median | max | (mean over seeds) |
|---|---|---|---|---|---|
| 2x16 | 64@128 | 0.3682 | 0.3965 | 0.4312 | |
| 2x16 | 16@512 | 0.2040 | 0.2195 | 0.2354 | |
| 2x16 | 4@2048 | 0.1285 | 0.1359 | 0.1430 | |
| 2x16 | 1@8192 | 0.1131 | 0.1131 | 0.1131 | |
| 2x32 | 64@128 | 0.3774 | 0.3999 | 0.4258 | |
| 2x32 | 16@512 | 0.1933 | 0.2092 | 0.2214 | |
| 2x32 | 4@2048 | 0.1041 | 0.1106 | 0.1159 | |
| 2x32 | 1@8192 | 0.0858 | 0.0858 | 0.0858 | |

Total wall time of the full experiment batch (4 parallel single-thread workers): 892 s
