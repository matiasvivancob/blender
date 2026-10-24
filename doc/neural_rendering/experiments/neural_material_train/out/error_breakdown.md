
### best 2x16 model: baseline_w16_pow_lsmelu_s5 (val pow-loss 0.0949)

| subset | share of samples | rel L1 | pred/target energy ratio |
|---|---|---|---|
| angle max(th_i,th_o) 0-30 | 5.5% | 0.129 | 0.907 |
| 30-60 | 28.5% | 0.089 | 0.985 |
| 60-75 | 27.5% | 0.164 | 0.875 |
| 75-85 | 24.9% | 0.150 | 0.933 |
| 85-90 | 13.6% | 0.291 | 0.804 |
| target <= median | 50.0% | 0.198 | 0.996 |
| median-90% | 40.0% | 0.096 | 1.015 |
| 90%-99% | 9.0% | 0.091 | 1.003 |
| top 1% | 1.0% | 0.321 | 0.749 |
| coat rough < 0.2 | 6.6% | 0.455 | 0.648 |
| coat rough 0.2-0.4 | 74.1% | 0.085 | 0.983 |
| coat rough >= 0.4 | 19.3% | 0.099 | 0.999 |
| metallic > 0.5 (gold checker) | 54.4% | 0.262 | 0.819 |
| dielectric base (metallic<0.5) | 45.6% | 0.100 | 0.988 |
| target lum > 100 (sharp peaks) | 0.49% | 0.360 | 0.699 |
| top 0.1% | 0.10% | 0.480 | 0.551 |

### best 2x32 model: baseline_w32_pow_lsmelu_s6 (val pow-loss 0.0617)

| subset | share of samples | rel L1 | pred/target energy ratio |
|---|---|---|---|
| angle max(th_i,th_o) 0-30 | 5.5% | 0.106 | 0.927 |
| 30-60 | 28.5% | 0.070 | 0.977 |
| 60-75 | 27.5% | 0.136 | 0.904 |
| 75-85 | 24.9% | 0.134 | 0.918 |
| 85-90 | 13.6% | 0.226 | 0.862 |
| target <= median | 50.0% | 0.105 | 1.001 |
| median-90% | 40.0% | 0.067 | 1.010 |
| 90%-99% | 9.0% | 0.060 | 1.004 |
| top 1% | 1.0% | 0.269 | 0.799 |
| coat rough < 0.2 | 6.6% | 0.404 | 0.720 |
| coat rough 0.2-0.4 | 74.1% | 0.046 | 0.987 |
| coat rough >= 0.4 | 19.3% | 0.077 | 0.991 |
| metallic > 0.5 (gold checker) | 54.4% | 0.218 | 0.857 |
| dielectric base (metallic<0.5) | 45.6% | 0.061 | 0.986 |
| target lum > 100 (sharp peaks) | 0.49% | 0.306 | 0.761 |
| top 0.1% | 0.10% | 0.413 | 0.635 |
