# Gate tables (session of 2026-10-02)

The browser-agent demo's travel section is published as aggregates in manifest.json; everything else here is computed from this folder's files and runs/2026-09-30_served-default.

## The suite (1,400 items, C2 to C4, one request at a time)

| server | C2 | C3 | C4 | max \|dp\| against the default server | choice changes | accuracy | paired difference, points [95%] | pooled ECE |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| default | pass | pass | pass | 0.0e+00 | 0 | 76.21% | - | 0.0201 |
| sequential | pass | pass | pass | 1.2e-03 | 0 | 76.21% | +0.00 [+0.00, +0.00] | 0.0201 |
| describe | pass | pass | pass | 1.2e-03 | 0 | 76.21% | +0.00 [+0.00, +0.00] | 0.0201 |
| padshared | pass | pass | pass | 4.4e-01 | 99 | 75.71% | -0.50 [-1.71, +0.64] | 0.0232 |

## JevBench published items, paired against the served default's records (231 items)

| arm | items | accuracy | paired difference, points [95%] | choice changes | ECE (10 equal-mass bins) | I_open_A / I_open_B |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| records | 231 | 84.42% | - | - | 0.0404 | 43.07 / 52.17 |
| default | 231 | 84.42% | +0.00 [+0.00, +0.00] | 0 | 0.0404 | 43.07 / 52.17 |
| describe | 231 | 83.98% | -0.43 [-2.60, +1.30] | 6 | 0.0479 | 42.69 / 51.60 |
| padshared | 231 | 83.98% | -0.43 [-3.03, +2.16] | 11 | 0.0830 | 42.28 / 51.30 |

## Decision Index, paired against the served default's records

| benchmark | arm | items | accuracy | paired difference, points [95%] | choice changes | ECE |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| BANKING77 | records | 3080 | 73.99% | - | - | 0.0421 |
| BANKING77 | padshared | 3080 | 73.47% | -0.52 [-1.33, +0.29] | 261 | 0.0663 |
| CLINC150+OOS | records | 5500 | 82.42% | - | - | 0.1704 |
| CLINC150+OOS | padshared | 5500 | 81.65% | -0.76 [-1.47, -0.04] | 536 | 0.1824 |
| GPQA-Diamond | records | 198 | 45.45% | - | - | 0.1439 |
| GPQA-Diamond | describe | 198 | 48.99% | +3.54 [-1.01, +8.08] | 29 | 0.1157 |
| GPQA-Diamond | padshared | 198 | 46.97% | +1.52 [-3.03, +6.06] | 29 | 0.1293 |
| MMLU-Pro | records | 12032 | 60.93% | - | - | 0.0093 |
| MMLU-Pro | describe | 12032 | 60.85% | -0.07 [-0.52, +0.37] | 1403 | 0.0093 |
| MMLU-Pro | padshared | 12032 | 60.23% | -0.70 [-1.16, -0.24] | 1462 | 0.0125 |

## Intent sets with heads (draw 0, 10 examples per intent; held-out test items)

| task | server | head | items | accuracy | paired difference, points [95%] | choice changes | max \|dp\| |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| banking77 | default | True | 150 | 84.67% | - | 0 | 0.0e+00 |
| banking77 | padshared | True | 150 | 81.33% | -3.33 [-7.33, +0.67] | 13 | 9.3e-01 |
| v3_clinc | default | True | 100 | 87.00% | - | 0 | 0.0e+00 |
| v3_clinc | padshared | True | 100 | 92.00% | +5.00 [+1.00, +10.00] | 6 | 8.9e-01 |

## Fresh-state latency (one request at a time, a never-seen state each repeat; medians of 20, ms)

Engine floor (vLLM `generate`, one fresh unpadded prompt, one output token, the served engine's settings, no server; median of 20). With prefix caching on (as served), a cold prompt longer than a 1,056-token block is prefilled in two engine steps, split at the last block boundary; off, in one:

| prompt tokens | prefix caching on (as served) | off (one step) | the split's cost |
| ---: | ---: | ---: | ---: |
| 300 | 27.2 | 26.2 | +1.0 |
| 1000 | 36.1 | - | - |
| 1056 | 36.1 / 36.1 | 35.4 | +0.7 |
| 1100 | 50.0 | 35.8 | +14.2 |
| 1150 | 53.1 | 36.3 | +16.8 |
| 2112 | 50.9 | 50.0 | +0.8 |
| 2200 | 67.8 | 51.9 | +15.9 |
| 3000 | 84.7 | 66.7 | +18.0 |
| 3168 | 70.3 | 69.2 | +1.1 |

| server | state tokens | questions | client wall | HTTP and JSON | route (server - engine) | prepare | warm-up | questions | readout | engine | server |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 300 | 1 | 49.5 | 1.1 | 0.5 | 0.5 | 0.0 | 47.4 | 0.0 | 48.0 | 48.5 |
| default | 300 | 4 | 63.1 | 1.0 | 0.7 | 0.7 | 39.0 | 21.5 | 0.0 | 61.2 | 62.0 |
| default | 1000 | 1 | 52.3 | 1.0 | 0.9 | 1.0 | 0.0 | 49.3 | 0.0 | 50.4 | 51.3 |
| default | 1000 | 4 | 79.5 | 1.2 | 1.2 | 1.2 | 41.0 | 34.8 | 0.0 | 77.1 | 78.4 |
| default | 3000 | 1 | 87.5 | 1.2 | 3.9 | 2.3 | 0.0 | 80.0 | 0.0 | 82.4 | 86.3 |
| default | 3000 | 4 | 118.1 | 1.3 | 2.5 | 2.5 | 72.4 | 38.9 | 0.0 | 114.2 | 116.8 |
| sequential | 300 | 4 | 111.1 | 0.8 | 0.7 | 0.7 | 40.0 | 68.6 | 0.1 | 109.6 | 110.3 |
| sequential | 1000 | 4 | 113.4 | 1.2 | 1.2 | 1.2 | 41.5 | 68.3 | 0.1 | 111.2 | 112.3 |
| sequential | 3000 | 4 | 153.6 | 1.3 | 2.5 | 2.6 | 73.2 | 73.5 | 0.1 | 149.6 | 152.4 |
| padshared | 300 | 1 | 31.3 | 0.9 | 0.4 | 0.5 | 0.0 | 29.5 | 0.0 | 30.1 | 30.4 |
| padshared | 300 | 4 | 62.9 | 1.0 | 0.7 | 0.7 | 38.9 | 21.5 | 0.0 | 61.2 | 61.8 |
| padshared | 1000 | 1 | 38.8 | 0.9 | 0.9 | 1.0 | 0.0 | 36.0 | 0.0 | 37.0 | 38.0 |
| padshared | 1000 | 4 | 65.6 | 1.1 | 1.2 | 1.2 | 40.5 | 21.5 | 0.0 | 63.3 | 64.5 |
| padshared | 3000 | 1 | 93.7 | 0.9 | 5.8 | 2.3 | 0.0 | 84.2 | 0.1 | 86.8 | 92.8 |
| padshared | 3000 | 4 | 117.9 | 1.0 | 2.5 | 2.6 | 72.5 | 39.0 | 0.0 | 114.3 | 116.9 |
| batch | 300 | 4 | 57.6 | 1.0 | 0.5 | 0.6 | 0.0 | 55.3 | 0.0 | 56.0 | 56.6 |
| batch | 1000 | 4 | 59.3 | 0.8 | 0.9 | 1.0 | 0.0 | 56.4 | 0.0 | 57.5 | 58.5 |
| batch | 3000 | 4 | 95.8 | 1.1 | 5.0 | 2.2 | 0.0 | 87.5 | 0.1 | 90.0 | 94.8 |

