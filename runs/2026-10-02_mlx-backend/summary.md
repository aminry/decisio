<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

## Gates

| Gate | 8bit | 6bit | 4bit |
| --- | --- | --- | --- |
| P prompts (1,400 rows identical) | pass (1400 of 1400) | pass (1400 of 1400) | pass (1400 of 1400) |
| C1 CPU checks; G2 isolation exact | pass; pass | pass; pass | pass; pass |
| G1 prefix path vs whole prompt (reported) | max 4.8e-03, 0 flips | max 1.1e-02, 0 flips | max 4.1e-02, 0 flips |
| C2-C4 conformance | pass | pass | pass |
| S1 accuracy MLX - FP8 (points) | pass: 0.7650 vs 0.7621, +0.3 [-0.6, +1.2] | pass: 0.7657 vs 0.7621, +0.4 [-0.5, +1.2] | pass: 0.7521 vs 0.7621, -1.0 [-2.3, +0.3] |
| S2 top-answer flips | 56 of 1400 | 56 of 1400 | 128 of 1400 |
| S3 pooled ECE, |MLX - FP8| <= 0.01 | **fail**: 0.0301 vs 0.0201 (+0.0101) | pass: 0.0252 vs 0.0201 (+0.0052) | pass: 0.0208 vs 0.0201 (+0.0008) |
| H2 BANKING77 mean of 3 draws (points) | pass: 0.842 vs 0.847, -0.4 [-2.7, +1.6] | pass: 0.833 vs 0.847, -1.3 [-4.0, +1.3] | pass: 0.831 vs 0.847, -1.6 [-4.2, +0.7] |
| H1 BANKING77 exact arithmetic | pass | pass | pass |
| H2 CLINC150 mean of 3 draws (points) | pass: 0.920 vs 0.893, +2.7 [-1.0, +6.7] | pass: 0.913 vs 0.893, +2.0 [-1.3, +5.7] | pass: 0.927 vs 0.893, +3.3 [+0.0, +7.3] |
| H1 CLINC150 exact arithmetic | pass | pass | pass |

## Suite per task (accuracy MLX / FP8, difference in points; ECE MLX / FP8; flips)

**8bit**

| Task | n | Accuracy | MLX - FP8 | ECE | Flips |
| --- | --- | --- | --- | --- | --- |
| banking77 | 150 | 0.753 / 0.740 | +1.3 [-1.3, +4.0] | 0.092 / 0.092 | 5 |
| boolq | 150 | 0.853 / 0.860 | -0.7 [-2.0, +0.0] | 0.074 / 0.052 | 1 |
| mmlu | 150 | 0.887 / 0.880 | +0.7 [+0.0, +2.0] | 0.063 / 0.069 | 2 |
| mmlu_pro_text | 150 | 0.620 / 0.600 | +2.0 [-1.3, +5.3] | 0.093 / 0.083 | 16 |
| mmlu_pro_text_x350 | 350 | 0.649 / 0.657 | -0.9 [-3.4, +1.7] | 0.031 / 0.050 | 26 |
| scifact | 150 | 0.747 / 0.740 | +0.7 [+0.0, +2.0] | 0.155 / 0.125 | 1 |
| scifact_clarified | 150 | 0.813 / 0.820 | -0.7 [-2.7, +1.3] | 0.095 / 0.081 | 3 |
| toxicchat | 150 | 0.953 / 0.940 | +1.3 [+0.0, +3.3] | 0.029 / 0.032 | 2 |
| all | 1400 | 0.765 / 0.762 | +0.3 [-0.6, +1.2] | 0.030 / 0.020 | 56 |

**6bit**

| Task | n | Accuracy | MLX - FP8 | ECE | Flips |
| --- | --- | --- | --- | --- | --- |
| banking77 | 150 | 0.747 / 0.740 | +0.7 [-2.7, +3.3] | 0.077 / 0.092 | 7 |
| boolq | 150 | 0.867 / 0.860 | +0.7 [+0.0, +2.0] | 0.059 / 0.052 | 1 |
| mmlu | 150 | 0.887 / 0.880 | +0.7 [+0.0, +2.0] | 0.073 / 0.069 | 3 |
| mmlu_pro_text | 150 | 0.613 / 0.600 | +1.3 [-1.3, +4.0] | 0.095 / 0.083 | 12 |
| mmlu_pro_text_x350 | 350 | 0.649 / 0.657 | -0.9 [-3.4, +1.7] | 0.028 / 0.050 | 28 |
| scifact | 150 | 0.747 / 0.740 | +0.7 [+0.0, +2.0] | 0.164 / 0.125 | 1 |
| scifact_clarified | 150 | 0.820 / 0.820 | +0.0 [-2.0, +2.0] | 0.075 / 0.081 | 2 |
| toxicchat | 150 | 0.953 / 0.940 | +1.3 [+0.0, +3.3] | 0.027 / 0.032 | 2 |
| all | 1400 | 0.766 / 0.762 | +0.4 [-0.5, +1.2] | 0.025 / 0.020 | 56 |

**4bit**

| Task | n | Accuracy | MLX - FP8 | ECE | Flips |
| --- | --- | --- | --- | --- | --- |
| banking77 | 150 | 0.707 / 0.740 | -3.3 [-8.0, +1.3] | 0.076 / 0.092 | 18 |
| boolq | 150 | 0.853 / 0.860 | -0.7 [-3.3, +2.0] | 0.070 / 0.052 | 5 |
| mmlu | 150 | 0.893 / 0.880 | +1.3 [-1.3, +4.0] | 0.086 / 0.069 | 6 |
| mmlu_pro_text | 150 | 0.627 / 0.600 | +2.7 [-1.3, +6.7] | 0.043 / 0.083 | 19 |
| mmlu_pro_text_x350 | 350 | 0.626 / 0.657 | -3.1 [-6.9, +0.6] | 0.033 / 0.050 | 69 |
| scifact | 150 | 0.727 / 0.740 | -1.3 [-4.0, +1.3] | 0.174 / 0.125 | 4 |
| scifact_clarified | 150 | 0.813 / 0.820 | -0.7 [-3.3, +2.0] | 0.065 / 0.081 | 5 |
| toxicchat | 150 | 0.940 / 0.940 | +0.0 [-2.0, +2.0] | 0.032 / 0.032 | 2 |
| all | 1400 | 0.752 / 0.762 | -1.0 [-2.3, +0.3] | 0.021 / 0.020 | 128 |

## Intent heads per draw (MLX / FP8)

- 8bit banking77: 0.847 / 0.847, 0.847 / 0.853, 0.833 / 0.840; lambda 0.001, 0.01, 0.01; registration s 1935, 1265, 1255
- 8bit v3_clinc: 0.900 / 0.870, 0.930 / 0.900, 0.930 / 0.910; lambda 0.0001, 0.0001, 0.0001; registration s 2701, 2688, 2735
- 6bit banking77: 0.853 / 0.847, 0.827 / 0.853, 0.820 / 0.840; lambda 0.001, 0.01, 0.01; registration s 1198, 1703, 1260
- 6bit v3_clinc: 0.890 / 0.870, 0.920 / 0.900, 0.930 / 0.910; lambda 0.0001, 0.0001, 0.0001; registration s 2833, 2716, 3014
- 4bit banking77: 0.853 / 0.847, 0.827 / 0.853, 0.813 / 0.840; lambda 0.001, 0.01, 0.001; registration s 1085, 1082, 1081
- 4bit v3_clinc: 0.910 / 0.870, 0.940 / 0.900, 0.930 / 0.910; lambda 0.0001, 0.0001, 0.0001; registration s 2411, 2408, 2406

## Latency (server ms, p50 / p95; per decision p50)

| Cell | 8bit | 6bit | 4bit |
| --- | --- | --- | --- |
| 1q | 485 / 494 (484.6 per q) | 482 / 490 (482.1 per q) | 430 / 436 (430.4 per q) |
| 10q | 1164 / 1183 (116.4 per q) | 1148 / 1159 (114.8 per q) | 958 / 968 (95.8 per q) |
| 100q | 9829 / 10779 (98.3 per q) | 9747 / 10594 (97.5 per q) | 7404 / 7481 (74.0 per q) |
| 100q_8k | 14716 / 14874 (147.2 per q) | 14489 / 14619 (144.9 per q) | 11240 / 11477 (112.4 per q) |

Idle rule, reported pass: 8bit: not idle, max foreign 93% (/usr/libexec/searchpartyd); 6bit: not idle, max foreign 61% (/usr/libexec/searchpartyd); 4bit: not idle, max foreign 41% (.app/Contents/MacOS/Google Chrome Helper (Aperitif Renderer))

## Memory (peak GB; seconds for 1 / 10 questions; fits)

| State tokens | 8bit | 6bit | 4bit |
| --- | --- | --- | --- |
| 8192 | 8473 tokens: 38.9 GB; 3.6 / 4.3 s; fits | 8473 tokens: 30.2 GB; 3.5 / 4.6 s; fits | 8473 tokens: 21.6 GB; 3.1 / 4.1 s; fits |
| 16384 | 16921 tokens: 39.1 GB; 7.4 / 8.2 s; fits | 16921 tokens: 30.4 GB; 8.2 / 9.1 s; fits | 16921 tokens: 21.7 GB; 7.4 / 8.1 s; fits |
| 32768 | 32761 tokens: 39.4 GB; 16.3 / 17.5 s; fits | 32761 tokens: 30.7 GB; 17.9 / 19.4 s; fits | 32761 tokens: 22.0 GB; 16.2 / 17.4 s; fits |
