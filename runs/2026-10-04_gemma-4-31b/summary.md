# Gemma 4 31B as a base, FP8 on load, on one card in one session (2026-10-04)

`--base gemma-4-31b`: `google/gemma-4-31B-it` at `842da37`, quantized to FP8 by vLLM 0.30.0 when it loads, read through decisio's text class of the checkpoint (the vision tower left out at load), with the 12B base's prompt.
The server ran at T 1 with `--debug-readout`; the temperatures were fitted offline on the plain readouts of the 1,400-item suite, which is exact.

## Numbers

| Measure | Result |
| --- | --- |
| Temperatures | 5.252 global, 4.672 for choice questions (kept: out-of-fold log loss −0.0115 [−0.0211, −0.0015]) |
| Suite 1,400, accuracy / tie-robust ECE | 0.799 / 0.054 |
| JevBench 231, correct | 213 (easy 48/48, standard 72/72, hard 93/111) |
| JevBench v1.5: I_open A; choice / yes-no / score | 81.3; 86.3 / 85.6 / 71.9; 1.4% of yes/no answers between 0.20 and 0.80 |
| Decision Index BANKING77, macro-F1 | 0.7845 |
| Decision Index CLINC150+OOS, macro-F1 | 0.9015 |
| Decision Index GPQA Diamond (196 scored), accuracy | 0.5204 |
| Decision Index MMLU-Pro, accuracy | 0.6939 |
| Intent heads from 10 examples per intent, three draws: BANKING77 / CLINC150 | 0.844 / 0.970 (plain readout 0.787 / 0.940); registration 308 s / 699 s |
| One question, cached 1,000-token state (server median of 20) | 40.8 ms |
| One question, new 300 / 1,000 / 3,000-token state | 61.8 / 170.1 / 476.1 ms |
| Four questions, new 300 / 1,000 / 3,000-token state | 182.2 / 282.3 / 614.5 ms |
| Memory at the 0.90 share | weights 30.61 GiB, KV cache 48.6 GiB (57,882 tokens, 1.77 requests at 32,768) |
| Conformance C2 / C3 / C4 | pass |
| The intent head exact (250 items, log space) | 250/250 bit for bit |
| Repeatability (231 JevBench items, each twice in a row) | max \|dp\| 5.6e-8, no choice changed |
| `/health` cache units | `cache_hit_unit` 32, `hash_unit` 16 |

## At bf16

The checkpoint at bf16 did not start at a 32,768-token context: its weights (57.91 GiB) left 20.24 GiB for the KV cache at the 0.90 share, while one 32,768-token request needs 27.51 GiB; vLLM estimated 24,096 tokens as the most that fits (`bf16_start.txt`).
So the base is served in FP8 on load; a pinned FP8 checkpoint would replace this when one exists.

## The decision

The session's pre-registered go rule compared the four Decision Index values with the board's "Decider chat · Gemma-4-31B" row (BANKING77 0.791, CLINC150+OOS 0.912, GPQA Diamond 0.490, MMLU-Pro 0.696) and the 300-token single question with 108 ms.
The 31B passed GPQA Diamond and the latency, and missed BANKING77 by 0.0065, CLINC150+OOS by 0.0105 and MMLU-Pro by 0.0021.
The maintainer made it a base all the same: those thresholds were one row taken as a proxy for the top of the open board, the misses lie within each benchmark's sampling noise, and against the two served bases it is stronger on every accuracy measure (`EVAL_CARD.md` section 7).
