# Gemma 4 31B on MLX at 4 bits: the gates, MMLU-Pro, memory and latency (2026-10-15 to 2026-10-16)

**Status: complete.** Every pre-registered stage ran except three optional MMLU-style families, which a pre-registered 14-hour rule skipped (GPQA Diamond, BANKING77 and CLINC150+OOS in the Decision Index).

`--backend mlx --base gemma-4-31b` serving a 4-bit conversion of `google/gemma-4-31B-it@842da37` (`mlx_lm.convert -q --q-bits 4 --q-group-size 32`, mlx 0.32.3, mlx-lm 0.32.0; the sha256 of every file is in `build.json`), prompts built with the base's tokenizer and chat template, on an Apple M5 Pro (64 GB), macOS 26.6.2, under a 27 GiB MLX memory limit.
Reference: the base's vLLM FP8 record, `runs/2026-10-04_gemma-4-31b`.
The group size is 32, not mlx-lm's default of 64, so a community "4-bit" conversion is a different build, and none was measured.

## Gates

| Gate | 4-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| Answers equal to the earlier build of the same recipe (200 suite items) | 200 of 200 equal | | pass |
| Suite accuracy (1,400 items) | 0.7957 | 0.7993 | level: -0.4 [-1.4, +0.7] points |
| Top-answer flips against FP8 | 80 of 1,400 (largest probability difference 0.777) | | counted |
| Suite pooled ECE (allowed: within 0.01) | 0.0550 | 0.0540 | pass |
| JevBench 231, ECE at the served temperatures | 0.0379 | 0.0349 | pass |
| Conformance C2 to C4 | pass | | pass |
| Head exactness (largest probability difference, 250 items) | 6.66e-16 | | pass |
| Answers on a second fresh start (200 items) | 200 of 200 equal | | pass |
| Intent head, BANKING77, three draws (of 150) | 124 / 126 / 125, mean 0.833 | 125 / 130 / 125, mean 0.844 | level: -1.1 [-3.6, +1.1] points |
| Intent head, CLINC150, three draws (of 100) | 96 / 97 / 97, mean 0.967 | 97 / 97 / 97, mean 0.970 | level: -0.3 [-2.3, +1.7] points |
| MMLU-Pro (12,032 items), the Decision Index | 0.6785 | 0.6939 | **below: -1.54 [-1.99, -1.08] points** (rule: lower bound at or above -1.0) |

The suite, ECE and JevBench rows were read on an earlier build of the same recipe (a first conversion); this build's answers equal that build's on 200 of 200 suite items, which is why they carry over.
The pairing for MMLU-Pro is item by item against the FP8 record's answers (95% bootstrap, 10,000 resamples, seed 0); each item's correctness on both sides is in `mmlu_pro_pairs.jsonl.gz`.
1,331 of 12,032 top answers differ; 307 move from wrong to right and 492 from right to wrong.
The FP8 record did not reproduce exactly in a later session (28 top-answer changes in 1,631 items), which is small beside 1,331 of 12,032.

**Where the MMLU-Pro loss is** (4-bit minus FP8, points): engineering -3.61, math -3.48, chemistry -3.36, physics -2.85, history -2.62, health -1.59, business -1.14, computer science -0.98, philosophy -0.80, other -0.43, law -0.09, psychology +0.63, biology +0.70, economics +0.83 (`mmlu_pro.json`).
The 6-bit conversion's MMLU-Pro was not measured, so the loss is stated against FP8 only.

## Memory

decisio's memory planner on this build (working set 9 GB at a 32k-token state, default cache limit 2 GiB): 32 GB and 36 GB refuse to start; 48 GB starts with a 32 GiB limit and a 1,789 MiB prefix cache; 64 GB with 33 GiB and 2,048 MiB (`memory.json`).
Derived minimum Mac: 48 GB, from the 32k peak (27.80e9 bytes on the server), the 0.5 GiB cache limit and 2 GiB of headroom, 30.48e9 bytes in all.
It was not measured on a 48 GB Mac.

In-process reading (`memory_in_process.json`, the server stopped, the 27 GiB limit; a length fits if it completes, peaks at most 90% of the limit and pressure stays normal):

| State tokens | Questions | MLX peak | Share of the limit | Time | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8,056 | 1 | 23.86 GB | 82.3% | 25.6 s | fits |
| 8,056 | 10 | 23.86 GB | 82.3% | 29.3 s | fits |
| 16,383 | 1 | 25.33 GB | 87.4% | 58.5 s | fits |
| 16,383 | 10 | 25.34 GB | 87.4% | 61.7 s | fits |
| 32,687 | 1 | 27.97 GB | 96.5% | 137.7 s | completes, over 90% |
| 32,687 | 10 | 27.97 GB | 96.5% | 133.7 s | completes, over 90% |

Pressure was normal in all 93 samples and swap stayed flat.
On the decisio server under the same limit (prefix cache cut to 0 MiB to fit): a 32,379-token state completed with an MLX peak of 27.80 GB, 95.9% of the limit, in 125 s; a 16,425-token state peaked at 23.88 GB, 82.4%, in 55 s.
The longest fitting length by the rule is 16,383 tokens, the same as the 6-bit conversion's documented length.

## Latency (reported, not gated)

Server time (`x-decisio-server-ms`), median of 5 new documents per length, each asked two different questions in turn:

| State | First question | Later question |
| --- | --- | --- |
| 3,000 tokens | 9.36 s | 9.55 s |
| 16,000 tokens | 66.3 s | 66.3 s |
| 32,000 tokens | 152.1 s | 149.9 s |

The Mac was not quiet: a desktop renderer and the window server used about 40% and 37% CPU throughout the 60 minutes the quiet rule waited, and the cells ran anyway, with the load recorded per document (`latency.json`).
The later question re-reads the document because the prefix cache was cut to 0 MiB under the 27 GiB limit.
With a 1,789 or 2,048 MiB cache a 3,000-token state's entry would fit and a later question would come from the cache; that was not measured.

## Not measured

No 48 GB Mac; the 6-bit conversion's MMLU-Pro; the later-question latency with a prefix cache; GPQA Diamond, BANKING77 and CLINC150+OOS in the Decision Index.
