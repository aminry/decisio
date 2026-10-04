# The Gemma base beside the Qwen base, on one card in one session (2026-10-04)

Both bases ran on the same code, card and items, one server at a time, each started with `--base` and its own defaults plus `--debug-readout`.
Pairs are over the same items; intervals are 95% paired bootstraps (10,000 resamples, seed 0; I_open 2,000).
The gates were pre-registered before the session.

## Paired, Gemma minus Qwen

| Measure | Gemma 4 12B | Qwen3.6-35B-A3B | Gemma minus Qwen |
| --- | ---: | ---: | --- |
| JevBench 231, correct | 200 | 200 | 0.0 points [-4.3, +4.3] |
| JevBench v1.5, I_open (equal types) | 64.4 | 49.4 | +15.0 [+7.0, +23.6] |
| v1.5 by type: choice / yes/no / score | 80.6 / 49.0 / 63.7 | 77.3 / 13.7 / 57.2 | |
| Yes/no answers between 0.20 and 0.80 (74) | 15% | 39% | |
| JevBench choice ECE | 0.064 | 0.026 | |
| Suite 1,400, accuracy | 0.735 | 0.770 | -3.5 points [-5.5, -1.6] |
| Suite, ECE (tie-robust) | 0.029 | 0.033 | |
| Decision Index BANKING77, accuracy (3,080) | 0.741 | 0.755 | -1.4 [-2.7, -0.1] |
| Decision Index CLINC150+OOS, accuracy (5,500) | 0.872 | 0.827 | +4.5 [+3.6, +5.5] |
| Decision Index GPQA Diamond, accuracy (198) | 0.374 | 0.510 | -13.6 [-21.7, -5.6] |
| Decision Index MMLU-Pro, accuracy (12,032) | 0.549 | 0.613 | -6.4 [-7.2, -5.5] |
| Intent heads, six draws: BANKING77 head (plain) | 0.832 (0.747) | 0.840 (0.747) | |
| Intent heads, six draws: CLINC150 head (plain) | 0.908 (0.850) | 0.912 (0.820) | |
| Registration, BANKING77 / CLINC150 | 195 s / 441 s | 165 s / 346 s | |
| Latency, one question, cached state (server p50) | 26.7 ms | 20.9 ms | |
| Latency, one question, new state 300 / 1,000 / 3,000 | 39.1 / 102.4 / 293.5 ms | 48.4 / 51.8 / 91.1 ms | |
| Latency, four questions, new state 300 / 1,000 / 3,000 | 131.0 / 195.8 / 391.6 ms | 110.0 / 113.8 / 157.4 ms | |
| Weights / KV cache at the 0.90 share | 22.8 / 57.4 GiB | 33.3 / 47.0 GiB | |

## Gates for the Gemma base

| Gate | Result |
| --- | --- |
| Conformance C2 / C3 / C4 | pass / pass / pass (C4 max \|dp\| 0) |
| The head is exact | 250 of 250 intent items bit-identical in log space (`samelog_gemma.json`, after the session); the session's check read 5.55e-17, the rounding of the exponential it took (`sameforward_gemma.json`) |
| Intent heads, six draws | gain +8.6 points (BANKING77), +5.8 (CLINC150); BANKING77's draw 0 reproduced every row |
| Several questions in one request | 272 of 272 four-question requests, 5 repeats, the same choice as alone; largest probability difference 0.035 |
| Prefix sharing | questions 2 to 4 read at least P minus 64 tokens from the cache, 9 of 9 (the pipeline's own verdict used the 16-token hashing step and printed FAIL) |
| Feature checks F1 to F6 | pass |
| The Qwen base unchanged | its plain suite readouts choose what `runs/2026-10-03_qiv-default` chose on 1,400 of 1,400 items, largest difference 1.1e-16 |

The Qwen base's own gates: conformance C2 and C3 pass; C4 failed in the session because the check applied one temperature to every type while choice questions are served at their own, and passed (max |dp| 0) when rerun with the fixed check (#60, `conformance_qwen_c4fix.json.gz`); same-forward 3.17e-8, travel 272 of 272 (max |dp| 0), cache, features and intents pass.

## The Gemma base under the Qwen base's prompt (reported, not gated)

Gemma with the Qwen base's prompt (no system turn, "Answer:" prefill, one token per label, yes/no as named letters), its temperature fitted on its own suite readouts the same way (3.396; no separate choice temperature), against the Gemma base as served:

| Measure | Under the Qwen prompt | As served | Difference |
| --- | ---: | ---: | --- |
| Suite accuracy | 0.736 | 0.735 | +0.14 points [-1.21, +1.43] |
| Suite ECE (tie-robust) | 0.041 | 0.029 | |
| JevBench correct | 204 | 200 | +1.7 [-1.3, +4.8] |
| JevBench hard tier | 86 | 82 | +3.6 [-1.8, +9.0] |
| I_open | 65.8 | 64.4 | +1.3 [-5.4, +8.4] |
| JevBench choice ECE | 0.074 | 0.064 | |

Level on accuracy, worse on calibration: the Gemma base keeps its own prompt.

## Repeatability (`repro.json`)

The Gemma base's logits are bf16 after its soft cap; near the top they lie on a grid of 0.0625 to 0.125.
Within this session, JevBench answers read on first contact and read again later differed by up to 0.035 (49 of 231 items, no choice changed); the Qwen base's were identical.
Against an earlier session on another card of the same type, with identical prompts and temperature, they differed by up to 0.128 (212 of 231 items) and two near-tied choices changed, one from right to wrong (201 correct then, 200 now).

## The 32 GB fit (simulated)

With the engine's share cut to 0.294 of the card (28.8 GB, vLLM's 0.90 of 32 GB), the Gemma base did not start at 32,768 tokens (0.16 GiB left for the KV cache, 10.5 GiB needed) or at 16,384 (no memory for cache blocks).

## After the session

- `/health` cache units (`health_*_cache_units.json`): Gemma `cache_hit_unit` 64 and `hash_unit` 16, `match_unit` 16 unchanged; Qwen 1,056 for all, and its cache rows identical to the session's (`cache_qwen_cache_units.json`).
- The head's log-space check and the Qwen C4 rerun, above.
