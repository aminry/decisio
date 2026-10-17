# The stored Gemma 4 31B did not repeat across two fresh starts in one card session (2026-10-16)

**Status: complete; the cause is not known.**
The stored FP8 31B (`tachara-ai/decisio-gemma-4-31b` at its 0.14.0 pin, decisio 0.14.0, vLLM 0.31.0, `--max-model-len 32768`, `VLLM_BATCH_INVARIANT` unset) was started twice and read twice, on a 776-item set of multiple-choice questions and a 500-item development set of public-source questions of three types.
The flags, the revision, the compiled function and the graph captures of the two starts were identical in the server logs.
Start 1 compiled the model and start 2 loaded the saved function, as in the earlier session where the same pair repeated exactly.

## How far the answers moved

| Pair of fresh starts | Items | Differ at all | Differ by more than 1e-3 | Median of the largest difference over the options | Largest | Top answers changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stored 31B, 776-item read | 776 | 776 | 717 | 0.0098 | 0.670 | 24 |
| Stored 31B, 500-item development read | 500 | 500 | 406 | 0.0045 | 0.292 | 10 |
| Same card, same session: R2 (a stored FP8 build of Google's QAT-trained weights), no adapter named in the request, 776 items | 776 | 0 | 0 | 0 | 0 | 0 |
| Same, 500-item development read | 500 | 0 | 0 | 0 | 0 | 0 |
| Same card, same session: R2 with an adapter, 500-item development read | 500 | 500 | 455 | 0.0071 | 0.050 | 5 |
| Earlier session (another box), stored 31B, same 776-item read | 776 | 0 | 0 | 0 | 0 | 0 |

Every item moved a little, on both reads, so the difference is in the whole server's numeric path, not in a few items or in the order of the requests.
The two sessions ran on the same card model at 600 W; the earlier one on an AMD EPYC 9334 with driver 615.71.09, this one on a Threadripper PRO 7965WX with driver 595.99.02.
The logs show no other difference between the sessions' servers, and the same session's R2 server repeated exactly, so the box is a candidate and not a cause.

## What a restart does to a confidence threshold

Raw served probabilities of the 776-item read (not a registered task): the top probability against a threshold t, the items at or above it counted as certified.

| t | At or above on start 1 | On start 2 | On one start only | At or above on both, with a different top answer |
| ---: | ---: | ---: | ---: | ---: |
| 0.80 | 535 | 533 | 26 | 0 |
| 0.90 | 415 | 412 | 35 | 0 |
| 0.95 | 290 | 288 | 24 | 0 |
| 0.98 | 67 | 69 | 8 | 0 |
| 0.99 | 13 | 10 | 5 | 0 |

The size of the set does not move, the items that swap sit at the threshold, and from t = 0.80 up no item above the threshold on both starts changed its top answer (at 0.5, 7 did; at 0.7, 1).
The 500-item read gives the same picture (`pair_statistics.json`, with the thresholds).
This is one pair on one box; a registered task's temperatures and calibration were not applied, and labels were not read.
