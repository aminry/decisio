<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# `--concurrency batched` on one RTX 5090: drift, a lone request, throughput

What this shows: decisio 0.23.0's batched mode (with the FP8 12B base of the next release) on one RTX 5090 (32 GB, sm_120, 575 W), vLLM 0.31.0, the engine in the server's process, the Gemma 4 12B in FP8 (`--base gemma-4-12b-fp8`, the stored conversion) and in bf16 (`--base gemma-4-12b`), both at `--max-model-len 4096` and 0.95 of the card (the only length the bf16 12B starts at on 32 GB), 2026-10-21.

## Drift across clients (`drift_fp8.json`, `drift_bf16.json`, `drift_and_g2.txt`)

The 200 single-question requests sent to one batched server one at a time (passes sA, sB, sC) and by 4 and 16 concurrent clients (c4, c16), each answer compared with the same server's one-at-a-time pass sB.

| | 4 clients | 16 clients | two passes one at a time |
| --- | --- | --- | --- |
| FP8 12B | 196 of 200 bit for bit; largest 7.5e-05 | 194 of 200; largest 0.00038; no choice changed | 200 of 200 |
| bf16 12B | 19 of 200; largest 0.064; 1 choice changed | 31 of 200; largest 0.073; 2 choices changed | 81 of 200; largest 0.017 (the 12B's prefix-cache noise) |

## A lone request (`drift_and_g2.txt`)

On a batched server a request that arrives alone is the serial server's answer bit for bit: the batched server's first and second one-at-a-time passes against a serial server's first and second, phase for phase, 200 of 200 on the FP8 12B and on the bf16 12B; on one engine with the mode switched between passes, 200 of 200 on the FP8 12B.
On the bf16 12B consecutive passes alternate between two states whatever the mode (two serial passes agree on 81 of 200), so the one-engine switch, which changed the mode in step with that alternation, is not the comparison for the bf16 12B; the phase-for-phase one is.

## Throughput (`throughput.json`, `throughput.txt`)

Requests per second from N concurrent clients; fresh: every request about a new state of about 1,000 tokens; mix: half about one of 8 states already in the cache.

| Clients | FP8 serial | FP8 batched | bf16 serial | bf16 batched |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.20 (mix 12.8) | 7.05 (12.1) | 4.67 (4.88) | 4.64 (4.93) |
| 4 | 6.73 (10.8) | 3.64 (15.0) | | 3.38 (5.18) |
| 8 | 6.66 (11.0) | 3.75 (5.38) | 4.70 (4.87) | 2.15 (5.50) |
| 16 | 6.68 (11.0) | 2.94 (11.9) | 4.60 (4.77) | 1.83 (3.34) |
| 32 | 6.66 (10.9) | 8.56 (17.1) | 4.58 | 2.79 (3.27) |

Serial throughput is flat with the clients and its 95th-percentile server time stays within about 2 percent of the median.
Batched mode is slower than serial on fresh states at 4 to 16 clients: some engine calls take seconds (up to 29 s for one call of one request) while the card is idle, and every request waiting behind them waits.
In those stalls one server thread uses a full core while the others wait on Python's interpreter lock; full garbage collections of 250 to 520 ms are part of it (and are not removed by `gc.freeze()` after start-up), turning vLLM's asynchronous scheduling off or the boundary registration off does not change it, and no kernel is compiled during them.
The rest of the cause is not found yet.
