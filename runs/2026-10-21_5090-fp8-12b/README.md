<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The Gemma 4 12B in FP8 on one RTX 5090: identity, memory, answers against bf16, speed

What this shows: the FP8 12B (`--base gemma-4-12b-fp8`) measured on one RTX 5090 (32,607 MiB, driver 615.71.09, 575 W), vLLM 0.31.0, the engine in the server's process, 2026-10-21.
The decisio measured is the local merge of this release's FP8 12B base (c055c2a) with `--concurrency batched` (568526b); nothing in it changes the readout of a single request.

## Identity (`identity.json`, `gates.json`)

The engine's own FP8 tensors (`tests/gpu/fp8_fingerprints.py`), read from a server that quantized the bf16 12B on load and from one that loaded the stored conversion (`python -m decisio.hub_fp8 convert`), differ in zero of 192 fused layers from each other and from the CPU prediction (`python -m decisio.hub_fp8 expected`).
A tuned 12B merged to FP8 on the card (`hub_fp8 merge`, an adapter of the 12B) loaded to the tensors its merge predicted.
A server started on the stored conversion answered every request (`gates.json`, G1), and that conversion is the repository this release pins.

## Memory (`memory_*.txt`, `bf16_ladder.jsonl`)

| | bf16 12B | FP8 12B |
| --- | --- | --- |
| Weights in memory | 22.83 GiB | 12.74 GiB |
| KV cache, 0.90 of the card (the default) | 0.48 GiB: refused at 32,768 (5.82 GiB needed) | 11.66 GiB: serves 65,536, the ceiling |
| KV cache, 0.95 of the card | 2.05 to 3.05 GiB: refused at 32,768, 16,384 and 8,192; starts at 4,096 | 12.15 GiB at 4,096 |

## Answers against bf16 (`answers_fp8_vs_bf16.json`, `answers_fp8_vs_bf16.txt`)

The 200 single-question requests on two fresh starts of each precision, all four at 4,096 tokens and 0.95 of the card, the same attention backend (Triton):
two starts of the same precision gave the same answers bit for bit (200 of 200, both precisions); FP8 against bf16, the top answer the same on 192 of 200 (96%), the largest probability difference 0.438, the mean 0.031.
The FP8 12B is its own function: its temperatures are the bf16 12B's until refitted, and certificates made on bf16 are suspended on it.

## Speed (`latency.json`)

Server time for one question on 40 fresh texts sent one at a time to each start, paired text by text: the FP8 12B's was 0.64 times the bf16 12B's (10th to 90th percentile 0.60 to 0.68), median 88.4 against 135.2 ms, 90th percentile 118.6 against 196.8 ms.
