<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# The batched-mode stall and the tokenizer fix, on one RTX 5090

What this shows: why `--concurrency batched` (0.23.0) was slower than serial on requests about new states, and that the engine's tokenizer used one call at a time fixes it.
One RTX 5090 (32 GB, driver 580.173.02, 600 W), vLLM 0.31.0, the engine in the server's process, the FP8 12B (`--base gemma-4-12b-fp8`, the stored conversion) at 0.95 of the card and `--max-model-len 4096`, every request about a new state of about 1,000 tokens, 2026-10-21.

## The cause (`holds_summary.txt`)

Each server ran under a sampler thread of its own, which woke every 20 ms and, when it woke more than 150 ms late (something held Python's interpreter lock), wrote what every thread was running just before and just after.
In batched mode at 8 and 16 clients there were 14 such holds, 56 s in all, the longest 11.6 s; in every one, request threads were inside `copy.deepcopy` called from vLLM's tokenizer (`vllm/tokenizers/hf.py`, `encode`), while the engine's thread waited for the lock mid-step with the card idle.
vLLM 0.31.0 makes the Hugging Face tokenizer thread-safe with a pool of copies, and a caller that finds every copy in use makes a new one with `copy.deepcopy` of the whole tokenizer (262,144 tokens for Gemma 4).
A batched engine call answers up to 16 requests at once, so their threads tokenize at once (the usage count) and each deep-copies the tokenizer; a serial server's requests end one at a time, and it had no hold at all.
Garbage collections ran alike in both modes (29 to 39 over 20 ms).

## The fix (`batched_fix_load_*.json`)

The engine's tokenizer is wrapped so that its calls run one at a time (`decisio.serve.tokenizer_lock.SerialTokenizer`): vLLM's pool never finds every copy in use and never copies.
On the same card, server and load:

| Clients | batched before (`batched_load_*.json`) | batched with the fix | serial (`serial_load_16.json`) |
| ---: | ---: | ---: | ---: |
| 8 | 1.64 requests a second; server p50 0.90 s, p95 16.7 s | **8.87; p50 0.89 s, p95 0.92 s** | |
| 16 | 2.25; p50 1.72 s, p95 24.8 s | **9.18; p50 1.74 s, p95 1.77 s** | 7.19; p50 2.21 s, p95 2.27 s |

With the fix there was no hold over 150 ms.
The decisio measured is 0.23.0's batched mode with the FP8 12B base of the next release (the local merge of c055c2a, #269, and 568526b, #270), with this module added and the one line that wraps the tokenizer changed.
