<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the decisio project -->

# Design: an MLX engine for the text route on a Mac

`--backend mlx` serves decisio's text route on Apple silicon from an MLX conversion of the served checkpoint.
Every route and feature of the text route works unchanged:
- the letters readout;
- the state prefix shared across a request's questions;
- the global temperature;
- task registration with calibration and the intent head;
- abstention;
- the rendering rules and the tie-break.

The image route, packed mode, LoRA adapters and the second-engine head are refused with this backend.

    uv sync --extra mlx
    uv run python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-6bit

The Mac default is the 6-bit conversion; the 4-bit one is the documented option for 32 GB machines; the 8-bit one is not shipped (Gates below).

## What is replaced, and what is not

`MLXLettersEngine` (`decisio.serve.mlx_engine`) subclasses `vllm_engine.LettersEngine`, as the CPU stand-in does, and replaces only model loading and the forward pass.
The request path before that is the served code: state tokenization, the split into state and question, the padding, the label tokens, derived questions, both routes, the temperature, tasks and abstention.

The prompts are the vLLM path's without the padding (below):
- **Tokenizer:** the official one (`--tokenizer`, default `Qwen/Qwen3.6-35B-A3B-FP8`). The mlx-community conversions ship a different `tokenizer.json` and `tokenizer_config.json`; the chat template and the vocabulary are the same. `GET /health` reports the tokenizer file's sha256 and whether it is the official one.
- **Padding:** none (`--pad-policy none`, the MLX default, `resolve_pad_policy`). MLX's cache does not need the served front padding to vLLM's 1,056-token block, which was about 70% of a single question's time (926 of 1,056 prefix tokens at the median).
  - The prompts are the vLLM path's without the padding, and a question asked alone and inside a request is one prompt.
  - `--pad-policy always` restores the served prompts byte for byte.

## The forward pass

For each request, the state prefix (the tokens every question shares, padding included) is prefilled once into a fresh prompt cache, in chunks of 2,048 tokens, and evaluated.
Each question then continues from a copy of that cache, one question at a time.

Copying the cache depends on the layer type:
- **Gated DeltaNet layers:** their cache entries (the convolution and recurrent states) are replaced by each forward, never written into, so a copy shares them.
- **Attention layers:** they write their keys and values into a buffer in place, so a copy slices the buffer to the prefix, and its first write allocates a new one.

Every question goes through the prefix, single-question requests included:
- **Isolation:** a question's answer does not depend on what else is in its request. Each question alone, the same question beside 16 others, and a repeat all agree bit for bit (`tests/mlx/serving_gates.py`, G2).
- **No batch effect:** the vLLM engine scores a request's questions in one batch and cannot make that promise (EVAL_CARD.md, the batch-forward finding).

The letters are read at the last position: the softmax over the label logits (bf16 out of the output layer, then float64).
The final-norm hidden state at the same position comes out of the same forward, so the intent head (`MLXHiddenReadout`) fits and serves on exactly the served readout, with no extra request and no second weight copy.

With the vLLM engine's newer flags:
- **`--multi-question`:** has no effect here, because the MLX engine always scores questions one at a time from the shared prefix, which is what `sequential` does on vLLM.
- **`--pad-policy`:** the row builder is shared with vLLM; the MLX default is `none` (below), vLLM's `always`.

## The MLX default: no padding (`--pad-policy none`)

Gated at 6 bits against the FP8 records with the backend's full gate set (`runs/2026-10-02_mlx-backend`: `6bit`, `6bit_shared`, `6bit_none`; pre-registered).

| | Padded (`always`) | `shared` | `none` (the default) |
| --- | --- | --- | --- |
| Prompts vs the vLLM path | identical | the padding of single-question requests removed | the padding of every request removed |
| A question alone vs inside a request | bit for bit | differs (max 0.021, no flip) | bit for bit |
| Suite accuracy vs FP8 0.762 (points) | +0.4 [-0.5, +1.2] | -0.1 [-1.4, +1.1] | -0.1 [-1.4, +1.1] |
| Top-answer flips vs FP8 | 56 of 1,400 | 98 | 98 |
| Pooled ECE vs FP8 0.020 (gate: within 0.01) | 0.025 | 0.030 (+0.0097) | 0.030 (+0.0097) |
| Brier vs FP8 0.329 | -0.004 [-0.009, +0.001] | +0.011 [+0.005, +0.018] | +0.011 [+0.005, +0.018] |
| Intent heads BANKING77 / CLINC150 (points) | -1.3 / +2.0 | 0.0 / +3.3 | 0.0 / +3.3 |
| A head question inside a two-question request | - | no answer changed | bit for bit (250 of 250) |
| Conformance C2-C4; head arithmetic | pass; exact | pass; exact | pass; exact |
| One question, server time p50 / p95 | 482 / 490 ms | 231 / 306 ms | 232 / 307 ms |

**How the policies compare:**
- **Single-question requests:** `shared` and `none` send the same prompt, so their suite and head results are identical.
- **Isolation:** `none` keeps a question's answer the same whether it is asked alone or with others, which `shared` gives up.
- **Against padded:** both cost more flips and a Brier score worse by 0.011, and halve a single question's time.
- **Multi-question requests:** without padding, the state prefill is about 130 tokens instead of 1,056. The latency run for those cells ran under heavy foreign load, so its multi-question numbers are not a measurement of this.

## The cross-request prefix cache

The evaluated cache of each state prefix is kept across requests (`PrefixCache`): least recently used first out, `--prefix-cache-mb`, default 2,048, 0 is off.

**Exact by construction:**
- an entry is returned only for exactly the token ids it was computed for (compared in full, not by hash alone);
- it is never written into, since every question continues from a copy;
- the forward is deterministic.

**Measured** on 6 bits with the defaults (no padding; `runs/2026-10-02_mlx-backend`, `6bit_prefix_cache`):
- the 1,400 suite answers with the cache on are bit-identical to those without it;
- on 30 repeated states every hit's answers equal its miss's.

| Questions per request | First request (miss), server p50 | Repeated state (hit), server p50 |
| --- | --- | --- |
| 1 | 264 ms | 85 ms |
| 10 | 939 ms | 734 ms |
| 100 | 10.70 s | 10.56 s |

**What the cache saves:**
- An entry holds the state's recurrent and attention caches, about 67 MB at 6 bits (2,048 MB kept 30 states).
- Without padding, a request with any number of questions on the same state hits the same entry.
- A hit saves the state's prefill and nothing else, so it matters for a few questions on a repeated state, and little at 100 questions, where the questions' own forwards dominate.

## The prefix path against the whole prompt

Scoring a question from the prefix cache is not bit-identical to running its whole prompt in one pass.
The bf16 arithmetic is chunked differently, which is the same effect vLLM's cached-against-cold gate (G1) bounds.
The MLX engine always serves the prefix path, so it never serves the other one; the difference is recorded, not gated (numbers below).

## Gates

Measured on an Apple M5 Pro (64 GB) against the FP8 served default's records (`runs/2026-09-30_plugin-verification`), pre-registered before any measurement.
The record is `runs/2026-10-02_mlx-backend` (`manifest.json`, `summary.md`).
FP8 reference: `runs/2026-09-30_plugin-verification`.
Intervals are paired 95% bootstrap intervals over items.

| Gate | 8-bit | 6-bit | 4-bit |
| --- | --- | --- | --- |
| Prompts identical to the served path (1,400 suite items) | 1,400 | 1,400 | 1,400 |
| Isolation (G2): alone, beside 16 others, repeated | bit for bit | bit for bit | bit for bit |
| Prefix path vs whole prompt (G1, reported) | max 4.8e-3 | max 1.1e-2 | max 4.1e-2 |
| Conformance C2-C4 | pass | pass | pass |
| Suite accuracy vs FP8 0.762 (points) | +0.3 [-0.6, +1.2] | +0.4 [-0.5, +1.2] | -1.0 [-2.3, +0.3] |
| Top-answer flips vs FP8 | 56 of 1,400 | 56 | 128 |
| Pooled ECE vs FP8 0.020 (gate: within 0.01) | 0.030 (fails, by 0.00006) | 0.025 | 0.021 |
| Intent head BANKING77, 3 draws (FP8 0.847) | 0.842, -0.4 [-2.7, +1.6] | 0.833, -1.3 [-4.0, +1.3] | 0.831, -1.6 [-4.2, +0.7] |
| Intent head CLINC150, 3 draws (FP8 0.893) | 0.920, +2.7 [-1.0, +6.7] | 0.913, +2.0 [-1.3, +5.7] | 0.927, +3.3 [0.0, +7.3] |
| Head arithmetic (stored fits; served answers) | exact | exact | exact |
| Peak memory at 32k tokens | 39.4 GB | 30.7 GB | 22.0 GB |
| One question, server time p50 (padded; 232 ms at 6 bits with the default `none`) | 485 ms | 482 ms | 430 ms |
| 100 questions sharing a state, per question p50 | 98 ms | 97 ms | 74 ms |

- **Shipped:**
  - 6-bit, the Mac default: it passes every gate.
  - 4-bit, the documented option for 32 GB machines: twice the flips and a Brier score worse by 0.009 [+0.001, +0.017].
- **Not shipped:** 8-bit, which misses the pre-registered ECE rule (reported as measured) and is no faster or more accurate than 6-bit.
- **Calibration:** the FP8 ECE is in-sample for the served temperature (fitted on the FP8 readouts of this suite), and nothing was refitted for MLX.
- **Speed:** the per-question time does not track the weight size, so it is not bandwidth. Batching a request's questions over the shared prefix and reading only the label rows of the output layer are the next steps (not done here).
- **Registration** of an intent task (770 to 1,500 examples) takes 18 to 50 minutes on the M5 Pro.
