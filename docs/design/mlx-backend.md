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

The Mac default is the 6-bit conversion; the 4-bit one is documented for Macs with more than 32 GB, or 32 GB with the GPU memory limit raised; the 8-bit one is not shipped (Gates below).

**Two prompt eras.** The Qwen figures in the sections on padding, the prefix cache and the gates were measured under the earlier served prompt (the compact layout, yes/no as words, T 1.307; `runs/2026-10-02_mlx-backend`).
The served prompt changed in #54 (the spaced layout, yes/no as named letters, T 1.506 with choice T 1.370), and the 6-bit gates were run again under it (`runs/2026-10-03_mlx-regate`).
Those are the current figures, in [Under the current served default](#under-the-current-served-default), and the ones `docs/running.md` quotes.

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

Gated at 6 bits against the FP8 records with the backend's full gate set, under the earlier served prompt (`runs/2026-10-02_mlx-backend`: `6bit`, `6bit_shared`, `6bit_none`; pre-registered).

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
- **Multi-question requests:** without padding, the state prefill is about 130 tokens instead of 1,056. Re-measured on `main` with the cache off (`6bit_none/latency_quiet`; the passes agreed within 5%), a request costs, server p50:

  | Questions | Padded | `none` |
  | --- | --- | --- |
  | 10 | 1,148 ms | 892 ms |
  | 100 | 9,747 ms | 9,484 ms |
  | 100, 8,000-token state | 14,489 ms | 14,288 ms |

  That is 200-260 ms less per request, the padding's prefill.

## The cross-request prefix cache

The evaluated cache of each state prefix is kept across requests (`PrefixCache`): least recently used first out, `--prefix-cache-mb`, default 2,048, 0 is off.

**Exact by construction:**
- an entry is returned only for exactly the token ids it was computed for (compared in full, not by hash alone);
- it is never written into, since every question continues from a copy;
- the forward is deterministic.

**Measured** on 6 bits with the defaults, under the earlier served prompt (no padding; `runs/2026-10-02_mlx-backend`, `6bit_prefix_cache`):
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

Measured on an Apple M5 Pro (64 GB) under the earlier served prompt, against that default's FP8 records (`runs/2026-09-30_plugin-verification`), pre-registered before any measurement.
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
  - 4-bit, documented then as the option for 32 GB machines: twice the flips and a Brier score worse by 0.009 [+0.001, +0.017]. Under the current default it is documented for Macs above 32 GB (below).
- **Not shipped:** 8-bit, which misses the pre-registered ECE rule (reported as measured) and is no faster or more accurate than 6-bit.
- **Calibration:** the FP8 ECE is in-sample for the served temperature (fitted on the FP8 readouts of this suite), and nothing was refitted for MLX.
- **Speed:** the per-question time does not track the weight size, so it is not bandwidth. Batching a request's questions over the shared prefix and reading only the label rows of the output layer are the next steps (not done here).
- **Registration** of an intent task (770 to 1,500 examples) takes 18 to 50 minutes on the M5 Pro.

## Under the current served default

The 6-bit gates run again on the same Mac under #54's served prompt, against the FP8 record of that default (`runs/2026-10-03_mlx-regate`, pre-registered; its `summary.md` has the intervals and the files).

| Gate | MLX 6-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| Suite accuracy (1,400) | 0.7721 | 0.7700 | pass: +0.2 [-1.0, +1.4] points |
| Top-answer flips | 102 of 1,400 | | counted |
| Pooled ECE (gate: within 0.01) | 0.0451 | 0.0327 | fails: +0.0125 |
| Intent heads, six draws, BANKING77 / CLINC150 | 0.841 / 0.912 | 0.840 / 0.912 | pass / pass |
| Conformance C2 to C4 | pass | | pass |

- **The ECE fail changes no setting.** The FP8 figure is in-sample for the served temperatures. An MLX-specific refit, judged on JevBench's 231 held-out items, made log loss worse, and at the served temperatures MLX is as well calibrated as FP8 there.
- **Server time, median,** with no other job running (`runs/2026-10-07_mlx-quiet-latency`): 258 ms for one question on a new state, 108 ms on a state seen before, and 123 ms per question when 100 share a state. macOS's indexing services ran during the passes, so the driver's idle rule did not hold; the passes agree within 3%.

The 4-bit conversion was gated the same way (`runs/2026-10-06_mlx-qwen-4bit-regate`):

| Gate | MLX 4-bit | FP8 | Verdict |
| --- | --- | --- | --- |
| Suite accuracy (1,400) | 0.7643 | 0.7700 | pass: -0.6 [-1.9, +0.8] points |
| Top-answer flips | 135 of 1,400 | | counted |
| Pooled ECE (gate: within 0.01) | 0.0297 | 0.0327 | pass |
| Intent heads, six draws, BANKING77 / CLINC150 | 0.834 / 0.928 | 0.840 / 0.912 | pass / pass |
| Conformance C2 to C4 | pass | | pass |

- **Memory:** GB are 1e9 bytes (the driver divides MLX's bytes by 1e9). The weights take 19.5 GB, and a request peaks at 21.6, 22.0 and 22.7 GB at about 8k, 16k and 32k tokens of state. Two thirds of a 32 GB Mac's memory (32 GiB, 34.4 GB) is 22.9 GB, a conservative stand-in for its default GPU limit, which was not measured on a 32 GB Mac; every peak is under it, by 1.3, 0.9 and 0.2 GB. At 32k that is 0.9% of the line, almost nothing for the system and other apps, so the conversion is documented for Macs with more than 32 GB, or 32 GB with the GPU memory limit raised; a 32 GB Mac runs the Gemma base (15.1 GB at 32k, 7.8 GB under the line).

## The Gemma base

The engine serves the Gemma base from an MLX conversion of `google/gemma-4-12B-it` with three additions; nothing in the Qwen path changes (50 suite answers bit-identical, `runs/2026-10-05_mlx-gemma/`).

- **Sliding-window layers.** Forty of Gemma 4's 48 layers keep a ring buffer of the last 1,024 tokens (mlx-lm's `RotatingKVCache`). `copy_cache` copies it whole with its write position into new array objects. A question's one-token steps write into the copy in place and longer ones concatenate; either way the prefix stays as it was.
- **The final-logit softcap.** Gemma 4 caps its logits at 30 (`tanh(z / 30) * 30`). The label logits pass through the same cap, in the logits' dtype, as mlx-lm applies it, and the intent head's base readout is that capped readout.
  - The readout takes the output layer at the last position alone. On quantised weights that matrix-vector product rounds differently from the model's own matrix-matrix product, by at most two bf16 steps at the label tokens; the tests bound it there.
- **The base's tokenizer.** The prompts are built with `google/gemma-4-12B-it` at the base's pinned revision. The conversions' tokenizer is the same file, but their chat template predates Google's 2026-07-15 fix.

The base and its settings (the system turn, the template's answer slot, summed label forms, the temperature, the yes/no rendering) come from `decisio.families`, as on vLLM.
The 6-bit conversion is the default; the 4-bit one fails the suite-accuracy gate, and the 8-bit one adds nothing over 6-bit on the suite.
